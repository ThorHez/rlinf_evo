# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import os
import threading
import time
from collections import deque
from typing import Any

from rlinf.utils.logging import get_logger

_logger = get_logger()


class KeyboardListener:
    """Read all accessible evdev keyboards, including newly connected devices.

    Set ``RLINF_KEYBOARD_DEVICE`` to restrict input to one device. Otherwise,
    key presses from all keyboards share one queue.
    """

    REQUIRED_KEY_NAMES = ("KEY_A", "KEY_B", "KEY_C", "KEY_Q")

    def __init__(self) -> None:
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError as exc:
            raise RuntimeError(
                "KeyboardListener requires the 'evdev' package. "
                "Install the real-world extras with evdev support."
            ) from exc

        self._input_device_cls = InputDevice
        self._ecodes = ecodes
        self._list_devices = list_devices

        self.state_lock = threading.Lock()
        self.latest_data = {"key": None}
        # Queue press edges so short taps are not lost between polls.
        self._press_events: deque[str] = deque()
        self._held_keys: dict[tuple[str, str], None] = {}
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._override_path = os.environ.get("RLINF_KEYBOARD_DEVICE")
        self._devices: dict[str, Any] = {}

        self.listener = threading.Thread(
            target=self._listen_loop,
            name="KeyboardListener",
            daemon=True,
        )
        try:
            self._refresh_devices(required=True)
            self.listener.start()
        except BaseException:
            self._stop.set()
            self._close_devices()
            raise
        finally:
            self._ready.set()
        self.last_intervene = 0

    def _refresh_devices(self, *, required: bool = False) -> None:
        paths = (
            [self._override_path]
            if self._override_path
            else sorted(self._list_devices())
        )
        permission_denied_paths: list[str] = []
        for device_path in paths:
            if self._stop.is_set():
                return
            if device_path in self._devices:
                continue
            try:
                device = self._input_device_cls(device_path)
            except OSError as exc:
                if required and self._override_path:
                    raise RuntimeError(
                        "KeyboardListener cannot open "
                        f"RLINF_KEYBOARD_DEVICE='{device_path}': {exc}"
                    ) from exc
                if isinstance(exc, PermissionError):
                    permission_denied_paths.append(device_path)
                # Devices may disappear while discovery is in progress.
                continue

            try:
                is_keyboard = self._is_keyboard_device(device)
            except OSError:
                device.close()
                continue
            except BaseException:
                device.close()
                raise
            if not is_keyboard or self._stop.is_set():
                device.close()
                if required and self._override_path and not is_keyboard:
                    raise RuntimeError(
                        f"RLINF_KEYBOARD_DEVICE='{device_path}' is not a keyboard."
                    )
                continue
            self._devices[device_path] = device
            _logger.info("Listening for keyboard input on %s.", device_path)

        if self._devices or not required:
            return
        if permission_denied_paths:
            denied = ", ".join(permission_denied_paths)
            raise RuntimeError(
                "KeyboardListener could not open any readable keyboard device under "
                f"/dev/input/event*. Permission denied for: {denied}. Grant the runtime "
                "user read access via ACLs, the input group, or udev rules."
            )

        raise RuntimeError(
            "KeyboardListener could not find a readable keyboard device under "
            "/dev/input/event*. Ensure a physical keyboard is connected and the "
            "runtime user has read access to its input device."
        )

    def _is_keyboard_device(self, device: Any) -> bool:
        required_codes = {
            getattr(self._ecodes, key_name) for key_name in self.REQUIRED_KEY_NAMES
        }
        capabilities = device.capabilities(verbose=False)
        supported_key_codes = set(capabilities.get(self._ecodes.EV_KEY, []))
        return required_codes.issubset(supported_key_codes)

    def _listen_loop(self) -> None:
        # Transfer descriptor ownership after initialization, including cleanup
        # if Thread.start() is interrupted before returning to the constructor.
        self._ready.wait()
        next_scan = time.monotonic() + 0.5
        try:
            # evdev reads are non-blocking. One unplugged keyboard must not
            # prevent polling the others while waiting for reconnection.
            while not self._stop.wait(0.01):
                for device_path, device in list(self._devices.items()):
                    try:
                        for event in device.read():
                            if self._stop.is_set():
                                return
                            self._record_event(device_path, event)
                    except BlockingIOError:
                        continue
                    except OSError as exc:
                        if exc.errno != errno.ENODEV:
                            _logger.error(
                                "Keyboard device %s read failed: %s", device_path, exc
                            )
                            raise
                        _logger.warning(
                            "Keyboard device %s disconnected; waiting for keyboards "
                            "to reconnect.",
                            device_path,
                        )
                        self._drop_device(device_path)
                if time.monotonic() >= next_scan:
                    self._refresh_devices()
                    next_scan = time.monotonic() + 0.5
        except Exception as exc:
            if not self._stop.is_set():
                with self.state_lock:
                    self._error = exc
                    self._held_keys.clear()
                    self.latest_data["key"] = None
                _logger.error("Keyboard input failed: %s", exc)
        finally:
            self._close_devices()

    def _record_event(self, device_path: str, event: Any) -> None:
        if event.type != self._ecodes.EV_KEY:
            return
        key = self._event_to_key(event.code)
        if key is None:
            return
        with self.state_lock:
            identity = (device_path, key)
            if event.value in (1, 2):
                self._held_keys.pop(identity, None)
                self._held_keys[identity] = None
                self.latest_data["key"] = key
                if event.value == 1:
                    # Enqueue the initial press only, not key-repeat events.
                    self._press_events.append(key)
            elif event.value == 0:
                self._held_keys.pop(identity, None)
                self._update_held_key()

    def _update_held_key(self) -> None:
        # Called with state_lock held; releases on one keyboard leave the
        # keys held on other keyboards available to get_key().
        self.latest_data["key"] = (
            next(reversed(self._held_keys))[1] if self._held_keys else None
        )

    def _drop_device(self, device_path: str) -> None:
        device = self._devices.pop(device_path)
        try:
            device.close()
        except Exception as exc:
            _logger.warning("Could not close keyboard %s: %s", device_path, exc)
        with self.state_lock:
            for identity in list(self._held_keys):
                if identity[0] == device_path:
                    del self._held_keys[identity]
            self._update_held_key()

    def _close_devices(self) -> None:
        for device_path in list(self._devices):
            self._drop_device(device_path)

    @property
    def error(self) -> Exception | None:
        """Return an unrecoverable input error for the control loop to handle."""
        with self.state_lock:
            return self._error

    def close(self) -> None:
        """Stop discovery and polling, and release all input descriptors."""
        self._stop.set()
        # The polling thread owns the descriptors and closes them in finally.
        if self.listener is not threading.current_thread():
            self.listener.join(timeout=1.0)
            if self.listener.is_alive():
                raise RuntimeError("Keyboard input thread did not stop")
        with self.state_lock:
            self._held_keys.clear()
            self.latest_data["key"] = None
            self._press_events.clear()

    def _event_to_key(self, key_code: int) -> str | None:
        key_name = self._ecodes.bytype[self._ecodes.EV_KEY].get(key_code)
        if isinstance(key_name, list):
            key_name = key_name[0]
        if not isinstance(key_name, str):
            return None

        if key_name.startswith("KEY_"):
            normalized_key = key_name.removeprefix("KEY_").lower()
            if len(normalized_key) == 1:
                return normalized_key
            return f"Key.{normalized_key}"
        return key_name.lower()

    def get_key(self) -> str | None:
        """Return the most recently active key still held, or ``None``.

        Only reflects held state; fast taps may be missed between polls.
        Use :meth:`pop_pressed_keys` when you need lossless press detection.
        """
        with self.state_lock:
            return self.latest_data["key"]

    def pop_pressed_keys(self) -> list[str]:
        """Return and clear key presses recorded since the previous call.

        Key-repeat events are omitted. The operation is thread-safe and
        non-blocking.
        """
        with self.state_lock:
            if not self._press_events:
                return []
            pressed = list(self._press_events)
            self._press_events.clear()
            return pressed
