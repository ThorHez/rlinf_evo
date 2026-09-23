# Copyright 2026 The RLinf Authors.
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

"""Bounded OpenCV/V4L2 capture for UVC cameras and HDMI capture cards."""

import glob
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .base import BaseCamera, Camera, CameraInfo


def _configure_capture(capture: Any, info: CameraInfo) -> None:
    import cv2

    expected = {
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*info.pixel_format),
        cv2.CAP_PROP_FRAME_WIDTH: info.resolution[0],
        cv2.CAP_PROP_FRAME_HEIGHT: info.resolution[1],
        cv2.CAP_PROP_FPS: info.fps,
    }
    for prop, value in expected.items():
        capture.set(prop, value)
    capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    # A request only: V4L2 devices can ignore this property.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for prop, value in expected.items():
        actual = capture.get(prop)
        tolerance = 0.1 if prop == cv2.CAP_PROP_FPS else 0
        if not np.isfinite(actual) or abs(actual - value) > tolerance:
            raise RuntimeError(
                f"UVC {info.serial_number}: property {prop} is {actual}, "
                f"requested {value}. Select a supported capture mode."
            )


def _put_latest(frames: Any, packet: tuple) -> None:
    """Keep bounded IPC storage even if the control process stops reading."""
    while True:
        try:
            frames.put_nowait(packet)
            return
        except queue.Full:
            try:
                frames.get(timeout=0.05)
            except queue.Empty:
                pass


def _capture_worker(
    info: CameraInfo,
    ready: Any,
    frames: Any,
    stop: Any,
    capture_factory: Optional[Callable],
) -> None:
    """Own the native handle so a blocked driver can be terminated safely."""
    import cv2

    capture = None
    announced = False
    try:
        factory = capture_factory or cv2.VideoCapture
        capture = factory(info.serial_number, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open UVC device {info.serial_number}.")
        _configure_capture(capture, info)
        ready.send(None)
        announced = True
        while not stop.is_set():
            success, frame = capture.read()
            received_at = time.monotonic()
            width, height = info.resolution
            if (
                not success
                or not isinstance(frame, np.ndarray)
                or frame.shape != (height, width, 3)
                or frame.dtype != np.uint8
            ):
                raise RuntimeError("UVC read failed or returned an invalid BGR frame.")
            _put_latest(frames, (None, received_at, frame))
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if announced:
            _put_latest(frames, (message, 0.0, None))
        else:
            ready.send(message)
    finally:
        if capture is not None:
            capture.release()
        ready.close()


@dataclass
class _CaptureSession:
    process: Any
    frames: Any
    stop: Any


@Camera.register("uvc")
class UVCCamera(BaseCamera):
    """Capture BGR uint8 frames from an explicitly selected Linux video device.

    ``serial_number`` must be an absolute ``/dev/`` path, preferably a stable
    ``/dev/v4l/by-id/`` link. Resolution, FPS and FOURCC are verified on open.
    Construction opens no device. A spawned process owns OpenCV so native open,
    read and release hangs have bounded lifetimes. ``capture_factory`` optionally
    replaces VideoCapture; it must be picklable for the spawn context.
    """

    SDK = "cv2"

    def __init__(
        self,
        camera_info: CameraInfo,
        *,
        open_timeout: float = 15.0,
        read_timeout: float = 1.0,
        capture_factory: Optional[Callable] = None,
    ) -> None:
        super().__init__(camera_info)
        if camera_info.enable_depth:
            raise ValueError("UVC capture does not provide depth.")
        path = camera_info.serial_number
        if (
            not isinstance(path, str)
            or not os.path.isabs(path)
            or not path.startswith("/dev/")
        ):
            raise ValueError("UVC serial_number must be an absolute /dev/ device path.")
        if (
            len(camera_info.resolution) != 2
            or any(int(v) != v or v <= 0 for v in camera_info.resolution)
            or not np.isfinite(camera_info.fps)
            or camera_info.fps <= 0
        ):
            raise ValueError("UVC resolution and fps must be positive.")
        if (
            not isinstance(camera_info.pixel_format, str)
            or len(camera_info.pixel_format) != 4
            or not camera_info.pixel_format.isascii()
        ):
            raise ValueError("UVC pixel_format must be a four-character ASCII FOURCC.")
        if any(not np.isfinite(v) or v <= 0 for v in (open_timeout, read_timeout)):
            raise ValueError("UVC timeouts must be finite and positive.")
        self._open_timeout = open_timeout
        self._read_timeout = read_timeout
        self._capture_factory = capture_factory

    def _open(self) -> _CaptureSession:
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        frames = context.Queue(maxsize=1)
        stop = context.Event()
        process = context.Process(
            target=_capture_worker,
            args=(self.camera_info, child, frames, stop, self._capture_factory),
            daemon=True,
        )
        session = _CaptureSession(process, frames, stop)
        try:
            process.start()
            child.close()
            if not parent.poll(self._open_timeout):
                raise TimeoutError("UVC open exceeded its timeout.")
            error = parent.recv()
            if error is not None:
                raise RuntimeError(error)
            return session
        except BaseException:
            self._release(session)
            raise
        finally:
            parent.close()
            child.close()

    def _read_frame_with_timestamp(
        self,
    ) -> tuple[bool, Optional[np.ndarray], float]:
        session = self._device
        try:
            error, received_at, frame = session.frames.get(timeout=self._read_timeout)
        except queue.Empty as exc:
            raise TimeoutError("UVC read exceeded its timeout.") from exc
        if error is not None:
            raise RuntimeError(error)
        return True, frame, received_at

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        success, frame, _ = self._read_frame_with_timestamp()
        return success, frame

    def _release(self, device: _CaptureSession) -> None:
        device.stop.set()
        process = device.process
        still_running = False
        if process.pid is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            if not process.is_alive():
                process.close()
            else:
                still_running = True
        device.frames.cancel_join_thread()
        device.frames.close()
        if still_running:
            raise TimeoutError("UVC capture process did not exit after SIGKILL.")

    @classmethod
    def discover(cls) -> set[str]:
        """List stable device paths without opening a video device."""
        return set(glob.glob("/dev/v4l/by-id/*"))
