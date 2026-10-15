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

"""Local ARX JSON protocol and SDK owner, importable with Python 3.10.

The SDK service imports this file directly, without importing RLinf or torch.
Timestamps use the host monotonic clock; clients must run on the same host.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1


class Arx5Error(RuntimeError):
    """A rejected command, unavailable service, or invalid hardware state."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def vector(value: Any, length: int, name: str) -> np.ndarray:
    """Validate a finite, fixed-size vector without silently reshaping it."""
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise Arx5Error("invalid_request", f"{name} must have {length} finite values")
    return result


def pose_matrix(value: Any) -> np.ndarray:
    """Convert XYZ + unit XYZW to a homogeneous transform."""
    pose = vector(value, 7, "tcp_pose")
    q = pose[3:]
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-4):
        raise Arx5Error("invalid_request", "tcp quaternion must be unit length")
    x, y, z, w = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a rigid transform to XYZ + XYZW, including half turns."""
    r = matrix[:3, :3]
    # Davenport's symmetric eigenproblem avoids a near-zero trace divisor.
    k = np.array(
        [
            [
                r[0, 0] - r[1, 1] - r[2, 2],
                r[0, 1] + r[1, 0],
                r[0, 2] + r[2, 0],
                r[2, 1] - r[1, 2],
            ],
            [
                r[0, 1] + r[1, 0],
                r[1, 1] - r[0, 0] - r[2, 2],
                r[1, 2] + r[2, 1],
                r[0, 2] - r[2, 0],
            ],
            [
                r[0, 2] + r[2, 0],
                r[1, 2] + r[2, 1],
                r[2, 2] - r[0, 0] - r[1, 1],
                r[1, 0] - r[0, 1],
            ],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
        ]
    )
    _, eigenvectors = np.linalg.eigh(k)
    q = eigenvectors[:, -1]
    if q[3] < 0:
        q = -q
    return np.concatenate([matrix[:3, 3], q])


def rpy_matrix(pose: Any) -> np.ndarray:
    """Decode SDK XYZ + RPY (Rz @ Ry @ Rx)."""
    x, y, z, roll, pitch, yaw = vector(pose, 6, "sdk_pose")
    sr, sp, sy = np.sin([roll, pitch, yaw])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = [x, y, z]
    return result


def matrix_rpy(matrix: np.ndarray) -> np.ndarray:
    """Encode a transform as SDK XYZ + RPY."""
    r = matrix[:3, :3]
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if math.hypot(r[0, 0], r[1, 0]) < 1e-8:
        roll, yaw = 0.0, math.atan2(-r[0, 1], r[1, 1])
    else:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    return np.concatenate([matrix[:3, 3], [roll, pitch, yaw]])


def local_endpoint(endpoint: str) -> None:
    """Refuse network clients whose monotonic clock is not this host's."""
    if not endpoint.startswith(("tcp://127.0.0.1:", "ipc://")):
        raise ValueError("ARX service requires tcp://127.0.0.1:PORT or ipc://PATH")


class Arx5Client:
    """Thread-serialized JSON RPC with bounded waits and no command retries."""

    def __init__(self, endpoint: str, timeout_ms: int = 1000) -> None:
        import zmq

        local_endpoint(endpoint)
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.endpoint, self.timeout_ms = endpoint, int(timeout_ms)
        self._zmq = zmq
        self._context = zmq.Context()
        self._lock = threading.Lock()
        self._closed = False
        self.server_id: str | None = None

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        """Issue one request; a timeout has an unknown command outcome."""
        with self._lock:
            if self._closed:
                raise Arx5Error("closed", "ARX client is closed")
            request_id = uuid.uuid4().hex
            message = {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "operation": operation,
                "server_id": self.server_id,
                "deadline": time.monotonic() + self.timeout_ms / 1000,
                **payload,
            }
            socket = self._context.socket(self._zmq.REQ)
            socket.setsockopt(self._zmq.LINGER, 0)
            socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
            socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
            try:
                socket.connect(self.endpoint)
                socket.send(json.dumps(message, allow_nan=False).encode())
                response = json.loads(socket.recv())
            except self._zmq.Again as exc:
                raise Arx5Error(
                    "timeout", "service did not reply; do not retry motion"
                ) from exc
            finally:
                socket.close(linger=0)
            if (
                not isinstance(response, dict)
                or response.get("request_id") != request_id
                or response.get("version") != PROTOCOL_VERSION
            ):
                raise Arx5Error("protocol", "mismatched ARX response")
            if (
                self.server_id is not None
                and response.get("server_id") != self.server_id
            ):
                raise Arx5Error(
                    "session",
                    "SDK service restarted; disconnect and reconnect explicitly",
                )
            if not response.get("ok"):
                error = response.get("error", {})
                raise Arx5Error(
                    error.get("code", "protocol"),
                    error.get("message", "invalid response"),
                )
            self.server_id = response["server_id"]
            return response["result"]

    def close(self) -> None:
        """Close transport only; disconnecting never changes robot mode."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._context.term()


@dataclass(frozen=True)
class MotionLimits:
    """Per-command Cartesian translation, rotation, and gripper limits."""

    translation: float
    rotation: float
    gripper: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.translation, self.rotation, self.gripper)
        ):
            raise ValueError("All motion limits must be finite and positive")


class Arx5Service:
    """Serialize all SDK access and retain measured history during inference.

    The adapter is the sole CAN owner. Reading never enables motion or clears
    a fault. Mode changes increment the epoch before touching the SDK.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        allow_motion: bool = False,
        limits: MotionLimits | None = None,
        sample_hz: float = 100.0,
        history_seconds: float = 10.0,
        state_timeout: float = 0.25,
        command_timeout: float = 2.0,
    ) -> None:
        if any(
            not math.isfinite(v) or v <= 0
            for v in (sample_hz, history_seconds, state_timeout, command_timeout)
        ):
            raise ValueError("Rates and timeouts must be finite and positive")
        if allow_motion and limits is None:
            raise ValueError("Motion requires explicit limits")
        self.adapter, self.allow_motion, self.limits = adapter, allow_motion, limits
        self.sample_hz, self.state_timeout, self.command_timeout = (
            sample_hz,
            state_timeout,
            command_timeout,
        )
        self.server_id, self.mode, self.control_epoch = uuid.uuid4().hex, "disabled", 0
        self._history: deque[dict[str, Any]] = deque(
            maxlen=max(2, int(sample_hz * history_seconds))
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._fault: str | None = None
        self._last_command_time = time.monotonic()
        self._last_target: np.ndarray | None = None
        self._last_gripper: float | None = None
        self._mode_since = -math.inf
        self._sample()

    def _sample(self) -> dict[str, Any]:
        with self._lock:
            raw = self.adapter.read_state()
            # A mode label must never be attached to feedback captured before
            # that mode was established. Wait for a new SDK tick, bounded by
            # the same feedback deadline used for stale-state detection.
            deadline = time.monotonic() + self.state_timeout
            while float(raw["timestamp"]) < self._mode_since:
                if self._stop.wait(min(0.001, self.state_timeout)):
                    raise Arx5Error(
                        "shutdown", "service stopped during mode confirmation"
                    )
                if time.monotonic() >= deadline:
                    raise Arx5Error(
                        "stale_state", "no measured feedback after control mode change"
                    )
                raw = self.adapter.read_state()
            pose = vector(raw["tcp_pose"], 7, "tcp_pose")
            pose_matrix(pose)
            joints = np.asarray(raw["arm_joint_position"], dtype=float)
            if joints.ndim != 1 or not len(joints):
                raise Arx5Error("state", "invalid joint position")
            joints = vector(joints, len(joints), "arm_joint_position")
            velocity = vector(
                raw["arm_joint_velocity"], len(joints), "arm_joint_velocity"
            )
            width, timestamp = float(raw["gripper_width"]), float(raw["timestamp"])
            now = time.monotonic()
            if (
                not math.isfinite(width)
                or not math.isfinite(timestamp)
                or not -0.01 <= now - timestamp <= self.state_timeout
            ):
                raise Arx5Error("stale_state", "invalid or expired SDK feedback")
            if self._history and timestamp <= self._history[-1]["timestamp"]:
                if timestamp < self._history[-1]["timestamp"]:
                    raise Arx5Error("state", "SDK feedback timestamp moved backwards")
                # RPC reads may outnumber physical feedback ticks. One tick
                # represents one observation, not several interpolator knots.
                return self._history[-1].copy()
            self._sequence += 1
            state = {
                "tcp_pose": pose.tolist(),
                "arm_joint_position": joints.tolist(),
                "arm_joint_velocity": velocity.tolist(),
                "gripper_width": width,
                "timestamp": timestamp,
                "mode": self.mode,
                "control_epoch": self.control_epoch,
                "sequence": self._sequence,
                "server_id": self.server_id,
                "fault": self._fault,
            }
            self._history.append(state)
            return state.copy()

    def start(self) -> None:
        """Start measured history sampling without changing control mode."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._sample_loop, name="arx-feedback", daemon=True
            )
            self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(1.0 / self.sample_hz):
            with self._lock:
                try:
                    self._sample()
                    if (
                        self.mode == "policy"
                        and time.monotonic() - self._last_command_time
                        > self.command_timeout
                    ):
                        self._transition("hold")
                except Exception as exc:
                    self._set_fault(str(exc))

    def _set_fault(self, message: str) -> None:
        if self.mode != "fault":
            self.control_epoch += 1
            self._mode_since = time.monotonic()
        self.mode, self._fault = "fault", message
        self._last_target = self._last_gripper = None
        # Do not use stale feedback to command a recovery pose. The SDK keeps
        # its last finite target; clearing the fault requires service restart.

    def _transition(self, mode: str) -> dict[str, Any]:
        if not self.allow_motion:
            raise Arx5Error(
                "disabled", "service was launched without motion permission"
            )
        if self.mode == "fault":
            raise Arx5Error("fault", self._fault or "latched hardware fault")
        current = self._sample()
        if mode == "policy" and self.mode != "hold":
            raise Arx5Error("mode", "enable_policy requires hold mode")
        self.control_epoch += 1
        self._last_target = self._last_gripper = None
        try:
            if mode == "drag":
                self.adapter.enter_drag()
            elif mode == "hold":
                self.adapter.resume_hold()
            elif mode != "policy":
                raise Arx5Error("mode", f"unsupported mode {mode}")
            self.mode = mode
            self._mode_since = time.monotonic()
            current = self._sample()
            self._last_target = np.asarray(current["tcp_pose"])
            self._last_gripper = current["gripper_width"]
            self._last_command_time = time.monotonic()
            return current
        except Exception as exc:
            self._set_fault(str(exc))
            raise Arx5Error("sdk", str(exc)) from exc

    def _check_epoch(self, request: dict[str, Any]) -> None:
        epoch = request.get("expected_epoch")
        if request.get("server_id") != self.server_id:
            raise Arx5Error(
                "session", "read state after connecting to this service instance"
            )
        if type(epoch) is not int or epoch != self.control_epoch:
            raise Arx5Error(
                "stale_epoch", "control changed since this command was planned"
            )

    def _command(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check_epoch(request)
        if self.mode != "policy" or not self.allow_motion:
            raise Arx5Error(
                "mode", "policy commands require explicitly enabled policy mode"
            )
        state = self._sample()
        target = vector(request.get("tcp_pose"), 7, "tcp_pose")
        target_matrix = pose_matrix(target)
        width = float(request["gripper_width"])
        if not math.isfinite(width):
            raise Arx5Error("invalid_request", "gripper_width must be finite")
        assert self.limits is not None
        for reference, gripper in (
            (self._last_target, self._last_gripper),
            (state["tcp_pose"], state["gripper_width"]),
        ):
            reference = np.asarray(reference)
            angle = math.acos(
                np.clip(
                    (
                        np.trace(
                            pose_matrix(reference)[:3, :3].T @ target_matrix[:3, :3]
                        )
                        - 1
                    )
                    / 2,
                    -1,
                    1,
                )
            )
            if (
                np.linalg.norm(target[:3] - reference[:3]) > self.limits.translation
                or angle > self.limits.rotation
                or abs(width - gripper) > self.limits.gripper
            ):
                raise Arx5Error(
                    "step_limit",
                    "translation, rotation, or gripper target exceeds configured limit",
                )
        self.adapter.command_pose(
            target,
            width,
            deadline=request["deadline"],
        )
        self._last_target, self._last_gripper = target.copy(), width
        self._last_command_time = time.monotonic()
        result = self._sample()
        result["accepted"] = True
        result["submitted_tcp_pose"] = target.tolist()
        result["submitted_gripper_width"] = width
        return result

    def handle(self, request: Any) -> dict[str, Any]:
        """Validate and execute one JSON request, always returning an envelope."""
        response = {
            "version": PROTOCOL_VERSION,
            "request_id": request.get("request_id")
            if isinstance(request, dict)
            else None,
            "server_id": self.server_id,
        }
        try:
            with self._lock:
                if (
                    not isinstance(request, dict)
                    or request.get("version") != PROTOCOL_VERSION
                    or not isinstance(request.get("request_id"), str)
                ):
                    raise Arx5Error(
                        "protocol", "invalid protocol version or request ID"
                    )
                deadline = request.get("deadline")
                if (
                    not isinstance(deadline, (int, float))
                    or not math.isfinite(deadline)
                    or deadline < time.monotonic()
                ):
                    raise Arx5Error("expired", "request deadline has passed")
                operation = request.get("operation")
                if operation == "state":
                    result = self._sample()
                elif operation == "history":
                    self._sample()
                    since = request.get("since")
                    if since is not None and (
                        not isinstance(since, (int, float)) or not math.isfinite(since)
                    ):
                        raise Arx5Error(
                            "invalid_request", "since must be a monotonic timestamp"
                        )
                    result = {
                        "states": [
                            s.copy()
                            for s in self._history
                            if since is None or s["timestamp"] >= since
                        ]
                    }
                elif operation in (
                    "enter_drag",
                    "resume_hold",
                    "hold",
                    "enable_policy",
                ):
                    self._check_epoch(request)
                    result = self._transition(
                        {
                            "enter_drag": "drag",
                            "resume_hold": "hold",
                            "hold": "hold",
                            "enable_policy": "policy",
                        }[operation]
                    )
                elif operation == "command_pose":
                    result = self._command(request)
                else:
                    raise Arx5Error("operation", "unknown operation")
                response.update(ok=True, result=result)
        except Arx5Error as exc:
            if exc.code in ("stale_state", "state", "sdk"):
                with self._lock:
                    self._set_fault(str(exc))
            response.update(ok=False, error={"code": exc.code, "message": str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            response.update(
                ok=False, error={"code": "invalid_request", "message": str(exc)}
            )
        except Exception as exc:
            with self._lock:
                self._set_fault(str(exc))
            response.update(ok=False, error={"code": "sdk", "message": str(exc)})
        return response

    def serve(self, endpoint: str, *, ready: threading.Event | None = None) -> None:
        """Serve local JSON requests until stop; malformed JSON is recoverable."""
        import zmq

        local_endpoint(endpoint)
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, 1_000_000)
        try:
            socket.bind(endpoint)
            self.start()
            if ready:
                ready.set()
            while not self._stop.is_set():
                if socket.poll(50):
                    try:
                        request = json.loads(socket.recv())
                    except (ValueError, UnicodeError):
                        request = None
                    socket.send(
                        json.dumps(self.handle(request), allow_nan=False).encode()
                    )
        finally:
            socket.close(linger=0)
            context.term()

    def stop(self) -> None:
        """Stop serving and history collection without an implicit home move."""
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise Arx5Error("shutdown", "SDK feedback did not stop")


class SdkArx5Adapter:
    """Adapt commands and feedback from the SDK Cartesian controller.

    ``eef_to_tcp`` right-multiplies the SDK EEF transform. Gripper calibration
    maps model metres to SDK metres as ``sdk = width * scale + offset``.
    The service launcher owns the CAN lock and constructs the controller.
    """

    def __init__(
        self,
        sdk: Any,
        controller: Any,
        solver: Any,
        *,
        eef_to_tcp: Any = None,
        gripper_scale: float = 1.0,
        gripper_offset: float = 0.0,
        drag_kd: Any = None,
        drag_gripper: str = "hold",
        feedback_timeout: float = 0.25,
        restore_tolerance: float = 0.01,
    ) -> None:
        self.sdk, self.controller, self.solver = sdk, controller, solver
        self.robot_config = controller.get_robot_config()
        self.controller_config = controller.get_controller_config()
        self.dof = int(self.robot_config.joint_dof)
        self.transform = (
            np.eye(4) if eef_to_tcp is None else np.asarray(eef_to_tcp, dtype=float)
        )
        if (
            self.transform.shape != (4, 4)
            or not np.isfinite(self.transform).all()
            or not np.allclose(self.transform[3], [0, 0, 0, 1])
            or not np.allclose(
                self.transform[:3, :3].T @ self.transform[:3, :3], np.eye(3), atol=1e-6
            )
            or not np.isclose(np.linalg.det(self.transform[:3, :3]), 1)
        ):
            raise ValueError("eef_to_tcp must be a rigid 4x4 transform")
        if (
            not math.isfinite(gripper_scale)
            or gripper_scale <= 0
            or not math.isfinite(gripper_offset)
        ):
            raise ValueError("invalid gripper calibration")
        if drag_gripper not in ("hold", "manual"):
            raise ValueError("drag_gripper must be hold or manual")
        self.gripper_scale, self.gripper_offset = gripper_scale, gripper_offset
        self.drag_gripper = drag_gripper
        self.feedback_timeout, self.restore_tolerance = (
            feedback_timeout,
            restore_tolerance,
        )
        self.drag_kd = vector(
            self.controller_config.default_kd if drag_kd is None else drag_kd,
            self.dof,
            "drag_kd",
        )
        if np.any(self.drag_kd < 0):
            raise ValueError("drag_kd cannot be negative")
        self._normal_gain = sdk.Gain(
            np.asarray(self.controller_config.default_kp).copy(),
            np.asarray(self.controller_config.default_kd).copy(),
            self.controller_config.default_gripper_kp,
            self.controller_config.default_gripper_kd,
        )
        # Both SDK steady_clock and Python monotonic use the host monotonic
        # clock. A fixed offset preserves identical timestamps for repeated
        # reads of a single feedback tick. Choose the least interrupted query.
        calibrations = []
        for _ in range(5):
            before = time.monotonic()
            sdk_now = self.controller.get_timestamp()
            after = time.monotonic()
            calibrations.append((after - before, (before + after) / 2 - sdk_now))
        self._clock_offset = min(calibrations)[1]

    def _joint_state(self) -> Any:
        state = self.controller.get_joint_state()
        age = self.controller.get_timestamp() - float(state.timestamp)
        if not math.isfinite(age) or not -0.01 <= age <= self.feedback_timeout:
            raise Arx5Error("stale_state", "SDK joint feedback is stale")
        vector(state.pos(), self.dof, "joint_position")
        vector(state.vel(), self.dof, "joint_velocity")
        if not math.isfinite(state.gripper_pos):
            raise Arx5Error("state", "invalid SDK gripper feedback")
        return state

    def read_state(self) -> dict[str, Any]:
        """Derive EEF from the same measured joint sample as the joint state."""
        state = self._joint_state()
        transform = (
            rpy_matrix(self.solver.forward_kinematics(state.pos())) @ self.transform
        )
        return {
            "tcp_pose": matrix_pose(transform),
            "arm_joint_position": state.pos().copy(),
            "arm_joint_velocity": state.vel().copy(),
            "gripper_width": (state.gripper_pos - self.gripper_offset)
            / self.gripper_scale,
            "timestamp": self._clock_offset + state.timestamp,
        }

    def _set_gain(self, gain: Any) -> None:
        self.controller.set_gain(gain)
        actual = self.controller.get_gain()
        if (
            not np.allclose(actual.kp(), gain.kp())
            or not np.allclose(actual.kd(), gain.kd())
            or not np.isclose(actual.gripper_kp, gain.gripper_kp)
            or not np.isclose(actual.gripper_kd, gain.gripper_kd)
        ):
            raise Arx5Error("sdk", "SDK gain change was not acknowledged")

    def _replace_target(self, state: Any, gripper: float) -> None:
        command = self.sdk.EEFState(
            self.solver.forward_kinematics(state.pos()), gripper
        )
        # The Cartesian controller schedules its preview after solving IK.
        # Wait for that interpolation before restoring joint stiffness.
        self.controller.set_eef_cmd(command)
        deadline = time.monotonic() + self.feedback_timeout
        while time.monotonic() < deadline:
            actual = self.controller.get_joint_cmd()
            if (
                np.max(np.abs(actual.pos() - state.pos())) <= self.restore_tolerance
                and abs(actual.gripper_pos - gripper) <= 1e-4
            ):
                return
            time.sleep(min(self.controller_config.controller_dt, 0.01))
        raise Arx5Error("sdk", "SDK target did not reset to measured position")

    def enter_drag(self) -> None:
        """Remove joint stiffness, preserving the configured gripper behavior."""
        if not self.controller_config.gravity_compensation:
            raise Arx5Error("sdk", "drag requires SDK gravity_compensation")
        state = self._joint_state()
        current = self.controller.get_gain()
        gain = self.sdk.Gain(self.dof)
        gain.kd()[:] = self.drag_kd
        if self.drag_gripper == "hold":
            gain.gripper_kp = current.gripper_kp or self._normal_gain.gripper_kp
            gain.gripper_kd = current.gripper_kd or self._normal_gain.gripper_kd
        self._set_gain(gain)
        # No set_to_damping(): that method drops gripper gains transiently.
        self._replace_target(state, state.gripper_pos)

    def resume_hold(self) -> None:
        """Rebuild the target from feedback before restoring joint stiffness."""
        current = self.controller.get_gain()
        passive = self.sdk.Gain(self.dof)
        passive.kd()[:] = self.controller_config.default_kd
        passive.gripper_kp, passive.gripper_kd = current.gripper_kp, current.gripper_kd
        self._set_gain(passive)
        state = self._joint_state()
        self._replace_target(state, state.gripper_pos)
        latest = self._joint_state()
        actual = self.controller.get_joint_cmd()
        if np.max(np.abs(latest.pos() - actual.pos())) > self.restore_tolerance:
            raise Arx5Error(
                "sdk", "arm moved during hold restoration; stiffness remains disabled"
            )
        self._set_gain(self._normal_gain)

    def command_pose(
        self,
        tcp_pose: Any,
        gripper_width: float,
        *,
        deadline: float,
    ) -> None:
        """Submit a converted EEF target through the SDK Cartesian controller."""
        preview = float(self.controller_config.default_preview_time)
        if (
            not math.isfinite(preview)
            or preview <= self.controller_config.controller_dt
        ):
            raise Arx5Error(
                "configuration",
                "set SDK default_preview_time above controller_dt for interpolated motion",
            )
        self._joint_state()
        sdk_pose = matrix_rpy(pose_matrix(tcp_pose) @ np.linalg.inv(self.transform))
        sdk_width = gripper_width * self.gripper_scale + self.gripper_offset
        if not 0 <= sdk_width <= self.robot_config.gripper_width:
            raise Arx5Error(
                "gripper_limit", "gripper target is outside calibrated stroke"
            )
        if time.monotonic() > deadline:
            raise Arx5Error("expired", "request expired before SDK submission")
        command = self.sdk.EEFState(sdk_pose, sdk_width)
        self.controller.set_eef_cmd(command)
