# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Synchronous RLT execution and measured, same-arm drag interventions."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from rlinf.algorithms.rlt.action_codec import EEFActionCodec
from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.utils.logging import get_logger

from .history import interpolate_states, prepare_rgb

if TYPE_CHECKING:
    from rlinf.robotics.discovery import RobotInfo
    from rlinf.robotics.parts.arms.arx5 import Arx5Arm, Arx5RobotState
    from rlinf.robotics.parts.cameras.base import BaseCamera
    from rlinf.scheduler import WorkerInfo

    from .preview import Arx5CameraPreview

StepResult = tuple[dict[str, Any], float, bool, bool, dict[str, Any]]


@dataclass
class Arx5RLTEnvConfig:
    """Model timing must be taken from the checkpoint's actual dataset."""

    norm_stats_path: str
    action_dt: float
    history_dt: float
    camera_latency: float
    gripper_width_min: float
    gripper_width_max: float
    task_description: str = "perform the manipulation task"
    camera_device: str = "/dev/video0"
    camera_resolution: tuple[int, int] = (1920, 1080)
    camera_fps: int = 30
    camera_pixel_format: str = "MJPG"
    enable_camera_preview: bool = False
    camera_preview_port: int = 8765
    image_crop: list[int] | None = None
    max_frame_age: float = 0.5
    max_state_gap: float = 0.05
    observation_timeout: float = 3.0
    history_seconds: float = 10.0
    max_episode_steps: int = 300
    enable_keyboard: bool = True
    start_paused: bool = True
    start_with_actor: bool = False
    return_on_episode_end: bool = False
    reset_pose: list[float] | str | None = None
    reset_gripper_width: float | None = None
    reset_duration: float = 4.0
    reset_frequency: float = 20.0
    reset_settle_timeout: float = 2.0
    record_dir: str | None = None
    is_dummy: bool = False

    def __post_init__(self) -> None:
        for name in (
            "action_dt",
            "history_dt",
            "max_frame_age",
            "max_state_gap",
            "observation_timeout",
            "history_seconds",
            "reset_duration",
            "reset_frequency",
            "reset_settle_timeout",
        ):
            value = float(getattr(self, name))
            setattr(self, name, value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.camera_latency = float(self.camera_latency)
        self.camera_fps = int(self.camera_fps)
        self.camera_preview_port = int(self.camera_preview_port)
        if not 1 <= self.camera_preview_port <= 65535:
            raise ValueError("camera_preview_port must be between 1 and 65535.")
        self.max_episode_steps = int(self.max_episode_steps)
        self.gripper_width_min = float(self.gripper_width_min)
        self.gripper_width_max = float(self.gripper_width_max)
        if (
            not np.isfinite([self.gripper_width_min, self.gripper_width_max]).all()
            or not 0 <= self.gripper_width_min < self.gripper_width_max
        ):
            raise ValueError(
                "Require calibrated 0 <= gripper_width_min < gripper_width_max."
            )
        if not np.isfinite(self.camera_latency) or self.camera_latency < 0:
            raise ValueError("camera_latency must be finite and nonnegative.")
        if self.history_seconds <= self.history_dt + self.camera_latency:
            raise ValueError(
                "history_seconds must cover camera latency and history_dt."
            )
        if self.camera_fps <= 0 or self.max_episode_steps <= 0:
            raise ValueError("camera_fps and max_episode_steps must be positive.")
        if isinstance(self.reset_pose, str):
            try:
                self.reset_pose = json.loads(self.reset_pose)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "reset_pose must be a JSON array of TCP XYZ + rotvec."
                ) from exc
        if self.reset_pose is not None:
            pose = np.asarray(self.reset_pose, dtype=np.float64)
            if pose.shape != (6,) or not np.isfinite(pose).all():
                raise ValueError("reset_pose must contain finite TCP XYZ + rotvec.")
            self.reset_pose = pose.tolist()
        elif self.return_on_episode_end:
            raise ValueError(
                "return_on_episode_end requires reset_pose in the service TCP frame; "
                "set ARX_RESET_POSE for the ARX example configuration."
            )
        if self.reset_gripper_width is not None:
            self.reset_gripper_width = float(self.reset_gripper_width)
            if (
                not np.isfinite(self.reset_gripper_width)
                or not self.gripper_width_min
                <= self.reset_gripper_width
                <= self.gripper_width_max
            ):
                raise ValueError("reset_gripper_width must be within gripper limits.")


class Arx5RLTEnv(gym.Env):
    """Execute normalized EEF chunks; sample drag actions from actual motion.

    Operator events use the same service as model commands, so its control
    epoch arbitrates even when inference is waiting. The optional injected
    arm and camera implement the real device interfaces for integration tests.
    """

    TELEOP = ()
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: WorkerInfo | None = None,
        robot_info: RobotInfo | None = None,
        env_idx: int = 0,
        *,
        arm: Arx5Arm | None = None,
        camera: BaseCamera | None = None,
    ) -> None:
        self.config = Arx5RLTEnvConfig(**override_cfg)
        self.codec = EEFActionCodec.from_stats_json(self.config.norm_stats_path)
        self._logger = get_logger()
        self.robot = None
        self._arm, self._camera = arm, camera
        self._closed = False
        self._stop = threading.Event()
        self._started = threading.Event()
        self._lock = threading.RLock()
        self._control_lock = threading.RLock()
        self._return_cancel = threading.Event()
        self._returning = False
        self._episode_finished = False
        self._pending_return: tuple[str, int | None] | None = None
        self._waiting_start = True
        self._frames = deque(
            maxlen=max(2, int(self.config.history_seconds * self.config.camera_fps))
        )
        self._background_error: Exception | None = None
        self._listener = None
        self._threads: list[threading.Thread] = []
        self._episode_start = None
        self._episode_id = 0
        self._preview: Arx5CameraPreview | None = None
        self._published = None
        self._chunk = None
        self._chunk_index = 0
        self._chunk_size = 1
        self._chunk_cancelled = False
        self._terminal = False
        self._reward = 0.0
        self._terminal_target = None
        self._terminal_reported = False
        self._terminal_human = False
        self._terminal_epoch = None
        self._terminal_time = None
        self._terminal_event = None
        self._terminal_source_epoch = None
        self._rlt_switch = self.config.start_with_actor
        self._step_count = 0
        self._record_path = None
        if self.config.record_dir:
            directory = Path(self.config.record_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            self._record_path = directory / f"arx-{time.time_ns()}.jsonl"
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32)
        model_spaces = {
            "robot0_eef_pos": gym.spaces.Box(-np.inf, np.inf, (2, 3)),
            "robot0_eef_rot_axis_angle": gym.spaces.Box(-1, 1, (2, 6)),
            "robot0_eef_rot_axis_angle_wrt_start": gym.spaces.Box(-1, 1, (2, 6)),
            "robot0_gripper_width": gym.spaces.Box(0, np.inf, (2, 1)),
            "left_wrist_0_rgb_1": gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {"proprio": gym.spaces.Box(-np.inf, np.inf, (32,))}
                ),
                "frames": gym.spaces.Dict(
                    {"wrist_1": model_spaces["left_wrist_0_rgb_1"]}
                ),
                "model_obs": gym.spaces.Dict(model_spaces),
            }
        )
        try:
            if (arm is None) != (camera is None):
                raise ValueError("Supply both test devices or neither.")
            if arm is None:
                if self.config.is_dummy:
                    raise ValueError(
                        "Use injected devices for ARX hardware-free tests."
                    )
                self._connect(robot_info, worker_info, env_idx)
            if self.config.enable_camera_preview:
                try:
                    from .preview import Arx5CameraPreview

                    self._preview = Arx5CameraPreview(self.config.camera_preview_port)
                except Exception as exc:
                    self._logger.warning("ARX camera preview unavailable: %s", exc)
                else:
                    self._logger.info("ARX camera preview: %s", self._preview.url)
            if self.config.enable_keyboard:
                from rlinf.envs.real.wrappers.episode.keyboard import KeyboardListener

                self._listener = KeyboardListener()
                self._start_thread(self._keyboard_loop, "arx-operator")
            self._start_thread(self._capture_loop, "arx-camera-history")
        except Exception:
            self.close()
            raise

    def _connect(
        self, robot_info: RobotInfo | None, worker_info: WorkerInfo | None, env_idx: int
    ) -> None:
        from rlinf.robotics.parts.cameras import CameraInfo
        from rlinf.robotics.robots.arx5 import Arx5Config, Arx5Robot

        hardware = get_hardware_config(Arx5Config, robot_info, is_dummy=False)
        node_rank = hardware.controller_node_rank
        if node_rank is None:
            node_rank = getattr(worker_info, "cluster_node_rank", 0)
        self.robot = Arx5Robot.build(
            endpoint=hardware.endpoint,
            timeout_ms=hardware.timeout_ms,
            max_state_age=hardware.max_state_age,
            node_rank=node_rank,
            env_idx=env_idx,
            worker_rank=getattr(worker_info, "rank", 0),
            cameras={
                "wrist_1": CameraInfo(
                    name="wrist_1",
                    serial_number=self.config.camera_device,
                    camera_type="uvc",
                    resolution=tuple(self.config.camera_resolution),
                    fps=self.config.camera_fps,
                    pixel_format=self.config.camera_pixel_format,
                )
            },
        )
        self.robot.connect()
        self._arm = self.robot.child("arm")
        self._camera = self.robot.child("wrist_1")

    def _start_thread(self, fn: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=fn, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    @property
    def task_description(self) -> str:
        return self.config.task_description

    @property
    def rlt_switch_flags(self) -> bool:
        return self._rlt_switch

    def _record(self, kind: str, **data) -> None:
        if self._record_path is not None:

            def convert(value: Any) -> Any:
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)

            with self._lock, self._record_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {"type": kind, "time": time.monotonic(), **data},
                        default=convert,
                        allow_nan=False,
                    )
                    + "\n"
                )

    def _capture_loop(self) -> None:
        last_frame = time.monotonic()
        while not self._stop.is_set():
            try:
                stamp, frame, seq = self._camera.get_frame_with_timestamp(timeout=0.2)
                age = time.monotonic() - (stamp - self.config.camera_latency)
                if (
                    not np.isfinite(stamp)
                    or age < -0.01
                    or age > self.config.max_frame_age
                ):
                    raise TimeoutError(
                        "Camera delivered an invalid or stale timestamp."
                    )
                rgb = prepare_rgb(frame, self.config.image_crop)
                with self._lock:
                    self._frames.append((stamp - self.config.camera_latency, rgb, seq))
                last_frame = stamp
                self._publish_preview(frame)
            except queue.Empty:
                if time.monotonic() - last_frame > self.config.max_frame_age:
                    self._background_error = TimeoutError(
                        "Camera stopped delivering frames."
                    )
                    self._hold_on_fault()
                    return
                continue
            except Exception as exc:
                self._background_error = exc
                self._hold_on_fault()
                return

    def _publish_preview(self, frame: np.ndarray) -> None:
        if self._preview is None:
            return
        with self._lock:
            episode = self._episode_id + int(self._waiting_start)
            if self._background_error is not None:
                status = "环境故障"
            elif self._returning:
                status = "回初始位置"
            elif self._terminal_event == "abort":
                status = "已中止"
            elif self._waiting_start:
                status = "等待开始（按 s）"
            elif (
                self._terminal
                or self._episode_finished
                or self._step_count >= self.config.max_episode_steps
            ):
                status = "本轮已结束"
            else:
                status = "进行中"
        self._preview.publish(frame, episode=episode, status=status)

    def _keyboard_loop(self) -> None:
        last_press: dict[str, float] = {}
        mapping = {
            "s": "start",
            "b": "actor",
            "v": "reference",
            "g": "drag",
            "r": "resume",
            "c": "success",
            "a": "failure",
            "q": "abort",
        }
        while not self._stop.wait(0.02):
            if getattr(self._listener, "error", None) is not None:
                self._background_error = self._listener.error
                self._hold_on_fault()
                return
            for key in self._listener.pop_pressed_keys():
                now = time.monotonic()
                if key not in mapping or now - last_press.get(key, -np.inf) < 0.2:
                    continue
                last_press[key] = now
                try:
                    self.handle_operator_event(mapping[key])
                except Exception as exc:
                    self._background_error = exc
                    self._hold_on_fault()
                    return

    def handle_operator_event(self, event: str) -> None:
        """Apply an explicit operator event, including while inference waits."""
        with self._control_lock:
            self._handle_operator_event(event)

    def _handle_operator_event(self, event: str) -> None:
        self._check_background()
        if self.config.return_on_episode_end and (
            self._returning or self._terminal or self._episode_finished
        ):
            if event in {"abort", "resume", "drag"}:
                self._return_cancel.set()
                state = self._arm.get_state()
                if event == "drag":
                    self._arm.enter_drag(expected_epoch=state.control_epoch)
                elif state.mode in {"drag", "policy"}:
                    self._arm.resume_hold(expected_epoch=state.control_epoch)
                self._record("return_cancel", event=event)
            else:
                self._logger.info("Ignoring %s until the next episode reset.", event)
            return
        if self._waiting_start and event == "abort":
            self._hold_on_fault()
            self._terminal_event = "abort"
            self._terminal = True
            self._started.set()
            return
        if self._waiting_start and event in {"success", "failure"}:
            self._logger.warning("Start an ARX episode before labelling its outcome.")
            return
        if self._terminal and event not in {"success", "failure", "abort"}:
            raise RuntimeError("Reset the episode before enabling motion again.")
        if event == "start":
            state = self._arm.resume_hold()
            self._arm.enable_policy(expected_epoch=state.control_epoch)
            self._started.set()
        elif event == "actor":
            self._rlt_switch = True
        elif event == "reference":
            self._rlt_switch = False
        elif event == "drag":
            self._arm.enter_drag()
        elif event == "resume":
            state = self._arm.resume_hold()
            if not self._waiting_start:
                self._arm.enable_policy(expected_epoch=state.control_epoch)
        elif event in {"success", "failure", "abort"}:
            if self._terminal:
                return
            state = self._arm.get_state()
            if state.mode not in {"drag", "policy"}:
                raise RuntimeError(f"Cannot label an episode in ARX mode {state.mode}.")
            if state.mode == "drag":
                held = self._arm.resume_hold(expected_epoch=state.control_epoch)
            elif state.mode == "policy":
                held = self._arm.hold(expected_epoch=state.control_epoch)
            else:
                held = state
            with self._lock:
                self._reward = {"success": 1.0, "failure": -1.0, "abort": 0.0}[event]
                self._terminal_target = (held.tcp_pose.copy(), held.gripper_width)
                self._terminal_human = state.mode == "drag"
                self._terminal_source_epoch = state.control_epoch
                self._terminal_epoch = held.control_epoch
                self._terminal_time = held.timestamp
                self._terminal_event = event
                self._terminal = True
        else:
            raise ValueError(f"Unknown ARX operator event: {event}")
        self._record("operator", event=event)

    def _check_background(self) -> None:
        if self._background_error is not None:
            self._hold_on_fault()
            raise RuntimeError(
                "ARX capture/operator loop failed."
            ) from self._background_error
        if self._closed:
            raise RuntimeError("ARX environment is closed.")

    def _hold_on_fault(self) -> Arx5RobotState | None:
        """Best-effort measured hold; the SDK service independently faults too."""
        if self._arm is None:
            return
        try:
            state = self._arm.get_state()
            if state.mode == "drag":
                return self._arm.resume_hold(expected_epoch=state.control_epoch)
            if state.mode == "policy":
                return self._arm.hold(expected_epoch=state.control_epoch)
            return state
        except Exception:
            self._logger.exception("Could not hold ARX after an environment failure.")
            return None

    def _return_control_changed(self, owner: Arx5RobotState) -> bool:
        current = self._arm.get_state()
        if current.mode == "fault":
            raise RuntimeError(f"ARX service fault during return: {current.fault}")
        changed = (
            current.server_id != owner.server_id
            or current.control_epoch != owner.control_epoch
            or current.mode != owner.mode
        )
        if changed:
            self._return_cancel.set()
        return changed

    def _return_waypoint(
        self, pose: np.ndarray, gripper: float, owner: Arx5RobotState
    ) -> Arx5RobotState | None:
        with self._control_lock:
            if self._stop.is_set() or self._return_cancel.is_set():
                return None
            self._check_background()
            if self._return_control_changed(owner):
                return None
            try:
                return self._arm.command_pose(
                    pose, gripper, expected_epoch=owner.control_epoch
                )
            except Exception:
                if self._return_control_changed(owner):
                    # Another client may take control between the read and send.
                    return None
                raise

    def _return_to_initial_pose(
        self, *, server_id: str, control_epoch: int | None
    ) -> None:
        """Return after capturing the terminal transition, outside replay."""
        if (
            not self.config.return_on_episode_end
            or self._terminal_event == "abort"
            or self._return_cancel.is_set()
        ):
            return
        target = np.asarray(self.config.reset_pose, dtype=np.float64)
        target_rotation = Rotation.from_rotvec(target[3:])
        target_pose = np.concatenate([target[:3], target_rotation.as_quat()])
        owner = None
        outcome = "failed"
        failure_active = False
        try:
            with self._control_lock:
                if self._stop.is_set() or self._return_cancel.is_set():
                    outcome = "cancelled"
                    return
                self._check_background()
                self._returning = True
                current = self._arm.get_state()
                if current.mode == "fault":
                    raise RuntimeError(
                        f"ARX service fault before return: {current.fault}"
                    )
                if (
                    current.server_id != server_id
                    or current.control_epoch != control_epoch
                    or current.mode != "hold"
                ):
                    self._return_cancel.set()
                    outcome = "cancelled"
                    return
                expected = current
                try:
                    held = self._arm.resume_hold(expected_epoch=expected.control_epoch)
                    expected = held
                    owner = self._arm.enable_policy(
                        expected_epoch=expected.control_epoch
                    )
                except Exception:
                    if self._return_control_changed(expected):
                        outcome = "cancelled"
                        return
                    raise
            start_pose = owner.tcp_pose.copy()
            start_gripper = owner.gripper_width
            target_gripper = (
                start_gripper
                if self.config.reset_gripper_width is None
                else self.config.reset_gripper_width
            )
            rotations = Slerp(
                [0.0, 1.0],
                Rotation.from_quat([start_pose[3:], target_pose[3:]]),
            )
            steps = max(
                2,
                int(np.ceil(self.config.reset_duration * self.config.reset_frequency)),
            )
            interval = self.config.reset_duration / steps
            self._logger.info(
                "Returning to the configured initial pose over at least %.2f s.",
                self.config.reset_duration,
            )
            self._record(
                "return_start",
                target_pose=target_pose,
                target_gripper=target_gripper,
                duration=self.config.reset_duration,
            )
            for index in range(1, steps + 1):
                if self._return_cancel.wait(interval) or self._stop.is_set():
                    outcome = "cancelled"
                    return
                progress = index / steps
                blend = (
                    2 * progress**2
                    if progress < 0.5
                    else 1 - (-2 * progress + 2) ** 2 / 2
                )
                pose = np.concatenate(
                    [
                        start_pose[:3] + blend * (target_pose[:3] - start_pose[:3]),
                        rotations(blend).as_quat(),
                    ]
                )
                gripper = start_gripper + blend * (target_gripper - start_gripper)
                if self._return_waypoint(pose, gripper, owner) is None:
                    outcome = "cancelled"
                    return

            # Keep the last waypoint active while the SDK's preview and motors
            # catch up. Holding immediately would truncate the final movement.
            deadline = time.monotonic() + self.config.reset_settle_timeout
            while time.monotonic() < deadline:
                state = self._return_waypoint(target_pose, target_gripper, owner)
                if state is None:
                    outcome = "cancelled"
                    return
                position_error = np.linalg.norm(state.tcp_pose[:3] - target_pose[:3])
                rotation_error = (
                    Rotation.from_quat(state.tcp_pose[3:]).inv() * target_rotation
                ).magnitude()
                if (
                    position_error <= 0.005
                    and rotation_error <= 0.05
                    and abs(state.gripper_width - target_gripper) <= 0.005
                ):
                    outcome = "completed"
                    return
                if self._return_cancel.wait(interval) or self._stop.is_set():
                    outcome = "cancelled"
                    return
            raise TimeoutError(
                "ARX did not reach the initial pose before reset timeout."
            )
        except BaseException:
            failure_active = True
            raise
        finally:
            try:
                with self._control_lock:
                    if owner is not None and not self._closed:
                        if self._return_control_changed(owner):
                            if not failure_active:
                                outcome = "cancelled"
                        else:
                            try:
                                self._arm.hold(expected_epoch=owner.control_epoch)
                            except Exception:
                                if self._return_control_changed(owner):
                                    if not failure_active:
                                        outcome = "cancelled"
                                else:
                                    raise
            except Exception:
                outcome = "failed"
                if not failure_active:
                    failure_active = True
                    raise
                self._logger.exception("Could not hold ARX after a return failure.")
            finally:
                with self._control_lock:
                    self._returning = False
                try:
                    self._record("return_end", outcome=outcome)
                    self._logger.info("Return to initial pose: %s.", outcome)
                except Exception:
                    if not failure_active:
                        raise
                    self._logger.exception("Could not record the ARX return outcome.")

    def _observe(self, target_time: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.observation_timeout
        last_error = None
        while time.monotonic() < deadline:
            self._check_background()
            with self._lock:
                frames = list(self._frames)
            if frames:
                if target_time is None:
                    stamp, rgb, seq = frames[-1]
                else:
                    stamp, rgb, seq = min(
                        frames, key=lambda item: abs(item[0] - target_time)
                    )
                    if abs(stamp - target_time) > 1.5 / self.config.camera_fps:
                        time.sleep(0.005)
                        continue
                if (
                    target_time is None
                    and time.monotonic() - stamp > self.config.max_frame_age
                ):
                    last_error = ValueError("Camera observation is stale.")
                else:
                    history = self._arm.get_state_history(
                        since=stamp - self.config.history_dt - self.config.max_state_gap
                    )
                    try:
                        times = (
                            [stamp, stamp]
                            if self._episode_start is None
                            else [stamp - self.config.history_dt, stamp]
                        )
                        poses, grips = interpolate_states(
                            history, times, max_gap=self.config.max_state_gap
                        )
                        at_observation = max(
                            (s for s in history if s.timestamp <= stamp),
                            key=lambda s: s.timestamp,
                        )
                        current = self._arm.get_state()
                        if current.mode == "fault":
                            raise RuntimeError(f"ARX service fault: {current.fault}")
                        if at_observation.server_id != current.server_id:
                            raise ValueError(
                                "Observation belongs to a previous SDK process."
                            )
                        if at_observation.control_epoch != current.control_epoch:
                            raise ValueError(
                                "Waiting for an image from the current control epoch."
                            )
                        if self._episode_start is None:
                            self._episode_start = poses[-1].copy()
                        fields = self.codec.encode_observation(
                            poses, grips, self._episode_start
                        )
                        fields = {
                            k: np.asarray(v, dtype=np.float32)
                            for k, v in fields.items()
                        }
                        fields["left_wrist_0_rgb_1"] = rgb.copy()
                        self._published = (
                            stamp,
                            poses[-1].copy(),
                            at_observation.control_epoch,
                            at_observation.server_id,
                        )
                        return {
                            "state": {
                                "proprio": self.codec.flatten_observation(
                                    fields
                                ).astype(np.float32)
                            },
                            "frames": {"wrist_1": rgb.copy()},
                            "model_obs": fields,
                        }
                    except ValueError as exc:
                        last_error = exc
            time.sleep(0.005)
        self._hold_on_fault()
        raise TimeoutError(f"Cannot form an aligned ARX observation: {last_error}")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Wait for the operator to start; automatic return belongs to episode end."""
        super().reset(seed=seed)
        with self._control_lock:
            # Preserve a drag takeover that cancelled the previous return.
            if not (
                self._return_cancel.is_set() and self._arm.get_state().mode == "drag"
            ):
                self._hold_on_fault()
            self._return_cancel.clear()
            self._episode_finished = False
            self._pending_return = None
            self._episode_start = None
            self._chunk = None
            self._step_count = 0
            self._terminal = False
            self._reward = 0.0
            self._terminal_target = None
            self._terminal_reported = False
            self._terminal_human = False
            self._terminal_epoch = None
            self._terminal_time = None
            self._terminal_event = None
            self._terminal_source_epoch = None
            self._rlt_switch = self.config.start_with_actor
            self._started.clear()
            self._waiting_start = True
        if not self.config.start_paused:
            self.handle_operator_event("start")
        while not self._started.wait(0.02):
            self._check_background()
        if self._terminal_event == "abort":
            raise InterruptedError("ARX episode aborted while waiting to start.")
        with self._lock:
            self._episode_id += 1
            self._waiting_start = False
            self._frames.clear()
        observation = self._observe()
        self._record("reset", anchor=self._published[1])
        return observation, {"rlt_switch_flags": self._rlt_switch}

    def set_action_chunk_size(self, chunk_size: int) -> None:
        """Finish a labelled episode after this many actually executed steps."""
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        self._chunk_size = chunk_size

    def on_action_chunk_begin(self) -> None:
        """Freeze the exact observation anchor throughout one predicted chunk."""
        if self._published is None:
            raise RuntimeError("reset must supply an observation before actions.")
        self._chunk = self._published
        self._chunk_index = 0
        self._chunk_cancelled = False

    def step(self, action: np.ndarray) -> StepResult:
        """Execute one model target or encode one measured human action."""
        try:
            result = self._step(action)
        except Exception:
            self._hold_on_fault()
            raise
        if self._pending_return is not None:
            server_id, control_epoch = self._pending_return
            self._pending_return = None
            # Return owns its cleanup; a new operator epoch must remain untouched.
            self._return_to_initial_pose(
                server_id=server_id, control_epoch=control_epoch
            )
        return result

    def _step(self, action: np.ndarray) -> StepResult:
        self._check_background()
        if self._episode_finished:
            raise RuntimeError("The ARX episode has ended; call reset before step.")
        if self._chunk is None:
            self.on_action_chunk_begin()
        stamp, anchor, epoch, server_id = self._chunk
        start = time.monotonic()
        state = self._arm.get_state()
        if state.mode == "fault":
            raise RuntimeError(f"ARX service fault: {state.fault}")
        valid = (
            not self._chunk_cancelled
            and not self._terminal
            and state.server_id == server_id
        )
        human = state.mode == "drag"
        info: dict[str, Any] = {"rlt_switch_flags": self._rlt_switch}
        actual = np.asarray(action, dtype=np.float32)
        action_sample_time = action_interval_end = None
        if self._terminal:
            # Complete controlled success/failure with real time spent holding.
            # This keeps the fixed-horizon critic's target and the final reward
            # valid, without recording the cancelled model targets as executed.
            valid = (
                not self._terminal_reported
                and not self._chunk_cancelled
                and self._terminal_event != "abort"
                and state.mode == "hold"
                and self._terminal_source_epoch == epoch
                and state.control_epoch == self._terminal_epoch
                and state.server_id == server_id
            )
            if self._terminal_target is not None:
                actual = self.codec.encode_action(*self._terminal_target, anchor)
            if valid:
                self._stop.wait(self.config.action_dt)
                held = self._arm.get_state()
                valid = (
                    held.mode == "hold"
                    and held.control_epoch == self._terminal_epoch
                    and held.server_id == server_id
                )
                if self._terminal_human:
                    human = True
                    info["intervene_action"] = actual.astype(np.float32)
            observation = self._observe()
        elif human:
            target = stamp + self._chunk_index * self.config.action_dt
            end = target + self.config.action_dt
            action_sample_time, action_interval_end = target, end
            if end > time.monotonic():
                self._stop.wait(end - time.monotonic())
            history = self._arm.get_state_history(
                since=target - self.config.max_state_gap
            )
            try:
                poses, grips = interpolate_states(
                    history, [target, end], max_gap=self.config.max_state_gap
                )
                times = np.asarray([s.timestamp for s in history])
                lo = max(0, np.searchsorted(times, target, side="right") - 1)
                hi = min(len(history), np.searchsorted(times, end, side="right") + 1)
                segment = history[lo:hi]

                def measured_human_or_terminal_hold(sample: Arx5RobotState) -> bool:
                    if sample.server_id != server_id:
                        return False
                    if sample.mode == "drag" and sample.control_epoch == epoch:
                        return True
                    return (
                        self._terminal
                        and self._terminal_event != "abort"
                        and sample.mode == "hold"
                        and sample.control_epoch == self._terminal_epoch
                        and sample.timestamp >= self._terminal_time
                    )

                valid = (
                    valid
                    and state.control_epoch == epoch
                    and bool(segment)
                    and all(measured_human_or_terminal_hold(s) for s in segment)
                )
                actual = self.codec.encode_action(poses[0], grips[0], anchor)
                info["intervene_action"] = actual.astype(np.float32)
                # Refresh to current time at every synchronous boundary. Using
                # an old end-time image would accumulate inference latency on
                # every drag chunk until the robot history was exhausted.
                observation = self._observe()
            except (ValueError, TimeoutError):
                valid = False
                observation = self._observe()
        elif valid and state.mode == "policy" and state.control_epoch == epoch:
            pose, grip = self.codec.decode_action(np.asarray(action), anchor)
            grip = float(
                np.clip(
                    grip, self.config.gripper_width_min, self.config.gripper_width_max
                )
            )
            actual = self.codec.encode_action(pose, grip, anchor)
            try:
                self._arm.command_pose(pose, float(grip), expected_epoch=epoch)
            except Exception:
                current = self._arm.get_state()
                if current.control_epoch == epoch and current.server_id == server_id:
                    raise
                valid = False
            self._stop.wait(max(0, self.config.action_dt - (time.monotonic() - start)))
            after = self._arm.get_state()
            controlled_hold = (
                self._terminal
                and self._terminal_event != "abort"
                and not self._terminal_human
                and self._terminal_source_epoch == epoch
                and after.mode == "hold"
                and after.control_epoch == self._terminal_epoch
            )
            valid = (
                valid
                and (after.control_epoch == epoch or controlled_hold)
                and after.server_id == server_id
            )
            observation = self._observe()
        else:
            valid = False
            observation = self._observe()
        self._chunk_cancelled |= not valid
        self._chunk_index += 1
        self._step_count += 1
        terminal = self._terminal and (
            self._chunk_index >= self._chunk_size
            or self._terminal_event == "abort"
            or self._terminal_reported
        )
        reward = self._reward if terminal and not self._terminal_reported else 0.0
        if terminal:
            self._terminal_reported = True
        truncated = (
            self._step_count >= self.config.max_episode_steps
            and self._chunk_index >= self._chunk_size
            and not self._terminal
        )
        return_epoch = self._terminal_epoch
        if truncated:
            with self._control_lock:
                held = self._hold_on_fault()
                return_epoch = None if held is None else held.control_epoch
        info["transition_valid"] = bool(valid)
        info["executed_action"] = np.asarray(actual, dtype=np.float32)
        info["control_epoch"] = state.control_epoch
        self._record(
            "step",
            observation_time=stamp,
            next_observation_time=self._published[0],
            anchor=anchor,
            action=np.asarray(action),
            executed_action=info["executed_action"],
            human=human,
            valid=bool(valid),
            measured_pose=state.tcp_pose,
            measured_gripper=state.gripper_width,
            mode=state.mode,
            control_epoch=state.control_epoch,
            chunk_index=self._chunk_index,
            measurement_time=state.timestamp,
            action_sample_time=action_sample_time,
            action_interval_end=action_interval_end,
            chunk_size=self._chunk_size,
            reward=reward,
            terminal=terminal,
            truncated=truncated,
        )
        if (
            self.config.return_on_episode_end
            and (terminal or truncated)
            and self._chunk_index >= self._chunk_size
        ):
            self._episode_finished = True
            self._pending_return = (server_id, return_epoch)
        return observation, reward, terminal, truncated, info

    def close(self) -> None:
        """Stop owned threads and release connections without commanding a reset."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._return_cancel.set()
        try:
            if self._listener is not None and hasattr(self._listener, "close"):
                self._listener.close()
            for thread in self._threads:
                thread.join(timeout=1.0)
            # An operator callback may already be inside an SDK request when close
            # starts. Its final mode change must precede our final measured hold.
            with self._control_lock:
                self._hold_on_fault()
                if self.robot is not None:
                    self.robot.disconnect()
        finally:
            if self._preview is not None:
                self._preview.close()
