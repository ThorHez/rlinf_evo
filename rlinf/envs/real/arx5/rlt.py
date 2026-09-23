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

from rlinf.algorithms.rlt.action_codec import EEFActionCodec
from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.utils.logging import get_logger

from .history import interpolate_states, prepare_rgb

if TYPE_CHECKING:
    from rlinf.robotics.discovery import RobotInfo
    from rlinf.robotics.parts.arms.arx5 import Arx5Arm, Arx5RobotState
    from rlinf.robotics.parts.cameras.base import BaseCamera
    from rlinf.scheduler import WorkerInfo

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
    image_crop: list[int] | None = None
    max_frame_age: float = 0.5
    max_state_gap: float = 0.05
    observation_timeout: float = 3.0
    history_seconds: float = 10.0
    max_episode_steps: int = 300
    enable_keyboard: bool = True
    start_paused: bool = True
    start_with_actor: bool = False
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
        ):
            value = float(getattr(self, name))
            setattr(self, name, value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.camera_latency = float(self.camera_latency)
        self.camera_fps = int(self.camera_fps)
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
        self._waiting_start = True
        self._frames = deque(
            maxlen=max(2, int(self.config.history_seconds * self.config.camera_fps))
        )
        self._background_error: Exception | None = None
        self._listener = None
        self._threads: list[threading.Thread] = []
        self._episode_start = None
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

    def _hold_on_fault(self) -> None:
        """Best-effort measured hold; the SDK service independently faults too."""
        if self._arm is None:
            return
        try:
            state = self._arm.get_state()
            if state.mode == "drag":
                self._arm.resume_hold(expected_epoch=state.control_epoch)
            elif state.mode == "policy":
                self._arm.hold(expected_epoch=state.control_epoch)
        except Exception:
            self._logger.exception("Could not hold ARX after an environment failure.")

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
        """Start an episode at the operator-established pose; never home implicitly."""
        super().reset(seed=seed)
        self._hold_on_fault()
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
        self._waiting_start = False
        if self._terminal_event == "abort":
            raise InterruptedError("ARX episode aborted while waiting to start.")
        with self._lock:
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
            return self._step(action)
        except Exception:
            self._hold_on_fault()
            raise

    def _step(self, action: np.ndarray) -> StepResult:
        self._check_background()
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
        if truncated:
            self._hold_on_fault()
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
        return observation, reward, terminal, truncated, info

    def close(self) -> None:
        """Stop owned threads and release connections without commanding a reset."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
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
