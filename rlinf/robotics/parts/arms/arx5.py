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

"""ARX arm and gripper sharing one local SDK service connection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlinf.robotics.parts.base import Action, Features, RobotPart
from rlinf.robotics.parts.views import MethodEndEffector

from .arx5_protocol import Arx5Client, Arx5Error, pose_matrix, vector
from .base import Arm, BaseArm


@dataclass(frozen=True)
class Arx5RobotState:
    """Measured state in metres/radians, with a host monotonic timestamp."""

    tcp_pose: np.ndarray
    arm_joint_position: np.ndarray
    arm_joint_velocity: np.ndarray
    gripper_width: float
    timestamp: float
    mode: str
    control_epoch: int
    sequence: int
    server_id: str
    fault: str | None = None

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> Arx5RobotState:
        """Validate the observation received across the process boundary."""
        pose = vector(state["tcp_pose"], 7, "tcp_pose")
        pose_matrix(pose)
        joints = np.asarray(state["arm_joint_position"], dtype=float)
        if joints.ndim != 1 or not len(joints):
            raise Arx5Error("protocol", "invalid joint state")
        joints = vector(joints, len(joints), "joint_position")
        velocity = vector(state["arm_joint_velocity"], len(joints), "joint_velocity")
        timestamp, width = float(state["timestamp"]), float(state["gripper_width"])
        if not np.isfinite([timestamp, width]).all():
            raise Arx5Error("protocol", "invalid timestamp or gripper width")
        if state["mode"] not in {"disabled", "hold", "policy", "drag", "fault"}:
            raise Arx5Error("protocol", "invalid control mode")
        return cls(
            pose,
            joints,
            velocity,
            width,
            timestamp,
            state["mode"],
            int(state["control_epoch"]),
            int(state["sequence"]),
            str(state["server_id"]),
            state.get("fault"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return measured fields without fabricating unavailable sensors."""
        return dict(vars(self))


@Arm.register("arx5_zmq")
class Arx5Arm(BaseArm):
    """Deferred connection to one local ARX SDK service.

    Connecting only reads state. Policy actions require an explicit epoch
    captured with the observation; stale inference results cannot regain
    control after manual intervention.
    """

    STATE_FIELDS = ("tcp_pose", "arm_joint_position", "arm_joint_velocity")

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5557",
        *,
        timeout_ms: int = 1000,
        max_state_age: float = 0.25,
    ) -> None:
        if not np.isfinite(max_state_age) or max_state_age <= 0:
            raise ValueError("max_state_age must be finite and positive")
        self.endpoint, self.timeout_ms, self.max_state_age = (
            endpoint,
            timeout_ms,
            max_state_age,
        )

    @property
    def action_features(self) -> Features:
        return {
            "tcp_pose": {"shape": (7,)},
            "gripper_width": {"shape": (1,)},
            "control_epoch": {"shape": ()},
        }

    @property
    def parts(self) -> dict[str, RobotPart]:
        return {
            "end_effector": MethodEndEffector(
                self,
                state_field="gripper_width",
                command="move_gripper",
                is_gripper=True,
            )
        }

    def _open(self) -> Arx5Client:
        client = Arx5Client(self.endpoint, self.timeout_ms)
        try:
            self._parse_state(client.request("state"))
        except BaseException:
            client.close()
            raise
        return client

    def _release(self, device: Arx5Client) -> None:
        device.close()

    def _client(self) -> Arx5Client:
        if self._device is None:
            raise Arx5Error("disconnected", "connect the ARX arm before using it")
        return self._device

    def _parse_state(
        self, result: dict[str, Any], *, check_age: bool = True
    ) -> Arx5RobotState:
        state = Arx5RobotState.from_dict(result)
        if (
            check_age
            and not -0.01 <= time.monotonic() - state.timestamp <= self.max_state_age
        ):
            raise Arx5Error("stale_state", "ARX feedback exceeds max_state_age")
        return state

    def get_state(self) -> Arx5RobotState:
        return self._parse_state(self._client().request("state"))

    def get_state_history(self, since: float | None = None) -> list[Arx5RobotState]:
        """Return bounded measured history, including mode and epoch boundaries."""
        result = self._client().request("history", since=since)
        return [self._parse_state(state, check_age=False) for state in result["states"]]

    def _mode(self, operation: str, expected_epoch: int | None) -> Arx5RobotState:
        if expected_epoch is None:
            expected_epoch = self.get_state().control_epoch
        return self._parse_state(
            self._client().request(operation, expected_epoch=expected_epoch)
        )

    def enter_drag(self, expected_epoch: int | None = None) -> Arx5RobotState:
        """Give control to the operator and invalidate in-flight policy results."""
        return self._mode("enter_drag", expected_epoch)

    def resume_hold(self, expected_epoch: int | None = None) -> Arx5RobotState:
        """Hold the measured position; policy control remains disabled."""
        return self._mode("resume_hold", expected_epoch)

    def hold(self, expected_epoch: int | None = None) -> Arx5RobotState:
        """Discard future targets and hold the measured position."""
        return self._mode("hold", expected_epoch)

    def enable_policy(self, expected_epoch: int) -> Arx5RobotState:
        """Enable policy commands after explicitly establishing hold mode."""
        return self._mode("enable_policy", expected_epoch)

    def command_pose(
        self,
        tcp_pose: Any,
        gripper_width: float,
        expected_epoch: int,
        deadline: float | None = None,
    ) -> Arx5RobotState:
        """Submit an absolute calibrated target and return measured feedback."""
        payload = {
            "tcp_pose": np.asarray(tcp_pose).tolist(),
            "gripper_width": float(gripper_width),
            "expected_epoch": int(expected_epoch),
        }
        if deadline is not None:
            payload["deadline"] = float(deadline)
        return self._parse_state(self._client().request("command_pose", **payload))

    def send_action(self, action: Action) -> dict[str, Any]:
        """Submit a versioned pose/width pair; never infer the epoch at send time."""
        required = {"tcp_pose", "gripper_width", "control_epoch"}
        if set(action) != required:
            raise ValueError(f"ARX action requires exactly {sorted(required)}")
        self.command_pose(
            action["tcp_pose"],
            float(np.asarray(action["gripper_width"]).item()),
            int(action["control_epoch"]),
        )
        return dict(action)

    def move_gripper(self, target: Any) -> None:
        """Command the shared gripper in metres through the current control epoch.

        This is an explicit operator operation. Policy execution must submit
        pose and gripper together through ``command_pose``.
        """
        width = float(vector(np.asarray(target).reshape(-1), 1, "gripper_width")[0])
        state = self.get_state()
        self.command_pose(state.tcp_pose, width, state.control_epoch)

    def is_robot_up(self) -> bool:
        if not self.is_connected:
            return False
        try:
            return self.get_state().mode != "fault"
        except Arx5Error:
            return False
