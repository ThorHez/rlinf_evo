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

"""Small ARX SDK simulation at the Cartesian controller boundary."""

import copy
import time
from types import SimpleNamespace

import numpy as np


class JointState:
    def __init__(self, dof):
        self._pos = np.zeros(dof)
        self._vel = np.zeros(dof)
        self._torque = np.zeros(dof)
        self.timestamp = 0.0
        self.gripper_pos = 0.04
        self.gripper_vel = self.gripper_torque = 0.0

    def pos(self):
        return self._pos

    def vel(self):
        return self._vel

    def torque(self):
        return self._torque


class Gain:
    def __init__(self, kp, kd=None, gripper_kp=0.0, gripper_kd=0.0):
        self._kp = np.zeros(kp) if kd is None else np.asarray(kp).copy()
        self._kd = np.zeros(kp) if kd is None else np.asarray(kd).copy()
        self.gripper_kp, self.gripper_kd = gripper_kp, gripper_kd

    def kp(self):
        return self._kp

    def kd(self):
        return self._kd


class EEFState:
    def __init__(self, pose, gripper_pos):
        self._pose = np.asarray(pose).copy()
        self.timestamp = 0.0
        self.gripper_pos = gripper_pos
        self.gripper_vel = self.gripper_torque = 0.0

    def pose_6d(self):
        return self._pose


class FakeArxSDK:
    JointState = JointState
    EEFState = EEFState
    Gain = Gain


class FakeArxController:
    """Emulate feedback, target buffers, gains, and SDK failure injection."""

    def __init__(self):
        self.robot = SimpleNamespace(
            joint_dof=6,
            joint_pos_min=np.full(6, -3.0),
            joint_pos_max=np.full(6, 3.0),
            gripper_width=0.085,
        )
        self.config = SimpleNamespace(
            default_kp=np.full(6, 30.0),
            default_kd=np.ones(6),
            default_gripper_kp=2.0,
            default_gripper_kd=0.1,
            gravity_compensation=True,
            controller_dt=0.002,
            default_preview_time=0.02,
        )
        self.gain = Gain(6)
        self.gain.kd()[:] = 1
        self.state = JointState(6)
        self.state.pos()[:] = [0.25, 0.0, 0.35, 0.0, 0.0, 0.0]
        self.command = copy.deepcopy(self.state)
        self.started = time.monotonic() - 1.0
        self.feedback_age = 0.0
        self.feedback_period = 0.0
        self.fail_read = self.fail_command = self.ignore_gain = False
        self.command_count = 0
        self.gain_history = []

    def get_timestamp(self):
        return time.monotonic() - self.started

    def get_robot_config(self):
        return self.robot

    def get_controller_config(self):
        return self.config

    def get_joint_state(self):
        if self.fail_read:
            raise RuntimeError("disconnected CAN")
        state = copy.deepcopy(self.state)
        state.timestamp = self.get_timestamp() - self.feedback_age
        if self.feedback_period:
            state.timestamp = (
                np.floor(state.timestamp / self.feedback_period) * self.feedback_period
            )
        return state

    def get_joint_cmd(self):
        return copy.deepcopy(self.command)

    def get_gain(self):
        return copy.deepcopy(self.gain)

    def set_gain(self, gain):
        if not self.ignore_gain:
            self.gain = copy.deepcopy(gain)
        self.gain_history.append(copy.deepcopy(self.gain))

    def set_eef_cmd(self, command):
        if self.fail_command:
            raise RuntimeError("CAN command failed")
        self.command_count += 1
        # Synthetic kinematics map each pose component to one joint directly.
        target = JointState(self.robot.joint_dof)
        target.pos()[:] = command.pose_6d()
        target.timestamp = command.timestamp or (
            self.get_timestamp() + self.config.default_preview_time
        )
        target.gripper_pos = command.gripper_pos
        target.gripper_vel = command.gripper_vel
        target.gripper_torque = command.gripper_torque
        self.command = target
        if np.any(self.gain.kp() > 0):
            self.state = copy.deepcopy(target)


class FakeArxSolver:
    """Forward kinematics matching the synthetic Cartesian controller."""

    def forward_kinematics(self, joints):
        return np.asarray(joints).copy()


class InProcessArxClient:
    """Transport boundary for environment tests using the real SDK service."""

    def __init__(self, service):
        self.service = service

    def request(self, operation, **payload):
        from rlinf.robotics.parts.arms.arx5_protocol import PROTOCOL_VERSION, Arx5Error

        reply = self.service.handle(
            {
                "version": PROTOCOL_VERSION,
                "request_id": "environment-test",
                "operation": operation,
                "server_id": self.service.server_id,
                "deadline": time.monotonic() + 1,
                **payload,
            }
        )
        if not reply["ok"]:
            raise Arx5Error(**reply["error"])
        return reply["result"]

    def close(self):
        pass


class TimedCamera:
    """Synthetic camera readings; processing/alignment remains real RLinf code."""

    def __init__(self, interval=0.005):
        self.interval = interval
        self.sequence = 0
        self.fail = False
        self.offset = 0.0

    def get_frame_with_timestamp(self, timeout=0.2):
        time.sleep(self.interval)
        if self.fail:
            raise OSError("camera disconnected")
        self.sequence += 1
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[:] = [11, 22, 33]
        return time.monotonic() + self.offset, frame, self.sequence
