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

"""Checkpoint-normalized EEF actions with a fixed observation-frame anchor.

Poses are XYZ in metres followed by an XYZW quaternion. The six rotation
numbers are the first two *rows*, matching the source OpenPI data converter.
No hardware calibration, gripper binarization, or world-Z correction is hidden
in this codec; those belong to explicit robot configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OBSERVATION_FIELDS = (
    ("robot0_eef_pos", (2, 3)),
    ("robot0_eef_rot_axis_angle", (2, 6)),
    ("robot0_eef_rot_axis_angle_wrt_start", (2, 6)),
    ("robot0_gripper_width", (2, 1)),
)


def _finite_array(value, *, name: str, last_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != last_dim:
        raise ValueError(f"{name} must end in {last_dim} values, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _pose_parts(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = _finite_array(pose, name="pose", last_dim=7)
    quat = pose[..., 3:]
    if np.any(np.linalg.norm(quat, axis=-1) < 1e-12):
        raise ValueError("Pose quaternion must be nonzero.")
    matrix = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix()
    return pose[..., :3], matrix.reshape(*pose.shape[:-1], 3, 3)


def row_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Project finite, nondegenerate row rot6d values onto SO(3)."""
    rot6d = _finite_array(rot6d, name="rot6d", last_dim=6)
    first = rot6d[..., :3]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("The first rot6d row must be nonzero.")
    first = first / first_norm
    second = rot6d[..., 3:]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("The rot6d rows must be linearly independent.")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def _xyz_affine(entry: dict, *, dim: int) -> tuple[np.ndarray, np.ndarray]:
    use_minmax = entry.get("min") is not None and entry.get("max") is not None
    lower = np.asarray(entry["min" if use_minmax else "q01"], dtype=np.float64)
    upper = np.asarray(entry["max" if use_minmax else "q99"], dtype=np.float64)
    if lower.shape != (dim,) or upper.shape != (dim,):
        raise ValueError(f"Checkpoint normalization requires {dim}-dimensional stats.")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Normalization statistics must be finite.")
    if np.any(upper < lower):
        raise ValueError("Normalization upper bounds must be >= lower bounds.")
    span = upper - lower
    constant = span < 1e-7
    scale = 2.0 / np.where(constant, 2.0, span)
    offset = np.where(constant, -lower, -1.0 - scale * lower)
    scale[3:] = 1.0
    offset[3:] = 0.0
    return scale, offset


class EEFActionCodec:
    """Convert physical EEF trajectories and original checkpoint action values.

    ``stats`` is the contents of the checkpoint's ``norm_stats`` object. The
    state entry is 16D (applied separately to both history frames), and actions
    are 10D. Min/max take precedence over q01/q99, without clipping; constant
    dimensions retain the source converter's unit scale.
    """

    def __init__(self, stats: dict):
        state_stats = stats.get("state", stats.get("observation.state"))
        if state_stats is None or "actions" not in stats:
            raise ValueError("norm_stats must contain state and actions entries.")
        self.state_scale, self.state_offset = _xyz_affine(state_stats, dim=16)
        self.action_scale, self.action_offset = _xyz_affine(stats["actions"], dim=10)

    @classmethod
    def from_stats_json(cls, path: str | Path) -> EEFActionCodec:
        """Load the same norm_stats.json configured for the Stage 1 model."""
        with Path(path).expanduser().open() as stream:
            data = json.load(stream)
        return cls(data.get("norm_stats", data))

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize XYZ while preserving row rotation and metric width."""
        action = _finite_array(action, name="action", last_dim=10)
        return action * self.action_scale + self.action_offset

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Restore metric relative XYZ without clipping model predictions."""
        action = _finite_array(action, name="action", last_dim=10)
        return (action - self.action_offset) / self.action_scale

    def encode_action(
        self, pose: np.ndarray, gripper: np.ndarray | float, anchor_pose: np.ndarray
    ) -> np.ndarray:
        """Encode absolute measured or target poses relative to one chunk anchor."""
        position, rotation = _pose_parts(pose)
        anchor_pose = np.asarray(anchor_pose)
        if anchor_pose.shape != (7,):
            raise ValueError("A chunk must have exactly one anchor pose of shape (7,).")
        anchor_position, anchor_rotation = _pose_parts(anchor_pose)
        local_position = (position - anchor_position) @ anchor_rotation
        local_rotation = anchor_rotation.T @ rotation
        gripper = np.asarray(gripper, dtype=np.float64)
        if gripper.shape == (*position.shape[:-1], 1):
            gripper = gripper[..., 0]
        gripper = np.broadcast_to(gripper, position.shape[:-1])[..., None]
        action = np.concatenate(
            (
                local_position,
                local_rotation[..., :2, :].reshape(*position.shape[:-1], 6),
                gripper,
            ),
            axis=-1,
        )
        return self.normalize_action(action)

    def decode_action(
        self, action: np.ndarray, anchor_pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode a normalized chunk using its original observation anchor."""
        action = self.unnormalize_action(action)
        anchor_pose = np.asarray(anchor_pose)
        if anchor_pose.shape != (7,):
            raise ValueError("A chunk must have exactly one anchor pose of shape (7,).")
        anchor_position, anchor_rotation = _pose_parts(anchor_pose)
        position = action[..., :3] @ anchor_rotation.T + anchor_position
        rotation = anchor_rotation @ row_rot6d_to_matrix(action[..., 3:9])
        quat = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat()
        pose = np.concatenate(
            (position, quat.reshape(*position.shape[:-1], 4)), axis=-1
        )
        return pose, action[..., 9].copy()

    def encode_observation(
        self, poses: np.ndarray, grippers: np.ndarray, episode_start_pose: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Build two raw 16D states relative to the latest pose and episode start."""
        poses = np.asarray(poses)
        if poses.shape != (2, 7):
            raise ValueError("Observation pose history must have shape (2, 7).")
        if np.asarray(episode_start_pose).shape != (7,):
            raise ValueError("Episode start pose must have shape (7,).")
        positions, rotations = _pose_parts(poses)
        _, start_rotation = _pose_parts(episode_start_pose)
        current_rotation = rotations[-1]
        grippers = np.asarray(grippers, dtype=np.float64)
        if grippers.shape not in ((2,), (2, 1)) or not np.isfinite(grippers).all():
            raise ValueError("Gripper history must contain two finite widths.")
        return {
            "robot0_eef_pos": (positions - positions[-1]) @ current_rotation,
            "robot0_eef_rot_axis_angle": (current_rotation.T @ rotations)[
                ..., :2, :
            ].reshape(2, 6),
            "robot0_eef_rot_axis_angle_wrt_start": (start_rotation.T @ rotations)[
                ..., :2, :
            ].reshape(2, 6),
            "robot0_gripper_width": grippers.reshape(2, 1),
        }

    def flatten_observation(self, fields: dict, normalize: bool = False) -> np.ndarray:
        """Flatten raw history in frame-major order, optionally scaling XYZ."""
        values = []
        for key, shape in OBSERVATION_FIELDS:
            value = np.asarray(fields[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must have finite shape {shape}.")
            values.append(value)
        state = np.concatenate(values, axis=-1)
        if normalize:
            state = state * self.state_scale + self.state_offset
        return state.reshape(32).astype(np.float32)
