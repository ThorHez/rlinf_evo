# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Time alignment for camera observations and measured end-effector poses."""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_states(
    states: Sequence[Any], timestamps: Sequence[float], *, max_gap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate measured poses without extrapolation or crossing restarts.

    Positions and gripper widths are linear; orientation follows the shortest
    quaternion arc. Timestamps must use the service's host monotonic clock.
    """
    if len(states) < 2:
        raise ValueError("At least two robot samples are required for alignment.")
    times = np.asarray([s.timestamp for s in states], dtype=np.float64)
    targets = np.asarray(timestamps, dtype=np.float64)
    poses = np.asarray([s.tcp_pose for s in states], dtype=np.float64)
    grips = np.asarray([s.gripper_width for s in states], dtype=np.float64)
    if (
        max_gap <= 0
        or not np.isfinite(max_gap)
        or not np.isfinite(times).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(poses).all()
        or not np.isfinite(grips).all()
        or np.any(np.diff(times) <= 0)
        or targets.ndim != 1
        or targets.size == 0
    ):
        raise ValueError("State history must be finite and strictly time ordered.")
    if poses.shape != (len(states), 7):
        raise ValueError("Robot poses must contain XYZ and an xyzw quaternion.")
    if targets.min() < times[0] or targets.max() > times[-1]:
        raise ValueError("Robot history does not bracket the observation time.")
    upper = np.clip(np.searchsorted(times, targets, side="right"), 1, len(times) - 1)
    lower = upper - 1
    if np.any(times[upper] - times[lower] > max_gap):
        raise ValueError("Robot history has a gap larger than max_state_gap.")
    participating = states[int(lower.min()) : int(upper.max()) + 1]
    if len({state.server_id for state in participating}) != 1:
        raise ValueError("Cannot interpolate across an SDK service restart.")
    xyz = np.stack([np.interp(targets, times, poses[:, i]) for i in range(3)], axis=-1)
    quats = Slerp(times, Rotation.from_quat(poses[:, 3:]))(targets).as_quat()
    return np.concatenate([xyz, quats], axis=-1), np.interp(targets, times, grips)


def prepare_rgb(frame: np.ndarray, crop: Sequence[int] | None = None) -> np.ndarray:
    """Crop an explicitly configured BGR rectangle and resize to model RGB."""
    frame = np.asarray(frame)
    if (
        frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[-1] != 3
        or min(frame.shape[:2]) == 0
    ):
        raise ValueError("Expected an HWC uint8 BGR frame.")
    if crop is not None:
        if len(crop) != 4 or any(int(v) != v for v in crop):
            raise ValueError("image_crop must be integer [left, top, width, height].")
        left, top, width, height = map(int, crop)
        if min(left, top) < 0 or min(width, height) <= 0:
            raise ValueError(
                "Image crop must have nonnegative origin and positive size."
            )
        if left + width > frame.shape[1] or top + height > frame.shape[0]:
            raise ValueError("Image crop is outside the captured image.")
        frame = frame[top : top + height, left : left + width]
    # Match the deployment transform exactly: preserve aspect ratio, resize
    # with AREA when shrinking / LINEAR when enlarging, then center crop.
    height, width = frame.shape[:2]
    if width >= height:
        resized_height, resized_width = 224, math.ceil(224 / height * width)
        enlarge = 224 > height
    else:
        resized_width, resized_height = 224, math.ceil(224 / width * height)
        enlarge = 224 > width
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR if enlarge else cv2.INTER_AREA,
    )
    top, left = (resized_height - 224) // 2, (resized_width - 224) // 2
    return np.ascontiguousarray(resized[top : top + 224, left : left + 224, ::-1])
