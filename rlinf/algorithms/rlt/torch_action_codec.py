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

"""Differentiable canonicalization of the ten-dimensional EEF action domain."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def validate_eef_action_config(
    action_dim: int,
    gripper_width_min: float,
    gripper_width_max: float | None,
    residual_scale: Sequence[float] | None = None,
) -> torch.Tensor:
    """Validate metric EEF bounds and return the ten latent residual scales."""
    if action_dim != 10:
        raise ValueError("eef_rot6d requires action_dim=10.")
    if (
        gripper_width_max is None
        or not math.isfinite(gripper_width_max)
        or not 0 <= gripper_width_min < gripper_width_max
    ):
        raise ValueError(
            "eef_rot6d requires calibrated gripper width bounds in metres."
        )
    scale = torch.tensor(
        residual_scale if residual_scale is not None else (0.1,) * 9 + (0.005,),
        dtype=torch.float32,
    )
    if scale.shape != (10,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("eef_residual_scale must contain ten finite positive values.")
    return scale


def parameterize_eef_residual(
    residual: torch.Tensor,
    reference: torch.Tensor,
    residual_scale: torch.Tensor,
    gripper_width_min: float,
    gripper_width_max: float,
) -> torch.Tensor:
    """Apply a reference residual while preserving the checkpoint action domain.

    Inputs end in ten dimensions. XYZ remains unclipped, the two rotation rows
    are projected onto SO(3), and width is bounded in metres. Exploration noise
    belongs in ``residual`` so each dimension uses its calibrated scale.
    """
    chunk = project_eef_actions(reference + residual * residual_scale)
    width = chunk[..., 9:].clamp(gripper_width_min, gripper_width_max)
    return torch.cat((chunk[..., :9], width), dim=-1)


def project_eef_actions(actions: torch.Tensor) -> torch.Tensor:
    """Keep XYZ/width unchanged and project the two rotation rows onto SO(3).

    Near-zero or parallel rows use deterministic orthogonal fallbacks. Valid
    rotations are unchanged; gradients flow through ordinary nondegenerate
    predictions, including batched chunks.
    """
    if actions.shape[-1] != 10:
        raise ValueError("EEF actions must have exactly ten dimensions.")
    first, second = actions[..., 3:6], actions[..., 6:9]
    norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    fallback_first = torch.zeros_like(first)
    fallback_first[..., 0] = 1.0
    first = torch.where(norm > 1e-6, first / norm.clamp_min(1e-6), fallback_first)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    # The axis least parallel to first gives a stable fallback for any pose.
    axis = F.one_hot(first.abs().argmin(dim=-1), num_classes=3).to(first.dtype)
    fallback_second = F.normalize(
        axis - (axis * first).sum(-1, keepdim=True) * first, dim=-1
    )
    second = torch.where(norm > 1e-6, second / norm.clamp_min(1e-6), fallback_second)
    return torch.cat((actions[..., :3], first, second, actions[..., 9:]), dim=-1)
