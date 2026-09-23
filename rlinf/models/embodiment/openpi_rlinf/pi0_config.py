# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi0 config for PyTorch, aligned with JAX models/pi0_config.py."""

from __future__ import annotations

import dataclasses
import math

import torch

from rlinf.models.embodiment.openpi_rlinf.modules import gemma, model, pointnet


@dataclasses.dataclass
class Pi0Config:
    dtype: str = "bfloat16"
    # Optional backbone compute precision, independent of parameter storage.
    compute_dtype: str | None = None
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    pointnet_variant: pointnet.Variant = "pcd"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    pi05: bool = False
    discrete_state_input: bool | None = None
    pcd: bool = False
    # Nonnegative per-dimension SFT weights; None averages all action dimensions.
    action_loss_mask: tuple[float, ...] | None = None
    # Match the UMI V4 source environment's augmax wrist-image augmentation.
    umi_v4_image_augmentation: bool = False

    def __post_init__(self):
        if self.compute_dtype not in (None, "float32", "bfloat16"):
            raise ValueError("compute_dtype must be float32, bfloat16, or None.")
        if self.action_loss_mask is not None:
            if len(self.action_loss_mask) != self.action_dim:
                raise ValueError(
                    "action_loss_mask must have one weight per model action "
                    f"dimension ({self.action_dim}); got {len(self.action_loss_mask)}."
                )
            if any(
                not math.isfinite(weight) or weight < 0
                for weight in self.action_loss_mask
            ):
                raise ValueError(
                    "action_loss_mask weights must be finite and nonnegative."
                )
            weight_sum = sum(self.action_loss_mask)
            if not math.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError("action_loss_mask must have a finite, positive sum.")
        if self.pi05 and self.max_token_len == 48:
            object.__setattr__(self, "max_token_len", 200)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    def fake_obs(self, batch_size: int = 1) -> model.Observation:
        image = torch.ones(batch_size, *model.IMAGE_RESOLUTION, 3)
        image_mask = torch.ones(batch_size, dtype=torch.bool)
        return model.Observation(
            images={
                "base_0_rgb": image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": image,
            },
            image_masks={
                "base_0_rgb": image_mask,
                "left_wrist_0_rgb": image_mask,
                "right_wrist_0_rgb": image_mask,
            },
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.ones(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
            pcd_xyz=torch.ones(batch_size, 16, 2025, 3) if self.pcd else None,
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        return torch.ones(batch_size, self.action_horizon, self.action_dim)
