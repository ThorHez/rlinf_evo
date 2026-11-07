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

"""OpenPI loader configuration for prechunked UMI V4 LeRobot datasets."""

import dataclasses
import json
import pathlib

import openpi.models.model as model
import openpi.models.tokenizer as tokenizer
import openpi.transforms as transforms
from openpi.training.config import DataConfig, DataConfigFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies.umi_policy import (
    UmiV4FlattenState,
    UmiV4Inputs,
    UmiV4ModelInputs,
    UmiV4Normalize,
    UmiV4Outputs,
)


@dataclasses.dataclass(frozen=True)
class LeRobotUmiV4DataConfig(DataConfigFactory):
    """Preserve UMI state history, prechunked actions and XYZ-only scaling.

    The configured asset directory must contain the original checkpoint's
    ``norm_stats.json``. UMI transforms own normalization in both directions;
    ``norm_stats={}`` makes OpenPI's generic quantile transform a no-op in the
    training loader and RLinf's evaluation transform builder.
    """

    from_start: bool = False
    image_horizon: int = 2
    image_slot: str = "left_wrist_0_rgb"
    flatten_state_before_tokenization: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: model.BaseModelConfig
    ) -> DataConfig:
        if (
            model_config.model_type != model.ModelType.PI05
            or model_config.action_dim != 32
            or model_config.action_horizon != 16
        ):
            raise ValueError(
                "UMI V4 requires Pi0.5 with action_dim=32 and action_horizon=16."
            )
        if self.image_horizon not in (1, 2):
            raise ValueError("UMI image_horizon must be 1 or 2.")
        asset_id = self.assets.asset_id or self.repo_id
        stats_path = (
            pathlib.Path(self.assets.assets_dir or assets_dirs)
            / asset_id
            / "norm_stats.json"
        )
        stats = json.loads(stats_path.read_text())["norm_stats"]
        normalizer = UmiV4Normalize.from_stats(
            stats, state_dim=10 if self.from_start else 16
        )
        action_scale, action_offset = normalizer.affine["actions"]
        repack = {
            "robot0_eef_rot_axis_angle_wrt_start": "observation.robot0_eef_rot_axis_angle_wrt_start",
            "robot0_gripper_width": "observation.robot0_gripper_width",
            "left_wrist_0_rgb_1": f"observation.left_wrist_0_rgb_{self.image_horizon - 1}",
            "actions": "actions",
            "prompt": "task",
        }
        if self.from_start:
            repack["robot0_eef_pos_wrt_start"] = "observation.robot0_eef_pos_wrt_start"
        else:
            repack.update(
                robot0_eef_pos="observation.robot0_eef_pos",
                robot0_eef_rot_axis_angle="observation.robot0_eef_rot_axis_angle",
            )
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=self.repo_id,
            asset_id=asset_id,
            norm_stats={},
            use_quantile_norm=False,
            prompt_from_task=True,
            action_sequence_keys=(),
            repack_transforms=transforms.Group(
                inputs=[transforms.RepackTransform(repack)]
            ),
            data_transforms=transforms.Group(
                inputs=[
                    UmiV4Inputs(from_start=self.from_start, image_slot=self.image_slot),
                    normalizer,
                ],
                outputs=[UmiV4Outputs(action_scale, action_offset)],
            ),
            model_transforms=transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    *(
                        [UmiV4FlattenState()]
                        if self.from_start and self.flatten_state_before_tokenization
                        else []
                    ),
                    transforms.TokenizePrompt(
                        tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=True,
                    ),
                    UmiV4ModelInputs(state_dim=20 if self.from_start else 32),
                ]
            ),
        )
