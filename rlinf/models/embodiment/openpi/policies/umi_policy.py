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

"""UMI V4 observations and checkpoint-specific XYZ normalization."""

import dataclasses

import numpy as np
from openpi import transforms


@dataclasses.dataclass(frozen=True)
class UmiV4Inputs(transforms.DataTransformFn):
    """Pack two 16-dimensional states and the latest wrist image."""

    def __call__(self, data: dict) -> dict:
        fields = (
            ("robot0_eef_pos", (2, 3)),
            ("robot0_eef_rot_axis_angle", (2, 6)),
            ("robot0_eef_rot_axis_angle_wrt_start", (2, 6)),
            ("robot0_gripper_width", (2, 1)),
        )
        states = []
        for key, shape in fields:
            value = np.asarray(data[key])
            if value.shape != shape:
                raise ValueError(
                    f"UMI V4 {key} must have shape {shape}, got {value.shape}."
                )
            states.append(value)
        image = np.asarray(data["left_wrist_0_rgb_1"])
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape == (3, 224, 224):
            image = image.transpose(1, 2, 0)
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError("UMI V4 wrist images must be 224x224 RGB images.")
        result = {
            "state": np.concatenate(states, axis=-1),
            "image": {
                "base_0_rgb": np.zeros_like(image),
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": data["prompt"],
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape != (16, 10):
                raise ValueError(
                    "UMI V4 actions must already contain one (16, 10) action chunk; "
                    f"got {actions.shape}."
                )
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class UmiV4Normalize(transforms.DataTransformFn):
    """Apply checkpoint min/max normalization to XYZ; preserve all other dims."""

    affine: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def from_stats(cls, stats: dict) -> "UmiV4Normalize":
        """Build the original UMI affine transform, including constant ranges."""
        affine = {}
        for key, dim in (("state", 16), ("actions", 10)):
            entry = stats[key]
            use_minmax = entry.get("min") is not None and entry.get("max") is not None
            lower = np.asarray(entry["min" if use_minmax else "q01"], dtype=np.float64)
            upper = np.asarray(entry["max" if use_minmax else "q99"], dtype=np.float64)
            if lower.shape != (dim,) or upper.shape != (dim,):
                raise ValueError(f"UMI V4 {key} norm stats must have {dim} dimensions.")
            input_range = upper - lower
            constant = input_range < 1e-7
            input_range[constant] = 2.0
            scale = 2.0 / input_range
            offset = -1.0 - scale * lower
            offset[constant] = -lower[constant]
            scale[3:] = 1.0
            offset[3:] = 0.0
            affine[key] = (scale, offset)
        return cls(affine)

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        for key, (scale, offset) in self.affine.items():
            if key in result:
                result[key] = np.asarray(result[key]) * scale + offset
        return result


@dataclasses.dataclass(frozen=True)
class UmiV4ModelInputs(transforms.DataTransformFn):
    """Flatten state after prompt tokenization and pad normalized actions."""

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        result["state"] = np.asarray(result["state"]).reshape(32).astype(np.float32)
        if "actions" in result:
            result["actions"] = transforms.pad_to_dim(
                np.asarray(result["actions"], dtype=np.float32), 32
            )
        return result


@dataclasses.dataclass(frozen=True)
class UmiV4Outputs(transforms.DataTransformFn):
    """Decode the ten physical action dimensions; discard model padding."""

    scale: np.ndarray
    offset: np.ndarray

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < 10:
            raise ValueError("UMI V4 model actions must have at least ten dimensions.")
        return {**data, "actions": (actions[..., :10] - self.offset) / self.scale}
