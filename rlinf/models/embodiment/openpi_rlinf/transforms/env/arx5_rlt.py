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

"""Online repack for the two-frame EEF RLT model contract."""

from typing import Callable

from rlinf.algorithms.rlt.action_codec import OBSERVATION_FIELDS


def repack_env_obs(env_obs: dict, *, select_state: Callable) -> dict:
    """Forward explicit raw model fields; never synthesize missing history."""
    del select_state
    keys = [key for key, _ in OBSERVATION_FIELDS] + ["left_wrist_0_rgb_1"]
    missing = [key for key in keys if key not in env_obs]
    if missing:
        raise ValueError(f"ARX RLT observations are missing model fields: {missing}.")
    result = {key: env_obs[key] for key in keys}
    for key, shape in OBSERVATION_FIELDS:
        if tuple(result[key].shape[1:]) != shape:
            raise ValueError(f"Batched {key} must have trailing shape {shape}.")
    result["prompt"] = env_obs["task_descriptions"]
    return result
