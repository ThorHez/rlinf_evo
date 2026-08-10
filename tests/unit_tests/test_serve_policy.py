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

from __future__ import annotations

import numpy as np
import pytest
import torch

from examples.embodiment import serve_policy


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.received = None

    def input_transform(self, inputs, transpose=False):
        assert transpose is False
        self.received = inputs
        return {"processed": torch.ones(1, 1)}

    def _observation_dict_to_device(self, processed):
        return processed

    def _predict_eval(self, observation, *, noise, rng):
        assert observation["processed"].shape == (1, 1)
        assert rng is None
        if noise is not None:
            assert noise.shape == (1, 2, 3)
        return torch.arange(6, dtype=torch.float32).reshape(1, 2, 3), {}


def test_policy_batches_openpi_request_and_unbatches_actions():
    model = _FakeModel()
    policy = serve_policy.RlinfOpenPiPolicy(model, default_prompt="pick it up")

    result = policy.infer(
        {
            "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation/state": np.zeros(7, dtype=np.float32),
        },
        noise=np.zeros((2, 3), dtype=np.float32),
    )

    assert model.received["observation/image"].shape == (1, 8, 8, 3)
    assert model.received["observation/state"].shape == (1, 7)
    assert model.received["prompt"] == ["pick it up"]
    np.testing.assert_array_equal(
        result["actions"], np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    assert result["policy_timing"]["infer_ms"] >= 0


def test_policy_requires_prompt_without_default():
    policy = serve_policy.RlinfOpenPiPolicy(_FakeModel())

    with pytest.raises(ValueError, match="no prompt"):
        policy.infer({"observation/state": np.zeros(7, dtype=np.float32)})


def test_build_model_config_uses_pi05_shape_and_norm_stats():
    args = serve_policy.Args(
        config_name="pi05_behavior",
        checkpoint_dir="/tmp/model",
        norm_stats="/tmp/model/assets/behavior/norm_stats.json",
        action_env_dim=23,
        action_horizon=32,
    )

    cfg = serve_policy.build_model_config(args)

    assert cfg.model_type == "openpi_rlinf"
    assert cfg.pi05 is True
    assert cfg.openpi.task == "eval"
    assert cfg.openpi.config_name == "pi05_behavior"
    assert cfg.openpi.model_action_dim == 32
    assert cfg.openpi_data.norm_stats_path.endswith("norm_stats.json")
