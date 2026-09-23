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

"""Shared helpers: metrics, entropy aggregation, checkpoint paths, and resume."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.utils import compute_entropy_loss
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.utils.metric_utils import compute_evaluate_metrics, compute_rollout_metrics


@pytest.mark.parametrize("params_collection", [False, True])
@pytest.mark.parametrize("leaf_format", ["plain", "nnx", "mixed"])
def test_load_jax_params_preserves_values_and_module_names(
    tmp_path, params_collection, leaf_format
):
    jax = pytest.importorskip("jax")
    ocp = pytest.importorskip("orbax.checkpoint")
    from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi_rlinf import _load_jax_params

    # These values would change if loading passed through bfloat16.
    kernel = np.array([[1.0001, -2.0003], [0.1234567, 17.12345]], dtype=np.float32)
    expected = {
        "PaliGemma": {"img": {"attention": {"value": {"kernel": kernel}}}},
        "action_in_proj": {"kernel": kernel.T, "bias": kernel[0]},
    }
    params = jax.tree_util.tree_map(jax.numpy.asarray, expected)
    if leaf_format == "nnx":
        params = jax.tree_util.tree_map(lambda array: {"value": array}, params)
    elif leaf_format == "mixed":
        params["action_in_proj"]["kernel"] = {
            "value": params["action_in_proj"]["kernel"]
        }
    saved = {"params": params} if params_collection else params
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(str(tmp_path / "params"), saved)

    restored = _load_jax_params(tmp_path)

    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
        expected
    )
    for actual, original in zip(
        jax.tree_util.tree_leaves(restored),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, original)


def test_compute_evaluate_metrics_reports_interact_delay_wait_time_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0]),
                "interact_delay": torch.tensor([0.10, 0.30]),
            },
            {
                "success": torch.tensor([0.0, 1.0]),
                "interact_delay": torch.tensor([0.20, 0.40]),
            },
        ]
    )

    assert math.isclose(float(metrics["success"]), 0.5)
    assert float(metrics["average_delay"]) == pytest.approx(0.25)
    assert float(metrics["median_delay"]) == pytest.approx(0.25)
    assert float(metrics["max_delay"]) == pytest.approx(0.40)
    assert float(metrics["min_delay"]) == pytest.approx(0.10)
    assert metrics["num_trajectories"] == 4


def test_compute_evaluate_metrics_ignores_delay_samples_for_trajectory_count():
    metrics = compute_evaluate_metrics(
        [{"interact_delay": torch.tensor([0.05, 0.15, 0.25])}]
    )

    assert float(metrics["average_delay"]) == pytest.approx(0.15)
    assert metrics["num_trajectories"] == 0


def test_compute_evaluate_metrics_reports_prefixed_interact_delay_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/success": torch.tensor([1.0]),
                "env/interact_delay": torch.tensor([0.12, 0.24]),
            }
        ]
    )

    assert float(metrics["env/average_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/median_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/max_delay"]) == pytest.approx(0.24)
    assert float(metrics["env/min_delay"]) == pytest.approx(0.12)


@pytest.fixture
def single_rank_reduction(monkeypatch):
    from rlinf.scheduler.worker.worker import Worker

    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(current_device=lambda: "cpu")
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)


def test_compute_rollout_metrics_reports_loss_mask_fraction(single_rank_reduction):
    metrics = compute_rollout_metrics(
        {
            "loss_mask": torch.tensor([[[True], [False]], [[True], [True]]]),
            "rewards": torch.tensor([[[1.0], [8.0]], [[2.0], [3.0]]]),
        }
    )

    assert metrics["loss_mask_fraction"] == pytest.approx(0.75)
    assert metrics["rewards"] == pytest.approx(2.0)


def test_compute_rollout_metrics_omits_loss_mask_fraction_without_mask(
    single_rank_reduction,
):
    metrics = compute_rollout_metrics({"rewards": torch.tensor([[[1.0], [3.0]]])})

    assert "loss_mask_fraction" not in metrics
    assert metrics["rewards"] == pytest.approx(2.0)


# The embodied actor's entropy bonus aggregation. The shapes below are the ones
# the shipped models actually emit: openpi, openpi_rlinf, lingbotvla,
# dexbotic_pi, dexbotic_dm0 and flow_policy all reduce entropy to [bsz, 1],
# cnn_policy returns [bsz, action_dim], openvla_oft returns [bsz, seq_len], and
# the StarVLA action heads return [bsz, num_action_chunks, action_dim].
# loss_mask is [bsz, 1] under reward_type: chunk_level and
# [bsz, num_action_chunks] otherwise.


def _entropy(*shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=generator) + 0.5


@pytest.mark.parametrize("batch_size", [4, 16, 64, 500])
def test_chunk_level_entropy_does_not_scale_with_batch_size(batch_size):
    """The bug signature: entropy_loss came out multiplied by the micro-batch size."""
    entropy = _entropy(batch_size, 1)
    loss_mask = torch.ones(batch_size, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_chunk_level_entropy_averages_only_the_valid_rows():
    entropy = _entropy(16, 1)
    loss_mask = torch.zeros(16, 1, dtype=torch.bool)
    loss_mask[:6] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:6].mean()), rel=1e-6)


def test_chunk_level_sums_a_wide_entropy_before_averaging():
    # cnn_policy shape: one entropy per action dimension.
    entropy = _entropy(12, 4)
    loss_mask = torch.zeros(12, 1, dtype=torch.bool)
    loss_mask[:5] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:5].sum(dim=-1).mean()), rel=1e-6)


def test_token_level_averages_over_every_valid_element():
    # openvla_oft shape: entropy per token, mask per chunk step.
    entropy = _entropy(10, 7)
    loss_mask = torch.zeros(10, 1, dtype=torch.bool)
    loss_mask[:4] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:4].mean()), rel=1e-6)


def test_action_level_sums_action_dim_then_averages():
    entropy = _entropy(6, 3 * 7)
    loss_mask = torch.zeros(6, 1, dtype=torch.bool)
    loss_mask[:2] = True

    got = compute_entropy_loss(
        entropy, "action_level", loss_mask, action_dim=7, batch_size=6
    )

    per_chunk = entropy.reshape(6, 3, 7).sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:2].mean()), rel=1e-6)


def test_a_wider_mask_than_entropy_still_weights_by_valid_steps():
    # lingbotvla: entropy is [bsz, 1] while reward_type != chunk_level keeps the
    # mask at [bsz, num_action_chunks]. Each sample is weighted by its valid steps.
    entropy = _entropy(5, 1)
    loss_mask = torch.zeros(5, 4, dtype=torch.bool)
    loss_mask[0, :4] = True
    loss_mask[1, :1] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    expected = (entropy[0, 0] * 4 + entropy[1, 0] * 1) / 5
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_three_dim_entropy_reduces_to_the_mask_rank():
    # StarVLA action heads return [bsz, num_action_chunks, action_dim]; the
    # chunk_level sum already lands on the mask's rank, so nothing is unsqueezed.
    entropy = _entropy(4, 8, 7)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:, :3].mean()), rel=1e-6)


def test_three_dim_entropy_is_right_when_batch_equals_num_chunks():
    # Same shape family with bsz == num_action_chunks, where a rank mismatch
    # broadcasts successfully instead of raising and would go unnoticed.
    entropy = _entropy(8, 8, 7)
    loss_mask = torch.zeros(8, 8, dtype=torch.bool)
    loss_mask[:3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:3].mean()), rel=1e-6)


def test_no_mask_averages_everything():
    entropy = _entropy(9, 1)

    got = compute_entropy_loss(entropy, "chunk_level", None)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_a_fully_masked_batch_contributes_zero():
    entropy = _entropy(8, 1)
    loss_mask = torch.zeros(8, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(0.0)


def test_entropy_loss_keeps_the_gradient_path():
    entropy = _entropy(8, 1).requires_grad_(True)
    loss_mask = torch.ones(8, 1, dtype=torch.bool)

    compute_entropy_loss(entropy, "chunk_level", loss_mask).backward()

    # A correct mean spreads 1/8 of the gradient onto each row; the outer-product
    # bug put 1.0 on each instead.
    assert torch.allclose(entropy.grad, torch.full((8, 1), 1 / 8))


def test_entropy_loss_rejects_a_model_that_computes_no_entropy():
    # gr00t, abot_m0 and evo1 return entropy=None; pairing one with a non-zero
    # entropy_bonus is a config error, not a zero bonus.
    with pytest.raises(ValueError, match="algorithm.entropy_bonus"):
        compute_entropy_loss(None, "chunk_level", torch.ones(4, 1, dtype=torch.bool))


def _load_checkpoint_utils():
    module_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "checkpoint.py"
    )
    assert module_path.exists(), "checkpoint path utilities are not implemented"
    spec = importlib.util.spec_from_file_location(
        "_rlinf_utils_checkpoint_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/global_step_30",
        "/tmp/checkpoints/global_step_30/",
        "/tmp/checkpoints/global_step_30///",
    ],
)
def test_parse_global_step_accepts_trailing_slashes(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    assert (
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path) == 30
    )


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/step_30",
        "/tmp/checkpoints/global_step_latest/",
        "/tmp/checkpoints/global_step_30/actor",
    ],
)
def test_parse_global_step_rejects_invalid_checkpoint_directories(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    with pytest.raises(ValueError, match="global_step_<step>"):
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path)


class _StubRunner:
    """Expose only the checkpoint helpers and state used by these tests."""

    def __init__(self, critic=None):
        self.critic = critic

    _is_complete_checkpoint = ReasoningRunner._is_complete_checkpoint


class _ImmediateHandle:
    def wait(self):
        return None


class _Actor:
    def save_checkpoint(self, path: str, _step: int):
        os.makedirs(path, exist_ok=True)
        return _ImmediateHandle()


class _Dataloader:
    def state_dict(self):
        return {"offset": 3}


def _write_checkpoint(
    root: Path, step: int, *, complete: bool, with_critic: bool = False
) -> Path:
    checkpoint_dir = root / f"global_step_{step}"
    (checkpoint_dir / "actor").mkdir(parents=True)
    if with_critic:
        (checkpoint_dir / "critic").mkdir()
    if complete:
        data_dir = checkpoint_dir / "data"
        data_dir.mkdir()
        (data_dir / "data.pt").write_bytes(b"dataloader-state")
    return checkpoint_dir


def _resolve_auto_resume(log_path: Path, *, critic=None) -> str | None:
    cfg = OmegaConf.create(
        {"runner": {"resume_dir": "auto", "logger": {"log_path": str(log_path)}}}
    )
    runner = _StubRunner(critic=critic)
    runner.cfg = cfg
    runner.init_rollout_workers = lambda: None
    runner.init_actor_critic_workers = lambda: None

    ReasoningRunner.init_workers(runner)
    return cfg.runner.resume_dir


def _saving_runner(tmp_path: Path) -> _StubRunner:
    runner = _StubRunner()
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "output_dir": str(tmp_path),
                "experiment_name": "experiment",
            }
        }
    )
    runner.global_steps = 8
    runner.actor = _Actor()
    runner.train_dataloader = _Dataloader()
    return runner


@pytest.mark.parametrize(
    "completeness,expected_step",
    [
        pytest.param({40: True, 80: False}, 40, id="skips-the-incomplete-newest"),
        pytest.param({40: True, 80: True}, 80, id="takes-the-newest-complete"),
        pytest.param({40: False}, None, id="starts-fresh-when-none-is-complete"),
    ],
)
def test_auto_resume_selects_the_newest_complete_checkpoint(
    tmp_path, completeness, expected_step
):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    for step, complete in completeness.items():
        _write_checkpoint(checkpoints_dir, step, complete=complete)

    expected = (
        None
        if expected_step is None
        else str(checkpoints_dir / f"global_step_{expected_step}")
    )
    assert _resolve_auto_resume(tmp_path) == expected


def test_checkpoint_requires_the_critic_only_when_configured(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    checkpoint = _write_checkpoint(checkpoints_dir, 40, complete=True)

    assert _StubRunner()._is_complete_checkpoint(str(checkpoint))
    assert not _StubRunner(critic=object())._is_complete_checkpoint(str(checkpoint))


def test_dataloader_state_is_published_atomically(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)
    written_paths = []

    def save(_state, path):
        written_paths.append(path)
        Path(path).write_bytes(b"complete")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", save)

    ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert written_paths == [f"{final_path}.tmp"]
    assert final_path.read_bytes() == b"complete"
    assert not Path(f"{final_path}.tmp").exists()
    assert runner._is_complete_checkpoint(str(checkpoint))


def test_interrupted_dataloader_save_does_not_publish_completion(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)

    def interrupted_save(_state, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", interrupted_save)

    with pytest.raises(RuntimeError, match="interrupted"):
        ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert not final_path.exists()
    assert not Path(f"{final_path}.tmp").exists()
    assert not runner._is_complete_checkpoint(str(checkpoint))


_REDIRECTED_ENTRYPOINT = textwrap.dedent(
    """
    import hydra

    from rlinf.scheduler import Cluster
    from rlinf.utils.utils import output_redirector


    @hydra.main(version_base="1.1", config_path=None)
    @output_redirector
    def main(cfg):
        Cluster(num_nodes=1)
        print("entrypoint ran")
        if cfg.outcome == "raise":
            raise RuntimeError("entrypoint failed")


    main()
    """
)


def _run_redirected_entrypoint(tmp_path, outcome):
    script = tmp_path / "entrypoint.py"
    script.write_text(_REDIRECTED_ENTRYPOINT)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            f"+outcome={outcome}",
            f"+runner.output_dir={tmp_path}",
            "+runner.experiment_name=exp",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_redirected_entrypoint_exits_zero_and_keeps_its_log(tmp_path):
    result = _run_redirected_entrypoint(tmp_path, "return")

    assert result.returncode == 0, result.stderr
    assert "entrypoint ran" in (tmp_path / "exp" / "log" / "main.log").read_text()


def test_redirected_entrypoint_failure_is_not_reported_as_success(tmp_path):
    # Hydra catches the exception and calls sys.exit(1) itself, so the failure
    # never reaches sys.excepthook.
    result = _run_redirected_entrypoint(tmp_path, "raise")

    assert result.returncode == 1, result.stderr
    assert "entrypoint failed" in (tmp_path / "exp" / "log" / "main.log").read_text()


def _eef_codec_stats():
    """Distinct XYZ ranges expose swapped state/action or min/quantile stats."""
    return {
        key: {
            "min": [-0.4, -0.1, 0.03] + [-9.0] * (dim - 3),
            "max": [0.6, 0.3, 0.03] + [9.0] * (dim - 3),
            "q01": [-0.2, -0.05, -0.08] + [-1.0] * (dim - 3),
            "q99": [0.2, 0.05, 0.08] + [1.0] * (dim - 3),
        }
        for key, dim in (("state", 16), ("actions", 10))
    }


def test_eef_codec_roundtrip_uses_fixed_anchor_and_rotation_rows():
    from scipy.spatial.transform import Rotation

    from rlinf.algorithms.rlt.action_codec import EEFActionCodec

    codec = EEFActionCodec(_eef_codec_stats())
    anchor_rotation = Rotation.from_euler("xyz", [0.4, -0.6, 1.2])
    relative_rotation = Rotation.from_euler(
        "xyz", [[-0.3, 0.2, 0.8], [0.7, 0.4, -0.2], [0.1, -0.9, 0.3]]
    )
    anchor = np.r_[[0.3, -0.2, 0.4], anchor_rotation.as_quat()]
    relative_xyz = np.array([[0.1, 0.2, 0.3], [0.3, -0.1, 0.4], [-0.2, 0.4, 0.1]])
    poses = np.c_[
        anchor_rotation.apply(relative_xyz) + anchor[:3],
        (anchor_rotation * relative_rotation).as_quat(),
    ]
    widths = np.array([0.012, 0.045, 0.083])
    encoded = codec.encode_action(poses, widths, anchor)
    raw = codec.unnormalize_action(encoded)
    np.testing.assert_allclose(raw[:, :3], relative_xyz, atol=1e-12)
    np.testing.assert_allclose(
        raw[:, 3:9], relative_rotation.as_matrix()[:, :2, :].reshape(3, 6), atol=1e-12
    )
    assert not np.allclose(
        raw[:, 3:9], relative_rotation.as_matrix()[:, :, :2].reshape(3, 6)
    )
    recovered, recovered_widths = codec.decode_action(encoded, anchor)
    np.testing.assert_allclose(recovered[:, :3], poses[:, :3], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(recovered[:, 3:]).as_matrix(),
        Rotation.from_quat(poses[:, 3:]).as_matrix(),
        atol=1e-12,
    )
    np.testing.assert_array_equal(recovered_widths, widths)
    one_pose, one_width = codec.decode_action(encoded[1], anchor)
    np.testing.assert_allclose(one_pose[:3], poses[1, :3], atol=1e-12)
    assert one_width == widths[1]


@pytest.mark.parametrize("use_minmax", [True, False])
def test_eef_codec_normalization_matches_openpi_transform_and_never_clips(use_minmax):
    from rlinf.algorithms.rlt.action_codec import EEFActionCodec
    from rlinf.models.embodiment.openpi.policies.umi_policy import UmiV4Normalize

    stats = _eef_codec_stats()
    if not use_minmax:
        for entry in stats.values():
            entry.pop("min")
            entry.pop("max")
    codec = EEFActionCodec(stats)
    actions = np.linspace(-0.6, 0.7, 160).reshape(16, 10)
    normalized = codec.normalize_action(actions)
    expected = UmiV4Normalize.from_stats(stats)({"actions": actions})["actions"]
    np.testing.assert_array_equal(normalized, expected)
    np.testing.assert_allclose(
        codec.unnormalize_action(normalized), actions, atol=1e-12
    )
    np.testing.assert_array_equal(normalized[:, 3:], actions[:, 3:])
    assert np.any(np.abs(normalized[:, :3]) > 1.0)
    if use_minmax:
        np.testing.assert_array_equal(normalized[:, 2], actions[:, 2] - 0.03)


def test_eef_codec_observation_history_and_checkpoint_file(tmp_path):
    import json

    from scipy.spatial.transform import Rotation

    from rlinf.algorithms.rlt.action_codec import EEFActionCodec
    from rlinf.models.embodiment.openpi.policies.umi_policy import UmiV4Normalize

    stats = _eef_codec_stats()
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    codec = EEFActionCodec.from_stats_json(path)
    rotations = Rotation.from_euler("xyz", [[0.2, 0.4, 0.7], [-0.3, 0.5, 0.2]])
    poses = np.c_[[[0.2, 0.3, 0.4], [0.3, 0.2, 0.5]], rotations.as_quat()]
    start_rotation = Rotation.from_euler("xyz", [-0.5, -0.1, 0.3])
    start = np.r_[[0.1, 0.2, 0.3], start_rotation.as_quat()]
    fields = codec.encode_observation(poses, np.array([0.02, 0.06]), start)
    np.testing.assert_allclose(fields["robot0_eef_pos"][-1], 0, atol=1e-12)
    np.testing.assert_allclose(
        fields["robot0_eef_rot_axis_angle"][-1], [1, 0, 0, 0, 1, 0], atol=1e-12
    )
    expected_start = (
        (start_rotation.inv() * rotations).as_matrix()[:, :2, :].reshape(2, 6)
    )
    np.testing.assert_allclose(
        fields["robot0_eef_rot_axis_angle_wrt_start"], expected_start, atol=1e-12
    )
    raw = codec.flatten_observation(fields).reshape(2, 16)
    normalized = codec.flatten_observation(fields, normalize=True)
    oracle = UmiV4Normalize.from_stats(stats)({"state": raw})["state"]
    np.testing.assert_allclose(normalized, oracle.reshape(32), atol=1e-7)
    np.testing.assert_allclose(raw[:, -1], [0.02, 0.06], atol=1e-8)


def test_eef_codec_rejects_bad_geometry_and_stats():
    from rlinf.algorithms.rlt.action_codec import EEFActionCodec

    codec = EEFActionCodec(_eef_codec_stats())
    anchor = np.array([0, 0, 0, 0, 0, 0, 1.0])
    for action, message in [
        (np.full(10, np.nan), "finite"),
        (np.zeros(10), "nonzero"),
        (np.array([0, 0, 0, 1, 0, 0, 2, 0, 0, 0]), "independent"),
    ]:
        with pytest.raises(ValueError, match=message):
            codec.decode_action(action, anchor)
    with pytest.raises(ValueError, match="quaternion"):
        codec.encode_action(np.zeros(7), 0.02, anchor)
    with pytest.raises(ValueError, match="one anchor"):
        codec.encode_action(anchor, 0.02, np.stack((anchor, anchor)))
    with pytest.raises(ValueError, match="shape"):
        codec.encode_observation(anchor[None], np.array([0.02]), anchor)
    stats = _eef_codec_stats()
    stats["actions"]["max"][0] = -2.0
    with pytest.raises(ValueError, match="upper bounds"):
        EEFActionCodec(stats)


_ARX_REQUIRED_ENV = {
    "RLT_STAGE1_CHECKPOINT": "/tmp/arx-stage1-test-checkpoint",
    "RLT_NORM_STATS": "/tmp/arx-test-norm_stats.json",
    "ARX_ACTION_DT": "0.05",
    "ARX_HISTORY_DT": "0.05",
    "ARX_CAMERA_LATENCY": "0.08",
    "ARX_GRIPPER_MIN": "0.0",
    "ARX_GRIPPER_MAX": "0.085",
    "ARX_CAMERA_DEVICE": "/dev/v4l/by-id/fake-arx-camera",
    "ARX_TASK_DESCRIPTION": "pick the test object",
    "RLT_STAGE2_WEIGHTS": "/tmp/arx-stage2-test-weights.pt",
}


def _compose_arx_config(monkeypatch, *, algorithm="ac", evaluation=False):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    monkeypatch.delenv("ARX_ENDPOINT", raising=False)
    suffix = (
        ("eval" if algorithm == "ac" else "td3_eval")
        if evaluation
        else f"{algorithm}_mlp"
    )
    name = f"realworld_arx5_rlt_stage2_{suffix}"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        return compose(config_name=name)


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("algorithm", ["ac", "td3"])
def test_arx_stage2_configs_share_model_contract_and_resolve_without_hardware(
    monkeypatch, algorithm, evaluation
):
    from rlinf.config import validate_only_eval_rollout_model
    from rlinf.envs.real.arx5.rlt import Arx5RLTEnvConfig
    from rlinf.robotics.robots.arx5 import Arx5Config

    for name, value in _ARX_REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    cfg = _compose_arx_config(monkeypatch, algorithm=algorithm, evaluation=evaluation)
    OmegaConf.resolve(cfg)
    assert cfg.cluster.num_nodes == 1
    assert cfg.runner.only_eval is evaluation
    assert cfg.runner.rtc.enabled is False
    assert cfg.runner.enable_decoupled_mode is False
    assert cfg.algorithm.loss_type == f"rlt_{algorithm}"
    assert cfg.algorithm.replay_buffer.filter_invalid_transitions is True
    assert cfg.algorithm.reference_dropout_prob == 0.0
    actor = cfg.actor.model
    assert actor.model_type == (
        "rlt_mlp_policy" if algorithm == "ac" else "rlt_td3_mlp_policy"
    )
    if algorithm == "td3":
        assert cfg.algorithm.adv_type == "rlt_td3"
        assert cfg.algorithm.actor_agg_q == "min"
        assert cfg.algorithm.critic_actor_ratio == 2
        assert cfg.algorithm.actor_update_action_noise is True
        assert actor.actor_noise_sigma == 0.1
    else:
        assert cfg.algorithm.actor_agg_q == "q1"
        assert cfg.algorithm.critic_actor_ratio == 4
        assert actor.fixed_std == 0.002
    assert (actor.action_dim, actor.proprio_dim, actor.num_action_chunks) == (
        10,
        32,
        16,
    )
    assert actor.ref_num_action_chunks == 16
    assert actor.action_parameterization == "eef_rot6d"
    assert list(actor.eef_residual_scale) == [0.1] * 9 + [0.005]
    assert (actor.gripper_width_min, actor.gripper_width_max) == (0.0, 0.085)
    validate_only_eval_rollout_model(cfg.rollout.model)
    assert cfg.rollout.model == actor
    feature = cfg.rollout.rlt_feature_model
    assert feature.model_path == _ARX_REQUIRED_ENV["RLT_STAGE1_CHECKPOINT"]
    assert feature.openpi.config_name == "pi05_umi_v4"
    assert feature.openpi.task == "eval"
    assert feature.openpi.use_rlt is True
    assert (feature.openpi.action_horizon, feature.action_dim) == (16, 10)
    assert feature.openpi.model_action_dim == 32
    hardware = Arx5Config(
        **OmegaConf.to_container(cfg.cluster.node_groups[1].hardware.configs[0])
    )
    assert hardware.node_rank == 0
    assert hardware.endpoint == "tcp://127.0.0.1:5557"
    for mode in ("train", "eval"):
        env = cfg.env[mode]
        assert env.ros_setup is False
        assert env.teleop == "none"
        assert env.keyboard_reward_wrapper is None
        assert env.init_params.id == "Arx5RLTEnv-v1"
        assert env.max_episode_steps % actor.num_action_chunks == 0
        assert env.max_steps_per_rollout_epoch % actor.num_action_chunks == 0
        assert len(env.model_obs_keys) == 5
        parsed = Arx5RLTEnvConfig(**OmegaConf.to_container(env.override_cfg))
        assert parsed.norm_stats_path == feature.openpi_data.norm_stats_path
        assert parsed.start_paused is True
        assert parsed.action_dt == 0.05
        assert parsed.camera_latency == 0.08
        assert parsed.camera_pixel_format == "MJPG"
        assert parsed.gripper_width_min == actor.gripper_width_min
        assert parsed.gripper_width_max == actor.gripper_width_max
        assert parsed.start_with_actor is (evaluation and mode == "eval")
    if evaluation:
        assert cfg.runner.ckpt_path == _ARX_REQUIRED_ENV["RLT_STAGE2_WEIGHTS"]
        assert cfg.rollout.collect_transitions is False
    else:
        assert cfg.runner.ckpt_path is None
        assert cfg.rollout.collect_transitions is True
    assert actor.model_path is None
    assert cfg.runner.resume_dir is None


@pytest.mark.parametrize("missing", list(_ARX_REQUIRED_ENV))
@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("algorithm", ["ac", "td3"])
def test_arx_stage2_configs_refuse_missing_model_and_calibration_inputs(
    monkeypatch, algorithm, evaluation, missing
):
    from omegaconf.errors import InterpolationResolutionError

    for name, value in _ARX_REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing, raising=False)
    cfg = _compose_arx_config(monkeypatch, algorithm=algorithm, evaluation=evaluation)
    if missing == "RLT_STAGE2_WEIGHTS" and not evaluation:
        OmegaConf.resolve(cfg)
        return
    with pytest.raises(InterpolationResolutionError, match=missing):
        OmegaConf.resolve(cfg)
