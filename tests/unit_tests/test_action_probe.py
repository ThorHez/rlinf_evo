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
"""Regression tests for episode isolation, controls, and probe numerics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.utils.action_probe import (
    RidgePath,
    cross_episode_permutation,
    episode_errors,
    episode_weights,
    fit_mlp,
    fit_scaler,
    metric_summary,
    mlp_predict,
    run_action_probes,
    run_method,
    source_record,
    transformed,
    validate_arrays,
)


def synthetic_arrays() -> tuple[dict, dict]:
    """Create an episode-separated task requiring information present in tokens."""
    rng = np.random.default_rng(9201)
    n, per_episode = 1200, 30
    state = rng.normal(size=(n, 4)).astype(np.float32)
    z = rng.normal(size=(n, 4)).astype(np.float32)
    # Feature-target relationship is shared across episodes, never memorized IDs.
    signal = 0.1 * state[:, :1] + np.sin(z[:, :1]) + 0.3 * z[:, 1:2]
    action = np.tile(signal[:, None], (1, 2, 10)).astype(np.float32)
    arrays = {
        "state": state,
        "z": z,
        "z_random": rng.normal(size=z.shape).astype(np.float32),
        "actions": action,
        "episode_id": np.repeat(np.arange(40), per_episode),
        "frame_index": np.tile(np.arange(per_episode), 40),
        "dataset_global_index": np.arange(n),
    }
    manifest = {
        "fit_episode_ids": list(range(26)),
        "dev_episode_ids": list(range(26, 32)),
        "holdout_episode_ids": list(range(32, 40)),
    }
    return arrays, manifest


@pytest.mark.parametrize("samples,features", [(80, 9), (19, 40)])
def test_weighted_ridge_matches_direct_solve(samples: int, features: int) -> None:
    """Both Gram-space branches must preserve the original weighted objective."""
    rng = np.random.default_rng(543)
    x, y = rng.normal(size=(samples, features)), rng.normal(size=(samples, 6))
    weights = rng.uniform(size=samples)
    weights /= weights.sum()
    new = RidgePath(x, y, weights)
    for alpha in (1e-5, 0.03, 100):
        centered_x, centered_y = x - weights @ x, y - weights @ y
        direct = np.linalg.solve(
            centered_x.T @ (weights[:, None] * centered_x) + alpha * np.eye(features),
            centered_x.T @ (weights[:, None] * centered_y),
        )
        np.testing.assert_allclose(
            new.coefficients(alpha)[0], direct, rtol=2e-7, atol=2e-8
        )


def test_scaler_never_fits_dev_or_holdout_and_state_padding_is_equal_width() -> None:
    """Changing held-out distributions cannot change preprocessing statistics."""
    arrays, manifest = synthetic_arrays()
    split = validate_arrays(arrays, manifest)
    scaler = fit_scaler(arrays, ("state", "z"), split["fit"], method="mlp")
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["z"][np.r_[split["dev"], split["holdout"]]] = 1e6
    changed_scaler = fit_scaler(changed, ("state", "z"), split["fit"], method="mlp")
    for first, second in zip(
        scaler.statistics["z"], changed_scaler.statistics["z"], strict=True
    ):
        np.testing.assert_array_equal(first, second)
    state_scaler = fit_scaler(arrays, ("state",), split["fit"], method="mlp")
    state = transformed(
        state_scaler, arrays, split["fit"], method="mlp", padded_width=8
    )
    full = transformed(scaler, arrays, split["fit"], method="mlp", padded_width=8)
    assert state.shape == full.shape
    assert np.all(state[:, 4:] == 0)
    with pytest.raises(ValueError, match="Intervention"):
        fit_scaler(arrays, ("state", "z_image_swapped"), split["fit"], method="mlp")


def test_overlap_and_row_identity_errors_are_rejected() -> None:
    """Leakage or duplicate source rows must stop fitting, even for finite data."""
    arrays, manifest = synthetic_arrays()
    bad = copy.deepcopy(manifest)
    bad["dev_episode_ids"][0] = bad["fit_episode_ids"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_arrays(arrays, bad)
    arrays["dataset_global_index"][1] = arrays["dataset_global_index"][0]
    with pytest.raises(ValueError, match="Duplicate dataset_global_index"):
        validate_arrays(arrays, manifest)


def test_informative_token_ridge_beats_state_and_shuffle_removes_gain() -> None:
    """The measured comparison must detect a real feature signal and its removal."""
    arrays, manifest = synthetic_arrays()
    split = validate_arrays(arrays, manifest)
    fit, holdout = split["fit"], split["holdout"]
    target = arrays["actions"].reshape(len(arrays["actions"]), -1)
    errors = {}
    for case, names in (("state", ("state",)), ("token", ("state", "z"))):
        scaler = fit_scaler(arrays, names, fit, method="ridge")
        x = transformed(scaler, arrays, fit, method="ridge", padded_width=8)
        path = RidgePath(x, target[fit], episode_weights(arrays["episode_id"][fit]))
        normal = transformed(scaler, arrays, holdout, method="ridge", padded_width=8)
        errors[case] = np.mean((path.predict(normal, 0.001) - target[holdout]) ** 2)
        if case == "token":
            permutation = cross_episode_permutation(arrays["episode_id"][holdout], 991)
            assert not np.any(
                arrays["episode_id"][holdout][permutation]
                == arrays["episode_id"][holdout]
            )
            shuffled = transformed(
                scaler,
                arrays,
                holdout,
                method="ridge",
                padded_width=8,
                donors=holdout[permutation],
            )
            errors["shuffled"] = np.mean(
                (path.predict(shuffled, 0.001) - target[holdout]) ** 2
            )
    assert errors["token"] < 0.3 * errors["state"]
    assert errors["shuffled"] > 3 * errors["token"]


def test_mlp_learns_token_signal_with_same_architecture_and_train_only_selection() -> (
    None
):
    """A nonlinear signal must be learnable with the selected head and optimizer."""
    arrays, manifest = synthetic_arrays()
    split = validate_arrays(arrays, manifest)
    fit, dev, holdout = (split[key] for key in ("fit", "dev", "holdout"))
    target = arrays["actions"].reshape(len(arrays["actions"]), -1)
    results = {}
    parameter_counts = []
    for case, names in (("state", ("state",)), ("token", ("state", "z"))):
        scaler = fit_scaler(arrays, names, fit, method="mlp")
        inputs = {
            key: transformed(scaler, arrays, indices, method="mlp", padded_width=8)
            for key, indices in split.items()
        }
        model, detail = fit_mlp(
            inputs["fit"],
            target[fit],
            arrays["episode_id"][fit],
            inputs["dev"],
            target[dev],
            arrays["episode_id"][dev],
            seed=121,
            max_epochs=100,
            patience=15,
            batch_size=128,
            hidden=64,
        )
        prediction = mlp_predict(model, inputs["holdout"])
        results[case] = np.mean((prediction - target[holdout]) ** 2)
        parameter_counts.append(detail["parameters"])
        assert detail["best_dev_mse"] == min(
            row["dev_mse"] for row in detail["history"]
        )
        assert detail["best_epoch"] <= detail["epochs_ran"] <= 100
    assert parameter_counts[0] == parameter_counts[1]
    assert results["token"] < 0.15 * results["state"]


def test_episode_balancing_and_shuffle_metrics_average_errors_not_predictions() -> None:
    """Correlated frame counts and opposite shuffle predictions cannot hide error."""
    target = np.zeros((4, 2, 10))
    prediction = np.stack([np.ones_like(target), -np.ones_like(target)])
    episodes = np.asarray([1, 1, 1, 2])
    squared, absolute = episode_errors(prediction, target, episodes)
    assert squared.shape == (2, 2, 10)
    metrics = metric_summary(squared, absolute)
    assert metrics["chunk"]["all"]["mse"] == 1.0
    prediction = np.ones_like(target)
    prediction[-1] *= 3
    squared, _ = episode_errors(prediction, target, episodes)
    assert squared.mean() == 5.0


def test_ridge_report_saved_heads_and_holdout_interventions_are_consistent(
    tmp_path: Path,
) -> None:
    """Exercise persisted predictions, provenance reuse, and image-only controls."""
    arrays, manifest = synthetic_arrays()
    arrays["z_image_swapped"] = arrays["z"].copy()
    split = validate_arrays(arrays, manifest)
    report = run_method(
        arrays,
        split,
        {"synthetic": True},
        tmp_path,
        method="ridge",
        max_epochs=2,
        patience=1,
    )
    saved = json.loads((tmp_path / "ridge" / "report.json").read_text())
    assert saved == report
    assert report["holdout_episodes"] == 8
    comparison = report["ensemble"]["paired_episode_comparisons"][
        "state_z_vs_state_only"
    ]["chunk"]["all"]
    assert comparison["relative_mse_improvement_ci95"][0] > 0.5
    with np.load(tmp_path / "ridge" / "predictions.npz") as predictions:
        np.testing.assert_array_equal(
            predictions["dataset_global_index"],
            arrays["dataset_global_index"][split["holdout"]],
        )
        np.testing.assert_array_equal(
            predictions["prediction_state_z"],
            predictions["prediction_state_z_image_swapped"],
        )
    assert (
        run_method(
            arrays,
            split,
            {"synthetic": True},
            tmp_path,
            method="ridge",
            max_epochs=2,
            patience=1,
        )
        == report
    )
    with pytest.raises(ValueError, match="incompatible provenance"):
        run_method(
            arrays,
            split,
            {"synthetic": False},
            tmp_path,
            method="ridge",
            max_epochs=2,
            patience=1,
        )


def _write_extraction(tmp_path: Path) -> tuple[Path, Path]:
    """Persist an authenticated extraction with a generic episode split."""
    arrays, manifest = synthetic_arrays()
    arrays["z_image_swapped"] = arrays["z"].copy()
    features, split = tmp_path / "features.npz", tmp_path / "probe_split.json"
    np.savez_compressed(features, **arrays)
    split.write_text(json.dumps(manifest))
    provenance = {
        "state": "completed",
        "source_kind": "live_fsdp_model",
        "model_identity": {"global_step": 123, "config_sha256": "test"},
        "features_sha256": source_record(features)["sha256"],
        "probe_split_sha256": source_record(split)["sha256"],
        "sample_count": len(arrays["state"]),
    }
    (tmp_path / "extraction_status.json").write_text(json.dumps(provenance))
    return features, split


def test_public_evaluation_restores_rng_and_settings_and_authenticates_outputs(
    tmp_path: Path,
) -> None:
    """Evaluation can run inside the driver without changing its random stream."""
    features, split = _write_extraction(tmp_path)
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    precision = torch.get_float32_matmul_precision()
    result = run_action_probes(
        features, split, tmp_path / "report", max_epochs=1, patience=1, threads=2
    )
    for first, second in zip(numpy_state, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(first, second)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert precision == torch.get_float32_matmul_precision()
    assert all(
        type(value) is float and np.isfinite(value)
        for value in result["metrics"].values()
    )
    assert result["metrics"]["ridge/holdout_episodes"] == 8.0
    assert result["metrics"]["mlp/holdout_episodes"] == 8.0
    assert (
        result["metrics"][
            "ridge/state_z_vs_state_only/chunk/all/relative_mse_improvement_ci95_low"
        ]
        > 0.5
    )
    protocol = json.loads((tmp_path / "report" / "protocol.json").read_text())
    assert protocol["source_kind"] == "live_fsdp_model"
    assert protocol["model_identity"]["global_step"] == 123
    assert "checkpoint_sha256" not in protocol
    assert protocol["mlp"]["input_dimensions"] == 8
    assert protocol["target"]["horizon"] == 2
    assert protocol["split_sizes"]["fit"]["episodes"] == 26
    for name, report_path in result["reports"].items():
        report = json.loads(Path(report_path).read_text())
        assert report["status"] == "completed"
        if name == "mlp":
            assert len(report["individual_seeds"]) == 3
            for path in (tmp_path / "report" / "mlp" / "heads").glob("*.pt"):
                values = torch.load(path, map_location="cpu", weights_only=True)
                assert all(value.device.type == "cpu" for value in values.values())
    assert (
        run_action_probes(
            features, split, tmp_path / "report", max_epochs=1, patience=1, threads=2
        )
        == result
    )
    # Provenance must be checked before fitting or reusing any saved report.
    with features.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Features differ"):
        run_action_probes(features, split, tmp_path / "report")


def test_fit_failure_restores_driver_rng(tmp_path: Path) -> None:
    """A failed persisted head must not leave the driver's RNG advanced."""
    features, split = _write_extraction(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    # A real filesystem error after MLP fitting exercises the cleanup path.
    (output / "mlp" / "heads").mkdir(parents=True)
    (output / "mlp" / "heads" / "state_only_seed_20260925.pt").mkdir()
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError):
        run_action_probes(
            features,
            split,
            output,
            methods=("mlp",),
            threads=2,
            max_epochs=1,
            patience=1,
        )
    assert torch.equal(state, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()


def test_public_ridge_selection_ignores_holdout_labels(tmp_path: Path) -> None:
    """Changing only holdout labels cannot select or fit a different head."""
    features, split = _write_extraction(tmp_path)
    run_action_probes(features, split, tmp_path / "before", methods=("ridge",))
    with np.load(features) as archive:
        arrays = dict(archive)
    manifest = json.loads(split.read_text())
    holdout = np.isin(arrays["episode_id"], manifest["holdout_episode_ids"])
    arrays["actions"][holdout] += 100
    np.savez_compressed(features, **arrays)
    extraction = tmp_path / "extraction_status.json"
    record = json.loads(extraction.read_text())
    record["features_sha256"] = source_record(features)["sha256"]
    extraction.write_text(json.dumps(record))
    run_action_probes(features, split, tmp_path / "after", methods=("ridge",))
    for before in (tmp_path / "before" / "ridge" / "heads").glob("*.npz"):
        after = tmp_path / "after" / "ridge" / "heads" / before.name
        with np.load(before) as first, np.load(after) as second:
            for name in first:
                np.testing.assert_array_equal(first[name], second[name])
    altered = tmp_path / "after" / "ridge" / "heads" / "state_z.npz"
    with altered.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="artifact differs"):
        run_action_probes(features, split, tmp_path / "after", methods=("ridge",))
