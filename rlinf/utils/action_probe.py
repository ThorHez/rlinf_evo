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

"""Fit CPU action probes on frozen UMI features with episode-isolated evaluation.

The public entry point consumes an extraction archive and its completed
provenance, fits heads using fit/dev episodes, and reports holdout-only errors.
Only small CPU heads are trained; this module never loads a policy model.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from scipy.linalg import eigh
from torch import nn

from rlinf.utils.logging import get_logger

logger = get_logger()
DEFAULT_ALPHAS = np.logspace(-8, 2, 16)
CASES = {
    "state_only": ("state",),
    "state_z": ("state", "z"),
    "state_z_random": ("state", "z_random"),
}
MLP_SEEDS = (20260925, 20261934, 20262943)
SHUFFLE_SEEDS = (710025, 710026, 710027, 710028, 710029)
BOOTSTRAP_SEED = 20260925
HORIZONS = {"chunk": slice(None), "first": slice(0, 1), "future": slice(1, None)}
ACTION_GROUPS = {
    "all": slice(None),
    "xyz": slice(0, 3),
    "rotation_6d": slice(3, 9),
    "gripper": slice(9, 10),
}


def episode_weights(episode_ids: np.ndarray) -> np.ndarray:
    """Return sample weights summing to one with equal total per episode."""
    _, inverse, counts = np.unique(episode_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


class FeatureScaler:
    """Standardize each feature using training data and balance modality sizes."""

    def __init__(self, feature_names: tuple[str, ...]):
        self.feature_names = feature_names
        self.statistics: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    def fit(
        self, features: dict[str, np.ndarray], indices: np.ndarray, weights: np.ndarray
    ) -> "FeatureScaler":
        """Fit weighted means/scales without accessing validation or test rows."""
        for name in self.feature_names:
            values = (
                features[name][indices].reshape(len(indices), -1).astype(np.float64)
            )
            mean = weights @ values
            std = np.sqrt(weights @ ((values - mean) ** 2))
            active = std > 1e-8
            # Suppress constant dimensions, including state padding.
            scale = np.where(active, std, np.inf)
            self.statistics[name] = (mean, scale, np.sqrt(max(1, int(active.sum()))))
        return self

    def transform(
        self,
        features: dict[str, np.ndarray],
        indices: np.ndarray,
        replacements: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Apply training statistics, optionally replacing one modality's rows."""
        blocks = []
        for name in self.feature_names:
            source = replacements.get(name, indices) if replacements else indices
            values = features[name][source].reshape(len(indices), -1).astype(np.float64)
            mean, scale, dimension_scale = self.statistics[name]
            blocks.append((values - mean) / scale / dimension_scale)
        return np.concatenate(blocks, axis=1)


def cross_episode_permutation(episode_ids: np.ndarray, seed: int) -> np.ndarray:
    """Permute observations while requiring every donor to be another episode."""
    rng = np.random.default_rng(seed)
    groups = [
        rng.permutation(np.flatnonzero(episode_ids == ep))
        for ep in rng.permutation(np.unique(episode_ids))
    ]
    largest = max(map(len, groups))
    if largest > len(episode_ids) // 2:
        raise ValueError(
            "An episode occupies more than half the test set; cannot derange episodes"
        )
    order = np.concatenate(groups)
    donors = np.empty(len(order), dtype=np.int64)
    donors[order] = np.roll(order, largest)
    if np.any(episode_ids[donors] == episode_ids):
        raise AssertionError("Cross-episode permutation contains same-episode donors")
    return donors


def paired_bootstrap(
    reference: np.ndarray, candidate: np.ndarray, seed: int, repetitions: int = 5000
) -> dict[str, Any]:
    """Bootstrap paired episode errors; positive improvements favor candidate."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(reference), size=(repetitions, len(reference)))
    ref_mean = reference[draws].mean(axis=1)
    cand_mean = candidate[draws].mean(axis=1)
    delta = ref_mean - cand_mean
    relative = delta / np.maximum(ref_mean, 1e-20)
    return {
        "reference_mse": float(reference.mean()),
        "candidate_mse": float(candidate.mean()),
        "absolute_mse_improvement": float((reference - candidate).mean()),
        "absolute_mse_improvement_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "relative_mse_improvement": float(
            1.0 - candidate.mean() / max(reference.mean(), 1e-20)
        ),
        "relative_mse_improvement_ci95": np.quantile(relative, [0.025, 0.975]).tolist(),
        "episode_fraction_improved": float(np.mean(candidate < reference)),
        "episodes": len(reference),
        "bootstrap_repetitions": repetitions,
        "uncertainty_scope": "Paired episode resampling of fixed out-of-fold predictions; excludes probe-refit uncertainty.",
    }


class RidgePath:
    """Weighted ridge with unpenalized intercept and the smaller Gram matrix.

    The objective is sum_i weight_i * squared_error_i + alpha * ||W||².
    Feature-space and sample-space solutions implement the same objective.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
        self.x_mean = weights @ x
        self.y_mean = weights @ y
        centered = (x - self.x_mean) * np.sqrt(weights)[:, None]
        target = (y - self.y_mean) * np.sqrt(weights)[:, None]
        self.primal = x.shape[1] <= x.shape[0]
        gram = centered.T @ centered if self.primal else centered @ centered.T
        values, vectors = eigh(gram, check_finite=False, driver="evr")
        self.eigenvalues = np.maximum(values, 0.0)
        self.feature_basis = vectors if self.primal else centered.T @ vectors
        self.projected_targets = vectors.T @ (
            centered.T @ target if self.primal else target
        )

    def coefficients(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Return feature coefficients and the unpenalized intercept."""
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and positive")
        coefficients = self.feature_basis @ (
            self.projected_targets / (self.eigenvalues[:, None] + alpha)
        )
        return coefficients, self.y_mean - self.x_mean @ coefficients

    def predict(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Predict with a fixed alpha without fitting on these rows."""
        coefficients, intercept = self.coefficients(alpha)
        return x @ coefficients + intercept

    def validation_losses(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, alphas: np.ndarray
    ) -> list[float]:
        """Evaluate alpha choices exclusively on the inner development episodes."""
        projection = (x - self.x_mean) @ self.feature_basis
        return [
            float(
                weights
                @ np.mean(
                    (
                        projection
                        @ (self.projected_targets / (self.eigenvalues[:, None] + alpha))
                        + self.y_mean
                        - y
                    )
                    ** 2,
                    axis=1,
                )
            )
            for alpha in alphas
        ]


def source_record(path: Path) -> dict[str, Any]:
    """Hash one provenance file without reading its entire content into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    """Publish JSON atomically so monitoring cannot read a partial report."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def validate_arrays(
    arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Reject overlap, missing episodes, ambiguous row identity, or bad shapes."""
    required = {
        "state",
        "actions",
        "z",
        "z_random",
        "episode_id",
        "frame_index",
        "dataset_global_index",
    }
    if missing := required - arrays.keys():
        raise ValueError(f"Missing arrays: {sorted(missing)}")
    count = len(arrays["episode_id"])
    for name in required | ({"z_image_swapped"} & arrays.keys()):
        value = arrays[name]
        if len(value) != count or not np.isfinite(value).all():
            raise ValueError(f"Invalid length or nonfinite data: {name}")
    for name in ("episode_id", "frame_index", "dataset_global_index"):
        if arrays[name].shape != (count,) or arrays[name].dtype.kind not in "iu":
            raise ValueError(f"Expected integer identity vector: {name}")
    if len(np.unique(arrays["dataset_global_index"])) != count:
        raise ValueError("Duplicate dataset_global_index rows")
    identities = np.stack((arrays["episode_id"], arrays["frame_index"]), axis=1)
    if len(np.unique(identities, axis=0)) != count:
        raise ValueError("Duplicate episode/frame rows")
    if (
        arrays["state"].ndim != 2
        or arrays["z"].ndim != 2
        or arrays["z_random"].shape != arrays["z"].shape
    ):
        raise ValueError("State and learned/random token arrays must be 2D")
    if (
        arrays["actions"].ndim != 3
        or arrays["actions"].shape[-1] < 10
        or arrays["actions"].shape[1] < 2
    ):
        raise ValueError("Actions must have at least two timesteps and ten dimensions")
    if (
        "z_image_swapped" in arrays
        and arrays["z_image_swapped"].shape != arrays["z"].shape
    ):
        raise ValueError("Image-swap token shape differs from normal token shape")
    groups = {}
    seen: set[int] = set()
    for name in ("fit", "dev", "holdout"):
        ids = manifest[f"{name}_episode_ids"]
        if (
            not ids
            or any(type(value) is not int for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                f"Episode ID list is empty, duplicated, or noninteger: {name}"
            )
        if seen.intersection(ids):
            raise ValueError("Episode overlap between fit/dev/holdout")
        seen.update(ids)
        index = np.flatnonzero(np.isin(arrays["episode_id"], ids))
        if set(arrays["episode_id"][index].tolist()) != set(ids):
            raise ValueError(f"Missing feature episodes in {name}")
        groups[name] = index
    if seen != set(arrays["episode_id"].tolist()):
        raise ValueError("Features contain episodes absent from the declared split")
    if "split" in arrays:
        for number, name in enumerate(("fit", "dev", "holdout")):
            if not np.all(arrays["split"][groups[name]] == number):
                raise ValueError(
                    "Feature split labels disagree with the episode manifest"
                )
    return groups


def fit_scaler(
    arrays: dict[str, np.ndarray],
    names: tuple[str, ...],
    fit: np.ndarray,
    *,
    method: str,
) -> FeatureScaler:
    """Fit preprocessing only on fit episodes, never on dev or holdout rows."""
    if any(name not in {"state", "z", "z_random"} for name in names):
        raise ValueError("Intervention features cannot be fitted")
    scaler = FeatureScaler(names).fit(
        arrays, fit, episode_weights(arrays["episode_id"][fit])
    )
    if method == "mlp":
        scaler.statistics = {
            name: (mean, scale, 1.0)
            for name, (mean, scale, _) in scaler.statistics.items()
        }
    return scaler


def scaler_arrays(scaler: FeatureScaler) -> dict[str, np.ndarray]:
    """Serialize fitted preprocessing as numeric arrays, without pickled code."""
    return {
        f"{name}_{suffix}": np.asarray(value)
        for name, (mean, scale, dimension) in scaler.statistics.items()
        for suffix, value in (
            ("mean", mean),
            ("scale", scale),
            ("dimension", dimension),
        )
    }


def transformed(
    scaler: FeatureScaler,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    method: str,
    padded_width: int,
    donors: np.ndarray | None = None,
    intervention: str | None = None,
) -> np.ndarray:
    """Apply fixed statistics; pad the state-only MLP to the same input width."""
    feature_source = arrays
    if intervention == "image_swapped":
        feature_source = {**arrays, "z": arrays["z_image_swapped"]}
    value = scaler.transform(
        feature_source, indices, {"z": donors} if donors is not None else None
    )
    if intervention == "mean_token":
        value[:, arrays["state"].shape[1] :] = 0.0
    if method == "mlp":
        if value.shape[1] < padded_width:
            value = np.pad(value, ((0, 0), (0, padded_width - value.shape[1])))
        value = value.astype(np.float32)
    return value


def make_head(inputs: int, outputs: int, hidden: int = 256) -> nn.Module:
    """Build the fixed two-hidden-layer regression head used by every case."""
    return nn.Sequential(
        nn.Linear(inputs, hidden, device="cpu", dtype=torch.float32),
        nn.SiLU(),
        nn.Linear(hidden, hidden, device="cpu", dtype=torch.float32),
        nn.SiLU(),
        nn.Linear(hidden, outputs, device="cpu", dtype=torch.float32),
    )


def fit_mlp(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    fit_episodes: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    dev_episodes: np.ndarray,
    *,
    seed: int,
    max_epochs: int = 100,
    patience: int = 15,
    batch_size: int = 512,
    hidden: int = 256,
) -> tuple[nn.Module, dict[str, Any]]:
    """Fit a CPU head and select the epoch with inner-dev episode-weighted MSE.

    This interface accepts no holdout data. All cases have identical seeded
    initialization, data order, architecture, optimizer, and fitting budget.
    """
    if len(set(fit_episodes).intersection(dev_episodes)):
        raise ValueError("MLP fit and development episodes overlap")
    if max_epochs < 1 or patience < 1:
        raise ValueError("Epoch budget and patience must be positive")
    torch.random.default_generator.manual_seed(seed)
    model = make_head(x_fit.shape[1], y_fit.shape[1], hidden)
    x = torch.as_tensor(np.ascontiguousarray(x_fit), dtype=torch.float32, device="cpu")
    y = torch.as_tensor(np.ascontiguousarray(y_fit), dtype=torch.float32, device="cpu")
    dev_x = torch.as_tensor(
        np.ascontiguousarray(x_dev), dtype=torch.float32, device="cpu"
    )
    dev_y = torch.as_tensor(
        np.ascontiguousarray(y_dev), dtype=torch.float32, device="cpu"
    )
    weights = torch.as_tensor(
        episode_weights(fit_episodes), dtype=torch.float32, device="cpu"
    )
    dev_weights = torch.as_tensor(
        episode_weights(dev_episodes), dtype=torch.float32, device="cpu"
    )
    with torch.no_grad():
        model[-1].bias.copy_(weights @ y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 999)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        train_loss = 0.0
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(order[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            row_losses = (model(x[index]) - y[index]).square().mean(1)
            # Unbiased minibatch estimator of the episode-balanced objective.
            loss = (row_losses * weights[index]).sum() * len(x) / len(index)
            loss.backward()
            optimizer.step()
            train_loss += float((row_losses.detach() * weights[index]).sum())
        model.eval()
        with torch.inference_mode():
            dev_loss = 0.0
            for start in range(0, len(dev_x), batch_size):
                index = slice(start, start + batch_size)
                rows = (model(dev_x[index]) - dev_y[index]).square().mean(1)
                dev_loss += float((rows * dev_weights[index]).sum())
        if not np.isfinite([train_loss, dev_loss]).all():
            raise ValueError("Nonfinite MLP loss")
        history.append(
            {"epoch": epoch, "fit_online_mse": train_loss, "dev_mse": dev_loss}
        )
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev_mse": best_loss,
        "epochs_ran": len(history),
        "max_epochs": max_epochs,
        "patience": patience,
        "early_stopped": len(history) < max_epochs,
        "history": history,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_dimensions": x.shape[1],
        "seconds": time.monotonic() - started,
    }


def mlp_predict(model: nn.Module, values: np.ndarray) -> np.ndarray:
    """Run CPU inference in bounded batches without changing fitted weights."""
    with torch.inference_mode():
        return np.concatenate(
            [
                model(torch.as_tensor(chunk, dtype=torch.float32, device="cpu")).numpy()
                for chunk in np.array_split(values, max(1, (len(values) + 511) // 512))
            ]
        )


def episode_errors(
    prediction: np.ndarray, target: np.ndarray, episodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average samples within each episode before averaging episodes.

    An extra leading axis means repeated token interventions; errors are then
    averaged over repetitions, rather than averaging their predictions.
    """
    difference = prediction - target
    squared, absolute = difference**2, np.abs(difference)
    if prediction.ndim == 4:
        squared, absolute = squared.mean(0), absolute.mean(0)
    unique = np.unique(episodes)
    return (
        np.stack([squared[episodes == episode].mean(0) for episode in unique]),
        np.stack([absolute[episodes == episode].mean(0) for episode in unique]),
    )


def metric_summary(squared: np.ndarray, absolute: np.ndarray) -> dict[str, Any]:
    """Report normalized action errors by chunk, first/future step, and group."""
    result = {}
    for horizon, times in HORIZONS.items():
        result[horizon] = {}
        for group, columns in ACTION_GROUPS.items():
            mse = float(squared[:, times, columns].mean())
            result[horizon][group] = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(absolute[:, times, columns].mean()),
            }
    result["mse_by_horizon"] = squared.mean(axis=(0, 2)).tolist()
    result["mse_by_dimension"] = squared.mean(axis=(0, 1)).tolist()
    return result


def compare_errors(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Bootstrap paired episodes with a fixed 95% interval protocol."""
    result = {}
    for horizon, times in HORIZONS.items():
        result[horizon] = {}
        for group, columns in ACTION_GROUPS.items():
            comparison = paired_bootstrap(
                reference[:, times, columns].mean((1, 2)),
                candidate[:, times, columns].mean((1, 2)),
                BOOTSTRAP_SEED,
            )
            comparison["uncertainty_scope"] = (
                "Paired holdout-episode resampling conditional on fixed fitted heads; excludes refit uncertainty and repeated checkpoint-selection effects."
            )
            result[horizon][group] = comparison
    return result


def summarize_predictions(
    predictions: dict[str, np.ndarray], target: np.ndarray, episodes: np.ndarray
) -> dict[str, Any]:
    """Summarize all fixed conditions and compare normal token heads to controls."""
    errors = {
        key: episode_errors(value, target, episodes)
        for key, value in predictions.items()
    }
    metrics = {key: metric_summary(*error) for key, error in errors.items()}
    comparisons = {
        f"state_z_vs_{reference}": compare_errors(
            errors[reference][0], errors["state_z"][0]
        )
        for reference in (
            "state_only",
            "state_z_random",
            "state_z_shuffled",
            "state_z_mean_token",
            "state_z_image_swapped",
        )
        if reference in errors
    }
    return {"metrics": metrics, "paired_episode_comparisons": comparisons}


def run_method(
    arrays: dict[str, np.ndarray],
    groups: dict[str, np.ndarray],
    protocol: dict[str, Any],
    output: Path,
    *,
    method: str,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    """Fit one prespecified head family, persist heads, and evaluate holdout once."""
    folder = output / method
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "report.json").exists():
        existing = json.loads((folder / "report.json").read_text())
        if existing["protocol"] != protocol or existing["status"] != "completed":
            raise ValueError(
                f"Existing {method} report has incompatible provenance/protocol"
            )
        for record in existing["artifacts"]:
            if source_record(Path(record["path"])) != record:
                raise ValueError(f"Saved {method} artifact differs from its report")
        logger.info("Action probe %s already completed", method)
        return existing
    started = time.monotonic()
    heads = folder / "heads"
    heads.mkdir(exist_ok=True)
    fit, dev, holdout = (groups[key] for key in ("fit", "dev", "holdout"))
    flat_target = (
        arrays["actions"][..., :10]
        .reshape(len(arrays["actions"]), -1)
        .astype(np.float32)
    )
    target = arrays["actions"][holdout, :, :10].astype(np.float64)
    episodes = arrays["episode_id"][holdout]
    width = arrays["state"].shape[1] + arrays["z"].shape[1]
    fit_weights, dev_weights = (
        episode_weights(arrays["episode_id"][index]) for index in (fit, dev)
    )
    donors = [
        holdout[cross_episode_permutation(episodes, seed)] for seed in SHUFFLE_SEEDS
    ]
    # Freeze all donor assignments before any fitted head or holdout metric exists.
    np.savez_compressed(
        folder / "shuffle_donors.npz",
        seeds=np.asarray(SHUFFLE_SEEDS),
        row_indices=np.stack(donors),
        donor_episode_id=arrays["episode_id"][np.stack(donors)],
        holdout_episode_id=episodes,
    )
    fits = {}
    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    seeds = MLP_SEEDS if method == "mlp" else (0,)
    for case, names in CASES.items():
        scaler = fit_scaler(arrays, names, fit, method=method)
        np.savez_compressed(heads / f"{case}_scaler.npz", **scaler_arrays(scaler))
        x_fit = transformed(scaler, arrays, fit, method=method, padded_width=width)
        x_dev = transformed(scaler, arrays, dev, method=method, padded_width=width)
        x_holdout = transformed(
            scaler, arrays, holdout, method=method, padded_width=width
        )
        ridge = (
            RidgePath(x_fit, flat_target[fit].astype(np.float64), fit_weights)
            if method == "ridge"
            else None
        )
        fits[case] = {}
        for seed in seeds:
            key = str(seed)
            all_predictions.setdefault(key, {})
            if ridge is not None:
                losses = ridge.validation_losses(
                    x_dev, flat_target[dev], dev_weights, DEFAULT_ALPHAS
                )
                best = int(np.argmin(losses))
                alpha = float(DEFAULT_ALPHAS[best])
                coefficients, intercept = ridge.coefficients(alpha)
                np.savez_compressed(
                    heads / f"{case}.npz",
                    coefficients=coefficients,
                    intercept=intercept,
                )
                detail = {
                    "alpha": alpha,
                    "dev_mse": losses[best],
                    "alpha_grid": DEFAULT_ALPHAS.tolist(),
                    "dev_losses": losses,
                    "gram_space": "feature" if ridge.primal else "sample",
                }

                def predict(value: np.ndarray) -> np.ndarray:
                    return value @ coefficients + intercept

            else:
                model, detail = fit_mlp(
                    x_fit,
                    flat_target[fit],
                    arrays["episode_id"][fit],
                    x_dev,
                    flat_target[dev],
                    arrays["episode_id"][dev],
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                torch.save(model.state_dict(), heads / f"{case}_seed_{seed}.pt")

                def predict(value: np.ndarray) -> np.ndarray:
                    return mlp_predict(model, value)

            all_predictions[key][case] = (
                predict(x_holdout).reshape(target.shape).astype(np.float64)
            )
            if case == "state_z":
                all_predictions[key]["state_z_shuffled"] = np.stack(
                    [
                        predict(
                            transformed(
                                scaler,
                                arrays,
                                holdout,
                                method=method,
                                padded_width=width,
                                donors=donor,
                            )
                        ).reshape(target.shape)
                        for donor in donors
                    ]
                ).astype(np.float64)
                for intervention in ("mean_token", "image_swapped"):
                    if (
                        intervention == "image_swapped"
                        and "z_image_swapped" not in arrays
                    ):
                        continue
                    all_predictions[key][f"state_z_{intervention}"] = (
                        predict(
                            transformed(
                                scaler,
                                arrays,
                                holdout,
                                method=method,
                                padded_width=width,
                                intervention=intervention,
                            )
                        )
                        .reshape(target.shape)
                        .astype(np.float64)
                    )
            fits[case][key] = detail
            write_json(
                folder / "fit_progress.json",
                {
                    "status": "fitting",
                    "method": method,
                    "fits": fits,
                    "seconds": time.monotonic() - started,
                },
            )
            logger.info(
                "Action probe %s/%s seed=%s selected_epoch=%s selected_alpha=%s",
                method,
                case,
                seed,
                detail.get("best_epoch"),
                detail.get("alpha"),
            )
    ensemble = {
        case: np.mean(
            [predictions[case] for predictions in all_predictions.values()], axis=0
        )
        for case in next(iter(all_predictions.values()))
    }
    prediction_payload = {
        "episode_id": episodes,
        "frame_index": arrays["frame_index"][holdout],
        "dataset_global_index": arrays["dataset_global_index"][holdout],
        "target": target,
        **{f"prediction_{case}": value for case, value in ensemble.items()},
        **{
            f"seed_{seed}_prediction_{case}": value
            for seed, values in all_predictions.items()
            for case, value in values.items()
        },
    }
    if not all(np.isfinite(value).all() for value in prediction_payload.values()):
        raise ValueError("Nonfinite predictions")
    np.savez_compressed(folder / "predictions.npz", **prediction_payload)
    report = {
        "status": "completed",
        "method": method,
        "protocol": protocol,
        "holdout_samples": len(holdout),
        "holdout_episodes": len(np.unique(episodes)),
        "fits": fits,
        "ensemble": summarize_predictions(ensemble, target, episodes),
        "individual_seeds": {
            seed: summarize_predictions(value, target, episodes)
            for seed, value in all_predictions.items()
        },
        "seconds": time.monotonic() - started,
    }
    report["artifacts"] = [source_record(path) for path in sorted(heads.iterdir())] + [
        source_record(folder / "predictions.npz"),
        source_record(folder / "shuffle_donors.npz"),
    ]
    write_json(folder / "report.json", report)
    logger.info("Action probe %s completed in %.1f s", method, report["seconds"])
    return report


@contextmanager
def _cpu_fit_context(threads: int) -> Iterator[None]:
    """Restore the driver's RNG and Torch settings after CPU head fitting."""
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    torch_rng = torch.random.get_rng_state()
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.random.set_rng_state(torch_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.set_float32_matmul_precision(old_precision)


def scalar_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Select logger scalars from episode-balanced holdout errors and intervals."""
    method = report["method"]
    result = {
        f"{method}/holdout_samples": float(report["holdout_samples"]),
        f"{method}/holdout_episodes": float(report["holdout_episodes"]),
    }
    for case, metrics in report["ensemble"]["metrics"].items():
        for horizon in HORIZONS:
            for group in ACTION_GROUPS:
                for name in ("mse", "mae"):
                    result[f"{method}/{case}/{horizon}/{group}/{name}"] = float(
                        metrics[horizon][group][name]
                    )
    for case, comparisons in report["ensemble"]["paired_episode_comparisons"].items():
        for horizon in HORIZONS:
            for group in ACTION_GROUPS:
                values = comparisons[horizon][group]
                prefix = f"{method}/{case}/{horizon}/{group}"
                for name in ("relative_mse_improvement", "episode_fraction_improved"):
                    result[f"{prefix}/{name}"] = float(values[name])
                low, high = values["relative_mse_improvement_ci95"]
                result[f"{prefix}/relative_mse_improvement_ci95_low"] = float(low)
                result[f"{prefix}/relative_mse_improvement_ci95_high"] = float(high)
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Action probe report contains nonfinite scalar metrics")
    return result


def run_action_probes(
    features_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    *,
    methods: tuple[str, ...] = ("ridge", "mlp"),
    threads: int = 4,
    max_epochs: int = 100,
    patience: int = 15,
) -> dict[str, Any]:
    """Fit CPU heads and evaluate token information on held-out episodes.

    The archive must contain state/actions/z/z_random and integer episode_id,
    frame_index, dataset_global_index vectors. Optional z_image_swapped features
    are evaluated only on holdout rows. Action targets use the first ten UMI
    coordinates; padded coordinates never contribute to errors. The split JSON
    declares disjoint fit_episode_ids, dev_episode_ids, and holdout_episode_ids.
    Each episode receives equal total weight even when row counts differ.

    Preprocessing and heads fit only on fit episodes; dev episodes select ridge
    regularization and MLP epochs. All three fixed MLP seeds use the same padded
    architecture for every input condition. Holdout rows only produce metrics.
    A completed extraction_status.json beside the archive must authenticate the
    archive and split hashes. Reports bind to that exact source and fitting
    protocol; incompatible previous output is rejected.

    The function restores Python, NumPy, and Torch CPU RNG plus Torch settings,
    including on failure. No CUDA device is used. ``threads`` controls Torch CPU
    threads; set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS before
    process startup to bound NumPy/SciPy BLAS threads as well.

    Args:
        features_path: NPZ archive of frozen model features and row identities.
        split_path: JSON manifest describing fit/dev/holdout episode membership.
        output_dir: Directory owning protocol, heads, predictions, and reports.
        methods: Nonempty tuple containing "ridge" and/or "mlp" without repeats.
        threads: Positive Torch CPU thread limit during fitting.
        max_epochs: Maximum epochs for each MLP head.
        patience: Dev-loss early-stopping patience for each MLP head.

    Returns:
        Mapping with ``metrics`` (finite float scalars named by method),
        ``reports`` (method to absolute JSON path), and ``summary_path``.
        Callers may prefix metric names, for example with ``rlt_probe/``.
    """
    import os

    if (
        not methods
        or len(set(methods)) != len(methods)
        or set(methods) - {"ridge", "mlp"}
    ):
        raise ValueError("Methods must select ridge and/or mlp without repeats")
    if any(
        type(value) is not int or value < 1 for value in (threads, max_epochs, patience)
    ):
        raise ValueError(
            "Thread count, epoch budget, and patience must be positive integers"
        )
    features_path, split_path, output_dir = (
        Path(path).resolve() for path in (features_path, split_path, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = features_path.parent / "extraction_status.json"
    extraction = json.loads(extraction_path.read_text())
    if extraction.get("state") != "completed":
        raise ValueError("Feature extraction must be completed before fitting probes")
    feature_record, split_record = (
        source_record(features_path),
        source_record(split_path),
    )
    if extraction.get("features_sha256") != feature_record["sha256"]:
        raise ValueError("Features differ from completed extraction provenance")
    if extraction.get("probe_split_sha256") != split_record["sha256"]:
        raise ValueError("Episode split differs from feature extraction provenance")
    with np.load(features_path, allow_pickle=False) as archive:
        arrays = dict(archive)
    manifest = json.loads(split_path.read_text())
    groups = validate_arrays(arrays, manifest)
    if extraction.get("sample_count", len(arrays["episode_id"])) != len(
        arrays["episode_id"]
    ):
        raise ValueError("Feature sample count differs from completed extraction")
    protocol = {
        "version": 2,
        "features": feature_record,
        "split": split_record,
        "extraction_provenance": source_record(extraction_path),
        "source_kind": extraction.get("source_kind", "unspecified"),
        "model_identity": extraction.get("model_identity"),
        "code": source_record(Path(__file__)),
        "methods": list(methods),
        "split_sizes": {
            name: {
                "episodes": len(manifest[f"{name}_episode_ids"]),
                "samples": len(index),
            }
            for name, index in groups.items()
        },
        "holdout_used_for_fitting_or_head_selection": False,
        "representation_updates": False,
        "target": {
            "space": "training normalization; no fitted target scaling",
            "horizon": arrays["actions"].shape[1],
            "coordinates": list(range(10)),
        },
        "fit_weights": "Equal total weight per episode for scalers, ridge, and MLP objective",
        "ridge": {
            "alpha_grid": DEFAULT_ALPHAS.tolist(),
            "selection": "minimum episode-balanced dev MSE; first alpha wins ties",
            "scaling": "fit-only coordinate mean/std divided by sqrt(active coordinates) per modality",
        },
        "mlp": {
            "seeds": list(MLP_SEEDS),
            "ensemble": "mean predictions of three fixed seeds; individual seeds also reported",
            "hidden": [256, 256],
            "input_dimensions": arrays["state"].shape[1] + arrays["z"].shape[1],
            "state_only_padding": arrays["z"].shape[1],
            "activation": "SiLU",
            "optimizer": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "max_epochs": max_epochs,
            "patience": patience,
            "selection": "best episode-balanced dev MSE; no refit on dev or holdout",
            "scaling": "fit-only coordinate mean/std; constant coordinates zeroed",
            "device": "cpu",
            "torch_threads": threads,
            "blas_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "interventions": {
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "shuffle": "five holdout-only bijective cross-episode permutations; average errors, not predictions",
            "mean_token": "fit-set token mean through the same fitted state_z head",
            "image_swapped": "raw images from another holdout episode; preserve state and prompt"
            if "z_image_swapped" in arrays
            else None,
        },
        "bootstrap": {
            "unit": "paired episode",
            "repetitions": 5000,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
        },
        "limitations": [
            "Offline head predictability is not closed-loop RL success.",
            "A negative result does not prove representation information is absent.",
            "Stage 1 episode isolation does not establish base-model pretraining isolation.",
            "Repeated checkpoint diagnostics reuse holdout; it is a monitoring set, not a final untouched test.",
            "Random encoder shares the current backbone and can carry useful state/image information.",
            "Intervals condition on fitted heads; they exclude refit and repeated-selection uncertainty.",
        ],
    }
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Output already contains a different frozen protocol")
    else:
        write_json(protocol_path, protocol)
    metrics: dict[str, float] = {}
    reports: dict[str, str] = {}
    with _cpu_fit_context(threads):
        for method in methods:
            report = run_method(
                arrays,
                groups,
                protocol,
                output_dir,
                method=method,
                max_epochs=max_epochs,
                patience=patience,
            )
            metrics.update(scalar_metrics(report))
            reports[method] = str(output_dir / method / "report.json")
    summary_path = output_dir / "summary.json"
    result = {"metrics": metrics, "reports": reports, "summary_path": str(summary_path)}
    write_json(summary_path, result)
    return result
