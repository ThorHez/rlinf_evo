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

"""Read immutable action-probe observations and merge distributed features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_METADATA = (
    "state",
    "actions",
    "episode_id",
    "frame_index",
    "dataset_global_index",
    "split",
)
_FEATURES = ("z", "z_random", "z_image_swapped")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_ids(value: Any, name: str) -> set[int]:
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if value is None:
        raise ValueError(f"Action probe requires explicit {name} episode IDs.")
    ids = list(value)
    if not ids or any(
        isinstance(item, bool) or not isinstance(item, int) for item in ids
    ):
        raise ValueError(f"Invalid {name} episode IDs.")
    if len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError(f"Duplicate or negative {name} episode IDs.")
    return set(ids)


@dataclass(frozen=True)
class ActionProbeBatch:
    """One equal-size distributed batch, including explicitly masked padding.

    ``indices`` address the prepared arrays, while ``dataset_global_index`` in
    ``metadata`` identifies the original dataset frame. Observation mappings
    retain uint8 images for the model's normal observation conversion. Every
    rank must execute an image-swap forward when ``globalbatch_has_holdout`` is
    true, including ranks whose local valid rows contain no holdout samples.
    """

    indices: np.ndarray
    valid: np.ndarray
    observation: dict[str, Any]
    donor_observation: dict[str, Any]
    metadata: dict[str, np.ndarray]
    globalbatch_has_holdout: bool


class PreparedActionProbeInputs:
    """Validate and memory-map fixed probe inputs without opening a data loader.

    Args:
        input_dir: Prepared arrays, split manifest, provenance, and completed QC.
        cfg: Full training configuration; episode IDs, normalization, and data
            path must match the preparation provenance.
        image_seed: Seed for a fixed cross-episode holdout image permutation.

    The adapter hashes small identity files and verifies every array header,
    size, and modification time. It trusts the completed preparation's image
    hashes instead of reading multi-gigabyte image files on every actor rank.
    """

    def __init__(self, input_dir: str | Path, cfg: Any, image_seed: int = 20260924):
        self.input_dir = Path(input_dir).resolve()
        self.provenance_path = self.input_dir / "provenance.json"
        self.provenance = json.loads(self.provenance_path.read_text())
        self.provenance_sha256 = _sha256(self.provenance_path)
        self.qc_path = self.input_dir / "input_quality_check.json"
        self.qc = json.loads(self.qc_path.read_text())
        if (
            self.provenance.get("complete") is not True
            or self.qc.get("passed") is not True
        ):
            raise ValueError("Action-probe preparation and QC must be complete.")
        split_path = self.input_dir / "probe_split.json"
        self.split_sha256 = _sha256(split_path)
        if self.split_sha256 != self.provenance.get("probe_split_sha256"):
            raise ValueError("Action-probe split provenance hash mismatch.")
        self.split = json.loads(split_path.read_text())
        if self.split.get(
            "parent_stage1_split_signature_sha256"
        ) != self.provenance.get("parent_stage1_split_signature_sha256"):
            raise ValueError("Action-probe parent split provenance mismatch.")
        samples_path = self.input_dir / "samples.jsonl"
        if _sha256(samples_path) != self.provenance.get("samples_sha256"):
            raise ValueError("Action-probe sample identity provenance hash mismatch.")
        self._validate_config(cfg)
        self._file_stats: dict[Path, tuple[int, int]] = {}
        for path in (self.qc_path, samples_path):
            stat = path.stat()
            self._file_stats[path] = (stat.st_size, stat.st_mtime_ns)
        self.arrays: dict[str, np.ndarray] = {}
        for name, record in self.provenance["arrays"].items():
            if Path(name).name != name:
                raise ValueError("Action-probe array names cannot contain paths.")
            path = self.input_dir / f"{name}.npy"
            stat = path.stat()
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                list(array.shape) != record["shape"]
                or str(array.dtype) != record["dtype"]
                or stat.st_size != record["logical_bytes"]
            ):
                raise ValueError(f"Action-probe array header/size mismatch: {name}")
            if stat.st_mtime_ns > self.provenance_path.stat().st_mtime_ns:
                raise ValueError(
                    f"Action-probe input changed after preparation: {name}"
                )
            if stat.st_size <= 8 * 1024 * 1024 and _sha256(path) != record["sha256"]:
                raise ValueError(f"Action-probe array provenance hash mismatch: {name}")
            self._file_stats[path] = (stat.st_size, stat.st_mtime_ns)
            self.arrays[name] = array
        self._validate_arrays(cfg)
        self.image_seed = int(image_seed)
        self.image_donors = self._image_donors()
        self.image_donor_sha256 = hashlib.sha256(
            self.image_donors.astype("<i8").tobytes()
        ).hexdigest()
        self.embedding_dim = int(
            OmegaConf.select(cfg, "actor.model.openpi.rlt_embed_dim", default=2048)
        )
        if self.embedding_dim < 1:
            raise ValueError("Action-probe token width must be positive.")

    def _validate_config(self, cfg: Any) -> None:
        from rlinf.data.storage.lerobot import resolve_lerobot_repo_id

        fit = _episode_ids(self.split.get("fit_episode_ids"), "probe fit")
        dev = _episode_ids(self.split.get("dev_episode_ids"), "probe dev")
        holdout = _episode_ids(self.split.get("holdout_episode_ids"), "probe holdout")
        if fit & dev or fit & holdout or dev & holdout:
            raise ValueError("Action-probe episode splits overlap.")
        train_ids = _episode_ids(
            OmegaConf.select(cfg, "data.train_episode_ids"), "training"
        )
        val_ids = _episode_ids(
            OmegaConf.select(cfg, "data.val_episode_ids"), "validation"
        )
        if train_ids != fit | dev or val_ids != holdout:
            raise ValueError(
                "Action-probe episode split does not match training configuration."
            )
        expected_path = Path(self.provenance["dataset_path"]).resolve()
        if Path(self.split["dataset_path"]).resolve() != expected_path:
            raise ValueError("Action-probe split dataset provenance mismatch.")
        for key in ("data.train_data_paths", "data.val_data_paths"):
            actual_path = resolve_lerobot_repo_id(OmegaConf.select(cfg, key))
            if actual_path is None or Path(actual_path).resolve() != expected_path:
                raise ValueError(f"Action-probe dataset differs from {key}.")
        norm = OmegaConf.select(cfg, "actor.model.openpi_data.norm_stats_path")
        if not norm or _sha256(Path(norm)) != self.provenance.get("norm_stats_sha256"):
            raise ValueError("Action-probe normalization provenance mismatch.")
        if self.provenance.get("image_augmentation") is not False:
            raise ValueError(
                "Action-probe inputs must be prepared without image augmentation."
            )
        for source, digest in self.provenance.get("source_files_sha256", {}).items():
            path = Path(source)
            if _sha256(path) != digest:
                raise ValueError(
                    f"Action-probe source provenance hash mismatch: {source}"
                )
            if path.name == "resolved_config.yaml":
                prepared_cfg = OmegaConf.load(path)
                for key in (
                    "actor.model.openpi.config_name",
                    "actor.model.openpi.model_action_dim",
                    "actor.model.openpi.action_horizon",
                    "actor.model.openpi.max_token_len",
                    "actor.model.openpi.discrete_state_input",
                ):
                    if OmegaConf.select(cfg, key) != OmegaConf.select(
                        prepared_cfg, key
                    ):
                        raise ValueError(
                            f"Action-probe model/data configuration mismatch: {key}"
                        )

    def _validate_arrays(self, cfg: Any) -> None:
        count = int(self.qc["sample_count"])
        if count < 1:
            raise ValueError("Action-probe input set is empty.")
        dimension = int(
            OmegaConf.select(cfg, "actor.model.openpi.model_action_dim", default=32)
        )
        horizon = int(
            OmegaConf.select(cfg, "actor.model.openpi.action_horizon", default=16)
        )
        prompt_length = int(
            OmegaConf.select(cfg, "actor.model.openpi.max_token_len", default=200)
        )
        expected = {
            "state": ((count, dimension), "float32"),
            "actions": ((count, horizon, dimension), "float32"),
            "tokenized_prompt": ((count, prompt_length), "int32"),
            "tokenized_prompt_mask": ((count, prompt_length), "bool"),
            **dict.fromkeys(
                ("episode_id", "frame_index", "dataset_global_index"),
                ((count,), "int64"),
            ),
            "split": ((count,), "uint8"),
            **{
                f"image_{camera}": ((count, 224, 224, 3), "uint8")
                for camera in _CAMERAS
            },
            **{f"image_mask_{camera}": ((count,), "bool") for camera in _CAMERAS},
        }
        for key, (shape, dtype) in expected.items():
            array = self.arrays.get(key)
            if array is None or array.shape != shape or str(array.dtype) != dtype:
                raise ValueError(f"Invalid action-probe array contract: {key}")
        indices = self.arrays["dataset_global_index"]
        if np.any(indices < 0) or len(np.unique(indices)) != count:
            raise ValueError(
                "Action-probe dataset indices must be nonnegative and unique."
            )
        if np.any(self.arrays["frame_index"] < 0):
            raise ValueError("Action-probe frame indices cannot be negative.")
        labels = self.arrays["split"]
        if not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("Invalid action-probe split labels.")
        for label, key in enumerate(
            ("fit_episode_ids", "dev_episode_ids", "holdout_episode_ids")
        ):
            actual = {
                int(item)
                for item in np.unique(self.arrays["episode_id"][labels == label])
            }
            if actual != set(self.split[key]):
                raise ValueError(f"Action-probe sample membership differs from {key}.")

    def _image_donors(self) -> np.ndarray:
        episodes, labels = self.arrays["episode_id"], self.arrays["split"]
        holdout = np.flatnonzero(labels == 2)
        groups = np.unique(episodes[holdout])
        rng = np.random.default_rng(self.image_seed)
        rng.shuffle(groups)
        ordered = np.concatenate(
            [
                rng.permutation(holdout[episodes[holdout] == episode])
                for episode in groups
            ]
        )
        shift = max(
            np.count_nonzero(episodes[holdout] == episode) for episode in groups
        )
        if len(groups) < 2 or 2 * shift > len(ordered):
            raise ValueError(
                "Holdout episodes cannot support a cross-episode image permutation."
            )
        donors = np.arange(len(self), dtype=np.int64)
        donors[ordered] = np.roll(ordered, shift)
        if np.any(episodes[donors[holdout]] == episodes[holdout]):
            raise ValueError("Image permutation contains a same-episode donor.")
        return donors

    def __len__(self) -> int:
        return len(self.arrays["state"])

    def batch_count(self, world_size: int, batch_size: int) -> int:
        """Return the same number of collective forward rounds for every rank."""
        if world_size < 1 or batch_size < 1:
            raise ValueError("Action-probe world and batch sizes must be positive.")
        return (len(self) + world_size * batch_size - 1) // (world_size * batch_size)

    def batch(
        self, round_index: int, rank: int, world_size: int, batch_size: int
    ) -> ActionProbeBatch:
        """Read one rank's batch; repeat the last row for invalid tail padding."""
        rounds = self.batch_count(world_size, batch_size)
        if not 0 <= rank < world_size or not 0 <= round_index < rounds:
            raise ValueError("Invalid action-probe rank or round index.")
        global_start = round_index * world_size * batch_size
        positions = global_start + rank * batch_size + np.arange(batch_size)
        valid = positions < len(self)
        indices = np.minimum(positions, len(self) - 1)
        observation = {
            "image": {
                camera: self.arrays[f"image_{camera}"][indices] for camera in _CAMERAS
            },
            "image_mask": {
                camera: self.arrays[f"image_mask_{camera}"][indices]
                for camera in _CAMERAS
            },
            **{
                key: self.arrays[key][indices]
                for key in ("state", "tokenized_prompt", "tokenized_prompt_mask")
            },
        }
        donor = {
            **observation,
            "image": {
                camera: self.arrays[f"image_{camera}"][self.image_donors[indices]]
                for camera in _CAMERAS
            },
        }
        end = min(global_start + world_size * batch_size, len(self))
        return ActionProbeBatch(
            indices=indices,
            valid=valid,
            observation=observation,
            donor_observation=donor,
            metadata={key: self.arrays[key][indices] for key in _METADATA},
            globalbatch_has_holdout=bool(
                np.any(self.arrays["split"][global_start:end] == 2)
            ),
        )

    def validate_unchanged(self) -> None:
        """Reject input replacement or modification during feature extraction."""
        if (
            _sha256(self.provenance_path) != self.provenance_sha256
            or _sha256(self.input_dir / "probe_split.json") != self.split_sha256
        ):
            raise ValueError("Action-probe provenance changed during extraction.")
        for path, expected in self._file_stats.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError(
                    f"Action-probe input changed during extraction: {path.name}"
                )


def merge_action_probe_shards(
    output_dir: str | Path, inputs: PreparedActionProbeInputs, world_size: int
) -> dict[str, Any]:
    """Merge rank archives after checking exact coverage and observation identity.

    Each ``rank_00000.npz`` archive contains prepared ``indices``, bool ``valid``,
    three float32 token matrices, and the fields in ``batch.metadata``. Padded
    rows are checked but excluded from the final archive. Existing completed
    archives are never overwritten. The caller owns rank-file cleanup.
    """
    if world_size < 1:
        raise ValueError("Action-probe world size must be positive.")
    inputs.validate_unchanged()
    output_dir = Path(output_dir)
    final_path = output_dir / "features.npz"
    if final_path.exists():
        raise FileExistsError(final_path)
    count = len(inputs)
    seen = np.zeros(count, dtype=np.int64)
    merged = {
        key: np.empty((count, inputs.embedding_dim), dtype=np.float32)
        for key in _FEATURES
    }
    for rank in range(world_size):
        with np.load(output_dir / f"rank_{rank:05d}.npz", allow_pickle=False) as shard:
            indices, valid = shard["indices"], shard["valid"]
            if (
                indices.ndim != 1
                or indices.dtype != np.int64
                or valid.shape != indices.shape
                or valid.dtype != np.bool_
                or np.any(indices < 0)
                or np.any(indices >= count)
            ):
                raise ValueError(
                    f"Invalid action-probe shard indices/mask on rank {rank}."
                )
            for key in _METADATA:
                if not np.array_equal(shard[key], inputs.arrays[key][indices]):
                    raise ValueError(
                        f"Action-probe shard metadata mismatch: rank {rank}, {key}"
                    )
            keep = indices[valid]
            np.add.at(seen, keep, 1)
            for key in _FEATURES:
                value = shard[key]
                if (
                    value.shape != (len(indices), inputs.embedding_dim)
                    or value.dtype != np.float32
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        f"Invalid action-probe features: rank {rank}, {key}"
                    )
                merged[key][keep] = value[valid]
    if not np.all(seen == 1):
        raise ValueError(
            "Action-probe shard indices must cover each prepared sample exactly once."
        )
    inputs.validate_unchanged()
    partial = output_dir / "features.partial.npz"
    with partial.open("xb") as stream:
        np.savez(
            stream,
            **merged,
            **{key: inputs.arrays[key] for key in _METADATA},
            image_donor_index=inputs.image_donors,
        )
    inputs.validate_unchanged()
    partial.replace(final_path)
    digest = _sha256(final_path)
    (output_dir / "features.sha256").write_text(f"{digest}  features.npz\n")
    return {
        "features_path": str(final_path),
        "features_sha256": digest,
        "sample_count": count,
        "probe_split_sha256": inputs.split_sha256,
        "input_provenance_sha256": inputs.provenance_sha256,
        "image_donor_sha256": inputs.image_donor_sha256,
    }
