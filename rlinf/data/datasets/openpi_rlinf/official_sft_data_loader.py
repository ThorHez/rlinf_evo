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

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.storage.lerobot import episode_boundaries, resolve_lerobot_repo_id


class EpisodeFilteredOpenPiLoader:
    """Expose official OpenPI batches and the immutable episode selection."""

    def __init__(self, loader: Any, selection: dict[str, Any]):
        self._loader = loader
        self.episode_selection = selection

    @property
    def _data_loader(self) -> Any:
        """Retain the official wrapper layout used by cursor recovery."""
        return self._loader._data_loader

    def data_config(self) -> Any:
        """Return the unchanged official transform configuration."""
        return self._loader.data_config()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._loader)


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build OpenPI batches, optionally selecting whole train/val episodes.

    Episode selections use ``data.train_episode_ids`` / ``val_episode_ids``.
    The full LeRobot dataset remains underneath a torch Subset, preserving its
    episode boundaries and reusing the original Hugging Face Arrow cache.
    ``val_frames_per_episode`` selects evenly spaced, interleaved frames;
    validation is finite, unshuffled, and capped by ``data.val_num_batches``.
    Image augmentation belongs to model preprocessing and must be disabled by
    the evaluation caller.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("OpenPI rank must be within a positive world size.")
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    model_type = SupportedModel(model_cfg.model_type)
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    if model_type == SupportedModel.OPENPI_RLINF:
        config = dataclasses.replace(
            config,
            num_workers=int(
                OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
            ),
            seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
        )
        _validate_openpi_rlinf_model_shape(model_cfg, config)

    episode_ids = _configured_episode_ids(cfg, eval_dataset)
    if episode_ids is not None or eval_dataset:
        return _build_selected_openpi_loader(
            cfg, config, episode_ids, world_size, rank, batch_size, eval_dataset
        )

    data_loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=not eval_dataset
    )
    return data_loader, data_loader.data_config()


def _episode_ids(value: Any, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).expanduser().read_text())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of nonnegative episode IDs.")
    if not value or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item < 0
        for item in value
    ):
        raise ValueError(f"{name} must contain nonnegative integer episode IDs.")
    ids = tuple(int(item) for item in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate episode IDs.")
    return ids


def _configured_episode_ids(cfg: Any, eval_dataset: bool) -> tuple[int, ...] | None:
    train_ids = _episode_ids(
        OmegaConf.select(cfg, "data.train_episode_ids"), "data.train_episode_ids"
    )
    val_ids = _episode_ids(
        OmegaConf.select(cfg, "data.val_episode_ids"), "data.val_episode_ids"
    )
    train_repo = resolve_lerobot_repo_id(OmegaConf.select(cfg, "data.train_data_paths"))
    val_repo = resolve_lerobot_repo_id(OmegaConf.select(cfg, "data.val_data_paths"))
    if train_ids is not None and val_ids is not None and train_repo == val_repo:
        overlap = set(train_ids) & set(val_ids)
        if overlap:
            raise ValueError(
                f"OpenPI train and val episodes overlap: {sorted(overlap)}"
            )
    return val_ids if eval_dataset else train_ids


def _select_episode_frames(
    dataset: Any, episode_ids: tuple[int, ...], frames_per_episode: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve selected frames against the full dataset's original boundaries."""
    import openpi.training.data_loader as openpi_data_loader

    original = dataset
    while isinstance(original, openpi_data_loader.TransformedDataset):
        original = original._dataset
    starts, ends = episode_boundaries(original)
    metadata = original.meta.episodes
    all_ids = (
        list(metadata)
        if isinstance(metadata, dict)
        else list(metadata["episode_index"])
    )
    if len(all_ids) != len(starts) or len(starts) != len(ends):
        raise ValueError("LeRobot episode metadata and frame boundaries disagree.")
    ranges = dict(zip(all_ids, zip(starts, ends, strict=True), strict=True))
    missing = set(episode_ids) - ranges.keys()
    if missing:
        raise ValueError(f"Unknown LeRobot episode IDs: {sorted(missing)}")
    rows, labels = [], []
    for episode_id in episode_ids:
        start, end = ranges[episode_id]
        if not 0 <= start < end <= len(dataset):
            raise ValueError(f"Invalid frame range for episode {episode_id}.")
        if frames_per_episode is None:
            indices = np.arange(start, end, dtype=np.int64)
        else:
            if end - start < frames_per_episode:
                raise ValueError(
                    f"Episode {episode_id} has {end - start} frames; "
                    f"cannot select {frames_per_episode} unique validation frames."
                )
            indices = (
                np.linspace(0, end - start - 1, frames_per_episode, dtype=np.int64)
                + start
            )
        rows.append(indices)
        labels.append(np.full(len(indices), episode_id, dtype=np.int64))
    if frames_per_episode is None:
        return np.concatenate(rows), np.concatenate(labels)
    # Interleave episodes so each local validation batch can use a different
    # episode as the donor when rolling tokens along its batch dimension.
    return np.stack(rows).T.reshape(-1), np.stack(labels).T.reshape(-1)


def _build_selected_openpi_loader(
    cfg: Any,
    config: Any,
    episode_ids: tuple[int, ...] | None,
    world_size: int,
    rank: int,
    batch_size: int,
    eval_dataset: bool,
) -> tuple[Any, Any]:
    import openpi.training.data_loader as openpi_data_loader
    from torch.utils.data import DistributedSampler, Subset

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise ValueError("Episode-filtered OpenPI loading requires a LeRobot dataset.")
    # Do not pass episodes= to LeRobotDataset: its data_files-based HF loader
    # would rebuild the large image cache instead of reusing the full dataset.
    dataset = openpi_data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    source_size = len(dataset)
    frames_per_episode = (
        OmegaConf.select(cfg, "data.val_frames_per_episode") if eval_dataset else None
    )
    if frames_per_episode is not None and (
        isinstance(frames_per_episode, bool)
        or not isinstance(frames_per_episode, Integral)
        or frames_per_episode < 2
    ):
        raise ValueError("data.val_frames_per_episode must be an integer >= 2 or null.")
    if frames_per_episode is not None and episode_ids is None:
        raise ValueError("val_frames_per_episode requires explicit val_episode_ids.")
    selected_indices = np.arange(source_size, dtype=np.int64)
    selected_episodes = None
    if episode_ids is not None:
        selected_indices, selected_episodes = _select_episode_frames(
            dataset, episode_ids, frames_per_episode
        )
        # Hugging Face Dataset rejects numpy integer keys, which Subset would
        # otherwise pass through to the underlying LeRobot dataset unchanged.
        dataset = Subset(dataset, selected_indices.tolist())
    dataset = openpi_data_loader.transform_dataset(dataset, data_config)
    seed = int(
        OmegaConf.select(cfg, "data.eval_seed", default=config.seed)
        if eval_dataset
        else config.seed
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=seed,
        drop_last=True,
    )
    epoch_batches = len(sampler) // batch_size
    if epoch_batches == 0:
        raise ValueError("OpenPI selection contains no complete distributed batch.")
    num_batches = None
    if eval_dataset:
        requested = OmegaConf.select(cfg, "data.val_num_batches")
        if requested is not None and (
            isinstance(requested, bool)
            or not isinstance(requested, Integral)
            or requested <= 0
        ):
            raise ValueError("data.val_num_batches must be a positive integer or null.")
        num_batches = min(requested, epoch_batches) if requested else epoch_batches
    cross_episode_verified = False
    if eval_dataset and frames_per_episode is not None:
        for replica in range(world_size):
            local_episodes = selected_episodes[
                replica : len(sampler) * world_size : world_size
            ]
            local_episodes = local_episodes[: num_batches * batch_size].reshape(
                -1, batch_size
            )
            if np.any(local_episodes == np.roll(local_episodes, 1, axis=1)):
                raise ValueError(
                    "Validation interleaving does not provide cross-episode token "
                    "donors for every local batch; adjust episode count or batch size."
                )
        cross_episode_verified = True
    inner = openpi_data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
        framework="pytorch",
    )
    loader = openpi_data_loader.DataLoaderImpl(data_config, inner)
    if episode_ids is not None:
        selection_digest = hashlib.sha256(
            json.dumps(episode_ids).encode() + selected_indices.astype("<i8").tobytes()
        ).hexdigest()
        selection = {
            "episode_ids": list(episode_ids),
            "frames_per_episode": frames_per_episode,
            "frame_selection_sha256": selection_digest,
            "source_frame_count": source_size,
            "selected_frame_count": len(selected_indices),
            "frame_order": "sample_slot_then_episode"
            if frames_per_episode
            else "episode_then_frame",
            "cross_episode_shuffle_verified": cross_episode_verified,
            "full_batches_per_rank": epoch_batches,
            "eval_batches_per_rank": num_batches,
            "tail_frames_dropped": len(selected_indices)
            - epoch_batches * batch_size * world_size,
        }
        loader = EpisodeFilteredOpenPiLoader(loader, selection)
    return loader, data_config


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    configured_batches = getattr(openpi_loader, "_num_batches", None)
    return (
        min(len(torch_loader), configured_batches)
        if configured_batches is not None
        else len(torch_loader)
    )


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def get_openpi_sft_data_order(data_loader: Any) -> dict[str, Any]:
    """Describe the sample order of an official distributed OpenPI loader."""
    from torch.utils.data import DistributedSampler

    torch_loader = data_loader._data_loader.torch_loader
    sampler = torch_loader.sampler
    if not isinstance(sampler, DistributedSampler):
        raise TypeError("OpenPI cursor recovery requires a DistributedSampler.")
    result = {
        "dataset_size": len(torch_loader.dataset),
        "dataset_repo_id": str(data_loader.data_config().repo_id),
        "sampler": {
            "class": type(sampler).__name__,
            "seed": sampler.seed,
            "epoch": sampler.epoch,
            "rank": sampler.rank,
            "num_replicas": sampler.num_replicas,
            "drop_last": sampler.drop_last,
            "shuffle": sampler.shuffle,
        },
    }
    selection = getattr(data_loader, "episode_selection", None)
    if selection is not None:
        result["episode_ids"] = selection["episode_ids"]
        result["frame_selection_sha256"] = selection["frame_selection_sha256"]
    return result


def restore_openpi_sft_iterator(
    data_loader: Any, saved_state: dict[str, Any]
) -> tuple[Any, int]:
    """Replay consumed samples for a deterministic, first-epoch UMI loader.

    A changed micro batch is supported when the consumed sample count is
    divisible by its new size. Random data transforms and crossing an epoch
    boundary require worker RNG and sampler state beyond this cursor format.
    """
    current = get_openpi_sft_data_order(data_loader)
    for key in ("dataset_size", "dataset_repo_id"):
        if current[key] != saved_state[key]:
            raise ValueError(f"OpenPI resume data mismatch: {key}.")
    for key in ("episode_ids", "frame_selection_sha256"):
        if current.get(key) != saved_state.get(key):
            raise ValueError(f"OpenPI resume episode selection mismatch: {key}.")
    for key, value in current["sampler"].items():
        expected = saved_state["sampler"][key]
        if key == "class":
            expected = expected.rsplit(".", 1)[-1]
        if value != expected:
            raise ValueError(f"OpenPI resume sampler mismatch: {key}.")
    torch_loader = data_loader._data_loader.torch_loader
    consumed = int(saved_state["consumed_samples"])
    batch_size = int(torch_loader.batch_size)
    if (
        saved_state["data_epoch"] != 0
        or consumed < 0
        or consumed > len(torch_loader) * batch_size
        or consumed % batch_size
    ):
        raise ValueError("OpenPI resume cursor must align within the first epoch.")
    iterator = iter(data_loader)
    skipped_batches = consumed // batch_size
    for _ in range(skipped_batches):
        next(iterator)
    generator_state = saved_state.get("loader_generator_state")
    if generator_state is not None:
        torch_loader.generator.set_state(generator_state)
    return iterator, skipped_batches


def _validate_openpi_rlinf_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config."""
    local_horizon = int(model_cfg.num_action_chunks)
    official_horizon = int(openpi_config.model.action_horizon)
    if local_horizon != official_horizon:
        raise ValueError(
            "openpi_rlinf SFT action horizon must match the official OpenPI "
            f"config: actor.model.num_action_chunks={local_horizon}, "
            f"{model_cfg.openpi.config_name}.model.action_horizon="
            f"{official_horizon}."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi_rlinf SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )
