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

"""Episode-separated OpenPI loading with weighted sampling across sources."""

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DistributedSampler, Subset


class DistributedWeightedSampler(DistributedSampler):
    """Partition one seeded, weighted draw across distributed replicas.

    Sampling uses replacement, matching OpenPI multi-dataset JAX training.
    Ranks own disjoint positions in that draw; repeated dataset indices are
    expected. The inherited epoch and seed are included in cursor recovery.
    """

    def __init__(self, dataset: Any, weights: np.ndarray, **kwargs: Any) -> None:
        super().__init__(dataset, **kwargs)
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        if (
            self.weights.shape != (len(dataset),)
            or not torch.isfinite(self.weights).all()
            or (self.weights <= 0).any()
        ):
            raise ValueError(
                "Sampling weights must be finite, positive and match the dataset."
            )

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        )
        return iter(indices[self.rank : self.total_size : self.num_replicas].tolist())


def build_multisource_openpi_loader(
    cfg: Any, world_size: int, rank: int, eval_dataset: bool
) -> tuple[Any, Any]:
    """Load source-specific transforms with disjoint episode splits.

    ``data.sources`` contains repo_id, image_horizon, weight and train/val
    episode ID lists for each source. A source's optional ``norm_stats_path``
    overrides the model's checkpoint normalization file. Validation interleaves
    all selected episodes
    globally and never uses weighted sampling.
    """
    import openpi.training.data_loader as openpi_loader

    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        EpisodeFilteredOpenPiLoader,
        _episode_ids,
        _select_episode_frames,
        _validate_openpi_rlinf_model_shape,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    sources = OmegaConf.to_container(cfg.data.sources, resolve=True)
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.sources must be a nonempty list.")
    paths = [str(Path(source["repo_id"]).resolve()) for source in sources]
    if len(paths) != len(set(paths)):
        raise ValueError("Multi-source dataset paths must be distinct.")
    frames = cfg.data.get("val_frames_per_episode", 36) if eval_dataset else None
    if eval_dataset and (type(frames) is not int or frames < 2):
        raise ValueError(
            "Multi-source validation requires val_frames_per_episode >= 2."
        )
    batch_size = (
        cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size)
        if eval_dataset
        else cfg.actor.micro_batch_size
    )
    seed = int(cfg.data.get("eval_seed", 0) if eval_dataset else cfg.actor.seed)
    datasets, indices, labels, weights, identities = [], [], [], [], []
    configs = []
    frame_offset = 0
    episode_offset = 0
    for source, path in zip(sources, paths, strict=True):
        train_ids = _episode_ids(source.get("train_episode_ids"), "train_episode_ids")
        val_ids = _episode_ids(source.get("val_episode_ids"), "val_episode_ids")
        if train_ids is None or val_ids is None or set(train_ids) & set(val_ids):
            raise ValueError(
                "Each source needs nonempty, disjoint train and validation episodes."
            )
        weight = float(source.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Each source weight must be finite and positive.")
        options = OmegaConf.to_container(cfg.actor.model.openpi_data, resolve=True)
        options["image_horizon"] = source.get("image_horizon", 2)
        if source.get("norm_stats_path") is not None:
            options["norm_stats_path"] = source["norm_stats_path"]
        norm_hash = hashlib.sha256(
            Path(options["norm_stats_path"]).read_bytes()
        ).hexdigest()
        config = get_openpi_config(
            cfg.actor.model.openpi.config_name,
            model_path=cfg.actor.model.model_path,
            batch_size=batch_size * world_size,
            repo_id=path,
            data_kwargs=options,
        )
        _validate_openpi_rlinf_model_shape(cfg.actor.model, config)
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = openpi_loader.create_torch_dataset(
            data_config, config.model.action_horizon, config.model
        )
        selected = val_ids if eval_dataset else train_ids
        local_indices, local_labels = _select_episode_frames(dataset, selected, frames)
        episode_ids = sorted(set(train_ids) | set(val_ids))
        mapping = {episode: episode_offset + i for i, episode in enumerate(episode_ids)}
        indices.append(local_indices + frame_offset)
        labels.append(
            np.asarray([mapping[int(x)] for x in local_labels], dtype=np.int64)
        )
        weights.append(np.full(len(local_indices), weight, dtype=np.float64))
        identities.append(
            {
                "repo_id": path,
                "image_horizon": options["image_horizon"],
                "weight": weight,
                "episode_ids": list(selected),
                "norm_sha256": norm_hash,
            }
        )
        for key in ("image_slot", "flatten_state_before_tokenization"):
            if key in options:
                identities[-1][key] = options[key]
        frame_offset += len(dataset)
        episode_offset += len(episode_ids)
        datasets.append(openpi_loader.transform_dataset(dataset, data_config))
        configs.append(data_config)
    if eval_dataset:
        selected_indices = np.concatenate(
            [x.reshape(frames, -1) for x in indices], axis=1
        ).reshape(-1)
        selected_labels = np.concatenate(
            [x.reshape(frames, -1) for x in labels], axis=1
        ).reshape(-1)
    else:
        selected_indices = np.concatenate(indices)
        selected_labels = np.concatenate(labels)
    dataset = Subset(ConcatDataset(datasets), selected_indices.tolist())
    sampler_args = {
        "num_replicas": world_size,
        "rank": rank,
        "seed": seed,
        "drop_last": True,
        "shuffle": not eval_dataset,
    }
    sampler = (
        DistributedSampler(dataset, **sampler_args)
        if eval_dataset
        else DistributedWeightedSampler(
            dataset, np.concatenate(weights), **sampler_args
        )
    )
    epoch_batches = len(sampler) // batch_size
    if epoch_batches < 1:
        raise ValueError("Multi-source selection has no complete distributed batch.")
    num_batches = None
    if eval_dataset:
        limit = cfg.data.get("val_num_batches")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("val_num_batches must be a positive integer or null.")
        num_batches = min(limit, epoch_batches) if limit is not None else epoch_batches
    cross_episode = False
    if eval_dataset and cfg.data.get("eval_token_shuffle", True):
        for replica in range(world_size):
            local = selected_labels[replica : sampler.total_size : world_size][
                : num_batches * batch_size
            ].reshape(-1, batch_size)
            if np.any(local == np.roll(local, 1, axis=1)):
                raise ValueError(
                    "Validation batches do not provide cross-episode token donors."
                )
        cross_episode = True
    inner = openpi_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_batches=num_batches,
        num_workers=int(cfg.data.get("num_workers", 1)),
        seed=seed,
        framework="pytorch",
    )
    shared_config = dataclasses.replace(
        configs[0],
        repo_id="multi:" + hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
    )
    digest = hashlib.sha256(
        json.dumps(identities, sort_keys=True).encode()
        + selected_indices.astype("<i8").tobytes()
    ).hexdigest()
    selection = {
        "episode_ids": sorted(set(selected_labels.tolist())),
        "sources": identities,
        "frame_selection_sha256": digest,
        "source_frame_count": frame_offset,
        "selected_frame_count": len(dataset),
        "frames_per_episode": frames,
        "cross_episode_shuffle_verified": cross_episode,
        "frame_order": "sample_slot_then_episode"
        if eval_dataset
        else "weighted_with_replacement",
        "full_batches_per_rank": epoch_batches,
        "eval_batches_per_rank": num_batches,
        "tail_frames_dropped": len(dataset) - epoch_batches * batch_size * world_size,
    }
    return EpisodeFilteredOpenPiLoader(
        openpi_loader.DataLoaderImpl(shared_config, inner), selection
    ), shared_config
