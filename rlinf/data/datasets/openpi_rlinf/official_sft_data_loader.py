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
from typing import Any

from omegaconf import OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    del rank
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

    data_loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=not eval_dataset
    )
    return data_loader, data_loader.data_config()


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
    return len(torch_loader)


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
    return {
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
