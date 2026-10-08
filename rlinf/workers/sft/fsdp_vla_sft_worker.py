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
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_RLINF:
            from rlinf.data.datasets.openpi_rlinf import (
                build_openpi_rlinf_sft_dataloader,
            )

            return build_openpi_rlinf_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi_rlinf import (
                build_official_openpi_sft_dataloader,
            )

            return build_official_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.COSMOS3:
            from rlinf.data.datasets.cosmos3 import (
                build_cosmos3_sft_dataloader,
            )

            return build_cosmos3_sft_dataloader(self.cfg, data_paths, eval_dataset)
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.FASTWAM:
            from rlinf.data.datasets.fastwam import build_fastwam_sft_dataloader

            return build_fastwam_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: Any) -> dict[str, torch.Tensor]:
        """Evaluate native OpenPI through the same FSDP wrapper as training."""
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            != SupportedModel.OPENPI_RLINF
        ):
            raise NotImplementedError("SFT validation currently supports openpi_rlinf.")
        with torch.no_grad(), self.amp_context:
            return self.model(
                forward_type=ForwardType.SFT,
                data=batch,
                evaluation=True,
                eval_num_steps=int(self.cfg.data.get("eval_num_steps", 10)),
                eval_token_shuffle=bool(self.cfg.data.get("eval_token_shuffle", True)),
            )

    def run_eval(self) -> dict[str, float]:
        """Validate a fixed finite subset without advancing training RNG streams.

        Sample metrics are weighted by batch size; token reconstruction metrics
        are weighted by valid embedding elements. All ranks check batch
        availability before each FSDP forward and abort together on loader failure.
        """
        if self.eval_data_loader is None:
            raise ValueError("eval_data_loader is not set")
        from rlinf.data.datasets.openpi_rlinf import (
            get_official_openpi_sft_num_batches,
            is_official_openpi_sft_dataloader,
        )

        if is_official_openpi_sft_dataloader(self.eval_data_loader):
            batches = get_official_openpi_sft_num_batches(self.eval_data_loader)
        else:
            batches = len(self.eval_data_loader)
        limit = self.cfg.data.get("val_num_batches")
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("data.val_num_batches must be positive or null.")
            batches = min(batches, int(limit))
        device = next(self.model.parameters()).device
        distributed = self._world_size > 1
        counts = torch.tensor([batches, -batches], device=device, dtype=torch.int64)
        if distributed:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.MIN)
        if counts[0].item() < 1 or counts[0].item() != -counts[1].item():
            raise ValueError(
                "Validation requires equal, nonempty batch counts on all ranks."
            )

        rng_state = get_rng_state()
        module_modes = [(module, module.training) for module in self.model.modules()]
        numerators: dict[str, float] = {}
        denominators: dict[str, float] = {}
        sample_count = 0.0
        try:
            seed = int(self.cfg.data.get("eval_seed", 0)) + self._rank
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if self.torch_platform.is_available():
                self.torch_platform.manual_seed(seed)
            self.model.eval()
            with self.worker_timer(), torch.no_grad():
                iterator = iter(self.eval_data_loader)
                for _ in range(batches):
                    loader_error = None
                    try:
                        batch = next(iterator)
                    except Exception as error:
                        loader_error = error
                    available = torch.tensor(
                        int(loader_error is None), device=device, dtype=torch.int64
                    )
                    if distributed:
                        torch.distributed.all_reduce(
                            available, op=torch.distributed.ReduceOp.MIN
                        )
                    if not available.item():
                        raise RuntimeError(
                            "Validation loader failed before all scheduled batches completed."
                        ) from loader_error
                    output = self.get_eval_model_output(batch)
                    samples = float(output["sample_count"].item())
                    elements = float(
                        output.get("rlt_valid_elements", output["sample_count"]).item()
                    )
                    sample_count += samples
                    for name, value in output.items():
                        if name in ("loss", "sample_count", "rlt_valid_elements"):
                            continue
                        weight = elements if name.startswith("rlt_") else samples
                        numerators[name] = (
                            numerators.get(name, 0.0) + float(value.item()) * weight
                        )
                        denominators[name] = denominators.get(name, 0.0) + weight

                names = sorted(numerators)
                totals = torch.tensor(
                    [numerators[name] for name in names]
                    + [denominators[name] for name in names]
                    + [sample_count],
                    dtype=torch.float64,
                    device=device,
                )
                if distributed:
                    torch.distributed.all_reduce(
                        totals, op=torch.distributed.ReduceOp.SUM
                    )
                if (
                    not torch.isfinite(totals).all()
                    or (totals[len(names) :] <= 0).any()
                ):
                    raise ValueError(
                        "Validation produced nonfinite metrics or empty reductions."
                    )
                values = totals.cpu().tolist()
                metrics = {
                    name: values[index] / values[index + len(names)]
                    for index, name in enumerate(names)
                }
                metrics["samples"] = values[-1]
                metrics["loss"] = metrics["vla_loss"]
                if "rlt_loss" in metrics:
                    alpha = float(self.cfg.actor.model.openpi.get("rlt_alpha", 1.0))
                    metrics["loss"] = metrics["rlt_loss"] + alpha * metrics["vla_loss"]
                    if "rlt_shuffle_loss" in metrics:
                        metrics["rlt_shuffle_ratio"] = metrics[
                            "rlt_shuffle_loss"
                        ] / max(metrics["rlt_loss"], 1e-12)
                        selection = getattr(
                            self.eval_data_loader, "episode_selection", {}
                        )
                        metrics["rlt_shuffle_cross_episode"] = float(
                            selection.get("cross_episode_shuffle_verified", False)
                        )
                return metrics
        finally:
            for module, training in module_modes:
                module.training = training
            set_rng_state(rng_state)

    def extract_action_probe_features(
        self, output_dir: str, global_step: int
    ) -> dict[str, Any]:
        """Extract a fixed probe dataset from the current sharded model.

        Every actor participates in the same FSDP forwards. Rank zero merges
        compact tokens on the shared output filesystem. This RPC never consumes
        the training iterator and restores RNG streams and module modes even
        when input loading or output publication fails.
        """
        from rlinf.data.datasets.openpi_rlinf.action_probe import (
            PreparedActionProbeInputs,
            merge_action_probe_shards,
        )
        from rlinf.models.embodiment.modules.rlt_token_transformer import (
            RLTTokenEncoder,
        )
        from rlinf.models.embodiment.openpi_rlinf.rlt_config import build_rlt_config

        if (
            SupportedModel(self.cfg.actor.model.model_type)
            != SupportedModel.OPENPI_RLINF
        ):
            raise NotImplementedError("Action probes require native OpenPI RLT.")
        options = self.cfg.runner.action_probe
        rlt = build_rlt_config(self.cfg.actor.model.openpi)
        if not rlt.use_rlt:
            raise ValueError("Action probes require actor.model.openpi.use_rlt.")
        batch_size = int(options.get("batch_size", 9))
        if batch_size < 1 or global_step < 0:
            raise ValueError("Probe batch size must be positive and step nonnegative.")
        device = next(self.model.parameters()).device
        distributed = self._world_size > 1
        directory = Path(output_dir)
        started = time.monotonic()
        rng_state = get_rng_state()
        module_modes = [(module, module.training) for module in self.model.modules()]
        random_encoder = None
        record: dict[str, Any] = {"global_step": int(global_step)}

        def synchronize_error(error: Exception | None, stage: str) -> None:
            success = torch.tensor(int(error is None), device=device, dtype=torch.int64)
            if distributed:
                torch.distributed.all_reduce(success, op=torch.distributed.ReduceOp.MIN)
            if not success.item():
                raise RuntimeError(
                    f"Action probe {stage} failed on an actor."
                ) from error

        def write_status(record: dict[str, Any]) -> None:
            temporary = directory / "extraction_status.json.tmp"
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            temporary.replace(directory / "extraction_status.json")

        def tensors(value):
            if isinstance(value, dict):
                return {key: tensors(item) for key, item in value.items()}
            return torch.from_numpy(np.asarray(value).copy())

        try:
            error = None
            try:
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "features.npz").exists():
                    raise FileExistsError(
                        "Completed probe features must not be overwritten."
                    )
                inputs = PreparedActionProbeInputs(
                    options.input_dir,
                    self.cfg,
                    image_seed=int(options.get("image_seed", 20260925)),
                )
                random.seed(int(options.get("random_seed", 271828)))
                np.random.seed(int(options.get("random_seed", 271828)) % (2**32))
                torch.random.default_generator.manual_seed(
                    int(options.get("random_seed", 271828))
                )
                random_encoder = (
                    RLTTokenEncoder(
                        input_dim=rlt.rlt_input_dim,
                        embed_dim=rlt.rlt_embed_dim,
                        prefix_seq_len=rlt.rlt_prefix_seq_len,
                        num_layers=rlt.rlt_num_layers,
                        num_heads=rlt.rlt_num_heads,
                        mlp_ratio=rlt.rlt_mlp_ratio,
                    )
                    .requires_grad_(False)
                    .eval()
                    .to(device)
                )
                self.model.eval()
            except Exception as caught:
                error = caught
            synchronize_error(error, "initialization")
            batches = inputs.batch_count(self._world_size, batch_size)
            counts = torch.tensor([batches, -batches], device=device, dtype=torch.int64)
            if distributed:
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.MIN)
            if counts[0].item() < 1 or counts[0].item() != -counts[1].item():
                raise ValueError("Action probes require equal nonempty rank batches.")
            record = {
                "state": "extracting",
                "source_kind": "live_fsdp_model",
                "global_step": int(global_step),
                "model_identity": {
                    "global_step": int(global_step),
                    "config_sha256": hashlib.sha256(
                        OmegaConf.to_yaml(self.cfg, resolve=True).encode()
                    ).hexdigest(),
                },
                "data_directory": str(Path(options.input_dir).resolve()),
                "input_provenance_sha256": inputs.provenance_sha256,
                "input_quality_sha256": hashlib.sha256(
                    (Path(options.input_dir) / "input_quality_check.json").read_bytes()
                ).hexdigest(),
                "probe_split_sha256": inputs.split_sha256,
                "sample_count": len(inputs),
                "random_encoder_seed": int(options.get("random_seed", 271828)),
                "image_shuffle_seed": int(options.get("image_seed", 20260925)),
                "world_size": self._world_size,
                "batch_size": batch_size,
                "done": 0,
            }
            metadata_keys = (
                "state",
                "actions",
                "episode_id",
                "frame_index",
                "dataset_global_index",
                "split",
            )
            collected = {
                key: []
                for key in (
                    "indices",
                    "valid",
                    "z",
                    "z_random",
                    "z_image_swapped",
                    *metadata_keys,
                )
            }
            for batch_index in range(batches):
                error = None
                try:
                    batch = inputs.batch(
                        batch_index, self._rank, self._world_size, batch_size
                    )
                    observation = tensors(batch.observation)
                    donor = (
                        tensors(batch.donor_observation)
                        if batch.globalbatch_has_holdout
                        else None
                    )
                except Exception as caught:
                    error = caught
                synchronize_error(error, "input loading")
                with torch.no_grad(), self.amp_context:
                    output = self.model(
                        forward_type=ForwardType.SFT,
                        data=observation,
                        feature_only=True,
                        random_encoder=random_encoder,
                        image_donor_observation=donor,
                    )
                error = None
                try:
                    for key in ("z", "z_random", "z_image_swapped"):
                        value = output[key].detach().float().cpu().numpy()
                        if (
                            value.shape != (batch_size, rlt.rlt_embed_dim)
                            or not np.isfinite(value).all()
                        ):
                            raise ValueError(f"Invalid action probe feature {key}.")
                        collected[key].append(value)
                    collected["indices"].append(batch.indices)
                    collected["valid"].append(batch.valid)
                    for key in metadata_keys:
                        collected[key].append(
                            np.asarray(inputs.arrays[key][batch.indices])
                        )
                    if self._rank == 0 and (
                        batch_index % 20 == 0 or batch_index == batches - 1
                    ):
                        record.update(
                            done=min(
                                (batch_index + 1) * batch_size * self._world_size,
                                len(inputs),
                            ),
                            elapsed_seconds=time.monotonic() - started,
                        )
                        write_status(record)
                except Exception as caught:
                    error = caught
                synchronize_error(error, "feature collection")
            error = None
            try:
                shard = directory / f"rank_{self._rank:05d}.npz"
                temporary = shard.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez(
                        stream,
                        **{
                            key: np.concatenate(value)
                            for key, value in collected.items()
                        },
                    )
                temporary.replace(shard)
            except Exception as caught:
                error = caught
            synchronize_error(error, "shard writing")
            result = {}
            error = None
            if self._rank == 0:
                try:
                    result = merge_action_probe_shards(
                        directory, inputs, self._world_size
                    )
                    record.update(
                        result,
                        state="completed",
                        done=len(inputs),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    write_status(record)
                except Exception as caught:
                    error = caught
            synchronize_error(error, "archive publication")
            return (
                record
                if self._rank == 0
                else {"state": "completed", "global_step": int(global_step)}
            )
        except Exception as error:
            if self._rank == 0 and not (directory / "features.npz").exists():
                try:
                    write_status({**record, "state": "failed", "error": str(error)})
                except OSError:
                    # Preserve the original distributed failure if storage is unavailable.
                    pass
            raise
        finally:
            # The random control is never registered in the training model or optimizer.
            del random_encoder
            for module, training in module_modes:
                module.training = training
            set_rng_state(rng_state)

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if self.cfg.data.get("save_openpi_resume_state", False):
            from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
                get_openpi_sft_data_order,
            )

            if self.cfg.actor.model.openpi.config_name != "pi05_umi_v4":
                raise ValueError(
                    "OpenPI cursor recovery requires deterministic UMI V4 transforms."
                )
            state = get_openpi_sft_data_order(self.data_loader)
            state.update(
                rank=self._rank,
                optimizer_steps=step,
                data_epoch=self._data_epoch,
                data_iter_offset=self._data_iter_offset,
                micro_batch_size=self.micro_batch_size,
                consumed_samples=self._data_iter_offset * self.micro_batch_size,
                rng_state=get_rng_state(),
                loader_generator_state=self.data_loader._data_loader.torch_loader.generator.get_state(),
            )
            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)
            if self._rank == 0:
                torch.save(
                    {
                        "version": 1,
                        "global_step": step,
                        "world_size": self._world_size,
                        "ranks": all_states,
                    },
                    os.path.join(save_path, "openpi_resume_state.pt"),
                )
            torch.distributed.barrier()

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        cursor_path = os.path.join(load_path, "openpi_resume_state.pt")
        if self.cfg.data.get("save_openpi_resume_state", False) and os.path.exists(
            cursor_path
        ):
            from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
                restore_openpi_sft_iterator,
            )

            if self.cfg.actor.model.openpi.config_name != "pi05_umi_v4":
                raise ValueError(
                    "OpenPI cursor recovery requires deterministic UMI V4 transforms."
                )
            saved = torch.load(cursor_path, weights_only=False, map_location="cpu")
            if saved["version"] != 1 or saved["world_size"] != self._world_size:
                raise ValueError("OpenPI resume cursor version or world size mismatch.")
            state = saved["ranks"][self._rank]
            if state["rank"] != self._rank:
                raise ValueError("OpenPI resume cursor rank mismatch.")
            self.data_iter, self._data_iter_offset = restore_openpi_sft_iterator(
                self.data_loader, state
            )
            self._data_epoch = 0
            self.optimizer_steps = int(saved["global_step"])
            set_rng_state(state["rng_state"])
            self.log_info(
                f"Restored OpenPI data cursor: {state['consumed_samples']} samples "
                f"per rank, optimizer step {self.optimizer_steps}."
            )
            torch.distributed.barrier()

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in (SupportedModel.OPENPI_RLINF, SupportedModel.OPENPI):
            if model_type == SupportedModel.OPENPI_RLINF:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                    is_official_openpi_sft_dataloader,
                )

                num_batches = (
                    get_official_openpi_sft_num_batches(self.data_loader)
                    if is_official_openpi_sft_dataloader(self.data_loader)
                    else len(self.data_loader)
                )
            else:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                )

                num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
