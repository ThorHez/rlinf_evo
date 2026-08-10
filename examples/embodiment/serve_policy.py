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

"""Serve an RLinf OpenPI_RLinf checkpoint over the OpenPI WebSocket protocol.

The wire protocol is compatible with
``openpi_client.websocket_client_policy.WebsocketClientPolicy``. Unlike
OpenPI's native server entrypoint, this script builds RLinf's JAX-aligned
PyTorch Pi0/Pi0.5 implementation and loads an RLinf ``model.safetensors``
checkpoint.

Example::

    python examples/embodiment/serve_policy.py \
        --config-name pi05_behavior \
        --checkpoint-dir /data/checkpoints/pi05_openpi_rlinf \
        --norm-stats /data/checkpoints/pi05_openpi_rlinf/physical-intelligence/behavior/norm_stats.json \
        --action-env-dim 23 \
        --action-horizon 32
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import socket
import threading
import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Arguments for the RLinf OpenPI policy server."""

    # RLinf/OpenPI data configuration used to construct input/output transforms.
    config_name: str
    # Directory containing model.safetensors from jax_to_openpi_rlinf.
    checkpoint_dir: str
    # Matching normalization-statistics JSON file.
    norm_stats: str

    # Network settings. The defaults match OpenPI's native policy server.
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str | None = None
    record: bool = False
    record_dir: str = "policy_records"

    # Runtime and Pi0.5 model-shape settings.
    device: str = "cuda"
    precision: str = "bf16"
    action_env_dim: int = 23
    action_horizon: int = 32
    model_action_dim: int = 32
    max_token_len: int = 200
    num_steps: int = 5
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    pi05: bool = True


def _batch_tree(value: Any) -> Any:
    """Add one batch dimension to every non-string leaf in a nested mapping."""
    if isinstance(value, dict):
        return {key: _batch_tree(item) for key, item in value.items()}
    if isinstance(value, str):
        return [value]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.expand_dims(np.asarray(value), axis=0)


def _to_unbatched_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Move a single-batch tensor to CPU and remove its batch dimension."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 0 or value.shape[0] != 1:
        raise ValueError(
            "RLinf policy server expects a single request per infer call; "
            f"received output shape {value.shape}."
        )
    return value[0]


def build_model_config(args: Args) -> DictConfig:
    """Build the model-only OmegaConf consumed by ``rlinf.models.get_model``."""
    return OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "model_path": args.checkpoint_dir,
            "precision": args.precision,
            "load_to_device": False,
            "is_lora": False,
            "lora_rank": 32,
            "pi05": args.pi05,
            "num_action_chunks": args.action_horizon,
            "action_dim": args.action_env_dim,
            "use_proprio": True,
            "num_steps": args.num_steps,
            "add_value_head": False,
            "openpi_data": {"norm_stats_path": args.norm_stats},
            "openpi": {
                "task": "eval",
                "config_name": args.config_name,
                "action_chunk": args.action_horizon,
                "action_env_dim": args.action_env_dim,
                "model_action_dim": args.model_action_dim,
                "paligemma_variant": args.paligemma_variant,
                "action_expert_variant": args.action_expert_variant,
                "max_token_len": args.max_token_len,
            },
        }
    )


class RlinfOpenPiPolicy:
    """Adapt an RLinf eval model to OpenPI's single-observation policy API."""

    def __init__(
        self,
        model: Any,
        *,
        default_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._default_prompt = default_prompt
        self._metadata = metadata or {}
        # The upstream server can host multiple clients concurrently. Serialize
        # access because the transform/sampling path is intended for one robot
        # control stream at a time and GPU inference is not re-entrant here.
        self._infer_lock = threading.Lock()

    def infer(
        self, obs: dict[str, Any], *, noise: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Infer one action chunk from an unbatched OpenPI observation."""
        inputs = dict(obs)
        if inputs.get("prompt") is None:
            if self._default_prompt is None:
                raise ValueError(
                    "Observation has no prompt and --default-prompt was not set."
                )
            inputs["prompt"] = self._default_prompt

        batched_inputs = _batch_tree(inputs)
        noise_tensor = None
        if noise is not None:
            noise_tensor = torch.as_tensor(
                noise, device=self._model.device, dtype=torch.float32
            )
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor.unsqueeze(0)

        with self._infer_lock, torch.inference_mode():
            processed = self._model.input_transform(batched_inputs, transpose=False)
            observation = self._model._observation_dict_to_device(processed)
            start_time = time.monotonic()
            actions, _ = self._model._predict_eval(
                observation, noise=noise_tensor, rng=None
            )
            infer_ms = (time.monotonic() - start_time) * 1000

        return {
            "actions": _to_unbatched_numpy(actions),
            "policy_timing": {"infer_ms": infer_ms},
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata sent to clients when they connect."""
        return self._metadata


class PolicyRecorder:
    """Record requests and responses without importing OpenPI's JAX policy."""

    def __init__(self, policy: RlinfOpenPiPolicy, record_dir: str) -> None:
        self._policy = policy
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0
        logger.info("Recording policy requests in %s", self._record_dir)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Run inference and persist the request/response pair."""
        result = self._policy.infer(obs)
        output_path = self._record_dir / f"step_{self._record_step:06d}.npy"
        self._record_step += 1
        np.save(output_path, {"inputs": obs, "outputs": result}, allow_pickle=True)
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        """Forward wrapped-policy metadata."""
        return self._policy.metadata


def create_policy(args: Args) -> RlinfOpenPiPolicy:
    """Load the converted RLinf checkpoint and create its server adapter."""
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {args.device!r}, but PyTorch cannot access CUDA."
        )

    from rlinf.models import get_model

    model = get_model(build_model_config(args))
    if model is None:
        raise RuntimeError("Failed to construct model_type='openpi_rlinf'.")
    model = model.to(args.device).eval()

    metadata = {
        "model_type": "openpi_rlinf",
        "config_name": args.config_name,
        "precision": args.precision,
        "pi05": args.pi05,
        "action_horizon": args.action_horizon,
        "action_env_dim": args.action_env_dim,
        "model_action_dim": args.model_action_dim,
        "num_steps": args.num_steps,
    }
    return RlinfOpenPiPolicy(
        model, default_prompt=args.default_prompt, metadata=metadata
    )


def main(args: Args) -> None:
    """Create the RLinf policy and serve it until interrupted."""
    from openpi.serving import websocket_policy_server

    policy = create_policy(args)
    policy_metadata = policy.metadata
    served_policy = PolicyRecorder(policy, args.record_dir) if args.record else policy

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    logger.info(
        "Creating RLinf policy server (host: %s, ip: %s, listen: %s:%d)",
        hostname,
        local_ip,
        args.host,
        args.port,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=served_policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
