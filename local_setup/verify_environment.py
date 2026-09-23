#!/usr/bin/env python3
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

"""Check the local RL Token runtime without a robot or model checkpoint.

Run with the installed environment's Python. CUDA is required; JAX is kept on
CPU, and the small PyTorch checks cap allocator usage at 512 MiB. Ray starts a
separate local instance and is shut down before this process exits.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    """Fail a check even when Python runs with optimizations enabled."""
    if not condition:
        raise RuntimeError(message)


def versions() -> dict[str, Any]:
    packages = (
        "rlinf",
        "torch",
        "torchvision",
        "torchcodec",
        "ray",
        "rlinf-openpi",
        "rlinf-transformer-openpi",
        "transformers",
        "tokenizers",
        "lerobot",
        "jax",
        "jaxlib",
        "orbax-checkpoint",
        "flax",
        "peft",
        "diffusers",
        "numpy",
        "datasets",
        "protobuf",
        "av",
    )
    return {
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def imports() -> dict[str, Any]:
    modules = (
        "openpi.training.data_loader",
        "lerobot.common.datasets.lerobot_dataset",
        "rlinf.models.embodiment.openpi_rlinf.pi0",
        "rlinf.models.embodiment.openpi_rlinf.tasks.eval",
        "rlinf.workers.actor.fsdp_rlt_ac_policy_worker",
        "rlinf.workers.rollout.hf.huggingface_worker",
        "rlinf.workers.env.env_worker",
        "rlinf.workers.sft.fsdp_vla_sft_worker",
    )
    loaded = {name: str(importlib.import_module(name).__file__) for name in modules}
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config("pi05_umi_v4")
    require(config.model.action_horizon == 16, "Unexpected UMI action horizon")
    require(config.model.action_dim == 32, "Unexpected padded UMI action dimension")
    return {"modules": loaded, "dataconfig": config.name}


def cuda_and_rlt() -> dict[str, Any]:
    import torch

    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder

    require(torch.cuda.is_available(), "PyTorch cannot access an NVIDIA GPU")
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(
        min(0.5, (512 * 1024**2) / properties.total_memory), device
    )
    torch.cuda.reset_peak_memory_stats(device)
    values = torch.arange(16, dtype=torch.float32, device=device)
    require(values.square().sum().item() == 1240.0, "CUDA arithmetic failed")

    encoder = RLTTokenEncoder(
        input_dim=16,
        embed_dim=16,
        prefix_seq_len=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
    ).to(device)
    policy = RLTMLPPolicy(
        z_dim=16,
        proprio_dim=16,
        action_dim=10,
        num_action_chunks=16,
    ).to(device)
    prefix = torch.randn(2, 8, 16, device=device, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool, device=device)
    z_rl = encoder(prefix, mask).reshape(2, 16)
    observations = {
        "z_rl": z_rl,
        "proprio": torch.zeros(2, 16, device=device),
        "ref_chunk": torch.zeros(2, 16, 10, device=device),
    }
    actions, logprobs, _ = policy.sac_forward(observations, deterministic=True)
    q_values = policy.sac_q_forward(observations, actions)
    require(tuple(actions.shape) == (2, 160), "Unexpected action chunk shape")
    require(q_values.shape[-1] == 2, "RLT critic did not produce twin Q values")
    for name, value in (("actions", actions), ("logprobs", logprobs), ("Q", q_values)):
        require(bool(torch.isfinite(value).all()), f"Non-finite {name}")
    loss = actions.square().mean() + q_values.square().mean()
    loss.backward()
    require(prefix.grad is not None, "RLT encoder backward did not reach its input")
    for name, module in (("encoder", encoder), ("actor_critic", policy)):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        require(bool(gradients), f"No gradients for {name}")
        require(
            all(bool(torch.isfinite(g).all()) for g in gradients),
            f"Non-finite gradients for {name}",
        )
        require(
            any(bool(g.abs().sum() > 0) for g in gradients),
            f"Zero gradients for {name}",
        )
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    require(peak < 512 * 1024**2, "CUDA allocator usage exceeded the self-check budget")
    result = {
        "gpu": properties.name,
        "torch_cuda": torch.version.cuda,
        "action_shape": list(actions.shape),
        "q_shape": list(q_values.shape),
        "loss": loss.item(),
        "peak_torch_reserved_mib": round(peak / 1024**2, 2),
    }
    del (
        encoder,
        policy,
        prefix,
        mask,
        z_rl,
        observations,
        actions,
        logprobs,
        q_values,
        loss,
    )
    torch.cuda.empty_cache()
    return result


def tokenizer() -> dict[str, Any]:
    from openpi.models.tokenizer import PaligemmaTokenizer

    cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    model_file = cache / "big_vision" / "paligemma_tokenizer.model"
    require(model_file.is_file(), f"Missing local tokenizer asset: {model_file}")
    tokenizer_instance = PaligemmaTokenizer(max_len=48)
    tokens, mask = tokenizer_instance.tokenize("pick up the object")
    require(tokens.shape == mask.shape == (48,), "Unexpected tokenizer output shape")
    require(bool(mask.any()), "Tokenizer produced no valid tokens")
    return {"asset": str(model_file.resolve()), "valid_tokens": int(mask.sum())}


def video_codecs() -> dict[str, Any]:
    import av
    import numpy as np
    import torch
    from torchcodec.decoders import VideoDecoder
    from torchcodec.encoders import VideoEncoder

    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    for index in range(4):
        frames[index, :, :, index % 3] = 64 + 32 * index
    with tempfile.TemporaryDirectory(prefix="rlt-video-") as directory:
        pyav_path = Path(directory) / "pyav.mp4"
        with av.open(str(pyav_path), mode="w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
            for pixels in frames:
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(pixels, format="rgb24")
                ):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(pyav_path)) as container:
            decoded = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            ]
        require(np.stack(decoded).shape == frames.shape, "PyAV video round trip failed")
        decoder = VideoDecoder(str(pyav_path), device="cpu", dimension_order="NHWC")
        decoded_tensor = decoder[:]
        require(
            tuple(decoded_tensor.shape) == frames.shape, "TorchCodec decoding failed"
        )
        difference = (
            (decoded_tensor.to(torch.float32) - torch.from_numpy(frames).float())
            .abs()
            .mean()
        )
        require(difference.item() < 10, "Decoded video differs excessively from input")

        torchcodec_path = Path(directory) / "torchcodec.mp4"
        VideoEncoder(
            torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),
            frame_rate=10,
        ).to_file(
            torchcodec_path, codec="libx264", pixel_format="yuv420p", preset="ultrafast"
        )
        with av.open(str(torchcodec_path)) as container:
            encoded_frames = list(container.decode(video=0))
        require(len(encoded_frames) == len(frames), "TorchCodec encoding failed")
        del decoder
    return {
        "frames": 4,
        "size": [32, 32],
        "codec": "h264",
        "mean_absolute_error": difference.item(),
    }


def local_ray() -> dict[str, Any]:
    import ray

    require(not ray.is_initialized(), "Self-check requires its own fresh Ray instance")
    with tempfile.TemporaryDirectory(prefix="rlt-ray-", dir="/tmp") as directory:
        try:
            ray.init(
                address="local",
                include_dashboard=False,
                num_cpus=2,
                num_gpus=0,
                log_to_driver=False,
                _temp_dir=directory,
                object_store_memory=80 * 1024**2,
            )

            @ray.remote
            def probe() -> dict[str, Any]:
                import sys

                return {"interpreter": sys.executable, "sum": sum(range(10))}

            result = ray.get(probe.remote(), timeout=30)
            require(result["sum"] == 45, "Ray remote task returned an incorrect result")
            require(
                result["interpreter"] == sys.executable,
                "Ray used a different Python environment",
            )
            return result
        finally:
            ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-ray", action="store_true", help="Skip the isolated local Ray check"
    )
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    checks = [versions, imports, cuda_and_rlt, tokenizer, video_codecs]
    if not args.skip_ray:
        checks.append(local_ray)
    for check in checks:
        start = time.monotonic()
        try:
            result = check()
        except Exception as error:
            print(
                json.dumps(
                    {"check": check.__name__, "status": "FAIL", "error": str(error)}
                ),
                flush=True,
            )
            raise
        print(
            json.dumps(
                {
                    "check": check.__name__,
                    "status": "PASS",
                    "seconds": round(time.monotonic() - start, 2),
                    "details": result,
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {"status": "PASS", "checks": len(checks), "ray_skipped": args.skip_ray}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
