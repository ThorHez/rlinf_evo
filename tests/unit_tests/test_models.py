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

"""Model registration, embeddings, and the reward-model helpers."""

from __future__ import annotations

import asyncio
import copy
import datetime
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.config import SupportedModel, validate_fsdp_cfg
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.scheduler import Worker
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _DummyFSDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = torch.nn.Linear(4, 2)
        self.head._fsdp_wrap_name = "custom_head"


def test_custom_model_registration_smoke():
    model_type = f"custom_model_smoke_{int(time.time() * 1000)}"
    received = {"torch_dtype": None}

    def _builder(cfg, torch_dtype):
        received["torch_dtype"] = torch_dtype
        return _DummyModel()

    register_model(model_type, _builder, category="embodied")

    supported_model = SupportedModel(model_type)
    assert supported_model.value == model_type

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    model = get_model(cfg)

    assert isinstance(model, _DummyModel)
    assert received["torch_dtype"] == torch.float32


def test_custom_model_registration_with_fsdp_wrap_policy():
    model_type = f"custom_model_fsdp_{int(time.time() * 1000)}"

    def _builder(cfg, torch_dtype):
        return _DummyFSDPModel()

    register_model(
        model_type,
        _builder,
        category="embodied",
    )

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    fsdp_cfg = OmegaConf.create(
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": ["_DummyBlock"],
                "module_classes_to_wrap": ["_DummyBlock"],
                "no_split_names": ["custom_head"],
            },
            "use_orig_params": True,
        }
    )
    model = get_model(cfg)
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)


def _make_model(*, prefix_seq_len: int = 5) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0


@pytest.fixture
def tiny_openpi_checkpoint(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    from rlinf.models.embodiment.openpi_rlinf import get_model as build_openpi
    from rlinf.models.embodiment.openpi_rlinf.modules import gemma, siglip
    from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
    from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config

    class TinySigLIP(siglip.SigLIPViT):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, "variant": "mu/112"})

    # Keep real attention, vision, flow matching, and RLT; reduce only capacity.
    monkeypatch.setattr(siglip, "SigLIPViT", TinySigLIP)
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    precision = torch.get_float32_matmul_precision()
    config = Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=2,
        pi05=True,
    )
    torch.manual_seed(0)
    save_file(Pi0(config).state_dict(), tmp_path / "model.safetensors")

    def build(
        *,
        action_loss_mask=None,
        use_rlt=False,
        action_chunk=3,
        env_dim=4,
        umi_v4_image_augmentation=False,
        compute_dtype=None,
        precision="fp32",
    ):
        cfg = OmegaConf.create(
            {
                "model_path": str(tmp_path),
                "precision": precision,
                "pi05": True,
                "action_dim": env_dim,
                "num_action_chunks": action_chunk,
                "num_steps": 10,
                "openpi": {
                    "task": "sft",
                    "model_action_dim": 4,
                    "action_horizon": 3,
                    "max_token_len": 2,
                    "paligemma_variant": "dummy",
                    "action_expert_variant": "dummy",
                    "action_loss_mask": action_loss_mask,
                    "umi_v4_image_augmentation": umi_v4_image_augmentation,
                    "compute_dtype": compute_dtype,
                    "use_rlt": use_rlt,
                    "rlt_input_dim": 64,
                    "rlt_embed_dim": 16,
                    "rlt_prefix_seq_len": 16,
                    "rlt_num_layers": 1,
                    "rlt_num_heads": 2,
                    "rlt_image_only": False,
                    "rlt_use_mask": True,
                    "rlt_alpha": 0.5,
                },
            }
        )
        return build_openpi(cfg), (config.fake_obs(), config.fake_act())

    yield build
    torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("compute_dtype", [None, "bf16"])
@pytest.mark.parametrize("use_rlt", [False, True])
def test_openpi_compute_precision_preserves_fp32_training_state(
    tiny_openpi_checkpoint, compute_dtype, use_rlt
):
    model, batch = tiny_openpi_checkpoint(compute_dtype=compute_dtype, use_rlt=use_rlt)
    expected_backbone_dtype = (
        torch.bfloat16 if compute_dtype == "bf16" else torch.float32
    )
    observed = {}

    def record(name):
        def hook(module, inputs, output):
            del module
            tensor = output[0] if isinstance(output, tuple) else output
            observed[name] = (inputs[0].dtype, tensor.dtype)

        return hook

    targets = {
        "stem": model.img.stem,
        "vision_attention": model.img.encoder.layers[0].attn,
        "vision_projector": model.img.head,
        "language_query": model.llm.layers[0].attn.q_proj[0],
        "language_mlp": model.llm.layers[0].mlps[0],
        "action_input": model.action_in_proj,
        "time_input": model.time_mlp_in,
        "action_output": model.action_out_proj,
    }
    if use_rlt:
        targets["rlt_attention"] = model.rlt_module.encoder.layers[0].self_attn
    handles = [
        module.register_forward_hook(record(name)) for name, module in targets.items()
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    try:
        output = model(data=batch)
        loss = output["loss"] if use_rlt else output
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    finally:
        for handle in handles:
            handle.remove()

    for name in ("stem", "action_input", "time_input", "action_output"):
        assert observed[name] == (torch.float32, torch.float32)
    for name in (
        "vision_attention",
        "vision_projector",
        "language_query",
        "language_mlp",
    ):
        assert observed[name] == (expected_backbone_dtype, expected_backbone_dtype)
    if use_rlt:
        assert observed["rlt_attention"][1] == expected_backbone_dtype
        assert output["rlt_loss"].dtype == output["vla_loss"].dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(
        gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    states = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    ]
    assert states
    assert all(
        value.dtype == torch.float32 and torch.isfinite(value).all() for value in states
    )


def test_openpi_compute_precision_rejects_reduced_parameter_storage(
    tiny_openpi_checkpoint,
):
    with pytest.raises(ValueError, match="compute_dtype requires precision=fp32"):
        tiny_openpi_checkpoint(compute_dtype="bf16", precision="bf16")


def _openpi_full_shard_actor_config(**overrides):
    cfg = OmegaConf.create(
        {
            "training_backend": "fsdp",
            "model": {
                "model_type": "openpi_rlinf",
                "precision": "fp32",
                "is_lora": False,
                "openpi": {"task": "sft", "compute_dtype": "bf16"},
            },
            "fsdp_config": {
                "strategy": "fsdp",
                "sharding_strategy": "full_shard",
                "mixed_precision": {
                    "param_dtype": "fp32",
                    "reduce_dtype": "fp32",
                    "buffer_dtype": "fp32",
                },
            },
        }
    )
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


@pytest.mark.parametrize("compute_dtype", [None, "fp32", "bf16"])
def test_openpi_native_sft_fp32_master_allows_full_shard(compute_dtype):
    cfg = _openpi_full_shard_actor_config(
        **{"model.openpi.compute_dtype": compute_dtype}
    )
    result = validate_fsdp_cfg(cfg)
    assert result.fsdp_config.sharding_strategy == "full_shard"
    assert result.model.precision == "fp32"


@pytest.mark.parametrize(
    "overrides",
    [
        {"model.openpi.task": "rl"},
        {"model.openpi.task": "dsrl"},
        {"model.openpi.task": "eval"},
        {"model.openpi.task": "dagger"},
        {"model.precision": "bf16"},
        {"model.precision": "fp16"},
        {"model.is_lora": True},
        {"training_backend": "fsdp2"},
        {"fsdp_config.strategy": "fsdp2"},
        {"fsdp_config.sharding_strategy": "shard_grad_op"},
        {"fsdp_config.sharding_strategy": "hybrid_shard"},
    ],
)
def test_openpi_full_shard_rejects_unvalidated_paths(overrides):
    with pytest.raises(AssertionError, match="full_shard requires"):
        validate_fsdp_cfg(_openpi_full_shard_actor_config(**overrides))


def test_openpi_full_shard_keeps_fsdp_mixed_precision_guard():
    cfg = _openpi_full_shard_actor_config(
        **{"fsdp_config.mixed_precision.param_dtype": "bf16"}
    )
    with pytest.raises(AssertionError, match="does not support FSDP mixed precision"):
        validate_fsdp_cfg(cfg)


@pytest.mark.parametrize("task", ["sft", "rl", "dsrl", "eval", "dagger"])
def test_openpi_existing_no_shard_paths_remain_accepted(task):
    cfg = _openpi_full_shard_actor_config(
        **{
            "fsdp_config.sharding_strategy": "no_shard",
            "model.openpi.task": task,
        }
    )
    assert validate_fsdp_cfg(cfg).fsdp_config.sharding_strategy == "no_shard"


def _activate_tiny_openpi_trunk(model):
    # Tiny random checkpoints start with zero vision output and expert gates.
    # Nonzero values exercise gradients through both experts and the image stem.
    with torch.no_grad():
        model.img.head.weight.normal_(std=0.02)
        for name, parameter in model.named_parameters():
            if name.endswith("ada_modulation.bias"):
                parameter.add_(0.1)


@pytest.mark.parametrize("compute_dtype", [None, "bf16"])
@pytest.mark.parametrize("use_rlt", [False, True])
def test_openpi_gradient_checkpointing_preserves_sft_training(
    tiny_openpi_checkpoint, compute_dtype, use_rlt
):
    baseline, batch = tiny_openpi_checkpoint(
        compute_dtype=compute_dtype,
        use_rlt=use_rlt,
        umi_v4_image_augmentation=True,
    )
    _activate_tiny_openpi_trunk(baseline)
    checkpointed = copy.deepcopy(baseline)
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    results = []
    for model in (baseline, checkpointed):
        calls = {"vision": 0, "language": 0}

        def count(name):
            def hook(module, inputs):
                del module, inputs
                calls[name] += 1

            return hook

        handles = [
            model.img.encoder.layers[0].register_forward_pre_hook(count("vision")),
            model.llm.layers[0].register_forward_pre_hook(count("language")),
        ]
        saved_elements = 0

        def pack(tensor):
            nonlocal saved_elements
            saved_elements += tensor.numel()
            return tensor

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.manual_seed(37)
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                output = model(data=batch)
                forward_calls = calls.copy()
                loss = output["loss"] if use_rlt else output
                loss.backward()
            assert model.img.stem.weight.grad.abs().sum() > 0
            gradients = {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in model.named_parameters()
            }
            optimizer.step()
            results.append(
                {
                    "output": output,
                    "gradients": gradients,
                    "parameters": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(),
                    "saved_elements": saved_elements,
                }
            )
        finally:
            for handle in handles:
                handle.remove()
        if model is checkpointed:
            assert all(calls[name] > forward_calls[name] for name in calls)
        else:
            assert calls == forward_calls

    assert results[1]["saved_elements"] < results[0]["saved_elements"]
    for key in ("output", "gradients", "parameters", "optimizer", "rng"):
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("compute_dtype", [None, "bf16"])
def test_openpi_gradient_checkpointing_preserves_cached_suffix(
    tiny_openpi_checkpoint, compute_dtype
):
    baseline, (observation, actions) = tiny_openpi_checkpoint(
        compute_dtype=compute_dtype
    )
    _activate_tiny_openpi_trunk(baseline)
    checkpointed = copy.deepcopy(baseline)
    checkpointed.gradient_checkpointing_enable()
    results = []
    for model in (baseline, checkpointed):
        prefix, mask, cache = model.build_prefix_cache(observation)
        assert len(cache) == len(model.llm.layers)
        assert prefix.shape[:2] == mask.shape
        suffix = model.run_suffix(
            observation, actions, torch.full((1,), 0.5), cache, mask
        )
        velocity = model.velocity_from_suffix(suffix)
        velocity.square().mean().backward()
        results.append(
            (
                prefix,
                cache,
                velocity,
                {name: parameter.grad for name, parameter in model.named_parameters()},
            )
        )
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize("compute_dtype", [None, "bf16"])
def test_openpi_checkpointed_rlt_loss_detaches_vla(
    tiny_openpi_checkpoint, compute_dtype
):
    model, batch = tiny_openpi_checkpoint(compute_dtype=compute_dtype, use_rlt=True)
    model.gradient_checkpointing_enable()
    output = model(data=batch)
    output["rlt_loss"].backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("rlt_module.")
    )
    for module in (model.rlt_module.encoder, model.rlt_module.decoder):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_openpi_rejects_reentrant_gradient_checkpointing(tiny_openpi_checkpoint):
    model, _ = tiny_openpi_checkpoint()
    with pytest.raises(ValueError, match="requires non-reentrant"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )


def test_umi_v4_color_jitter_matches_source_augmax():
    augmax = pytest.importorskip("augmax")
    jax = pytest.importorskip("jax")
    from rlinf.models.embodiment.openpi_rlinf.modules.model import (
        apply_umi_v4_color_jitter,
    )

    jitter = augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
    images = np.random.default_rng(41).random((16, 2, 4, 3), dtype=np.float32)
    images[:, 0, 0] = 0
    images[:, 0, 1] = 1
    images[:, 0, 2] = 0.5
    keys = jax.random.split(jax.random.key(7), len(images))
    parameters = []
    reference = []
    for key, image in zip(keys, images):
        subkeys = jax.random.split(key, jitter.keys_needed)
        parameters.append(
            [
                float(jax.random.uniform(subkeys[0], minval=-0.3, maxval=0.3)),
                float(jax.random.uniform(subkeys[1], minval=-0.4, maxval=0.4)),
                float(jax.random.uniform(subkeys[2], minval=-0.1, maxval=0.1)),
                bool(jax.random.bernoulli(key, 0.5)),
            ]
        )
        reference.append(np.asarray(jitter(key, image)))
    parameters = torch.tensor(parameters)
    actual = apply_umi_v4_color_jitter(
        torch.from_numpy(images),
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 2],
        parameters[:, 3].bool(),
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(np.stack(reference)), atol=2e-6, rtol=2e-6
    )


def test_umi_v4_augmentation_uses_seeded_rng_and_preserves_evaluation():
    from rlinf.models.embodiment.openpi_rlinf.modules.model import (
        Observation,
        preprocess_observation,
    )

    image = torch.tensor([0.2, 0.4, 0.8]).expand(256, 8, 8, 3).clone() * 2 - 1
    observation = Observation(
        images=dict.fromkeys(
            ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), image
        ),
        image_masks={
            "base_0_rgb": torch.zeros(256, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(256, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(256, dtype=torch.bool),
        },
        state=torch.zeros(256, 32),
    )
    kwargs = {"image_resolution": (8, 8), "umi_v4_image_augmentation": True}
    torch.manual_seed(19)
    first = preprocess_observation(observation, train=True, **kwargs)
    torch.manual_seed(19)
    repeated = preprocess_observation(observation, train=True, **kwargs)
    torch.testing.assert_close(
        first.images["left_wrist_0_rgb"],
        repeated.images["left_wrist_0_rgb"],
        rtol=0,
        atol=0,
    )
    changed = (first.images["left_wrist_0_rgb"] - image).abs().amax(
        dim=(1, 2, 3)
    ) > 1e-6
    assert 90 < changed.sum() < 170
    assert first.images["left_wrist_0_rgb"].unique(dim=0).shape[0] > 90
    for key in ("base_0_rgb", "right_wrist_0_rgb"):
        torch.testing.assert_close(first.images[key], image, rtol=0, atol=0)
    evaluated = preprocess_observation(observation, train=False, **kwargs)
    for key in observation.images:
        torch.testing.assert_close(evaluated.images[key], image, rtol=0, atol=0)


@pytest.mark.parametrize("use_rlt", [False, True])
def test_openpi_sft_uses_umi_v4_augmentation(tiny_openpi_checkpoint, use_rlt):
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.openpi_rlinf.modules.model import (
        preprocess_observation,
    )

    model, (observation, actions) = tiny_openpi_checkpoint(
        use_rlt=use_rlt, umi_v4_image_augmentation=True
    )
    image = torch.linspace(-1, 1, 224 * 224 * 3).reshape(1, 224, 224, 3)
    observation.images["left_wrist_0_rgb"] = image
    torch.manual_seed(2)
    expected = preprocess_observation(
        observation, train=True, umi_v4_image_augmentation=True
    ).images["left_wrist_0_rgb"]
    assert not torch.allclose(expected, image)
    captured = []
    handle = model.img.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().clone())
    )
    try:
        torch.manual_seed(2)
        model(ForwardType.SFT, data=(observation, actions))
    finally:
        handle.remove()
    torch.testing.assert_close(captured[1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_rlt", [False, True])
def test_openpi_sft_action_mask_excludes_padded_output_gradients(
    tiny_openpi_checkpoint, use_rlt
):
    from rlinf.models.embodiment.base_policy import ForwardType

    model, batch = tiny_openpi_checkpoint(
        action_loss_mask=[1.0, 2.0, 0.0, 0.0], use_rlt=use_rlt
    )
    torch.manual_seed(42)
    per_element_loss = model.compute_loss(*batch, train=True)
    expected = (per_element_loss[..., 0] + 2 * per_element_loss[..., 1]).mean() / 3

    torch.manual_seed(42)
    output = model(forward_type=ForwardType.SFT, data=batch)
    vla_loss = output["vla_loss"] if use_rlt else output
    loss = output["loss"] if use_rlt else output
    torch.testing.assert_close(vla_loss, expected)
    if use_rlt:
        torch.testing.assert_close(loss, output["rlt_loss"] + 0.5 * vla_loss)
    loss.backward()
    projection_grad = model.action_out_proj.weight.grad.clone()
    assert torch.count_nonzero(projection_grad[:2]) > 0
    assert torch.count_nonzero(projection_grad[2:]) == 0
    assert torch.count_nonzero(model.action_out_proj.bias.grad[2:]) == 0
    if use_rlt:
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in model.rlt_module.encoder.parameters()
        )

    # Changes only to excluded outputs must preserve both loss and valid gradients.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.action_out_proj.bias[2:] += 100.0
    torch.manual_seed(42)
    changed_output = model(forward_type=ForwardType.SFT, data=batch)
    changed_loss = changed_output["loss"] if use_rlt else changed_output
    torch.testing.assert_close(changed_loss, loss)
    changed_loss.backward()
    torch.testing.assert_close(model.action_out_proj.weight.grad, projection_grad)


@pytest.mark.parametrize("use_action_chunk_loss", [False, True])
def test_openpi_sft_without_action_mask_preserves_loss(
    tiny_openpi_checkpoint, use_action_chunk_loss
):
    from rlinf.models.embodiment.base_policy import ForwardType

    model, batch = tiny_openpi_checkpoint(action_chunk=2, env_dim=2)
    torch.manual_seed(42)
    expected = model.compute_loss(*batch, train=True)
    if use_action_chunk_loss:
        expected = expected[:, :2, :2]
    torch.manual_seed(42)
    loss = model(
        forward_type=ForwardType.SFT,
        data=batch,
        use_action_chunk_loss=use_action_chunk_loss,
    )
    torch.testing.assert_close(loss, expected.mean())


def test_openpi_sft_action_mask_respects_dagger_chunk(tiny_openpi_checkpoint):
    from rlinf.models.embodiment.base_policy import ForwardType

    model, batch = tiny_openpi_checkpoint(
        action_loss_mask=[1.0, 0.0, 2.0, 0.0], action_chunk=2, env_dim=2
    )
    torch.manual_seed(42)
    expected = model.compute_loss(*batch, train=True)[:, :2, 0].mean()
    torch.manual_seed(42)
    loss = model(forward_type=ForwardType.SFT, data=batch, use_action_chunk_loss=True)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(model.action_out_proj.weight.grad[0]) > 0
    assert torch.count_nonzero(model.action_out_proj.weight.grad[1:]) == 0


def test_openpi_sft_action_mask_rejects_empty_dagger_overlap(tiny_openpi_checkpoint):
    from rlinf.models.embodiment.base_policy import ForwardType

    model, batch = tiny_openpi_checkpoint(
        action_loss_mask=[0.0, 0.0, 1.0, 0.0], env_dim=2
    )
    with pytest.raises(ValueError, match="no positive weights"):
        model(forward_type=ForwardType.SFT, data=batch, use_action_chunk_loss=True)


@pytest.mark.parametrize(
    "mask",
    [
        (1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (1.0, -1.0, 0.0, 0.0),
        (1.0, float("nan"), 0.0, 0.0),
        (1.0, float("inf"), 0.0, 0.0),
    ],
)
def test_openpi_config_rejects_invalid_action_loss_mask(mask):
    from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config

    with pytest.raises(ValueError, match="action_loss_mask"):
        Pi0Config(action_dim=4, action_loss_mask=mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_siglip_stem_preserves_float32_convolution_and_gradients(dtype, autocast):
    from rlinf.models.embodiment.openpi_rlinf.modules.siglip import SigLIPViT

    torch.manual_seed(42)
    stem = SigLIPViT(variant="mu/14").get_submodule("stem").to(dtype=dtype)
    image = torch.randn(2, 3, 28, 42, dtype=dtype, requires_grad=True)
    reference_image = image.detach().clone().requires_grad_()
    reference_weight = stem.weight.detach().clone().requires_grad_()
    reference_bias = stem.bias.detach().clone().requires_grad_()
    expected = torch.nn.functional.conv2d(
        reference_image.float(),
        reference_weight.float(),
        reference_bias.float(),
        stride=stem.stride,
        padding=stem.padding,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = stem(image)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.randn_like(expected)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    for actual_grad, expected_grad in (
        (image.grad, reference_image.grad),
        (stem.weight.grad, reference_weight.grad),
        (stem.bias.grad, reference_bias.grad),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def _run_fsdp_siglip_replica_check(rank, init_path):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardingStrategy,
    )

    from rlinf.models.embodiment.openpi_rlinf.modules.siglip import SigLIPViT

    class VisionPolicy(torch.nn.Module):
        _no_split_modules = ["SigLIPViT", "Encoder", "Encoder1DBlock"]
        _no_split_names = ["stem", "head"]

        def __init__(self):
            super().__init__()
            self.img = SigLIPViT(variant="mu/112", num_classes=16)
            torch.nn.init.normal_(self.img.head.weight, std=0.02)
            for name, module in self.named_modules():
                module._fsdp_wrap_name = name.rsplit(".", 1)[-1]

        def forward(self, images):
            # Pi0 reuses the same encoder for its three camera slots.
            return (
                sum(
                    self.img(images + camera * 0.1)[0].square().mean()
                    for camera in range(3)
                )
                / 3
            )

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        torch.manual_seed(42)
        policy = VisionPolicy().cuda(rank)
        initial_stem = policy.img.stem.weight.detach().clone()
        wrap_policy = get_fsdp_wrap_policy(
            module=policy,
            config=OmegaConf.create({"use_orig_params": False}),
            is_lora=False,
            model_type="openpi_rlinf",
        )
        wrapped = FSDP(
            policy,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=rank,
            sync_module_states=True,
            use_orig_params=False,
        )
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3, fused=True)
        torch.manual_seed(100 + rank)
        for _ in range(2):
            images = torch.randn(1, 224, 224, 3, device=f"cuda:{rank}")
            optimizer.zero_grad(set_to_none=True)
            wrapped(images).backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), max_norm=1.0)
            optimizer.step()

        state = wrapped.state_dict()
        assert not torch.equal(state["img.stem.weight"], initial_stem)
        for name in ("img.stem.weight", "img.stem.bias", "img.head.weight"):
            value = state[name].contiguous()
            replicas = [torch.empty_like(value) for _ in range(2)]
            torch.distributed.all_gather(replicas, value)
            torch.testing.assert_close(
                replicas[0], replicas[1], rtol=0, atol=0, msg=f"Unsynced {name}"
            )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs.")
def test_siglip_fsdp_no_shard_keeps_replicas_equal(tmp_path):
    torch.multiprocessing.spawn(
        _run_fsdp_siglip_replica_check,
        args=(str(tmp_path / "process_group"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires one CUDA GPU.")
@pytest.mark.parametrize("fsdp_version", ["fsdp", "fsdp2"])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_fsdp_checkpoint_restore_releases_old_optimizer_buffers(
    tmp_path, fsdp_version, cpu_offload
):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    from rlinf.hybrid_engines.fsdp import FSDP, ShardingStrategy, fully_shard
    from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
    from rlinf.hybrid_engines.fsdp.utils import FSDPVersion, to_local_if_dtensor

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'process_group'}",
        rank=0,
        world_size=1,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        torch.manual_seed(12)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
        ).cuda()
        if fsdp_version == "fsdp":
            model = FSDP(
                model, sharding_strategy=ShardingStrategy.NO_SHARD, device_id=0
            )
        else:
            fully_shard(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, fused=fsdp_version == "fsdp"
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        inputs = torch.randn(4, 16, device="cuda")

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            optimizer.step()
            scheduler.step()

        def snapshot():
            return (
                {
                    name: to_local_if_dtensor(value).detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                [
                    {
                        name: to_local_if_dtensor(value).detach().cpu().clone()
                        for name, value in state.items()
                    }
                    for state in optimizer.state.values()
                ],
                scheduler.state_dict(),
            )

        for _ in range(3):
            train_step()
        checkpoint = Checkpoint(
            model,
            optimizer,
            scheduler,
            StateDictOptions(cpu_offload=cpu_offload, full_state_dict=False),
            FSDPVersion(fsdp_version),
        )
        dcp.save({"actor": checkpoint}, checkpoint_id=tmp_path / "checkpoint")
        saved = snapshot()
        train_step()
        uninterrupted = snapshot()
        train_step()

        old_state_devices = []
        unexpected_steps = []

        def inspect_old_state(opt, state_dict):
            assert opt.state
            old_state_devices.extend(
                state["exp_avg"].device.type for state in opt.state.values()
            )

        load_hook = optimizer.register_load_state_dict_pre_hook(inspect_old_state)
        step_hook = optimizer.register_step_pre_hook(
            lambda *args: unexpected_steps.append(True)
        )
        try:
            dcp.load({"actor": checkpoint}, checkpoint_id=tmp_path / "checkpoint")
        finally:
            load_hook.remove()
            step_hook.remove()
        assert old_state_devices
        assert set(old_state_devices) == {"cpu" if cpu_offload else "cuda"}
        assert not unexpected_steps
        torch.testing.assert_close(snapshot(), saved, rtol=0, atol=0)
        train_step()
        torch.testing.assert_close(snapshot(), uninterrupted, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


class _FakeValueExpert:
    def __init__(self, image_emb, lang_emb):
        self.image_emb = image_emb
        self.lang_emb = lang_emb

    def embed_image(self, image):
        return self.image_emb.to(device=image.device)

    def embed_language_tokens(self, tokens):
        return self.lang_emb.to(device=tokens.device)


def _load_value_critic_model(monkeypatch):
    value_model_dir = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/value_model/recap"
    )
    package_name = "value_model_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(value_model_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_name = f"{package_name}.modeling_critic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        value_model_dir / "modeling_critic.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ValueCriticModel


def test_value_model_does_not_rescale_gemma3_language_embeddings(monkeypatch):
    """Gemma3 embed_tokens already applies sqrt(hidden_size) internally."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.Gemma3ForCausalLM")

    ValueCriticModel = _load_value_critic_model(monkeypatch)

    hidden_size = 4
    image_emb = torch.zeros(1, 2, hidden_size)
    lang_emb = torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden_size)

    model = SimpleNamespace(
        gradient_checkpointing_enabled=False,
        training=False,
        value_expert=_FakeValueExpert(image_emb=image_emb, lang_emb=lang_emb),
        _apply_checkpoint=lambda func, *args: func(*args),
    )

    prefix_embs, prefix_pad_masks = ValueCriticModel.embed_prefix(
        model,
        images=[torch.empty(1, 3, 8, 8)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix_embs[:, 2:], lang_emb)
    torch.testing.assert_close(
        prefix_pad_masks,
        torch.tensor([[True, True, True, True, False]]),
    )


_STARVLA_UTILS_DIR = (
    Path(__file__).resolve().parents[2] / "rlinf/models/embodiment/starvla/utils"
)
_FRANKA_ACTION_STATS = {
    "q01": [-0.5] * 7,
    "q99": [0.5] * 7,
    "min": [-1.0] * 7,
    "max": [1.0] * 7,
    "mask": [True] * 6 + [False],
}


def _load_starvla_util(name: str) -> ModuleType:
    # The starvla package __init__ imports starVLA, which only its venv has.
    spec = importlib.util.spec_from_file_location(
        f"starvla_{name}_under_test", _STARVLA_UTILS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("source", "bound"), [("q01q99", 0.5), ("minmax", 1.0)])
def test_starvla_action_stats_follow_the_configured_source(source, bound):
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    stats = action_space.resolve_action_norm_stats(
        model, "franka", action_dim=7, action_stats_source=source
    )

    np.testing.assert_array_equal(stats["q99"], [bound] * 7)
    np.testing.assert_array_equal(stats["q01"], [-bound] * 7)
    np.testing.assert_array_equal(stats["mask"], [True] * 6 + [False])


def test_starvla_action_stats_name_the_available_keys_for_an_unknown_key():
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    with pytest.raises(RuntimeError, match=r"available keys: \['franka'\]"):
        action_space.resolve_action_norm_stats(model, "libero_spatial", action_dim=7)


def test_starvla_action_stats_require_a_norm_stats_mapping():
    action_space = _load_starvla_util("action_space")

    with pytest.raises(RuntimeError, match="no usable 'norm_stats' mapping"):
        action_space.resolve_action_norm_stats(
            SimpleNamespace(norm_stats=None), "franka", action_dim=7
        )


def test_starvla_env_actions_keep_their_shape_and_map_the_libero_gripper(monkeypatch):
    action_space = _load_starvla_util("action_space")
    received_shapes = []

    def unnormalize_actions(actions, action_norm_stats):
        received_shapes.append(actions.shape)
        return actions

    tools = ModuleType("starVLA.model.tools")
    tools.FrameworkTools = SimpleNamespace(unnormalize_actions=unnormalize_actions)
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    normalized = np.zeros((2, 3, 7), dtype=np.float32)
    normalized[..., 0] = 0.25
    normalized[0, :, 6] = 1.0
    stats = {"q99": np.ones(7), "q01": -np.ones(7), "mask": np.ones(7, dtype=bool)}

    env_actions = action_space.unnormalize_actions_for_env(
        normalized, stats, policy_setup="libero"
    )

    # starVLA unnormalizes [T, action_dim]; the chunk layout comes back intact.
    assert received_shapes == [(6, 7)]
    assert env_actions.shape == (2, 3, 7)
    np.testing.assert_array_equal(env_actions[..., 0], 0.25)
    # LIBERO wants the 0/1 gripper as -1 (open) / +1 (closed).
    np.testing.assert_array_equal(env_actions[0, :, 6], -1.0)
    np.testing.assert_array_equal(env_actions[1, :, 6], 1.0)


def test_starvla_autocast_targets_the_worker_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    # CPU stands in for a non-CUDA accelerator such as an Ascend NPU.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_starvla_autocast_is_a_noop_without_an_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    monkeypatch.setattr(Worker, "torch_device_type", None)

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")


def test_starvla_gaussian_is_float32_and_keeps_the_gradient_path():
    accelerator = _load_starvla_util("accelerator")
    mean = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    log_std = torch.nn.Parameter(torch.zeros(3))

    dist = accelerator.build_gaussian(mean, log_std.exp())
    sample = dist.rsample()

    assert dist.loc.dtype == dist.scale.dtype == sample.dtype == torch.float32
    dist.log_prob(sample.detach()).sum().backward()
    assert mean.grad is not None and mean.grad.dtype == torch.bfloat16
    assert log_std.grad is not None


class _QwenVisionPatchEmbed(torch.nn.Module):
    """Shape contract of Qwen2.5-VL PatchEmbed: Conv3d kernel == stride."""

    def __init__(self, in_channels=3, temporal=2, patch=4, embed_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal
        self.patch_size = patch
        kernel = (temporal, patch, patch)
        self.proj = torch.nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))
        return hidden_states.view(-1, self.proj.out_channels)


def test_qwen_vl_linear_patch_embed_matches_conv3d_and_backprops():
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
    )

    torch.manual_seed(0)
    module = _QwenVisionPatchEmbed()
    patches = torch.randn(5, 3 * 2 * 4 * 4, requires_grad=True)

    conv_out = module(patches)
    linear_out = _linear_patch_embed_forward(module, patches)
    torch.testing.assert_close(linear_out, conv_out, rtol=1e-5, atol=1e-5)

    linear_out.sum().backward()
    assert module.proj.weight.grad is not None
    assert patches.grad is not None


def test_qwen_vl_linear_patch_embed_is_rebound_on_npu(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())
    original_forward = model[0].forward

    assert patch_vision_patch_embed(model) == 1
    assert model[0].forward.__func__ is _linear_patch_embed_forward
    assert original_forward.__func__ is not _linear_patch_embed_forward


def test_qwen_vl_linear_patch_embed_is_left_alone_on_nvidia(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())

    assert patch_vision_patch_embed(model) == 0
    assert model[0].forward.__func__ is _QwenVisionPatchEmbed.forward


_WAN_NPU_PATCHES = "rlinf.envs.sim.world_model.backend.npu_patches"


@pytest.fixture
def wan_dit(monkeypatch):
    """diffsynth's Wan DiT module, whose operators the NPU patches rebind."""
    from rlinf.utils.patcher import Patcher

    def flash_attention(q, k, v, num_heads, compatibility_mode=False):
        return q

    def rope_apply(x, freqs, num_heads):
        return x

    class RMSNorm(torch.nn.Module):
        def forward(self, x):
            return x

    dit = ModuleType("diffsynth.models.wan_video_dit")
    dit.flash_attention, dit.rope_apply, dit.RMSNorm = (
        flash_attention,
        rope_apply,
        RMSNorm,
    )
    for name in ("diffsynth", "diffsynth.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, dit.__name__, dit)
    yield dit
    Patcher.clear()


def _import_wan_npu_patches(monkeypatch, *, mindiesd: bool) -> ModuleType:
    """Import the Wan NPU patches on an Ascend stack with or without MindIE-SD."""
    vendor = {"torch_npu": ModuleType("torch_npu"), "mindiesd": None}
    if mindiesd:
        names = (
            "mindiesd",
            "mindiesd.layers",
            "mindiesd.layers.flash_attn",
            "mindiesd.layers.flash_attn.attention_forward",
        )
        vendor.update({name: ModuleType(name) for name in names})
        vendor["mindiesd"].rotary_position_embedding = lambda *args, **kwargs: None
        vendor[names[-1]].attention_forward = lambda *args, **kwargs: None
    for name, module in vendor.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.find_spec(_WAN_NPU_PATCHES)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _WAN_NPU_PATCHES, module)
    spec.loader.exec_module(module)
    return module


def _patch_like_wan_backend(npu_patches: ModuleType) -> None:
    """Run the patch sequence ``WanBackend._build_pipeline`` runs."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    npu_patches.apply_npu_patches(Patcher)
    Patcher.apply()


def _wan_operators(dit: ModuleType) -> tuple:
    return dit.flash_attention, dit.rope_apply, dit.RMSNorm.forward


def test_wan_npu_patches_leave_diffsynth_alone_on_nvidia(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators


def test_wan_npu_patches_log_why_mindiesd_is_unavailable(wan_dit, monkeypatch, caplog):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=False)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators
    assert "import of mindiesd halted" in caplog.text


def test_wan_npu_patches_rebind_the_dit_operators_on_every_build(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    kernels = (
        npu_patches.npu_flash_attention,
        npu_patches.npu_rope_apply,
        npu_patches.npu_rmsnorm_forward,
    )

    # Every WanBackend in the process repeats the sequence on the patched module.
    for _ in range(2):
        _patch_like_wan_backend(npu_patches)
        assert _wan_operators(wan_dit) == kernels


def _history_cfg():
    return OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "main": {
                        "history_size": 2,
                        "min_history_size": 1,
                        "input_interval": 3,
                        "history_keys": ["main_images"],
                        "input_on_done": True,
                    }
                }
            }
        }
    )


def _append_step(manager: HistoryManager, value: int) -> None:
    manager.append_to_history_entries(
        {"main_images": torch.tensor([[value], [value + 10]])}
    )


def test_build_history_input_skips_between_interval_ticks():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_input == {}
    assert history_length == {}
    assert manager.history_counts == [2, 2]


def test_build_history_input_emits_on_interval_tick():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)
    _append_step(manager, 3)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_length == {"main": [2, 2]}
    assert history_input["main"]["main_images"][0] == [
        torch.tensor([2]),
        torch.tensor([3]),
    ]
    assert history_input["main"]["main_images"][1] == [
        torch.tensor([12]),
        torch.tensor([13]),
    ]


def _success_potential_state_machine():
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        ShapedVLMRewardModel,
    )

    model = ShapedVLMRewardModel.__new__(ShapedVLMRewardModel)
    model.potential_gamma = 1.0
    model.potential_scale = 1.0
    model.potential_ema_alpha = 0.5
    model.potential_clip = 0.0
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model.gt_success_bonus = 0.0
    model.infer_micro_batch_size = 0
    model._previous_potentials = None
    model._success_fired = None
    model._success_streak = None
    return model


def test_empty_history_input_still_resets_shaping_state_on_done():
    model = _success_potential_state_machine()
    model._previous_potentials = torch.tensor([0.4, 0.8])
    model._success_fired = torch.tensor([True, True])
    model._success_streak = torch.tensor([3, 1], dtype=torch.int32)

    rewards = model.compute_reward(
        {
            "history_input": {},
            "dones": torch.tensor([True, False]),
        }
    )

    assert rewards.tolist() == pytest.approx([0.0, 0.0])
    assert torch.isnan(model._previous_potentials[0])
    assert float(model._previous_potentials[1]) == pytest.approx(0.8)
    assert model._success_fired.tolist() == [False, True]
    assert model._success_streak.tolist() == [0, 1]


VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def _critic_metrics(values, prev_values, returns, loss_mask=None):
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=loss_mask,
    )
    return metrics


def test_value_clip_ratio_is_zero_when_no_update_is_clipped():
    prev_values = torch.zeros(4, 8)
    values = torch.full((4, 8), VALUE_CLIP / 2)
    returns = torch.zeros(4, 8)

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_clip_ratio_reports_the_fraction_of_clipped_updates():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    # Half of the entries move outside the trust region, half stay inside.
    values = torch.full((4, 8), VALUE_CLIP / 2)
    values[:, :4] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_grows_with_the_size_of_the_value_update():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)

    ratios = [
        float(
            _critic_metrics(torch.full((4, 8), scale), prev_values, returns)[
                "critic/value_clip_ratio"
            ]
        )
        for scale in (0.5 * VALUE_CLIP, 2 * VALUE_CLIP)
    ]

    assert ratios == [pytest.approx(0.0), pytest.approx(1.0)]


def test_value_clip_ratio_ignores_masked_out_entries():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :2] = True

    # Every valid entry is clipped; every padded entry is not.
    values = torch.zeros(4, 8)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(1.0)


def test_value_clip_ratio_broadcasts_a_narrower_loss_mask():
    prev_values = torch.zeros(4, 8, 3)
    returns = torch.zeros(4, 8, 3)
    loss_mask = torch.zeros(4, 8, 1, dtype=torch.bool)
    loss_mask[:, :4] = True

    values = torch.zeros(4, 8, 3)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    # 2 of the 4 unmasked steps are clipped.
    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_is_zero_when_every_entry_is_masked_out():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    values = torch.full((4, 8), 10 * VALUE_CLIP)

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_loss_is_unchanged_by_the_metric_computation():
    torch.manual_seed(0)
    prev_values = torch.randn(4, 8)
    values = torch.randn(4, 8, requires_grad=True)
    returns = torch.randn(4, 8)

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=None,
    )

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -VALUE_CLIP, VALUE_CLIP
    )
    expected = torch.max(
        torch.nn.functional.huber_loss(
            values, returns, delta=HUBER_DELTA, reduction="none"
        ),
        torch.nn.functional.huber_loss(
            value_pred_clipped, returns, delta=HUBER_DELTA, reduction="none"
        ),
    ).mean()

    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert loss.requires_grad
    assert not metrics["critic/value_clip_ratio"].requires_grad


def test_create_builds_expected_sampler_types():
    constant = DelaySampler.create(
        OmegaConf.create({"type": "constant", "delay": 0.12})
    )
    uniform = DelaySampler.create(
        OmegaConf.create({"type": "uniform", "min_delay": 0.03, "max_delay": 0.08})
    )
    exponential = DelaySampler.create(
        OmegaConf.create({"type": "exponential", "rate": 0.5})
    )
    gaussian = DelaySampler.create(
        OmegaConf.create({"type": "gaussian", "mean": 0.20, "stddev": 0.03})
    )

    assert isinstance(constant, ConstantDelaySampler)
    assert isinstance(uniform, UniformDelaySampler)
    assert isinstance(exponential, ExponentialDelaySampler)
    assert isinstance(gaussian, GaussianDelaySampler)


def test_create_accepts_none():
    assert DelaySampler.create(None) is None


def test_same_seed_produces_same_sequence_per_sampler():
    first = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)
    second = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)

    assert first.sample(8) == second.sample(8)


def test_constant_sampler_uses_seconds_helpers():
    sampler = ConstantDelaySampler(delay=0.25)

    assert sampler.sample(3) == [0.25, 0.25, 0.25]
    assert sampler.sample_one() == 0.25


def test_gaussian_sampler_never_returns_negative_seconds():
    sampler = GaussianDelaySampler(mean=0, stddev=0.1, seed=0)

    assert all(delay >= 0 for delay in sampler.sample(100))


def test_invalid_ranges_raise_clear_errors():
    with pytest.raises(ValueError, match="min_delay must be <="):
        UniformDelaySampler(min_delay=0.2, max_delay=0.1)

    with pytest.raises(ValueError, match="rate must be > 0"):
        ExponentialDelaySampler(rate=0)


def test_num_samples_must_be_non_negative_int():
    sampler = ConstantDelaySampler(delay=1)

    with pytest.raises(TypeError, match="num_samples must be an int"):
        sampler.sample(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        sampler.sample(-1)


class _FakeEnv:
    """Minimal non-gym env exposing the chunk_step/reset surface."""

    def chunk_step(self, *args, **kwargs):
        return "stepped"

    def reset(self, *args, **kwargs):
        return "obs", {}


# Mock gymnasium and its transitive imports for unit-test environments that
# do not install the embodied extras. A minimal gym.Wrapper shim is enough
# because InsertDelay only delegates to self.env.


class _FakeGymEnv:
    pass


class _FakeGymWrapper:
    def __init__(self, env):
        self.env = env


_fake_gym = MagicMock()
_fake_gym.Env = _FakeGymEnv
_fake_gym.Wrapper = _FakeGymWrapper

if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = _fake_gym
if "imageio" not in sys.modules:
    sys.modules["imageio"] = MagicMock()


def _delayed_env(delay: float):
    from rlinf.envs.wrappers import InsertDelay

    return InsertDelay(
        _FakeEnv(), OmegaConf.create({"type": "constant", "delay": delay})
    )


def test_chunk_step_does_not_block_the_caller():
    env = _delayed_env(0.5)

    start = time.monotonic()
    assert env.chunk_step() == "stepped"
    elapsed = time.monotonic() - start

    # The delay is sampled, not slept: blocking here would stall the event loop.
    assert elapsed < 0.05


def test_wait_delay_waits_out_the_accumulated_delay():
    env = _delayed_env(0.05)
    env.chunk_step()
    env.chunk_step()

    start = time.monotonic()
    asyncio.run(env.wait_delay())
    elapsed = time.monotonic() - start

    # Both sampled delays are paid, never dropped.
    assert elapsed == pytest.approx(0.1, abs=0.03)


def test_wait_delay_yields_to_other_coroutines():
    env = _delayed_env(0.2)
    env.chunk_step()
    progressed = []

    async def main():
        async def ticker():
            for _ in range(4):
                await asyncio.sleep(0.01)
                progressed.append(1)

        await asyncio.gather(env.wait_delay(), ticker())

    asyncio.run(main())
    # A blocking sleep would have starved the ticker entirely.
    assert len(progressed) == 4


def test_wait_delay_is_a_noop_when_nothing_is_pending():
    env = _delayed_env(0.5)

    start = time.monotonic()
    asyncio.run(env.wait_delay())

    assert time.monotonic() - start < 0.05


def test_delay_metrics_report_every_sample():
    env = _delayed_env(0.03)
    env.chunk_step()
    env.reset()

    metrics = env.insert_delay_metrics()

    assert metrics.tolist() == pytest.approx([0.03, 0.03])
    assert env.insert_delay_metrics().numel() == 0
