# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import dataclasses
import inspect
import math

import pytest
import torch

from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    MAGI2_GENERATION_CONFIG,
    MAGI2_REFINER_CONFIG,
    MAGI2_REFINER_GENERATION_CONFIG,
    Magi2RefinerConfig,
    Magi2RefinerWindowConfig,
)
from vllm_omni.diffusion.models.magi2.layers import ElementWiseFourierEmbed, MultiModalityRMSNorm, swiglu7

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

# sand-ai/MAGI-2-preview refiner/model.safetensors.index.json
_CHECKPOINT_TENSORS = 281
_CHECKPOINT_PARAMETERS = 6_720_075_280
_CHECKPOINT_BYTES = 13_484_611_648


def _checkpoint_shapes(cfg: Magi2RefinerConfig) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Checkpoint tensor names, shapes and dtypes implied by the config."""

    h, d = cfg.hidden_size, cfg.head_dim
    hq, hkv, inter = cfg.num_heads_q, cfg.num_heads_kv, cfg.intermediate_size
    bf16, f32 = cfg.params_dtype, torch.float32
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for i in range(cfg.num_layers):
        m = cfg.num_modalities(i)
        prefix = f"block.layers.{i}"
        shapes[f"{prefix}.attention.pre_norm.weight"] = ((m * h,), f32)
        shapes[f"{prefix}.attention.linear_qkv.weight"] = ((m * (hq + 2 * hkv) * d, h), bf16)
        shapes[f"{prefix}.attention.linear_g.weight"] = ((m * hq, h), bf16)
        shapes[f"{prefix}.attention.q_norm.weight"] = ((m * d,), f32)
        shapes[f"{prefix}.attention.k_norm.weight"] = ((m * d,), f32)
        shapes[f"{prefix}.attention.linear_proj.weight"] = ((m * h, hq * d), bf16)
        shapes[f"{prefix}.mlp.pre_norm.weight"] = ((m * h,), f32)
        shapes[f"{prefix}.mlp.up_gate_proj.weight"] = ((m * 2 * inter, h), bf16)
        shapes[f"{prefix}.mlp.down_proj.weight"] = ((m * h, inter), bf16)
    for name, channels in (
        ("video", cfg.video_in_channels),
        ("text", cfg.text_in_channels),
        ("audio", cfg.audio_in_channels),
    ):
        shapes[f"pre_adapter.{name}_embedder.weight"] = ((h, channels), f32)
        shapes[f"pre_adapter.{name}_embedder.bias"] = ((h,), f32)
    shapes["pre_adapter.rope.bands"] = ((cfg.num_rope_bands,), f32)
    shapes["post_adapter.final_norm_video.weight"] = ((h,), f32)
    shapes["post_adapter.final_norm_audio.weight"] = ((h,), f32)
    shapes["post_adapter.final_linear_video.weight"] = ((cfg.video_in_channels, h), f32)
    shapes["post_adapter.final_linear_audio.weight"] = ((cfg.audio_in_channels, h), f32)
    return shapes


def test_refiner_architecture_matches_released_config() -> None:
    cfg = MAGI2_REFINER_CONFIG
    cfg.validate()

    assert (cfg.num_layers, cfg.hidden_size, cfg.head_dim) == (30, 4096, 128)
    assert (cfg.num_heads_q, cfg.num_heads_kv) == (32, 8)
    assert (cfg.video_in_channels, cfg.audio_in_channels, cfg.text_in_channels) == (48, 64, 5120)
    assert cfg.multimodal_layers == (0, 1, 28, 29)
    assert [i for i in range(cfg.num_layers) if cfg.num_modalities(i) == 3] == [0, 1, 28, 29]
    assert cfg.params_dtype == torch.bfloat16
    assert cfg.attention_gating
    # int(4096 * 4 * 2 / 3) // 4 * 4; the Preview's //128 rounding would give 10880.
    assert cfg.intermediate_size == 10920


def test_refiner_config_reproduces_checkpoint_census() -> None:
    shapes = _checkpoint_shapes(MAGI2_REFINER_CONFIG)

    assert len(shapes) == _CHECKPOINT_TENSORS == 9 * MAGI2_REFINER_CONFIG.num_layers + 11
    assert sum(math.prod(shape) for shape, _ in shapes.values()) == _CHECKPOINT_PARAMETERS
    assert sum(math.prod(shape) * dtype.itemsize for shape, dtype in shapes.values()) == _CHECKPOINT_BYTES
    # Spot checks against the checkpoint headers.
    assert shapes["block.layers.0.attention.linear_qkv.weight"][0] == (18432, 4096)
    assert shapes["block.layers.0.attention.q_norm.weight"][0] == (384,)
    assert shapes["block.layers.0.attention.linear_g.weight"][0] == (96, 4096)
    assert shapes["block.layers.29.mlp.up_gate_proj.weight"][0] == (65520, 4096)
    assert shapes["block.layers.2.attention.linear_qkv.weight"][0] == (6144, 4096)
    assert shapes["block.layers.2.attention.k_norm.weight"][0] == (128,)
    assert shapes["block.layers.2.mlp.up_gate_proj.weight"][0] == (21840, 4096)
    assert shapes["block.layers.2.mlp.down_proj.weight"][0] == (4096, 10920)


def test_refiner_rope_norm_and_activation_constants_match_layers() -> None:
    cfg = MAGI2_REFINER_CONFIG

    assert (cfg.num_rope_bands, cfg.rotary_dim) == (16, 96)
    assert cfg.rope_temperature == 10000.0
    rope = ElementWiseFourierEmbed(cfg.head_dim, temperature=cfg.rope_temperature)
    expected_bands = 1.0 / 10000.0 ** (torch.arange(16, dtype=torch.float32) / 16)
    torch.testing.assert_close(rope.bands.detach(), expected_bands, rtol=0, atol=0)
    coords = torch.tensor([[1.0, 2.0, 3.0, 8.0, 4.0, 4.0, 8.0, 4.0, 4.0]])
    assert rope(coords).shape == (1, cfg.rotary_dim)

    assert cfg.norm_eps == 1e-6
    assert MultiModalityRMSNorm(8).eps == cfg.norm_eps
    activation = inspect.signature(swiglu7).parameters
    assert (cfg.swiglu7_alpha, cfg.swiglu7_limit) == (1.702, 7.0)
    assert activation["alpha"].default == cfg.swiglu7_alpha
    assert activation["limit"].default == cfg.swiglu7_limit


def test_refiner_window_config() -> None:
    window = MAGI2_REFINER_CONFIG.window

    assert window.block_size == (8, 4, 4)
    assert window.radius == (2, 2, 2)
    assert window.dense_query_chunk == 128
    with pytest.raises(ValueError, match="positive"):
        Magi2RefinerWindowConfig(block_size=(8, 0, 4))
    with pytest.raises(ValueError, match="non-negative"):
        Magi2RefinerWindowConfig(radius=(2, -1, 2))
    with pytest.raises(ValueError, match="three"):
        Magi2RefinerWindowConfig(block_size=(4, 4))


def test_refiner_generation_contract_follows_the_preview_timeline() -> None:
    gen = MAGI2_REFINER_GENERATION_CONFIG

    assert (gen.num_inference_steps, gen.shift, gen.video_guidance_scale) == (5, 5.0, 2.0)
    assert gen.noise_index == 220
    assert (gen.zero_snr_linear_start, gen.zero_snr_linear_end, gen.zero_snr_num_timesteps) == (0.00085, 0.012, 1000)
    assert gen.video_vae_stride == (4, 16, 16)
    assert (gen.fps, gen.output_frames) == (25.0, 249)

    # The Preview runs on the internal 25-FPS timeline with temporal stride 8.
    preview = MAGI2_GENERATION_CONFIG
    internal_frames = int(round(preview.duration_seconds * preview.fps)) * 2
    preview_latent_frames = (internal_frames - 1) // preview.video_vae_stride[0] + 1
    assert preview_latent_frames == 32
    # The refiner doubles the temporal grid (2T - 1) and decodes at stride 4.
    refiner_latent_frames = 2 * preview_latent_frames - 1
    assert refiner_latent_frames == 63
    assert (refiner_latent_frames - 1) * gen.video_vae_stride[0] + 1 == gen.output_frames
    assert gen.output_frames / gen.fps == pytest.approx(9.96)


def test_refiner_config_is_frozen_and_validated() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        MAGI2_REFINER_CONFIG.num_layers = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="multiple of key/value heads"):
        Magi2RefinerConfig(hidden_size=96, head_dim=32, num_query_groups=2).validate()
    with pytest.raises(ValueError, match="outside the transformer"):
        Magi2RefinerConfig(num_layers=4, multimodal_layers=(0, 4)).validate()
    tiny = Magi2RefinerConfig(num_layers=4, hidden_size=256, head_dim=32, num_query_groups=2, multimodal_layers=(0, 3))
    tiny.validate()
    assert (tiny.num_heads_q, tiny.num_heads_kv, tiny.rotary_dim) == (8, 2, 24)
