# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for ABot-World configuration and helpers."""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.abot_world import abot_world_transformer as abot_transformer
from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import (
    _DEFAULT_HEIGHT,
    _DEFAULT_WIDTH,
    ABOT_DMD_TIMESTEPS,
    _build_shifted_flow_schedule,
    _convert_wan_umt5_encoder_state_dict,
    _fix_wan22_residual_vae_keys,
    _paged_kv_tokens_per_frame,
    _positive_finite_flow_shift,
    _resolve_local_model_path,
    _validate_latent_channel_contract,
    _validate_local_model_files,
    _validate_parallel_config,
)
from vllm_omni.diffusion.models.abot_world.abot_world_transformer import (
    ABotCausalHead,
    ABotWorldCausalTransformer3DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_parallel_config_warns_when_eager_execution_is_disabled(enforce_eager: bool) -> None:
    config = SimpleNamespace(enforce_eager=enforce_eager, quantization_config=None, parallel_config=None)

    if enforce_eager:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_parallel_config(config)
    else:
        with pytest.warns(UserWarning, match=r"NaNs or black frames.*--enforce-eager"):
            _validate_parallel_config(config)


def test_parallel_config_still_rejects_quantization() -> None:
    config = SimpleNamespace(enforce_eager=True, quantization_config=object(), parallel_config=None)

    with pytest.raises(NotImplementedError, match="does not support quantization"):
        _validate_parallel_config(config)


def test_abot_world_packed_modules_constant_is_model_prefixed() -> None:
    assert abot_transformer.ABOT_WORLD_WAN_PACKED_MODULES == {"qkv": ["q", "k", "v"]}
    assert not hasattr(abot_transformer, "_WAN_PACKED_MODULES")


def test_default_resolution_is_flash_attention_page_aligned() -> None:
    tokens_per_frame = _paged_kv_tokens_per_frame(
        _DEFAULT_HEIGHT,
        _DEFAULT_WIDTH,
        vae_scale_factor=16,
        patch_height=2,
        patch_width=2,
    )

    assert (_DEFAULT_HEIGHT, _DEFAULT_WIDTH) == (512, 832)
    assert tokens_per_frame == 416
    assert tokens_per_frame % 16 == 0


@pytest.mark.parametrize(
    ("height", "expected_tokens"),
    [(480, 390), (448, 364)],
)
def test_unaligned_paged_kv_resolutions_are_rejected(height: int, expected_tokens: int) -> None:
    with pytest.raises(ValueError, match=rf"got {expected_tokens} for"):
        _paged_kv_tokens_per_frame(
            height,
            832,
            vae_scale_factor=16,
            patch_height=2,
            patch_width=2,
        )


def test_wan22_vae_requires_48_channel_transformer_contract() -> None:
    _validate_latent_channel_contract(
        vae_z_dim=48,
        transformer_in_channels=48,
        transformer_out_channels=48,
    )

    with pytest.raises(ValueError, match="VAE z_dim=16.*48/48"):
        _validate_latent_channel_contract(
            vae_z_dim=16,
            transformer_in_channels=48,
            transformer_out_channels=48,
        )


def test_shifted_flow_schedule_is_finite_and_monotonic() -> None:
    schedule = _build_shifted_flow_schedule(flow_shift=5.0)

    assert len(schedule) == len(ABOT_DMD_TIMESTEPS) == 4
    assert all(math.isfinite(timestep) and math.isfinite(sigma) for timestep, sigma in schedule)
    assert all(0 < sigma <= 1 for _, sigma in schedule)
    assert [sigma for _, sigma in schedule] == sorted((sigma for _, sigma in schedule), reverse=True)


@pytest.mark.parametrize("value", [0, -1, True, math.inf, math.nan, "invalid"])
def test_flow_shift_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _positive_finite_flow_shift(value)


def test_local_model_path_accepts_existing_directory(tmp_path) -> None:
    assert _resolve_local_model_path(str(tmp_path)) == str(tmp_path.resolve())


def test_official_checkpoint_config_does_not_require_model_index(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"ti2v","dim":3072,"in_dim":48,"out_dim":48}',
        encoding="utf-8",
    )
    config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="ABotWorldCausalPipeline",
    )

    config.enrich_config()

    assert config.model_class_name == "ABotWorldCausalPipeline"


def test_incomplete_local_checkpoint_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="diffusion_pytorch_model.safetensors"):
        _validate_local_model_files(str(tmp_path))


def test_umt5_conversion_maps_gated_gelu_branches() -> None:
    source = {
        "token_embedding.weight": torch.zeros(2, 2),
        "norm.weight": torch.zeros(2),
    }
    suffixes = (
        "norm1.weight",
        "attn.q.weight",
        "attn.k.weight",
        "attn.v.weight",
        "attn.o.weight",
        "pos_embedding.embedding.weight",
        "norm2.weight",
        "ffn.gate.0.weight",
        "ffn.fc1.weight",
        "ffn.fc2.weight",
    )
    for value, suffix in enumerate(suffixes):
        source[f"blocks.0.{suffix}"] = torch.full((1,), value)

    converted = _convert_wan_umt5_encoder_state_dict(source, num_layers=1)

    assert converted["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].item() == suffixes.index("ffn.gate.0.weight")
    assert converted["encoder.block.0.layer.1.DenseReluDense.wi_1.weight"].item() == suffixes.index("ffn.fc1.weight")


def test_wan22_residual_vae_conversion_preserves_grouped_blocks() -> None:
    source = {
        "encoder.downsamples.0.downsamples.0.residual.0.gamma": torch.tensor(0),
        "encoder.downsamples.0.downsamples.2.resample.1.weight": torch.tensor(1),
        "decoder.upsamples.0.upsamples.2.shortcut.weight": torch.tensor(2),
        "decoder.upsamples.0.upsamples.3.time_conv.weight": torch.tensor(3),
        "decoder.upsamples.3.upsamples.0.residual.0.gamma": torch.tensor(4),
    }
    converted = _fix_wan22_residual_vae_keys(
        source,
        {
            "encoder.down_blocks.0.downsamples.0.norm1.gamma": torch.tensor(-1),
            "decoder.up_blocks.0.resnets.0.norm1.gamma": torch.tensor(-1),
            "decoder.upsamples.3.upsamples.0.residual.0.gamma": torch.tensor(-1),
            "quant_conv.weight": torch.tensor(4),
        },
    )

    assert set(converted) == {
        "encoder.down_blocks.0.resnets.0.norm1.gamma",
        "encoder.down_blocks.0.downsampler.resample.1.weight",
        "decoder.up_blocks.0.resnets.2.conv_shortcut.weight",
        "decoder.up_blocks.0.upsampler.time_conv.weight",
        "decoder.up_blocks.3.resnets.0.norm1.gamma",
        "quant_conv.weight",
    }


def test_abot_head_keeps_bfloat16_through_modulation() -> None:
    head = ABotCausalHead(dim=4, out_dim=2, eps=1e-6).to(torch.bfloat16)

    output = head(
        torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        torch.zeros(1, 1, 4, dtype=torch.bfloat16),
        tokens_per_frame=2,
    )

    assert output.dtype == torch.bfloat16


def test_from_config_uses_wan_text_width_not_text_length() -> None:
    model = ABotWorldCausalTransformer3DModel.from_config(
        {
            "patch_size": [1, 2, 2],
            "num_heads": 1,
            "dim": 8,
            "ffn_dim": 16,
            "num_layers": 1,
            "in_dim": 4,
            "out_dim": 4,
            "text_len": 512,
            "downscale_factor_control_adapter": 2,
        }
    )

    assert model.config.text_dim == 4096
    assert model.config.downscale_factor_control_adapter == 2
    names = dict(model.named_parameters())
    assert "blocks.0.modulation" in names
    assert "blocks.0.self_attn.qkv.weight" in names
    assert "blocks.0.self_attn.o.weight" in names
    assert "blocks.0.cross_attn.q.weight" in names
    assert "blocks.0.norm3.weight" in names
    assert "blocks.0.norm3.bias" in names
    assert "act_control_adapter.conv.weight" in names
    assert "act_control_adapter.residual_blocks.0.conv1.weight" in names
    assert "head.modulation" in names
    assert "head.head.weight" in names
