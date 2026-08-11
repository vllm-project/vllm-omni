# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness contracts for request-gated VAE decoder fast paths."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.vae_optimizations.gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
    use_pipeline_vae_fast_path,
    use_vae_fast_path,
)

pytestmark = [pytest.mark.diffusion]


class _PipelineWithVae(nn.Module):
    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae


class _DistributedExecutorConfig:
    def __init__(self, parallel_mode: str) -> None:
        self.parallel_mode = parallel_mode


@pytest.mark.core_model
@pytest.mark.cpu
def test_vae_fast_path_gate_restores_nested_state() -> None:
    vae = nn.Identity()
    gate = VaeFastPathGate()
    register_vae_fast_path_gate(vae, gate)

    with use_vae_fast_path(vae, True):
        assert gate.enabled
        with use_vae_fast_path(vae, False):
            assert not gate.enabled
        assert gate.enabled

    assert not gate.enabled


@pytest.mark.core_model
@pytest.mark.cpu
def test_pipeline_vae_fast_path_controls_discovered_vae() -> None:
    vae = nn.Identity()
    pipeline = _PipelineWithVae(vae)
    gate = VaeFastPathGate()
    register_vae_fast_path_gate(vae, gate)

    with use_pipeline_vae_fast_path(pipeline, True):
        assert gate.enabled

    assert not gate.enabled


@pytest.mark.core_model
@pytest.mark.cpu
def test_image_group_norm_fast_path_uses_compile_safe_fallback(monkeypatch) -> None:
    from vllm_omni.diffusion.vae_optimizations import image

    gate = VaeFastPathGate()
    gate.enabled = True
    norm = image.FusedGroupNormSiLU(nn.GroupNorm(2, 4), gate)
    inputs = torch.randn(1, 4, 3, 3)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        image,
        "group_norm_silu_4d",
        lambda *args, **kwargs: pytest.fail("eager Triton kernel entered during compilation"),
    )

    actual = norm(inputs)
    expected = F.silu(F.group_norm(inputs, 2, norm.weight, norm.bias, norm.eps))
    torch.testing.assert_close(actual, expected)


def _small_image_vae() -> nn.Module:
    from diffusers.models.autoencoders import AutoencoderKL

    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=16,
    )


def _small_wan_vae(vae_cls: type[nn.Module] | None = None) -> nn.Module:
    if vae_cls is None:
        from diffusers.models.autoencoders import AutoencoderKLWan

        vae_cls = AutoencoderKLWan

    return vae_cls(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        scale_factor_temporal=1,
        scale_factor_spatial=2,
    )


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
def test_image_vae_install_preserves_lossless_path_and_state_dict() -> None:
    from vllm_omni.diffusion.vae_optimizations import clear_pipeline_vae_fast_path_caches
    from vllm_omni.diffusion.vae_optimizations.image import (
        FusedGroupNormSiLU,
        FusedUpsample2xConv2d,
        maybe_optimize_image_vae,
    )

    torch.manual_seed(0)
    vae = _small_image_vae().to(device="cuda", dtype=torch.bfloat16).eval()
    latents = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.bfloat16)
    parameter_names = {name for name, _ in vae.named_parameters()}
    state_dict = {name: value.clone() for name, value in vae.state_dict().items()}
    reference = vae.decode(latents).sample

    maybe_optimize_image_vae(vae)
    maybe_optimize_image_vae(vae)

    assert {name for name, _ in vae.named_parameters()} == parameter_names
    vae.load_state_dict(state_dict, strict=True)
    assert any(isinstance(module, FusedGroupNormSiLU) for module in vae.modules())
    assert any(isinstance(module, FusedUpsample2xConv2d) for module in vae.modules())
    assert torch.equal(vae.decode(latents).sample, reference)

    with use_vae_fast_path(vae, True):
        fast = vae.decode(latents).sample

    torch.testing.assert_close(fast.float(), reference.float(), atol=0.1, rtol=0)
    assert any(
        buffer is not None
        for module in vae.modules()
        for name, buffer in module._buffers.items()
        if name in {"_fused_weight", "_vllm_folded_value_weight", "_vllm_folded_value_bias"}
    )
    clear_pipeline_vae_fast_path_caches(_PipelineWithVae(vae))
    assert all(
        buffer is None
        for module in vae.modules()
        for name, buffer in module._buffers.items()
        if name in {"_fused_weight", "_vllm_folded_value_weight", "_vllm_folded_value_bias"}
    )
    with use_vae_fast_path(vae, True):
        torch.testing.assert_close(vae.decode(latents).sample.float(), reference.float(), atol=0.1, rtol=0)
    assert torch.equal(vae.decode(latents).sample, reference)
    assert vae.decoder.conv_in.weight.is_contiguous()


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_channels_last_group_norm_silu_matches_reference(dtype: torch.dtype) -> None:
    from vllm_omni.diffusion.vae_optimizations.triton_group_norm_silu import group_norm_silu_4d

    torch.manual_seed(0)
    x = torch.randn(1, 64, 16, 16, device="cuda", dtype=dtype).contiguous(memory_format=torch.channels_last)
    weight = torch.randn(64, device="cuda", dtype=dtype)
    bias = torch.randn(64, device="cuda", dtype=dtype)

    actual = group_norm_silu_4d(x, weight, bias, num_groups=8, eps=1e-5)
    expected = F.silu(F.group_norm(x, 8, weight, bias))

    assert actual is not None
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (7e-2, 2e-2)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
@pytest.mark.parametrize(
    "activation_dtype,affine_dtype,atol,rtol",
    [
        (torch.float32, torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, torch.float32, 1.5e-1, 3e-2),
    ],
)
def test_wan_rmsnorm_silu_matches_reference(
    activation_dtype: torch.dtype,
    affine_dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    from vllm_omni.diffusion.vae_optimizations.triton_wan_rmsnorm_silu import wan_rmsnorm_silu

    torch.manual_seed(0)
    x = torch.randn(1, 96, 3, 10, 14, device="cuda", dtype=activation_dtype).contiguous(
        memory_format=torch.channels_last_3d
    )
    gamma = torch.randn(96, 1, 1, 1, device="cuda", dtype=affine_dtype)
    bias = torch.randn_like(gamma)
    normalized = F.normalize(x.float() if activation_dtype != torch.float32 else x, dim=1).to(activation_dtype)
    expected = F.silu(normalized * 96**0.5 * gamma + bias)

    actual = wan_rmsnorm_silu(x, gamma, bias)

    assert actual is not None
    assert actual.stride() == x.stride()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
def test_wan_vae_install_preserves_lossless_path_and_state_dict() -> None:
    from vllm_omni.diffusion.vae_optimizations.wan import (
        FusedWanRMSNormSiLU,
        maybe_optimize_wan_vae,
    )

    torch.manual_seed(0)
    vae = _small_wan_vae().cuda().eval()
    latents = torch.randn(1, 4, 1, 8, 8, device="cuda")
    parameter_names = {name for name, _ in vae.named_parameters()}
    state_dict = {name: value.clone() for name, value in vae.state_dict().items()}
    reference = vae.decode(latents).sample

    maybe_optimize_wan_vae(vae)
    maybe_optimize_wan_vae(vae)

    assert {name for name, _ in vae.named_parameters()} == parameter_names
    vae.load_state_dict(state_dict, strict=True)
    assert any(isinstance(module, FusedWanRMSNormSiLU) for module in vae.modules())
    assert torch.equal(vae.decode(latents).sample, reference)

    with use_vae_fast_path(vae, True):
        fast = vae.decode(latents).sample

    torch.testing.assert_close(fast, reference, atol=5e-4, rtol=1e-4)
    assert torch.equal(vae.decode(latents).sample, reference)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("parallel_mode", ["auto", "spatial_shard_height", "spatial_shard_width"])
def test_wan_fast_path_skips_spatial_shard_mode(monkeypatch: pytest.MonkeyPatch, parallel_mode: str) -> None:
    import vllm_omni.diffusion.vae_optimizations.wan as wan_optimizations

    monkeypatch.setattr(wan_optimizations, "_HAS_TRITON", True)
    vae = _small_wan_vae()
    vae.distributed_executor = _DistributedExecutorConfig(parallel_mode)

    wan_optimizations.maybe_optimize_wan_vae(vae)

    assert not any(isinstance(module, wan_optimizations.FusedWanRMSNormSiLU) for module in vae.modules())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
@pytest.mark.parametrize(
    ("precision", "decode_dtype"),
    [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_wan_decode_precision_keeps_encode_fp32(precision: str, decode_dtype: torch.dtype) -> None:
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
    from vllm_omni.diffusion.vae_optimizations.precision import configure_wan_decode_precision

    torch.manual_seed(0)
    vae = _small_wan_vae(DistributedAutoencoderKLWan).to(device="cuda", dtype=torch.bfloat16).eval()
    configure_wan_decode_precision(vae, precision)

    assert next(vae.encoder.parameters()).dtype == torch.float32
    assert next(vae.quant_conv.parameters()).dtype == torch.float32
    assert next(vae.decoder.parameters()).dtype == decode_dtype
    assert next(vae.post_quant_conv.parameters()).dtype == decode_dtype

    _, grid_spec = vae.tile_split(torch.randn(1, 4, 1, 8, 8, device="cuda", dtype=decode_dtype))
    assert grid_spec.output_dtype == decode_dtype

    encoded = vae.encode(torch.randn(1, 3, 1, 16, 16, device="cuda", dtype=torch.bfloat16))
    decoded = vae.decode(torch.randn(1, 4, 1, 8, 8, device="cuda", dtype=torch.float32))

    assert encoded.latent_dist.parameters.dtype == torch.float32
    assert decoded.sample.dtype == decode_dtype
