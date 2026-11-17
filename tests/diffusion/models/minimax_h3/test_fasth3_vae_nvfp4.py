# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released decoder schema, projection packing, and strict loading."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import (
    FastH3NVFP4Decoder,
    PackedNVFP4Linear,
    _QKVProjection,
    _SwiGLU,
    install_nvfp4_video_decoder,
    released_decoder_pixels,
    require_nvfp4_linear,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _host():
    decoder = nn.Module()
    decoder.spatial_parallel = decoder.t_causal = False
    decoder.config = SimpleNamespace(patch_size=2, patch_size_t=1)
    decoder.x_embedder = nn.Linear(24, 128)
    decoder.proj_out = nn.Linear(128, 12)
    decoder.norm_out = nn.LayerNorm(128)
    decoder.register_tokens = nn.Parameter(torch.randn(1, 4, 128))
    decoder.pos_embed = nn.Identity()
    block = nn.Module()
    block.norm1, block.norm2 = nn.RMSNorm(128), nn.RMSNorm(128)
    block.scale1, block.scale2 = nn.Parameter(torch.ones(128)), nn.Parameter(torch.ones(128))
    block.attn = nn.Module()
    block.attn.dim_head = 64
    block.attn.to_qkv, block.attn.to_out = nn.Linear(128, 384), nn.Linear(128, 128)
    block.ff = nn.Module()
    decoder.transformer_blocks = nn.ModuleList([block])
    vae = nn.Module()
    vae.decoder, vae.post_quant_conv = decoder, nn.Conv3d(24, 24, 1)
    vae.encoder = nn.Linear(24, 24)
    return vae


def _packed(n=128, k=128, value=0):
    return {
        "packed_weight": torch.full((n, k // 2), value, dtype=torch.uint8),
        "weight_inv_scale": torch.zeros(n, k // 16, dtype=torch.uint8),
        "weight_global_sf": torch.tensor(17.0),
        "bias": torch.full((n,), value, dtype=torch.bfloat16),
        "input_amax": torch.tensor(0.0),
    }


def _checkpoint(tmp_path, vae, *, metadata_override=None, extra=None, missing=None):
    state = {}
    for name, value in vae.state_dict().items():
        if name.startswith("encoder.") or ".attn." in name:
            continue
        name = name.replace("decoder.x_embedder.", "decoder.proj_in.")
        state[name] = value.clone()
    prefix = "decoder.transformer_blocks.0"
    for index, (suffix, n, k) in enumerate(
        (
            ("attn.to_q", 128, 128),
            ("attn.to_k", 128, 128),
            ("attn.to_v", 128, 128),
            ("attn.to_out.0", 128, 128),
            ("ff.net.0.proj", 256, 128),
            ("ff.net.2", 128, 128),
        ),
        start=1,
    ):
        state.update({f"{prefix}.{suffix}.{key}": value for key, value in _packed(n, k, index).items()})
    metadata = dict(
        block_size=16, act_scale="unit", rotation_group=None, skip_blocks=[], num_linears=6, student_layers=1
    )
    metadata.update(metadata_override or {})
    if extra:
        state[extra] = torch.zeros(1)
    if missing:
        del state[f"{prefix}.{missing}"]
    path = tmp_path / "decoder.safetensors"
    save_file(
        state,
        path,
        metadata={"format": "fastvideo_h3_decoder_nvfp4_packed_v1", "nvfp4_metadata": json.dumps(metadata)},
    )
    return path


def test_released_decoder_strict_load_preserves_encoder_and_dtypes(tmp_path):
    vae = _host()
    encoder = {key: value.clone() for key, value in vae.encoder.state_dict().items()}
    path = _checkpoint(tmp_path, vae)
    install_nvfp4_video_decoder(vae, path)
    assert isinstance(vae.decoder, FastH3NVFP4Decoder)
    linears = [module for module in vae.decoder.modules() if isinstance(module, PackedNVFP4Linear)]
    assert len(linears) == 6
    assert all(module.packed_weight.dtype == torch.uint8 and module.bias.dtype == torch.bfloat16 for module in linears)
    for index, name in enumerate(("attn.to_qkv.q", "attn.to_qkv.k", "attn.to_qkv.v", "attn.to_out", "ff.0", "ff.2"), 1):
        module = vae.decoder.transformer_blocks[0].get_submodule(name)
        assert torch.all(module.packed_weight == index) and torch.all(module.bias == index)
    assert vae.decoder.proj_in.weight.dtype == torch.float32
    for key, value in encoder.items():
        torch.testing.assert_close(vae.encoder.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "metadata", [{"act_scale": "dynamic"}, {"student_layers": 36, "num_linears": 216}, {"num_linears": 5}]
)
def test_rejects_incompatible_decoder_contract(tmp_path, metadata):
    vae = _host()
    with pytest.raises(ValueError):
        install_nvfp4_video_decoder(vae, _checkpoint(tmp_path, vae, metadata_override=metadata))


@pytest.mark.parametrize(
    "extra,error",
    [
        ("unknown.weight", RuntimeError),
        ("decoder.transformer_blocks.0.attn.to_qkv.q.packed_weight", ValueError),
    ],
)
def test_unknown_checkpoint_tensor_is_not_silently_dropped(tmp_path, extra, error):
    vae = _host()
    with pytest.raises(error, match="Unexpected"):
        install_nvfp4_video_decoder(vae, _checkpoint(tmp_path, vae, extra=extra))


def test_missing_packed_tensor_is_rejected(tmp_path):
    vae = _host()
    with pytest.raises(ValueError, match="Missing packed NVFP4 tensor"):
        install_nvfp4_video_decoder(vae, _checkpoint(tmp_path, vae, missing="attn.to_k.weight_global_sf"))


def test_qkv_global_scales_keep_separate_projection_outputs():
    class Constant(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x):
            return torch.full((*x.shape[:-1], 128), self.value, dtype=x.dtype)

    out = _QKVProjection(Constant(1), Constant(2), Constant(3), head_dim=64)(torch.zeros(2, 4, 128))
    q, k, v = out.unflatten(-1, (2, 192)).chunk(3, -1)
    assert torch.all(q == 1) and torch.all(k == 2) and torch.all(v == 3)


def test_released_swiglu_order():
    value = torch.tensor([2.0, -3.0])
    gate = torch.tensor([-1.0, 4.0])
    torch.testing.assert_close(_SwiGLU()(torch.cat((value, gate))), value * torch.nn.functional.silu(gate))


def test_released_pixels_normalize_in_fp32_before_rounding():
    decoded = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.bfloat16).view(1, 1, 1, 1, 3).expand(1, 3, 1, 1, 3)
    pixels = released_decoder_pixels(decoded)
    expected = torch.tensor([[65, 124, 182], [59, 116, 173], [46, 104, 161]], dtype=torch.uint8)
    assert pixels.dtype == torch.float32
    assert torch.equal((pixels * 255).round().to(torch.uint8).view(3, 3), expected)


@pytest.fixture
def clear_nvfp4_cache():
    require_nvfp4_linear.cache_clear()
    yield
    require_nvfp4_linear.cache_clear()


@pytest.mark.parametrize("capability,allowed", [((8, 9), False), ((12, 0), True), ((12, 1), True)])
def test_nvfp4_rejects_ada_and_accepts_spark(monkeypatch, capability, allowed, clear_nvfp4_cache):
    monkeypatch.setattr(current_omni_platform, "get_device_capability", lambda _: capability)
    module = ModuleType("flashinfer")
    setattr(module, "mm_fp4", Mock())
    setattr(module, "nvfp4_quantize", Mock())
    monkeypatch.setitem(sys.modules, "flashinfer", module)
    if allowed:
        require_nvfp4_linear(torch.device("cuda:0"))
    else:
        with pytest.raises(ValueError, match="SM121"):
            require_nvfp4_linear(torch.device("cuda:0"))
