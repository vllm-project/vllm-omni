# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused packed SigLIP layers and vision CUDA graphs match the eager packed encode."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import vision_fused
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    Resampler,
    SiglipVisionConfig,
    SiglipVisionTransformer,
    _encode_vision_packed,
)

_PATCH = 2

pytestmark = [pytest.mark.core_model]


def _towers(*, dtype: torch.dtype = torch.float32, device: str = "cpu") -> tuple[SiglipVisionTransformer, Resampler]:
    torch.manual_seed(0)
    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=72,
        num_hidden_layers=3,
        num_attention_heads=4,
        image_size=28,
        patch_size=_PATCH,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    vpm = SiglipVisionTransformer(config).eval()
    for module in vpm.modules():
        if isinstance(module, torch.nn.LayerNorm):
            # Non-trivial affines, so a LayerNorm fused into the wrong layer shows.
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.2)
            torch.nn.init.normal_(module.bias, std=0.2)
    resampler = Resampler(num_queries=4, embed_dim=64, num_heads=4, kv_dim=32, adaptive=True, max_size=(8, 8)).eval()
    for param in resampler.parameters():
        torch.nn.init.normal_(param, std=0.2)
    return vpm.to(device=device, dtype=dtype), resampler.to(device=device, dtype=dtype)


def _embeddings(vpm: SiglipVisionTransformer, tokens: int, *, seed: int = 1) -> torch.Tensor:
    weight = vpm.embeddings.patch_embedding.weight
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((tokens, 32), generator=generator).to(device=weight.device, dtype=weight.dtype)


def _eager(vpm: SiglipVisionTransformer, hidden: torch.Tensor, seq_groups) -> torch.Tensor:
    for layer in vpm.encoder.layers:
        hidden = layer.forward_packed(hidden, seq_groups)
    return vpm.post_layernorm(hidden)


@pytest.mark.cpu
def test_fused_layers_match_eager_packed_layers() -> None:
    vpm, _ = _towers()
    # Two runs of different sequence length, one with several items.
    seq_groups = [(0, 3, 16), (48, 2, 15)]
    hidden = _embeddings(vpm, 78)
    with torch.inference_mode():
        expected = _eager(vpm, hidden.clone(), seq_groups)
        fused = vision_fused.encode_packed_fused(vpm, hidden.clone(), seq_groups)
    torch.testing.assert_close(fused, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
def test_explicit_eager_attention_keeps_the_unfused_layers() -> None:
    vpm, _ = _towers()
    assert vision_fused.supports_fused_layers(vpm)
    vpm.config._attn_implementation = "eager"
    assert not vision_fused.supports_fused_layers(vpm)


@pytest.mark.cpu
def test_packed_qkv_repoints_the_projections_without_changing_them() -> None:
    vpm, _ = _towers()
    attn = vpm.encoder.layers[0].self_attn
    before = [p.weight.detach().clone() for p in (attn.q_proj, attn.k_proj, attn.v_proj)]
    weight, bias = vision_fused.packed_qkv(attn)
    assert weight.shape == (96, 32) and bias is not None
    for index, projection in enumerate((attn.q_proj, attn.k_proj, attn.v_proj)):
        assert projection.weight.data_ptr() == weight[index * 32].data_ptr()
        torch.testing.assert_close(projection.weight, before[index], rtol=0, atol=0)
    assert vision_fused.packed_qkv(attn)[0] is weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_layers_on_cuda_stay_as_close_to_fp32_as_eager(dtype: torch.dtype) -> None:
    reference, _ = _towers(device="cuda")
    vpm, _ = _towers(dtype=dtype, device="cuda")
    seq_groups = [(0, 4, 16)]
    hidden = _embeddings(reference, 64)
    with torch.inference_mode():
        expected = _eager(reference, hidden.clone(), seq_groups)
        eager = _eager(vpm, hidden.to(dtype), seq_groups).float()
        fused = vision_fused.encode_packed_fused(vpm, hidden.to(dtype), seq_groups).float()
    eager_error = (eager - expected).abs().max().item()
    fused_error = (fused - expected).abs().max().item()
    assert fused_error <= 1.5 * eager_error + 1e-5, (fused_error, eager_error)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fused", [False, True])
def test_vision_graph_matches_eager_and_ignores_batch_padding(monkeypatch, fused: bool) -> None:
    monkeypatch.setattr(vision_fused, "_CAPTURE_AFTER", 0)
    vpm, resampler = _towers(device="cuda")
    vpm.fused_layers = fused
    height, width = 4, 4
    generator = torch.Generator().manual_seed(2)
    pixels = [torch.randn((3, _PATCH, height * width * _PATCH), generator=generator).cuda() for _ in range(3)]
    layouts = [(height, width)] * 3
    encoder = vision_fused.VisionGraphEncoder(vpm, resampler, batch_sizes=(4,))
    with torch.inference_mode():
        expected = _encode_vision_packed(vpm, resampler, pixels, layouts, 16)
        # Three slices replay the batch-4 graph with one zero slice of padding.
        replayed = _encode_vision_packed(vpm, resampler, pixels, layouts, 16, encoder)
        # A second replay with other pixels in the same graph.
        again = _encode_vision_packed(vpm, resampler, pixels[:1], layouts[:1], 16, encoder)
    assert len(encoder._graphs) == 1
    torch.testing.assert_close(replayed, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(again, expected[:1], rtol=1e-4, atol=1e-4)
