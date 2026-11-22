# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend cache aliasing and mask parity for the shared CFM attention path."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import prepare_code2wav_graph_runtime
from vllm_omni.platforms.npu.models.step_audio2_token2wav import npu_token2wav_sdpa_context

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@pytest.mark.parametrize("cached,alias", [(0, False), (17, False), (17, True), (512, False), (512, True)])
@pytest.mark.parametrize("mask_kind", ["none", "padding", "ragged"])
@pytest.mark.parametrize("qkv_bias", [False, True])
@pytest.mark.parametrize("extra_capacity", [0, 3])
@torch.inference_mode()
def test_attention_cache_and_replay(cached, alias, mask_kind, qkv_bias, extra_capacity):
    decoder = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    torch.manual_seed(42)
    batch, frames, heads, dim = 2, 6, 2, 16
    attn = decoder.Attention(heads * dim, heads, dim, qkv_bias=qkv_bias, qk_norm=True).npu().eval()
    x = torch.randn(batch, frames, heads * dim, device="npu")
    kv = torch.randn(batch, heads, frames + cached + extra_capacity, 2 * dim, device="npu")
    tail = kv[:, :, frames + cached :].clone()
    cache = kv[:, :, frames : frames + cached] if cached else None
    if cache is not None and not alias:
        cache = cache.clone()
    mask = None
    if mask_kind != "none":
        mask = torch.ones(
            batch, 1 if mask_kind == "padding" else frames, frames + cached, device="npu", dtype=torch.bool
        )
        mask[..., -2:] = False

    with npu_token2wav_sdpa_context(require_math=True):
        expected, expected_cache = attn.forward_chunk(x, cache, mask)
        actual = BatchedToken2Wav._attend_in_place(attn, x, cache, mask, kv)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(kv[:, :, : frames + cached], expected_cache)
        torch.testing.assert_close(kv[:, :, frames + cached :], tail)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = BatchedToken2Wav._attend_in_place(attn, x, cache, mask, kv)
        # Change both graph inputs, including the aliased cache, between replays.
        for _ in range(2):
            x.copy_(torch.randn_like(x))
            if cache is not None:
                cache.copy_(torch.randn_like(cache))
            expected, expected_cache = attn.forward_chunk(x, cache, mask)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(replayed, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(kv[:, :, : frames + cached], expected_cache)
            torch.testing.assert_close(kv[:, :, frames + cached :], tail)


@pytest.mark.parametrize("inference", [False, True])
def test_packed_qkv_tracks_weight_and_bias_loads(inference):
    from types import SimpleNamespace

    from vllm_omni.platforms.npu.models.minicpmo_fused import packed_qkv

    with torch.inference_mode(inference), torch.no_grad():
        attn = SimpleNamespace(**{name: torch.nn.Linear(16, 16) for name in ("to_q", "to_k", "to_v")})
        x = torch.randn(2, 5, 16)

        def check():
            weight, bias = packed_qkv(attn)
            expected = torch.cat([getattr(attn, name)(x) for name in ("to_q", "to_k", "to_v")], dim=-1)
            torch.testing.assert_close(torch.nn.functional.linear(x, weight, bias), expected)
            return weight, bias

        weight, bias = check()
        attn.to_q.weight.add_(0.25)
        attn.to_v.bias.add_(0.5)
        reloaded = check()
        assert reloaded[0] is weight and reloaded[1] is bias
        attn.to_k.weight = torch.nn.Parameter(torch.randn_like(attn.to_k.weight))
        assert check()[0] is not weight
        weight, bias = check()
        attn.to_q.bias = torch.nn.Parameter(torch.randn_like(attn.to_q.bias))
        assert check()[1] is not bias


@pytest.mark.parametrize("affine", [False, True])
@torch.inference_mode()
def test_gated_residual_norm_replay(affine):
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import _gated_residual_norm

    prepare_code2wav_graph_runtime()
    norm = torch.nn.LayerNorm(512, eps=1e-6, elementwise_affine=affine).npu().eval()
    if affine:
        norm.weight.normal_()
        norm.bias.normal_()
    x = torch.randn(2, 50, 512, device="npu")
    branch = torch.randn_like(x)
    gate = torch.randn(2, 1, 512, device="npu")
    _gated_residual_norm(x, gate, branch, norm)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        residual, normalized = _gated_residual_norm(x, gate, branch, norm)
    for _ in range(2):
        for tensor in (x, branch, gate):
            tensor.copy_(torch.randn_like(tensor))
        expected = x + gate * branch
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(residual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(normalized, norm(expected), atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def test_original_and_packed_graphs_share_stable_weights():
    from types import SimpleNamespace

    from vllm_omni.platforms.npu.models.minicpmo_fused import prepare_estimator

    decoder = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    attn = decoder.Attention(32, 2, 16, qkv_bias=True, qk_norm=True).npu().eval()
    prepare_estimator(SimpleNamespace(blocks=[SimpleNamespace(attn=attn)]))
    pointers = [p.weight.data_ptr() for p in (attn.to_q, attn.to_k, attn.to_v)]
    x = torch.randn(2, 6, 32, device="npu")
    kv = torch.empty(2, 2, 6, 32, device="npu")
    with npu_token2wav_sdpa_context(require_math=True):
        attn.forward_chunk(x)
        old_graph = torch.npu.NPUGraph()
        with torch.npu.graph(old_graph):
            original, _ = attn.forward_chunk(x)
        BatchedToken2Wav._attend_in_place(attn, x, None, None, kv)
        new_graph = torch.npu.NPUGraph()
        with torch.npu.graph(new_graph):
            packed = BatchedToken2Wav._attend_in_place(attn, x, None, None, kv)
        assert pointers == [p.weight.data_ptr() for p in (attn.to_q, attn.to_k, attn.to_v)]
        for _ in range(2):
            x.copy_(torch.randn_like(x))
            expected, _ = attn.forward_chunk(x)
            old_graph.replay()
            new_graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(original, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(packed, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("affine", [False, True])
@torch.inference_mode()
def test_add_layer_norm_npugraph_matches_reference(affine):
    import torch.nn.functional as F

    from vllm_omni.platforms.npu.models.minicpmo_fused import _add_layer_norm

    prepare_code2wav_graph_runtime()
    rows, frames, channels = 2, 6, 32
    x = torch.randn(rows, frames, channels, device="npu")
    branch = torch.randn(rows, frames, channels, device="npu")
    weight = torch.randn(channels, device="npu") if affine else torch.ones(channels, device="npu")
    bias = torch.randn(channels, device="npu") if affine else torch.zeros(channels, device="npu")
    added_out = torch.empty_like(x)
    norm_out = torch.empty_like(x)

    def run():
        added, normalized = _add_layer_norm(x, branch, weight, bias, 1e-6)
        added_out.copy_(added)
        norm_out.copy_(normalized)

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for _ in range(2):
        x.normal_()
        branch.normal_()
        expected_added = x + branch
        expected_norm = F.layer_norm(expected_added, (channels,), weight, bias, 1e-6)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(added_out, expected_added, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(norm_out, expected_norm, rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def test_convolution_graph_storage_survives_fused_body_warmup():
    from vllm_omni.platforms.npu.models.minicpmo_fused import packed_conv_columns as packed_conv_taps
    from vllm_omni.platforms.npu.models.minicpmo_fused import prepare_estimator

    decoder = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    estimator = (
        decoder.DiT(in_channels=7, out_channels=2, depth=2, num_heads=2, head_dim=8, hidden_size=16, mlp_ratio=2.0)
        .npu()
        .eval()
    )
    conv = estimator.blocks[0].conv.block[1]
    original_pointer = conv.weight.data_ptr()
    prepare_estimator(estimator)
    assert conv.weight.data_ptr() == original_pointer
    prepare_estimator(estimator, pack_convs=True)
    pointer = conv.weight.data_ptr()
    x = torch.randn(2, conv.in_channels, 12, device="npu")
    for _ in range(3):
        conv(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = conv(x)
    packed_conv_taps(conv)
    assert conv.weight.data_ptr() == pointer
    for _ in range(2):
        x.normal_()
        expected = conv(x)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
