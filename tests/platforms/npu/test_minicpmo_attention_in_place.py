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
@torch.inference_mode()
def test_attention_cache_and_replay(cached, alias, mask_kind, qkv_bias):
    decoder = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    torch.manual_seed(42)
    batch, frames, heads, dim = 2, 6, 2, 16
    attn = decoder.Attention(heads * dim, heads, dim, qkv_bias=qkv_bias, qk_norm=True).npu().eval()
    x = torch.randn(batch, frames, heads * dim, device="npu")
    # _attend_in_place keeps a split (2, N, H, frames, D) K/V buffer:
    # K in the leading half, V in the trailing one.
    kv = torch.randn(2, batch, heads, frames + cached, dim, device="npu")
    span = frames + cached
    cache = kv[:, :, :, frames:span] if cached else None
    if cache is not None and not alias:
        cache = cache.clone()

    def packed(cache):
        # The eager reference still takes the packed [K | V] cache width.
        return None if cache is None else torch.cat((cache[0], cache[1]), dim=-1)

    mask = None
    if mask_kind != "none":
        mask = torch.ones(batch, 1 if mask_kind == "padding" else frames, span, device="npu", dtype=torch.bool)
        mask[..., -2:] = False

    with npu_token2wav_sdpa_context(require_math=True):
        expected, expected_cache = attn.forward_chunk(x, packed(cache), mask)
        actual = BatchedToken2Wav._attend_in_place(attn, x, cache, mask, kv)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(packed(kv[:, :, :, :span]), expected_cache)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = BatchedToken2Wav._attend_in_place(attn, x, cache, mask, kv)
        # Change both graph inputs, including the aliased cache, between replays.
        for _ in range(2):
            x.copy_(torch.randn_like(x))
            if cache is not None:
                cache.copy_(torch.randn_like(cache))
            expected, expected_cache = attn.forward_chunk(x, packed(cache), mask)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(replayed, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(packed(kv[:, :, :, :span]), expected_cache)


@pytest.mark.parametrize("cached", [0, 17])
@torch.inference_mode()
def test_attention_rejects_a_buffer_wider_than_the_live_span(cached):
    decoder = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    torch.manual_seed(42)
    batch, frames, heads, dim = 2, 6, 2, 16
    attn = decoder.Attention(heads * dim, heads, dim, qkv_bias=False, qk_norm=True).npu().eval()
    x = torch.randn(batch, frames, heads * dim, device="npu")
    # A padded buffer would attend tail frames nobody wrote. The refusal now
    # runs before either write branch (the in-place cache copy and the cat
    # path), so the whole buffer must stay untouched -- snapshot it instead of
    # only the tail, which is never written in either order.
    kv = torch.randn(2, batch, heads, frames + cached + 3, dim, device="npu")
    span = frames + cached
    before = kv.clone()
    cache = kv[:, :, :, frames:span] if cached else None
    with npu_token2wav_sdpa_context(require_math=True):
        with pytest.raises(ValueError, match="live span"):
            BatchedToken2Wav._attend_in_place(attn, x, cache, None, kv)
        torch.testing.assert_close(kv, before)


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
    kv = torch.empty(2, 2, 2, 6, 16, device="npu")
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


@pytest.mark.parametrize("batch,frames", [(2, 50), (4, 100), (8, 300)])
@pytest.mark.parametrize("affine", [False, True])
@torch.inference_mode()
def test_qkv_norm_npu_graph_matches_native(batch, frames, affine):
    from vllm_omni.model_executor.models.minicpmo_4_5.ops.fused_qkv_layer_norm import (
        _use_npu_triton,
        qkv_head_layer_norm,
    )

    prepare_code2wav_graph_runtime()
    torch.manual_seed(42)
    heads, dim, history = 8, 64, 17
    # Exercise explicit strides, including a cache capacity larger than the live span.
    qkv = torch.randn(batch, frames, 3 * heads * dim + 16, device="npu")[..., : 3 * heads * dim]
    kv = torch.randn(2, batch, heads, frames + history + 9, dim, device="npu")[:, :, :, :-9]
    q_norm = torch.nn.LayerNorm(dim, eps=1e-6, elementwise_affine=affine, device="npu")
    k_norm = torch.nn.LayerNorm(dim, eps=1e-5, elementwise_affine=affine, device="npu")
    assert _use_npu_triton(qkv, kv, q_norm, k_norm, dim), "This test must exercise the NPU Triton kernel"

    def run():
        return qkv_head_layer_norm(qkv, kv, num_heads=heads, head_dim=dim, q_norm=q_norm, k_norm=k_norm)

    run()  # Compile before capture.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        query = run()
    for _ in range(2):
        qkv.normal_()
        kv.normal_()
        if affine:
            for norm in (q_norm, k_norm):
                norm.weight.uniform_(0.5, 1.5)
                norm.bias.uniform_(-0.5, 0.5)
        packed = qkv.view(batch, frames, 3, heads, dim)
        expected_q = q_norm(packed[:, :, 0].transpose(1, 2))
        expected_k = k_norm(packed[:, :, 1].transpose(1, 2))
        expected_v = packed[:, :, 2].transpose(1, 2).clone()
        old_history = kv[:, :, :, frames:].clone()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(query, expected_q, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(kv[0, :, :, :frames], expected_k, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(kv[1, :, :, :frames], expected_v, atol=0, rtol=0)
        torch.testing.assert_close(kv[:, :, :, frames:], old_history, atol=0, rtol=0)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("qkv_bias", [False, True])
@pytest.mark.parametrize("cached", [0, 17])
@pytest.mark.parametrize("mask_kind", ["none", "ragged"])
@torch.inference_mode()
def test_fused_qkv_attention_npu_graph_preserves_cache_and_mask(monkeypatch, alias, qkv_bias, cached, mask_kind):
    from vllm_omni.platforms.npu.models import minicpmo_fused

    decoder = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    torch.manual_seed(42)
    batch, frames, heads, dim = 2, 50, 8, 64
    gap, tail = (13, 9) if mask_kind == "ragged" else (0, 0)
    attn = decoder.Attention(heads * dim, heads, dim, qkv_bias=qkv_bias, qk_norm=True).npu().eval()
    x = torch.randn(batch, frames, heads * dim, device="npu")
    kv = torch.randn(2, batch, heads, frames + cached + gap + tail, dim, device="npu")
    cache = kv[:, :, :, frames : frames + cached] if cached else None
    if cache is not None and not alias:
        cache = cache.clone()
    mask = None
    if mask_kind == "ragged":
        mask = torch.ones(batch, frames, frames + cached + gap, device="npu", dtype=torch.bool)
        mask[..., frames + cached :] = False
        mask[1, -3:, :3] = False  # Per-query masks leave other valid keys.

    with npu_token2wav_sdpa_context(require_math=True):
        monkeypatch.setattr(attn, "_npu_fused_qkv_norm", True, raising=False)
        minicpmo_fused.attention_npu(attn, x, cache, mask, kv)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = minicpmo_fused.attention_npu(attn, x, cache, mask, kv)
        for _ in range(2):
            x.normal_()
            kv.normal_()
            if cache is not None and not alias:
                cache.normal_()
            expected_kv = kv.clone()
            monkeypatch.setattr(attn, "_npu_fused_qkv_norm", False)
            expected = minicpmo_fused.attention_npu(attn, x, cache, mask, expected_kv)
            monkeypatch.setattr(attn, "_npu_fused_qkv_norm", True)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(kv, expected_kv, atol=2e-5, rtol=2e-5)
            assert torch.count_nonzero(kv[:, :, :, frames + cached :]).item() == 0


@torch.inference_mode()
def test_qkv_norm_npu_partial_affine_uses_native():
    from vllm_omni.model_executor.models.minicpmo_4_5.ops.fused_qkv_layer_norm import (
        _use_npu_triton,
        qkv_head_layer_norm,
    )

    qkv = torch.randn(2, 6, 3 * 8 * 64, device="npu")
    kv = torch.randn(2, 2, 8, 10, 64, device="npu")
    q_norm = torch.nn.LayerNorm(64, bias=False, device="npu")
    k_norm = torch.nn.LayerNorm(64, device="npu")
    assert not _use_npu_triton(qkv, kv, q_norm, k_norm, 64)
    packed = qkv.view(2, 6, 3, 8, 64)
    old_history = kv[:, :, :, 6:].clone()
    query = qkv_head_layer_norm(qkv, kv, num_heads=8, head_dim=64, q_norm=q_norm, k_norm=k_norm)
    torch.testing.assert_close(query, q_norm(packed[:, :, 0].transpose(1, 2)))
    torch.testing.assert_close(kv[0, :, :, :6], k_norm(packed[:, :, 1].transpose(1, 2)))
    torch.testing.assert_close(kv[1, :, :, :6], packed[:, :, 2].transpose(1, 2), atol=0, rtol=0)
    torch.testing.assert_close(kv[:, :, :, 6:], old_history, atol=0, rtol=0)
