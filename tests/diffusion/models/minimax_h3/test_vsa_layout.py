# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import math
import sys
import types
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
    VideoTokenSpan,
)
from vllm_omni.diffusion.models.minimax_h3.attention.vsa import (
    MiniMaxH3VSAImpl,
    _build_h3_block_map,
    _get_h3_tile_metadata,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _h3_metadata(prefix_segments, video_shape, *, gate=None, packed_padding=None):
    extra: dict[str, Any] = {
        "vsa_h3_prefix_segments": prefix_segments,
    }
    if gate is not None:
        extra["gate_compress"] = gate
    return AttentionMetadata(
        packed_padding=packed_padding,
        video_layout=VideoTokenLayout(
            used_len=sum(prefix_segments) + math.prod(video_shape),
            video_spans=(VideoTokenSpan(start=sum(prefix_segments), latent_grid=video_shape, role="target"),),
        ),
        extra=extra,
    )


def _fake_block_sparse_kernel(monkeypatch, calls):
    """Stand in for fastvideo_kernel.block_sparse_attn with a mask-obeying SDPA."""

    def block_sparse_attn(q, k, v, block_map, variable_block_sizes):
        calls["block_map"] = block_map.clone()
        calls["variable_block_sizes"] = variable_block_sizes.tolist()
        calls["q_shape"] = tuple(q.shape)
        blocks = block_map.shape[-1]
        assert blocks * 64 == q.shape[2]
        # Expand the per-block map to a per-token mask so the stand-in honours
        # the sparsity contract the real kernel implements, and drop the rows
        # each partial block pads with, which the real kernel skips through
        # variable_block_sizes.
        token_mask = block_map.repeat_interleave(64, dim=-1).repeat_interleave(64, dim=-2)
        within = torch.arange(64, device=q.device)[None, :] < variable_block_sizes[:, None]
        token_mask = token_mask & within.reshape(1, 1, 1, -1)
        out = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=token_mask)
        return out.to(q.dtype), None

    module = types.ModuleType("fastvideo_kernel.block_sparse_attn")
    setattr(module, "block_sparse_attn", block_sparse_attn)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", types.ModuleType("fastvideo_kernel"))
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn", module)


def _h3_impl(**backend_kwargs):
    kwargs = {"topk": 1, "min_seq_len": 1, "disable_when_sp_active": False}
    kwargs.update(backend_kwargs)
    return MiniMaxH3VSAImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5, backend_kwargs=kwargs)


def test_h3_forward_tiles_untiles_and_restores_the_original_row_order(monkeypatch):
    calls: dict[str, Any] = {}
    _fake_block_sparse_kernel(monkeypatch, calls)
    impl = _h3_impl(topk=64)
    prefix_segments, video_shape = (5, 70), (2, 4, 4)
    seq_len = sum(prefix_segments) + 2 * 4 * 4
    torch.manual_seed(0)
    query = torch.randn(1, seq_len, 2, 8, dtype=torch.bfloat16)

    output = impl.forward_cuda(query, query, query, _h3_metadata(prefix_segments, video_shape))

    assert output.shape == query.shape
    # topk covers every video block, so the block map is dense and the tiled
    # kernel must reproduce plain attention over the same rows.
    reference = torch.nn.functional.scaled_dot_product_attention(
        *(x.float().transpose(1, 2) for x in (query, query, query))
    ).transpose(1, 2)
    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)
    # 5 + 70 splits into segment-pure 5 | 64 | 6, then one (2, 4, 4) video tile.
    assert calls["variable_block_sizes"] == [5, 64, 6, 32]


def test_h3_forward_keeps_prefix_dense_and_selects_top_k_video_tiles(monkeypatch):
    calls: dict[str, Any] = {}
    _fake_block_sparse_kernel(monkeypatch, calls)
    impl = _h3_impl(topk=1)
    prefix_segments, video_shape = (5,), (8, 4, 4)
    seq_len = 5 + 8 * 4 * 4
    torch.manual_seed(0)
    query = torch.randn(1, seq_len, 2, 8, dtype=torch.bfloat16)

    impl.forward_cuda(query, query, query, _h3_metadata(prefix_segments, video_shape))

    # 1 prefix block + 2 video blocks. The odd count is padded to an even one
    # for the paired-CTA kernel contract, and the Triton route slices that
    # transport-only partner off again before launching.
    block_map = calls["block_map"]
    assert block_map.shape[-1] == 3
    assert calls["variable_block_sizes"] == [5, 64, 64]
    assert block_map[:, :, :1, :].all(), "prefix queries stay dense"
    assert block_map[..., :1].all(), "prefix keys are exempt from selection"
    assert (block_map[:, :, 1:, 1:].sum(dim=-1) == 1).all(), "one video tile per video query"

    # The same odd geometry must still untile back onto the original rows.
    dense = _h3_impl(topk=64).forward_cuda(query, query, query, _h3_metadata(prefix_segments, video_shape))
    reference = torch.nn.functional.scaled_dot_product_attention(
        *(x.float().transpose(1, 2) for x in (query, query, query))
    ).transpose(1, 2)
    torch.testing.assert_close(dense.float(), reference, atol=2e-2, rtol=2e-2)


def test_h3_forward_applies_the_learned_compression_gate(monkeypatch):
    _fake_block_sparse_kernel(monkeypatch, {})
    impl = _h3_impl(topk=64)
    prefix_segments, video_shape = (5,), (2, 4, 4)
    seq_len = 5 + 2 * 4 * 4
    torch.manual_seed(0)
    query = torch.randn(1, seq_len, 2, 8, dtype=torch.bfloat16)

    without_gate = impl.forward_cuda(query, query, query, _h3_metadata(prefix_segments, video_shape))
    zero_gate = torch.zeros_like(query)
    with_zero_gate = impl.forward_cuda(query, query, query, _h3_metadata(prefix_segments, video_shape, gate=zero_gate))
    live_gate = torch.full_like(query, 0.5)
    with_live_gate = impl.forward_cuda(query, query, query, _h3_metadata(prefix_segments, video_shape, gate=live_gate))

    torch.testing.assert_close(with_zero_gate.float(), without_gate.float())
    assert not torch.allclose(with_live_gate.float(), without_gate.float())


def test_h3_forward_restores_rows_the_packed_padding_excludes(monkeypatch):
    _fake_block_sparse_kernel(monkeypatch, {})
    impl = _h3_impl(topk=64)
    prefix_segments, video_shape = (5,), (2, 4, 4)
    valid = 5 + 2 * 4 * 4
    torch.manual_seed(0)
    query = torch.randn(1, valid + 24, 2, 8, dtype=torch.bfloat16)
    padding = PackedPaddingMetadata(
        q_length=valid,
        kv_length=valid,
        cu_seqlens_q=torch.tensor([0, valid], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, valid], dtype=torch.int32),
    )

    output = impl.forward_cuda(query, query, query, _h3_metadata(prefix_segments, video_shape, packed_padding=padding))

    assert output.shape == query.shape
    assert torch.count_nonzero(output[:, valid:]) == 0


def test_sdpa_fallback_never_attends_the_structural_padding(monkeypatch):
    # The backend advertises supports_packed_mask_free, so MiniMax-H3 skips
    # building the padding mask. Every fallback out of the VSA path must honour
    # that contract itself; SDPA reads attn_mask and nothing else.
    impl = MiniMaxH3VSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        backend_kwargs={"topk": 1, "min_seq_len": 4096},
    )
    monkeypatch.setattr(impl.sdpa_fallback, "forward", impl.sdpa_fallback.forward_cuda)
    valid = 40
    torch.manual_seed(0)
    query = torch.randn(1, 64, 2, 8)
    padding = PackedPaddingMetadata(
        q_length=valid,
        kv_length=valid,
        cu_seqlens_q=torch.tensor([0, valid], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, valid], dtype=torch.int32),
    )

    baseline = impl.forward_cuda(query, query, query, AttentionMetadata(packed_padding=padding, extra={}))
    perturbed_input = query.clone()
    perturbed_input[:, valid:] += 100.0
    perturbed = impl.forward_cuda(
        perturbed_input, perturbed_input, perturbed_input, AttentionMetadata(packed_padding=padding, extra={})
    )

    assert baseline.shape == query.shape
    torch.testing.assert_close(baseline[:, :valid], perturbed[:, :valid])
    assert torch.count_nonzero(baseline[:, valid:]) == 0


def test_h3_geometry_keeps_prefix_segments_pure_and_tiles_video_3d():
    partition, sizes, non_pad, untile, prefix_blocks, video_blocks = _get_h3_tile_metadata(
        (5, 70, 9), (5, 6, 6), torch.device("cpu")
    )
    assert sizes[:4].tolist() == [5, 64, 6, 9]
    assert prefix_blocks == 4
    assert video_blocks == 8
    assert int(sizes.sum()) == 5 + 70 + 9 + 5 * 6 * 6
    assert partition.numel() == non_pad.numel() == untile.numel() == int(sizes.sum())


def test_h3_block_map_makes_prefix_queries_dense_and_prefix_keys_exempt():
    scores = torch.arange(1 * 2 * 5 * 5, dtype=torch.float32).reshape(1, 2, 5, 5)
    block_map = _build_h3_block_map(scores, num_prefix_blocks=2, num_video_blocks=3, topk=1)
    assert block_map[:, :, :2].all()
    assert block_map[..., :2].all()
    assert (block_map[:, :, 2:, 2:].sum(dim=-1) == 1).all()


@pytest.mark.parametrize("fallback_on_error", [False, True])
def test_h3_kernel_failure_preserves_fallback_policy(monkeypatch, fallback_on_error):
    impl = _h3_impl(fallback_on_error=fallback_on_error)
    query = torch.randn(1, 37, 2, 8)
    metadata = _h3_metadata((5,), (2, 4, 4))

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(impl, "_forward_h3", fail)
    monkeypatch.setattr(impl.sdpa_fallback, "forward", impl.sdpa_fallback.forward_cuda)
    if fallback_on_error:
        output = impl.forward_cuda(query, query, query, metadata)
        expected = impl.sdpa_fallback.forward_cuda(query, query, query, metadata)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    else:
        with pytest.raises(RuntimeError, match="kernel failure"):
            impl.forward_cuda(query, query, query, metadata)


def test_h3_accelerator_failure_is_not_swallowed(monkeypatch):
    impl = _h3_impl(fallback_on_error=True)
    query = torch.randn(1, 37, 2, 8)

    def fail(*args, **kwargs):
        raise torch.AcceleratorError("device failure")

    monkeypatch.setattr(impl, "_forward_h3", fail)
    with pytest.raises(torch.AcceleratorError, match="device failure"):
        impl.forward_cuda(query, query, query, _h3_metadata((5,), (2, 4, 4)))


def test_h3_without_layout_delegates_to_shared_backend(monkeypatch):
    from vllm_omni.diffusion.attention.backends.fastvideo_vsa import FastVideoVSAImpl

    impl = _h3_impl()
    query = torch.randn(1, 37, 2, 8)
    metadata = AttentionMetadata()
    expected = torch.empty_like(query)
    calls = []

    def forward(self, q, k, v, attn_metadata):
        calls.append((self, q, k, v, attn_metadata))
        return expected

    monkeypatch.setattr(FastVideoVSAImpl, "forward_cuda", forward)
    assert impl.forward_cuda(query, query, query, metadata) is expected
    assert len(calls) == 1
    assert all(actual is wanted for actual, wanted in zip(calls[0], (impl, query, query, query, metadata)))
