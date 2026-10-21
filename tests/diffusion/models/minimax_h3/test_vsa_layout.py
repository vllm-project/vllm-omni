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
from vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3 import (
    MiniMaxH3VSAImpl,
    _build_h3_region_block_map,
    _get_h3_multiregion_tile_metadata,
    _get_h3_tile_metadata,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("topk", [0, -1])
def test_h3_rejects_nonpositive_topk_at_construction(topk):
    with pytest.raises(ValueError, match="topk must be positive"):
        MiniMaxH3VSAImpl(num_heads=2, head_size=128, softmax_scale=128**-0.5, backend_kwargs={"topk": topk})


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


@pytest.mark.parametrize("topk", [1, 64])
def test_v2_uses_trained_sparsity_instead_of_legacy_fixed_topk(monkeypatch, topk):
    calls: dict[str, object] = {}
    _fake_block_sparse_kernel(monkeypatch, calls)
    metadata = _h3_metadata((5,), (8, 8, 8))
    metadata.extra.update(vsa_h3_tile_size=64, vsa_h3_sparsity=0.8)
    query = torch.randn(1, 517, 2, 8, dtype=torch.bfloat16)
    output = _h3_impl(topk=topk).forward_cuda(query, query, query, metadata)
    mask = calls["block_map"]
    assert isinstance(mask, torch.Tensor)
    assert mask.shape[-1] == 9
    assert mask[..., :1].all() and mask[..., :1, :].all()
    assert (mask[..., 1:, 1:].sum(-1) == 2).all()  # ceil(20% of eight video tiles)
    assert output.shape == query.shape and torch.isfinite(output).all()


@pytest.mark.parametrize("failure", ["missing-layout", "unequal-lengths", "kernel"])
def test_v2_never_replaces_trained_sparse_attention_with_dense(monkeypatch, failure):
    impl = _h3_impl(fallback_on_error=True)
    metadata = _h3_metadata((5,), (2, 4, 4))
    metadata.extra.update(vsa_h3_tile_size=64, vsa_h3_sparsity=0.8)
    query = torch.randn(1, 37, 2, 8, dtype=torch.bfloat16)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("V2 must preserve the trained sparse attention contract")

    monkeypatch.setattr(impl, "_fallback", unexpected_fallback)
    expected_error: type[Exception]
    if failure == "missing-layout":
        metadata.video_layout = None
        expected_error, match = ValueError, "complete packed"
    elif failure == "unequal-lengths":
        metadata.packed_padding = PackedPaddingMetadata(
            q_length=37,
            kv_length=36,
            cu_seqlens_q=torch.tensor([0, 37], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 36], dtype=torch.int32),
        )
        expected_error, match = ValueError, "Q/KV lengths"
    else:

        def fail(*args, **kwargs):
            raise RuntimeError("V2 kernel failure")

        monkeypatch.setattr(impl, "_forward_h3", fail)
        expected_error, match = RuntimeError, "V2 kernel failure"

    with pytest.raises(expected_error, match=match):
        impl.forward_cuda(query, query, query, metadata)


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


def test_h3_geometry_keeps_prefix_segments_pure_and_tiles_video_3d():
    partition, sizes, non_pad, untile, prefix_blocks, video_blocks = _get_h3_tile_metadata(
        (5, 70, 9), (5, 6, 6), torch.device("cpu")
    )
    assert sizes[:4].tolist() == [5, 64, 6, 9]
    assert prefix_blocks == 4
    assert video_blocks == 8
    assert int(sizes.sum()) == 5 + 70 + 9 + 5 * 6 * 6
    assert partition.numel() == non_pad.numel() == untile.numel() == int(sizes.sum())


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


def test_omni_multiregion_geometry_preserves_interleaved_reference_order():
    prefix = (5, 512, 3, 7, 128, 9)
    regions = ((5, (8, 4, 16)), (527, (4, 4, 8)), (664, (4, 4, 24)))
    partition, sizes, non_pad, untile, dense, spans = _get_h3_multiregion_tile_metadata(
        prefix, regions, torch.device("cpu")
    )
    assert sizes[:dense].tolist() == [5, 3, 7, 9]
    assert spans == ((4, 8), (8, 9), (9, 12))
    original = torch.arange(1048)
    tiled = torch.full((sizes.numel() * 128,), -1)
    tiled[non_pad] = original[partition]
    assert torch.equal(tiled[untile], original)


@pytest.mark.parametrize("reference_kinds", [("image", "image"), ("audio", "audio"), ("audio",)])
def test_omni_production_branch_preserves_reference_pooling_boundaries(reference_kinds):
    from vllm_omni.diffusion.attention.ops.block_sparse import mean_pool_tiles
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks

    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=5,
        latent_t=4,
        latent_h=8,
        latent_w=16,
        audio_t=32,
        ref_blocks=[
            {"kind": kind, "latent_h": 16, "latent_w": 16} if kind == "image" else {"kind": kind, "ref_audio_t": 32}
            for kind in reference_kinds
        ],
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(5, 4),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    prefix = branch.static_kwargs["packed_seq_params"]["vsa_prefix_segments"]
    layout = branch.static_kwargs["video_token_layout"]
    regions = tuple((span.start, span.latent_grid) for span in layout.video_spans)
    partition, sizes, non_pad, untile, dense, _ = _get_h3_multiregion_tile_metadata(
        prefix, regions, torch.device("cpu")
    )
    # Text, each reference and target audio must pool independently, even
    # when adjacent segments have the same modality and fit in one tile.
    expected_sizes = [5, *([64] * (len(reference_kinds) + 1)), 128]
    rows = torch.repeat_interleave(torch.arange(len(expected_sizes), dtype=torch.float32), torch.tensor(expected_sizes))
    tiled = torch.zeros(sizes.numel() * 128)
    tiled[non_pad] = rows[partition]
    pooled = mean_pool_tiles(tiled.view(1, -1, 1, 1), sizes, 128).flatten()
    assert sizes.tolist() == expected_sizes
    assert dense == len(expected_sizes) - 1
    torch.testing.assert_close(pooled, torch.arange(len(expected_sizes), dtype=torch.float32))
    torch.testing.assert_close(tiled[untile], rows)


def test_omni_selects_a_separate_budget_from_every_reference_region():
    torch.manual_seed(43)
    scores = torch.randn(1, 2, 40, 40)
    spans = ((3, 23), (23, 28), (28, 40))
    mask = _build_h3_region_block_map(scores, 3, spans, (0.1, 0.1, 0.1))
    assert mask[..., :3].all()
    assert mask[..., :3, :].all()
    for (start, end), expected in zip(spans, (2, 1, 2)):
        assert (mask[..., 3:, start:end].sum(-1) == expected).all()


@pytest.mark.parametrize("provider,precision", [("flashinfer", "sage"), ("auto", "sage")])
def test_omni_tile128_rejects_non_bf16_precision(monkeypatch, provider, precision):
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.ops.flashinfer_block_sparse.require_flashinfer_sparse",
        lambda *args: None,
    )
    impl = _h3_impl(provider=provider, precision=precision)
    metadata = _h3_metadata((5,), (4, 4, 8))
    metadata.extra["vsa_h3_tile_size"] = 128
    query = torch.zeros(1, 133, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16"):
        impl.forward_cuda(query, query, query, metadata)


def test_omni_tile128_uses_native_provider_without_opt_in(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3.resolve_block_sparse_provider",
        lambda *args, **kwargs: "fastvideo",
    )
    calls = {}

    def kernel(q, k, v, indices, counts, sizes, **kwargs):
        calls["sizes"] = sizes.tolist()
        calls["mask"] = indices.bool()
        mask = indices.bool().repeat_interleave(128, -1).repeat_interleave(128, -2)
        valid = (torch.arange(128)[None] < sizes[:, None]).reshape(1, 1, 1, -1)
        out = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask & valid)
        return out.to(q.dtype), None

    native = types.ModuleType("fastvideo_kernel.block_sparse_attn_sm100a")
    setattr(native, "is_supported", lambda *args: True)
    setattr(native, "block_sparse_attn_sm100a", kernel)
    package = types.ModuleType("fastvideo_kernel")
    setattr(package, "block_sparse_attn_sm100a", native)
    index_module = types.ModuleType("fastvideo_kernel.triton_kernels.index")
    setattr(index_module, "map_to_index", lambda mask: (mask, mask.sum(-1)))
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", package)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.triton_kernels", types.ModuleType("triton_kernels"))
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.triton_kernels.index", index_module)
    monkeypatch.delenv("FASTVIDEO_VSA_SM100A", raising=False)

    prefix, target = (5, 512, 3), (4, 4, 24)
    metadata = _h3_metadata(prefix, target)
    metadata.video_layout = VideoTokenLayout(
        used_len=904,
        video_spans=(
            VideoTokenSpan(start=5, latent_grid=(8, 4, 16), role="reference"),
            VideoTokenSpan(start=520, latent_grid=target, role="target"),
        ),
    )
    metadata.extra.update(
        vsa_h3_tile_size=128, vsa_h3_ref_policy="p2_multi_region", vsa_h3_sparsity=0.9, vsa_h3_ref_keep_rate=0.1
    )
    torch.manual_seed(43)
    q = torch.randn(1, 904, 2, 8, dtype=torch.bfloat16)
    output = _h3_impl().forward_cuda(q, q, q, metadata)
    assert output.shape == q.shape
    assert torch.isfinite(output).all()
    assert calls["sizes"] == [5, 3, 128, 128, 128, 128, 128, 128, 128, 0]
    mask = calls["mask"]
    assert not mask[..., -1].any() and not mask[..., -1, :].any()
    assert (mask[..., 2:9, 2:6].sum(-1) == 1).all()
    assert (mask[..., 2:9, 6:9].sum(-1) == 1).all()


def test_omni_tile128_does_not_fall_back_to_dense_on_kernel_failure(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3.resolve_block_sparse_provider",
        lambda *args, **kwargs: "fastvideo",
    )
    impl = _h3_impl(fallback_on_error=True)
    metadata = _h3_metadata((5,), (2, 4, 4))
    metadata.extra["vsa_h3_tile_size"] = 128

    def fail(*args, **kwargs):
        raise RuntimeError("native tile128 unavailable")

    monkeypatch.setattr(impl, "_forward_h3", fail)
    q = torch.randn(1, 37, 2, 8)
    with pytest.raises(RuntimeError, match="tile128 unavailable"):
        impl.forward_cuda(q, q, q, metadata)


@pytest.mark.parametrize("transport_padding", [False, True])
def test_omni_sm120_decomposition_preserves_the_original_token_mask(monkeypatch, transport_padding):
    from vllm_omni.diffusion.attention.ops import flashinfer_block_sparse
    from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

    torch.manual_seed(7)
    sizes = torch.tensor([128, 17, 64, 65, 0], dtype=torch.int32)
    blocks = sizes.numel()
    mask128 = torch.rand(2, 2, blocks, blocks) < 0.4
    mask128[..., 0] = True
    mask128[..., 0, :] = True
    mask128[..., -1, :] = False
    q, k, v = [torch.randn(2, blocks * 128, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    # A large value on a padded key catches incorrect handling of partial tiles.
    v[:, 128 + 17 : 256] = 100
    valid = (torch.arange(128)[None] < sizes[:, None]).flatten()
    expected_mask = mask128.repeat_interleave(128, -2).repeat_interleave(128, -1)
    expected_mask &= valid.view(1, 1, 1, -1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(x.transpose(1, 2) for x in (q, k, v)), attn_mask=expected_mask
    ).transpose(1, 2)
    calls = []

    def kernel(q, k, v, indices, capacity, *, block_sizes, q2k_block_nums, softmax_scale):
        calls.append(True)
        assert softmax_scale == 128**-0.5
        n = block_sizes.numel()
        mask = torch.zeros(*indices.shape[:3], n, dtype=torch.bool)
        for b in range(indices.shape[0]):
            for h in range(indices.shape[1]):
                for row in range(indices.shape[2]):
                    selected = indices[b, h, row, : q2k_block_nums[b, h, row]]
                    assert (selected >= 0).all()
                    assert (block_sizes[selected] > 0).all()
                    mask[b, h, row, selected] = True
        token_mask = mask.repeat_interleave(64, -2).repeat_interleave(64, -1)
        valid = (torch.arange(64)[None] < block_sizes[:, None]).flatten()
        token_mask &= valid.view(1, 1, 1, -1)
        torch.testing.assert_close(token_mask, expected_mask)
        return torch.nn.functional.scaled_dot_product_attention(
            *(x.transpose(1, 2) for x in (q, k, v)), attn_mask=token_mask
        ).transpose(1, 2), None

    monkeypatch.setattr(flashinfer_block_sparse, "require_flashinfer_sparse", lambda *args: kernel)
    if transport_padding:
        q, k, v = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 128)) for x in (q, k, v)]
        mask128 = torch.nn.functional.pad(mask128, (0, 1, 0, 1))
        sizes = torch.nn.functional.pad(sizes, (0, 1))
        expected = torch.nn.functional.pad(expected, (0, 0, 0, 0, 0, 128))
        expected_mask = torch.nn.functional.pad(expected_mask, (0, 128, 0, 128))

    actual = block_sparse_attn_bshd(q, k, v, mask128, sizes, 128**-0.5, provider="flashinfer", block_size=128)
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1, "all selected KV blocks must share a single attention reduction"


@pytest.mark.parametrize("error_type", [RuntimeError, torch.AcceleratorError])
def test_omni_sm120_flashinfer_error_is_not_swallowed(monkeypatch, error_type):
    from vllm_omni.diffusion.attention.ops import flashinfer_block_sparse
    from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

    def fail(*args, **kwargs):
        raise error_type("FlashInfer kernel failed")

    monkeypatch.setattr(flashinfer_block_sparse, "require_flashinfer_sparse", lambda *args: fail)
    q = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(error_type, match="FlashInfer kernel failed"):
        block_sparse_attn_bshd(
            q,
            q,
            q,
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            torch.tensor([128], dtype=torch.int32),
            128**-0.5,
            provider="flashinfer",
            block_size=128,
        )


def test_omni_sm120_missing_flashinfer_has_actionable_error(monkeypatch):
    from vllm_omni.diffusion.attention.ops import flashinfer_block_sparse
    from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

    def missing_provider(*args):
        raise ImportError("FlashInfer bf16 requires a build with bsa_attn_sm120_blk64_fwd")

    monkeypatch.setattr(flashinfer_block_sparse, "require_flashinfer_sparse", missing_provider)
    q = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ImportError, match="requires a build with bsa_attn_sm120_blk64_fwd"):
        block_sparse_attn_bshd(
            q,
            q,
            q,
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            torch.tensor([128], dtype=torch.int32),
            128**-0.5,
            provider="flashinfer",
            block_size=128,
        )
