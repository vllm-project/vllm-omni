# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
@pytest.mark.parametrize("precision", ["bf16", "sage"])
def test_h3_auto_provider_matches_explicit_flashinfer(precision):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout, VideoTokenSpan
    from vllm_omni.diffusion.attention.ops.block_sparse import resolve_block_sparse_provider
    from vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3 import MiniMaxH3VSAImpl
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    q = torch.randn(1, 133, 2, 128, device="cuda", dtype=torch.bfloat16)
    metadata = AttentionMetadata(
        extra={"vsa_h3_prefix_segments": (5,)},
        video_layout=VideoTokenLayout(
            used_len=133, video_spans=(VideoTokenSpan(start=5, latent_grid=(8, 4, 4), role="target"),)
        ),
    )
    kwargs = dict(num_heads=2, head_size=128, softmax_scale=128**-0.5)
    auto = MiniMaxH3VSAImpl(**kwargs, backend_kwargs={"precision": precision, "topk": 1})
    explicit = MiniMaxH3VSAImpl(**kwargs, backend_kwargs={"provider": "flashinfer", "precision": precision, "topk": 1})
    assert auto.provider == "auto"
    assert resolve_block_sparse_provider(auto.provider, precision, q.device, q.dtype, 128) == "flashinfer"
    actual = auto.forward_cuda(q, q, q, metadata)
    expected = explicit.forward_cuda(q, q, q, metadata)
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
@pytest.mark.parametrize("rows,keys", [(128, 192), (129, 191), (256, 256)])
@pytest.mark.parametrize("precision", ["sage", "bf16"])
def test_sage_sparse_against_masked_sdpa(rows, keys, precision):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    from vllm_omni.diffusion.attention.ops.flashinfer_block_sparse import flashinfer_block_sparse_attention

    generator = torch.Generator(device="cuda").manual_seed(7415)
    q = torch.randn(1, rows, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(1, keys, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(1, keys, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    blocks = (keys + 63) // 64
    indices = torch.arange(blocks, device="cuda", dtype=torch.int32).repeat(1, 2, (rows + 63) // 64, 1)
    counts = torch.full(indices.shape[:3], blocks, device="cuda", dtype=torch.int32)
    counts[..., 1::2] -= 1
    sizes = torch.full((blocks,), 64, device="cuda", dtype=torch.int32)
    sizes[-1] = keys - (blocks - 1) * 64
    mask = torch.arange(keys, device="cuda")[None, :] < counts[0, 0].repeat_interleave(64)[:rows, None] * 64
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), attn_mask=mask
    ).transpose(1, 2)
    block_map = torch.arange(blocks, device="cuda")[None, None, None, :] < counts[..., None]
    actual = flashinfer_block_sparse_attention(q, k, v, block_map, sizes, 128**-0.5, precision=precision)
    torch.accelerator.synchronize()
    assert actual.shape == q.shape and actual.dtype == q.dtype
    relative_rms = (actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rms.item() < (0.08 if precision == "sage" else 0.01)
    assert torch.isfinite(actual).all()


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
@pytest.mark.parametrize("precision", ["sage", "bf16"])
@pytest.mark.parametrize("size_rank", [0, 1, 2, 3])
def test_generic_rectangular_sparse_layout_against_sdpa(precision, size_rank):
    """Exercise per-batch/head sparsity without any H3 layout or prefix."""
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

    generator = torch.Generator(device="cuda").manual_seed(7989)
    q = torch.randn(2, 129, 3, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(2, 223, 3, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    block_map = torch.rand(2, 3, 3, 4, device="cuda", generator=generator) > 0.5
    block_map[..., 0] = True
    block_map[0, 0, 0] = False  # Empty selections must return finite zeros.
    sizes = None
    if size_rank:
        sizes = torch.tensor([63, 17, 64, 31], device="cuda", dtype=torch.int32)
        if size_rank >= 2:
            sizes = sizes.repeat(2, 1)
            sizes[1, 1] = 9
        if size_rank == 3:
            sizes = sizes[:, None, :].repeat(1, 3, 1)
            sizes[:, 1, 2] = 29
    mask = block_map.repeat_interleave(64, -2).repeat_interleave(64, -1)[..., :129, :223]
    if sizes is not None:
        expanded = sizes.reshape((1, 1, 4) if size_rank == 1 else (2, 1, 4) if size_rank == 2 else (2, 3, 4))
        positions = torch.arange(223, device="cuda")
        valid_keys = positions % 64 < expanded.repeat_interleave(64, -1)[..., :223]
        mask = mask & valid_keys.unsqueeze(-2)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), attn_mask=mask, scale=0.125
    ).transpose(1, 2)
    actual = block_sparse_attn_bshd(q, k, v, block_map, sizes, 0.125, provider="flashinfer", precision=precision)
    torch.accelerator.synchronize()
    assert actual.shape == q.shape and actual.dtype == q.dtype and actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0, :64, 0]) == 0
    relative_rms = (actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rms.item() < (0.08 if precision == "sage" else 0.01)


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
def test_prepared_query_matches_inline_quantization():
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    from vllm_omni.diffusion.attention.ops.block_sparse import block_map_to_indices
    from vllm_omni.diffusion.attention.ops.sage_block_sparse_attention import sage_block_sparse_attention
    from vllm_omni.diffusion.attention.ops.sage_quantization import quantize_sage_q_sm120

    generator = torch.Generator(device="cuda").manual_seed(7989)
    q = torch.randn(1, 129, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(1, 191, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    block_map = torch.ones(1, 2, 3, 3, device="cuda", dtype=torch.bool)
    block_map[..., 1] = False
    indices, counts = block_map_to_indices(block_map)
    prepared = quantize_sage_q_sm120(q.transpose(1, 2))
    inline = sage_block_sparse_attention(q, k, v, indices, counts, None, 0.125)
    reused = sage_block_sparse_attention(q, k, v, indices, counts, None, 0.125, prepared_q=prepared)
    torch.testing.assert_close(inline, reused, rtol=0, atol=0)


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
def test_sage_value_layout_preserves_partial_tile_token_order():
    """Pin the V packing ABI using distinguishable tokens and partial tiles."""
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

    q = torch.zeros(1, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 192, 2, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(192, device="cuda")
    features = torch.arange(128, device="cuda")
    v = ((positions[:, None] % 64) / 16 + features[None, :] / 128).to(torch.bfloat16)
    v = v[None, :, None, :].expand(1, 192, 2, 128).contiguous()
    sizes = torch.tensor([17, 33, 3], device="cuda", dtype=torch.int32)
    block_map = torch.eye(3, device="cuda", dtype=torch.bool)[None, None].expand(1, 2, 3, 3).contiguous()
    mask = block_map.repeat_interleave(64, -2).repeat_interleave(64, -1)[..., :129, :192]
    mask &= (positions % 64 < sizes.repeat_interleave(64))[None, None, None, :]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), attn_mask=mask
    ).transpose(1, 2)
    actual = block_sparse_attn_bshd(q, k, v, block_map, sizes, 128**-0.5, provider="flashinfer", precision="sage")
    torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.04)
