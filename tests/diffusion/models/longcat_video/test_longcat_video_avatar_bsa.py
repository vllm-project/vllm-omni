# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.longcat_video.longcat_video_avatar_bsa import flash_attn_bsa_3d

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk_shape", [(4, 4, 8), (4, 4, 16)])
@pytest.mark.parametrize("sparsity", [0.0, 0.875])
def test_longcat_video_avatar_bsa_matches_dense_reference(dtype, chunk_shape, sparsity):
    latent_shape = (8, 8, 16)
    seq_len = math.prod(latent_shape)
    chunk_size = math.prod(chunk_shape)
    num_blocks = seq_len // chunk_size
    head_dim = 128
    generator = torch.Generator(device="cuda").manual_seed(42)
    q, k, v = [torch.randn(2, 2, seq_len, head_dim, device="cuda", dtype=dtype, generator=generator) for _ in range(3)]
    q, k = q * 0.1, k * 0.1

    # Assign block IDs in THW order independently of the kernel's rearrange helpers.
    t, h, w = torch.meshgrid(*(torch.arange(size, device="cuda") for size in latent_shape), indexing="ij")
    ct, ch, cw = chunk_shape
    block_ids = ((t // ct * (latent_shape[1] // ch) + h // ch) * (latent_shape[2] // cw) + w // cw).flatten()
    membership = F.one_hot(block_ids, num_classes=num_blocks).float()
    # Separate block scores so low-precision rounding cannot change the selected top block.
    q[..., :num_blocks] += membership.to(dtype) * 4
    k[..., :num_blocks] += membership.to(dtype) * 4

    if sparsity == 0.0:
        mask = None
    else:
        pooled_q = (torch.einsum("sn,bhsd->bhnd", membership, q.float()) / chunk_size).to(dtype).float()
        pooled_k = (torch.einsum("sn,bhsd->bhnd", membership, k.float()) / chunk_size).to(dtype).float()
        block_scores = (pooled_q @ pooled_k.transpose(-2, -1)).to(dtype)
        selected_count = max(1, int((1 - sparsity) * num_blocks))
        selected = block_scores.topk(selected_count, dim=-1).indices
        block_mask = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, selected, True)
        mask = block_mask[:, :, block_ids[:, None], block_ids[None, :]]

    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    expected = (scores.softmax(dim=-1) @ v.float()).to(dtype)
    actual = flash_attn_bsa_3d(
        q,
        k,
        v,
        latent_shape,
        latent_shape,
        sparsity=sparsity,
        chunk_3d_shape_q=chunk_shape,
        chunk_3d_shape_k=chunk_shape,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-3 if dtype == torch.float16 else 3e-3, rtol=1e-2)
