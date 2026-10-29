# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend tile128 sparse attention with ragged-tile normalization."""

import torch


def npu_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
) -> torch.Tensor:
    import torch_npu

    if query.dtype != torch.bfloat16 or query.shape[-1] != 128:
        raise ValueError("NPU VSA requires BF16 with head dimension 128")
    if block_sizes is None or block_sizes.shape != (key.shape[1] // 128,):
        raise ValueError("NPU VSA requires one valid-token count per tile128")
    # Zero padding has score zero and contributes one to the softmax denominator.
    # The native operator exposes FP32 LSE, allowing that contribution to be removed.
    valid = torch.arange(key.shape[1], device=key.device) % 128 < block_sizes.repeat_interleave(128)
    k = key.transpose(1, 2).masked_fill(~valid[None, None, :, None], 0).contiguous()
    v = value.transpose(1, 2).masked_fill(~valid[None, None, :, None], 0).contiguous()
    output, lse = torch_npu.npu_block_sparse_attention(
        query.transpose(1, 2).contiguous(),
        k,
        v,
        block_map.to(torch.int8),
        [128, 128],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=key.shape[2],
        scale_value=softmax_scale,
        inner_precise=0,
        softmax_lse_flag=1,
    )
    padding = torch.matmul(block_map.float(), (128 - block_sizes).float())
    padding = padding.repeat_interleave(128, -1).unsqueeze(-1)
    valid_mass = 1 - padding * torch.exp(-lse)
    output = output.float() / valid_mass
    has_keys = (block_map & (block_sizes > 0)).any(dim=-1)
    output = output.unflatten(2, (-1, 128)).masked_fill(~has_keys[..., None, None], 0).flatten(2, 3)
    return output.to(query.dtype).transpose(1, 2).contiguous()
