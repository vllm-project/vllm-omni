# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = ["update_out_and_lse", "flatten_varlen_lse", "unflatten_varlen_lse"]

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    block_out = block_out.to(torch.float32)
    # block_lse shape: (batch_size, num_heads, seq_len)
    # lse shape: (batch_size, seq_len, num_heads, 1)
    
    # Check if block_lse needs transposition
    # If block_lse is (B, H, S), we need to transpose to (B, S, H) then unsqueeze to (B, S, H, 1)
    # If block_lse is already (B, S, H), we just unsqueeze
    
    if block_lse.dim() == 3:
        # Assuming (B, H, S) from Flash Attention / SDPA
        # But we need to be careful if it is already (B, S, H)
        # Check against out shape. out is (B, S, H, D)
        # So lse should be (B, S, H, 1)
        
        B, S, H, D = out.shape
        
        # If block_lse is (B, H, S)
        if block_lse.shape[-1] == S and block_lse.shape[-2] == H:
             block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        # If block_lse is (B, S, H)
        elif block_lse.shape[-2] == S and block_lse.shape[-1] == H:
             block_lse = block_lse.unsqueeze(dim=-1)
        else:
            # Fallback to original behavior but adding a check might be good
             block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    
    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        
        # Initialize LSE
        # block_lse from FA/SDPA is usually (B, H, S)
        # We want internal LSE state to be (B, S, H, 1) to match out (B, S, H, D) broadcasting
        
        if block_lse.dim() == 3:
             B, S, H, D = out.shape
             # If block_lse is (B, H, S)
             if block_lse.shape[-1] == S and block_lse.shape[-2] == H:
                  lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
             # If block_lse is (B, S, H)
             elif block_lse.shape[-2] == S and block_lse.shape[-1] == H:
                  lse = block_lse.unsqueeze(dim=-1)
             else:
                  # Original behavior fallback
                  lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        else:
             lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
             
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()
