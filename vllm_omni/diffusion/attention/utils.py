# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = ["update_out_and_lse", "flatten_varlen_lse", "unflatten_varlen_lse"]

# Remove torch.jit.script for debugging and flexible shape handling
# @torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    block_out = block_out.to(torch.float32)
    
    # Debug prints (will be visible in stdout/stderr of the test output)
    # print(f"DEBUG: out.shape={out.shape}, lse.shape={lse.shape}, block_out.shape={block_out.shape}, block_lse.shape={block_lse.shape}")

    # Check and adjust block_lse shape to match lse
    # lse expected to be (B, S, H, 1) or broadcastable to out
    
    # Case 1: block_lse is (B, H, S) - Typical from SDPA/FlashAttn
    if block_lse.dim() == 3 and block_lse.shape[-2] == out.shape[-2]: 
        # CAUTION: Heuristic assuming H is dim -2 for block_lse? No, SDPA returns (B, H, S)
        # But out is (B, S, H, D). So block_lse (B, H, S) needs transpose to (B, S, H)
        
        # Let's rely on matching S and H dimensions with out
        B, S, H, D = out.shape
        if block_lse.shape[-1] == S and block_lse.shape[-2] == H:
            block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        elif block_lse.shape[-2] == S and block_lse.shape[-1] == H:
             block_lse = block_lse.unsqueeze(dim=-1)
             
    # Case 2: block_lse is flattened? or mismatch
    
    # Ensure block_lse is broadcastable to lse
    # If lse is (B, S, H, 1) and block_lse ended up (B, H, S, 1) due to wrong transpose, it would fail
    
    # Safe guard: if shapes are just permuted (B, H, S) vs (B, S, H), force match to out
    if lse.shape != block_lse.shape:
        # Try to align block_lse to lse if possible
        if block_lse.dim() == 4 and lse.dim() == 4:
             if block_lse.shape[1] == lse.shape[2] and block_lse.shape[2] == lse.shape[1]:
                  block_lse = block_lse.transpose(1, 2)
    
    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    
    try:
        out = out - F.sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
    except RuntimeError as e:
        print(f"ERROR in _update_out_and_lse: {e}")
        print(f"out: {out.shape}, lse: {lse.shape}")
        print(f"block_out: {block_out.shape}, block_lse: {block_lse.shape}")
        raise e

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
        # Goal: lse shape should match out's spatial dims (B, S, H, 1)
        B, S, H, D = out.shape
        
        # block_lse from SDPA/FA is typically (B, H, S)
        # We need (B, S, H, 1)
        
        if block_lse.dim() == 3:
             if block_lse.shape[-1] == S and block_lse.shape[-2] == H:
                  lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
             elif block_lse.shape[-2] == S and block_lse.shape[-1] == H:
                  lse = block_lse.unsqueeze(dim=-1)
             else:
                  # Fallback or weird shape
                  lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        else:
             # Already 4D?
             lse = block_lse
             
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


# @torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


# @torch.jit.script
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
