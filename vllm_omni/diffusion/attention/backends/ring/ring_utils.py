# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention


import torch
import torch.nn.functional as F

__all__ = ["update_out_and_lse", "flatten_varlen_lse", "unflatten_varlen_lse"]


# Remove torch.jit.script for debugging and flexible shape handling
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)

    B, S, H, D = out.shape

    # --- Shape Correction Logic for block_lse ---
    # Goal: block_lse should be (B, S, H, 1) to match out (B, S, H, D)

    # Debug info
    # print(f"DEBUG _update: out={out.shape}, block_lse={block_lse.shape}")

    # Case 0: If block_lse is already 4D, check if it matches
    if block_lse.dim() == 4:
        if block_lse.shape[1] == S and block_lse.shape[2] == H:
            pass  # Good
        elif block_lse.shape[1] == H and block_lse.shape[2] == S:
            block_lse = block_lse.transpose(1, 2)
        elif block_lse.shape[1] == H and block_lse.shape[2] >= S:  # Padding case
            block_lse = block_lse[:, :, :S, :].transpose(1, 2)
        # If shape is (B, H, S, 1) but expected (B, S, H, 1) because out is (B, S, H, D)
        elif block_lse.shape[1] == H and block_lse.shape[2] == S and block_lse.shape[3] == 1:
            block_lse = block_lse.transpose(1, 2)

    # Case 1: block_lse is 3D (B, H, S) or (B, S, H) or (B, ?, ?)
    elif block_lse.dim() == 3:
        # Check for (B, H, S) - Standard SDPA/FA output
        if block_lse.shape[1] == H and block_lse.shape[2] == S:
            block_lse = block_lse.transpose(1, 2).unsqueeze(-1)

        # Check for (B, S, H)
        elif block_lse.shape[1] == S and block_lse.shape[2] == H:
            block_lse = block_lse.unsqueeze(-1)

        # Check for Padding: (B, H, S_pad) where S_pad >= S
        elif block_lse.shape[1] == H and block_lse.shape[2] >= S:
            # print(f"DEBUG: Trimming padding from lse. {block_lse.shape} -> S={S}")
            block_lse = block_lse[:, :, :S].transpose(1, 2).unsqueeze(-1)

        # Check for weird case: (B, S, H_pad) ? Unlikely for LSE but possible
        elif block_lse.shape[1] == S and block_lse.shape[2] >= H:
            block_lse = block_lse[:, :, :H].unsqueeze(-1)

        # Check for flipped weird case: (B, S_pad, H)
        elif block_lse.shape[1] >= S and block_lse.shape[2] == H:
            block_lse = block_lse[:, :S, :].unsqueeze(-1)

    # --- Shape Correction for lse (internal state) ---
    # Ensure lse matches block_lse's corrected shape (B, S, H, 1)
    if lse.shape != block_lse.shape:
        # If lse was initialized with wrong shape, try to fix it
        if lse.dim() == 4 and lse.shape[1] == block_lse.shape[2] and lse.shape[2] == block_lse.shape[1]:
            lse = lse.transpose(1, 2)
        elif lse.shape[1] >= S:  # slice if lse was initialized with padding
            lse = lse[:, :S, :, :]

    # Final check
    if lse.shape != block_lse.shape:
        # Force broadcast if possible?
        pass

    try:
        out = out - F.sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
    except RuntimeError as e:
        print(f"ERROR in _update_out_and_lse: {e}")
        print(f"out: {out.shape}, lse: {lse.shape}")
        print(f"block_out: {block_out.shape}, block_lse: {block_lse.shape}")
        # raise e
        raise e

    return out, lse


def update_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")

        out = block_out.to(torch.float32)

        # Initialize LSE with robust logic (same as _update)
        B, D1, D2, D3 = out.shape

        S_guess = D1
        H_guess = D2

        if block_lse.dim() == 3:
            if block_lse.shape[1] == H_guess and block_lse.shape[2] == S_guess:
                lse = block_lse.transpose(1, 2).unsqueeze(-1)
            elif block_lse.shape[1] == S_guess and block_lse.shape[2] == H_guess:
                lse = block_lse.unsqueeze(-1)
            elif block_lse.shape[1] == H_guess and block_lse.shape[2] >= S_guess:  # Padding
                lse = block_lse[:, :, :S_guess].transpose(1, 2).unsqueeze(-1)
            elif block_lse.shape[1] == S_guess and block_lse.shape[2] >= H_guess:  # Padding/Weird
                lse = block_lse[:, :, :H_guess].unsqueeze(-1)
            elif block_lse.shape[1] >= S_guess and block_lse.shape[2] == H_guess:
                lse = block_lse[:, :S_guess, :].unsqueeze(-1)

            # Reverse case: What if out is (B, H, S, D) so S=D2, H=D1?
            elif block_lse.shape[1] == D1 and block_lse.shape[2] >= D2:  # Matches (H, S)
                # Then out is (B, H, S, D). We should transpose out!
                out = out.transpose(1, 2)
                lse = block_lse[:, :, :D2].transpose(1, 2).unsqueeze(-1)  # (B, S, H, 1)

            else:
                # Fallback
                lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        else:
            # Case 0: If block_lse is already 4D, check if it matches
            if block_lse.dim() == 4:
                if block_lse.shape[1] == S_guess and block_lse.shape[2] == H_guess:
                    lse = block_lse
                elif block_lse.shape[1] == H_guess and block_lse.shape[2] == S_guess:
                    lse = block_lse.transpose(1, 2)
                elif block_lse.shape[1] == H_guess and block_lse.shape[2] >= S_guess:  # Padding case
                    lse = block_lse[:, :, :S_guess, :].transpose(1, 2)
                elif block_lse.shape[1] == D1 and block_lse.shape[2] >= D2:  # Matches (H, S)
                    # Then out is (B, H, S, D). We should transpose out!
                    out = out.transpose(1, 2)
                    lse = block_lse[:, :, :D2].transpose(1, 2)  # (B, S, H, 1)
                else:
                    lse = block_lse
            else:
                lse = block_lse

    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()
