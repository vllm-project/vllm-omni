# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

from typing import Literal

import torch
import torch.nn.functional as F

LseLayout = Literal["bhs", "bsh"]

__all__ = ["update_out_and_lse", "flatten_varlen_lse", "unflatten_varlen_lse"]


def _normalize_lse(block_lse: torch.Tensor, block_out: torch.Tensor, lse_layout: LseLayout) -> torch.Tensor:
    """Normalize a ring kernel LSE to ``(B, S, H, 1)``.

    ``lse_layout`` is required. When ``seq_len == num_heads``, ``(B, H, S)`` and
    ``(B, S, H)`` are the same shape, so guessing from dimension sizes would
    silently transpose a non-symmetric LSE.
    """
    if block_out.ndim != 4:
        raise ValueError(f"Ring attention output must be 4D (B, S, H, D), got {tuple(block_out.shape)}")

    batch, seq_len, num_heads, _ = block_out.shape
    if block_lse.shape[0] != batch:
        raise ValueError(f"Ring LSE batch dimension {block_lse.shape[0]} does not match output batch {batch}.")

    if block_lse.ndim == 4:
        if block_lse.shape[-1] != 1:
            raise ValueError(f"4D Ring LSE must end in a singleton dimension, got {tuple(block_lse.shape)}")
        block_lse = block_lse.squeeze(-1)
    elif block_lse.ndim != 3:
        raise ValueError(f"Ring LSE must be 3D or 4D, got {tuple(block_lse.shape)}")

    if lse_layout == "bhs":
        if block_lse.shape[1] != num_heads or block_lse.shape[2] < seq_len:
            raise ValueError(
                f"Ring LSE shape {tuple(block_lse.shape)} is incompatible with output "
                f"shape {tuple(block_out.shape)}; expected (B, H, S) for lse_layout='bhs'."
            )
        return block_lse[:, :, :seq_len].transpose(1, 2).unsqueeze(-1)
    if lse_layout == "bsh":
        if block_lse.shape[1] < seq_len or block_lse.shape[2] != num_heads:
            raise ValueError(
                f"Ring LSE shape {tuple(block_lse.shape)} is incompatible with output "
                f"shape {tuple(block_out.shape)}; expected (B, S, H) for lse_layout='bsh'."
            )
        return block_lse[:, :seq_len, :].unsqueeze(-1)

    raise ValueError(f"Unknown Ring LSE layout {lse_layout!r}; expected 'bhs' or 'bsh'.")


def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    lse_layout: LseLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    if out.shape != block_out.shape:
        raise ValueError(
            f"Ring attention block output shape {tuple(block_out.shape)} does not match "
            f"accumulated output shape {tuple(out.shape)}."
        )
    block_lse = _normalize_lse(block_lse, block_out, lse_layout)
    if lse.shape != block_lse.shape:
        raise ValueError(
            f"Ring attention block LSE shape {tuple(block_lse.shape)} does not match "
            f"accumulated LSE shape {tuple(lse.shape)}."
        )

    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
    *,
    lse_layout: LseLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")

        out = block_out.to(torch.float32)
        lse = _normalize_lse(block_lse, out, lse_layout)

    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse, lse_layout)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse, lse_layout)
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
