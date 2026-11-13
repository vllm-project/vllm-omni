"""Utilities for batching variable-length tensors in diffusion attention."""

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA KV from num_kv_heads to num_heads.

    Input:  (batch, seqlen, num_key_value_heads, head_dim)
    Output: (batch, seqlen, num_key_value_heads * n_rep, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def left_pad_stack(
    tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Left-pad and stack variable-length tensors. Only dim 0 may vary,
    and it's assumed that all tensors are the same dtype & use the same
    device.

    Returns (stacked, mask) where mask is a 2D boolean mask, and both
    tensors are on the device of the provided tensors. If all tensors are
    the same length, None is returned for mask.
    """
    trailing_dims = {ts.shape[1:] for ts in tensors}
    if len(trailing_dims) != 1:
        raise ValueError("Tensors must be non-empty and can only vary in dim 0")
    trailing = trailing_dims.pop()

    seq_lens = [ts.shape[0] for ts in tensors]
    max_len = max(seq_lens)

    # If everything is the same length, we don't need a mask / varlen
    if all(sl == max_len for sl in seq_lens):
        return torch.stack(tensors), None

    device = tensors[0].device
    dtype = tensors[0].dtype
    stacked = torch.zeros(len(tensors), max_len, *trailing, dtype=dtype, device=device)
    # Create the boolean mask for the input sequences
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool, device=device)
    for idx, (ts, sl) in enumerate(zip(tensors, seq_lens)):
        pad = max_len - sl
        stacked[idx, pad:] = ts
        mask[idx, pad:] = True

    return stacked, mask
