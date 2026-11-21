# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend block-sparse attention for 64-token tiles.

``torch_npu.npu_block_sparse_attention`` accepts query blocks of any size but
key blocks only in multiples of 128, so a 64-token key tile cannot be selected
on its own. Queries therefore stay on their 64-token tiles and keys are paired:
key block ``j`` holds tiles ``2j`` and ``2j + 1`` and is enabled for a query
tile when that tile selected either of them. Every query attends a superset of
the tiles it selected -- its selection plus their pair partners -- and never
loses one.

The operator has no per-key mask. A zero-padded key scores zero, so it adds
one to the softmax denominator and nothing to the numerator. The operator
returns the log-sum-exp of each query row, which lets that contribution be
removed exactly afterwards.
"""

from __future__ import annotations

import torch

TILE = 64
KEY_BLOCK = 128


def require_npu_tile64_sparse() -> None:
    """Fail before launch when the installed torch_npu lacks the block-sparse operator."""
    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError("NPU VSA requires torch_npu") from exc
    if not callable(getattr(torch_npu, "npu_block_sparse_attention", None)):
        raise ImportError("NPU VSA requires torch_npu.npu_block_sparse_attention")


def pair_key_tiles(block_map: torch.Tensor, block_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a ``[B,H,Qtiles,Ktiles]`` tile map onto 128-token key blocks.

    Returns the ``[B,H,Qtiles,ceil(Ktiles/2)]`` map and the number of valid
    keys in each key block. An odd tile count gains an empty, unselected
    partner so the last real tile still forms a whole key block.
    """
    if block_map.shape[-1] % 2:
        block_map = torch.nn.functional.pad(block_map, (0, 1), value=False)
        block_sizes = torch.nn.functional.pad(block_sizes, (0, 1), value=0)
    return block_map.unflatten(-1, (-1, 2)).any(dim=-1), block_sizes.view(-1, 2).sum(dim=-1)


def npu_tile64_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
) -> torch.Tensor:
    """Run tile64 block-sparse attention on BSHD tensors padded to whole tiles.

    Rows past a tile's ``block_sizes`` entry must be zero in ``key`` and
    ``value``: the denominator correction relies on them scoring zero.
    """
    import torch_npu

    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape or min(query.shape) < 1:
        raise ValueError("NPU VSA requires matching nonempty BSHD inputs")
    if query.dtype != torch.bfloat16 or query.shape[-1] != 128:
        raise ValueError("NPU VSA requires BF16 with head dimension 128")
    batch, rows, heads, _ = query.shape
    if rows % TILE:
        raise ValueError("NPU VSA inputs must be padded to complete 64-token tiles")
    tiles = rows // TILE
    if block_map.shape != (batch, heads, tiles, tiles) or block_map.dtype != torch.bool:
        raise ValueError("NPU VSA block_map must be boolean [B,H,tiles,tiles]")
    if block_sizes is None or block_sizes.shape != (tiles,):
        raise ValueError("NPU VSA requires one valid-token count per 64-token tile")

    pair_map, pair_valid = pair_key_tiles(block_map, block_sizes)
    if tiles % 2:
        key, value = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, TILE)) for t in (key, value))
    output, lse = torch_npu.npu_block_sparse_attention(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        pair_map.to(torch.int8),
        [TILE, KEY_BLOCK],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=heads,
        scale_value=softmax_scale,
        inner_precise=0,
        softmax_lse_flag=1,
    )
    # exp(lse) is the real mass plus one per zero-padded key the row was shown.
    pad_keys = torch.matmul(pair_map.float(), (KEY_BLOCK - pair_valid).float())
    pad_keys = pad_keys.repeat_interleave(TILE, dim=-1).unsqueeze(-1)
    real_mass = 1 - pad_keys * torch.exp(-lse.float().reshape(batch, heads, rows, 1))
    output = output.float() / real_mass
    # A query tile whose enabled key blocks hold no valid key has nothing to attend.
    has_keys = (pair_map & (pair_valid > 0)).any(dim=-1)
    output = output.unflatten(2, (tiles, TILE)).masked_fill(~has_keys[..., None, None], 0).flatten(2, 3)
    return output.to(query.dtype).transpose(1, 2).contiguous()
