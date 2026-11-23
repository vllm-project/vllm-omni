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

The operator has no per-key mask, so a padded key row still enters the
softmax. Its contribution is removed afterwards from the log-sum-exp the
operator returns, and what the row is padded *with* decides whether that
subtraction survives fp32.

A zero row scores logit zero, which is unbounded above the real mass: a block
holding one valid key at a scaled logit of -24 beside 127 zero rows gives
``exp(lse) = 127 + 3.8e-11``, which rounds to 127, and the real mass is gone
before the correction runs. Each padded row is therefore filled with one of
its own key block's valid keys. That row scores ``e* <= M``, because the key it
copies is itself one of the keys summed in ``M``, so the removed mass is at
most ``P * M`` and

    real_mass = M / (M + sum_b P_b e*_b)  >=  1 / (1 + P)

for any valid logits at all -- four orders of magnitude clear of the fp32
cancellation at ``2**-24`` even when ``P`` runs to thousands. Padded rows of
``value`` stay zero, so they still add nothing to the numerator.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

TILE = 64
KEY_BLOCK = 128


def require_npu_tile64_sparse() -> None:
    """Fail before launch when this device cannot run the route below.

    Having the operator installed is not enough. The call uses
    ``inner_precise=0``, and the supported modes differ by SoC version -- the
    950 series documents 4 as its only one -- so a device that resolves the
    symbol can still reject the call. This route is measured on Atlas A3 and
    nothing else, so A3 is what it admits.

    A device that cannot be classified at all is let through rather than
    refused: that is the behaviour without this guard, and an older
    ``vllm_ascend`` missing the classifier should not stop an A3 from starting.
    The guard refuses only a SoC version it positively identifies as another.
    """
    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError("NPU VSA requires torch_npu") from exc
    if not callable(getattr(torch_npu, "npu_block_sparse_attention", None)):
        raise ImportError("NPU VSA requires torch_npu.npu_block_sparse_attention")
    try:
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        device_type = get_ascend_device_type()
    except (ImportError, AttributeError):
        logger.warning(
            "NPU VSA could not read the Ascend SoC version; this route is validated on "
            "Atlas A3 only and calls npu_block_sparse_attention with inner_precise=0, "
            "which other SoC versions need not support."
        )
        return
    if device_type != AscendDeviceType.A3:
        raise ValueError(
            f"NPU VSA runs on Atlas A3 only, and this device reports {device_type}. The route "
            "calls npu_block_sparse_attention with inner_precise=0, whose support varies by SoC "
            "version. Select a dense diffusion attention backend to run MiniMax-H3 here."
        )


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


def padded_key_sources(tile_sizes: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index that fills every padded key row from its own 128-token block.

    Returns a gather index over all ``rows`` key rows -- a valid row maps to
    itself, a padded row to one of its block's valid keys -- and the row index
    of each block's representative. This is index arithmetic only: the caller
    applies it in whichever layout it already has to build, so filling the
    padded rows costs no pass of its own. Blocks without a valid key get index
    0; the caller drops them from the map, so that index is never used.
    """
    device = tile_sizes.device
    blocks = rows // KEY_BLOCK
    row = torch.arange(rows, device=device)
    valid = (row % TILE) < tile_sizes[row // TILE]
    low_tile_has_a_key = tile_sizes.view(blocks, 2)[:, 0] > 0
    representative = torch.arange(blocks, device=device) * KEY_BLOCK + torch.where(low_tile_has_a_key, 0, TILE)
    return torch.where(valid, row, representative[row // KEY_BLOCK]), representative


def npu_tile64_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
) -> torch.Tensor:
    """Run tile64 block-sparse attention on BSHD tensors padded to whole tiles.

    Rows past a tile's ``block_sizes`` entry must be zero in ``value``: they
    contribute to the softmax denominator and the correction removes exactly
    that, but nothing subtracts a padded numerator. Their ``key`` rows may hold
    anything, because they are overwritten before the operator sees them.
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
    tile_sizes = block_sizes
    if tiles % 2:
        key, value = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, TILE)) for t in (key, value))
        tile_sizes = torch.nn.functional.pad(block_sizes, (0, 1))
    # A key block holding no valid key has nothing to attend and no key to pad
    # with. Dropping it cannot change the result and keeps every selected block
    # supplied with a representative.
    pair_map = pair_map & (pair_valid > 0)
    source, representative = padded_key_sources(tile_sizes, key.shape[1])
    # The operator wants BNSD, so the keys have to be laid out again anyway.
    # Gathering the padded rows during that pass costs nothing extra, while
    # filling them in BSHD first and transposing afterwards walks the keys
    # twice. The query is laid out once here and reused by the correction
    # below for the same reason.
    query_bnsd = query.transpose(1, 2).contiguous()
    key_bnsd = key.transpose(1, 2).index_select(2, source)
    output, lse = torch_npu.npu_block_sparse_attention(
        query_bnsd,
        key_bnsd,
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
    # Every padded row of a selected block scores that block's representative
    # logit, so the mass to remove is that logit's weight times the number of
    # padded rows. The logits are taken in fp32: the quotient below divides by
    # a value that can sit at its 1/(1+P) floor, so an error here is amplified
    # by 1/real_mass, and bfloat16 would not survive it.
    lse = lse.float().reshape(batch, heads, rows, 1)
    representative_key = key_bnsd.index_select(2, representative)
    pad_mass = torch.matmul(query_bnsd.float(), representative_key.float().transpose(-1, -2)).mul_(softmax_scale)
    # ``lse`` covers the blocks a query selected, so only their representatives
    # are bounded by it. An unselected block's can sit far above, where ``exp``
    # overflows fp32 and a zero mask applied afterwards turns the infinity into
    # NaN for the whole row. Blocks with nothing to remove are masked before
    # the exponent instead.
    padded = KEY_BLOCK - pair_valid
    removable = pair_map.repeat_interleave(TILE, dim=-2) & (padded > 0)
    pad_mass = pad_mass.sub_(lse).masked_fill_(~removable, float("-inf")).exp_().mul_(padded.float())
    pad_mass = pad_mass.sum(dim=-1, keepdim=True)
    output = output.float() / (1 - pad_mass)
    # A query tile that enabled no key block with a valid key has nothing to attend.
    has_keys = pair_map.any(dim=-1)
    output = output.unflatten(2, (tiles, TILE)).masked_fill(~has_keys[..., None, None], 0).flatten(2, 3)
    return output.to(query.dtype).transpose(1, 2).contiguous()
