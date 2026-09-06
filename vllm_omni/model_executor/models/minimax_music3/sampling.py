# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic guided sampling for MiniMax Music 3.

Two things happen per code, in this order:

1. **Classifier-free guidance.** Every request decodes twice, once on the real
   prompt and once on a prompt whose caption and lyrics have been replaced by
   ``<|audio_cfg|>``. Codes are drawn from ``uncond + (cond - uncond) * 1.5``.
   For ``c0`` the candidate set is additionally masked to the conditioned
   row's own top 50, so the guided draw cannot wander outside what the
   conditioned branch already considered plausible.
2. **Seeded top-k.** A batched, order-independent categorical draw: each row
   gets its own RNG stream keyed by ``(seed, position)``, so a request's audio
   depends only on its own seed and never on who else happens to share the
   batch.

The RNG is a MurmurHash3 x86_32 of ``(seed_lo, seed_hi, position, column)``
turned into Gumbel noise, and the draw is an argmax over ``logits + gumbel``.
That is mathematically a categorical sample from ``softmax(logits)``, but
unlike ``torch.multinomial`` it needs no per-row generator state, so it is
CUDA-graph safe and reproducible across batch compositions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import AR_CFG_SCALE, AR_CFG_TOP_K, AR_TOP_K

_UINT32_MASK = 0xFFFFFFFF
_UINT32_MAX = 4_294_967_295

# MurmurHash3 x86_32 constants.
_C1 = 0xCC9E2D51
_C2 = 0x1B873593
_R1 = 15
_R2 = 13
_M = 5
_N = 0xE6546B64


def _rotl32(x: Tensor, r: int) -> Tensor:
    return ((x << r) | (x >> (32 - r))) & _UINT32_MASK


def _murmur3_mix(h: Tensor, k: Tensor) -> Tensor:
    """Mix one 32-bit key into the running hash state."""
    k = (k * _C1) & _UINT32_MASK
    k = _rotl32(k, _R1)
    k = (k * _C2) & _UINT32_MASK
    h = h ^ k
    h = _rotl32(h, _R2)
    return (h * _M + _N) & _UINT32_MASK


def _fmix32(h: Tensor) -> Tensor:
    """MurmurHash3 finalizer."""
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _UINT32_MASK
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _UINT32_MASK
    return h ^ (h >> 16)


# Memoized ``k`` term of the column block's mix. The column block is the same
# ``arange(width)`` on every draw, so its three ops are hoisted out of the per
# draw work and kept per (device, width). Small and bounded: two widths exist,
# the c0 candidate set and the residual codebook.
_COLUMN_MIX_CACHE: dict[tuple[torch.device, int], Tensor] = {}


def _column_mix(num_columns: int, device: torch.device) -> Tensor:
    """Return the mixed column block, ``[cols]`` int64 holding uint32."""
    key = (device, num_columns)
    cached = _COLUMN_MIX_CACHE.get(key)
    if cached is None:
        columns = torch.arange(num_columns, dtype=torch.int64, device=device) & _UINT32_MASK
        k = (columns * _C1) & _UINT32_MASK
        k = _rotl32(k, _R1)
        cached = (k * _C2) & _UINT32_MASK
        _COLUMN_MIX_CACHE[key] = cached
    return cached


def _absorb_mixed_key(h: Tensor, k: Tensor) -> Tensor:
    """Absorb a key whose own three-op transform is already applied."""
    h = h ^ k
    h = _rotl32(h, _R2)
    return (h * _M + _N) & _UINT32_MASK


def row_hash_state(seeds: Tensor, positions: Tensor) -> Tensor:
    """Hash state after the seed and position blocks, before the columns.

    Args:
        seeds: ``[rows]`` per-request seeds.
        positions: ``[rows]`` or ``[rows, draws]`` draw identifiers. The extra
            dimension lets every codebook of a frame be hashed in one batch
            instead of one dispatch per codebook; the arithmetic is elementwise
            integer, so a batched call is bit-identical to the loop it replaces.

    Returns:
        A tensor shaped like ``positions``, int64 holding uint32.
    """
    seeds = seeds.to(torch.int64)
    while seeds.dim() < positions.dim():
        seeds = seeds.unsqueeze(-1)
    h = torch.zeros((), dtype=torch.int64, device=seeds.device)
    h = _murmur3_mix(h, seeds & _UINT32_MASK)
    h = _murmur3_mix(h, (seeds >> 32) & _UINT32_MASK)
    return _murmur3_mix(h, positions.to(torch.int64) & _UINT32_MASK)


def murmur_hash32(seeds: Tensor, positions: Tensor, columns: Tensor) -> Tensor:
    """Hash ``(seed, position, column)`` to a uint32 held in int64.

    The 64-bit seed is consumed as two 32-bit blocks, then the position and
    the column, for a hashed length of 16 bytes.

    Args:
        seeds: ``[rows]`` per-request seeds.
        positions: ``[rows]`` draw identifiers, unique per (frame, codebook).
        columns: ``[cols]`` candidate indices, which must be ``arange(cols)``.

    Returns:
        ``[rows, cols]`` int64 tensor holding uint32 values.
    """
    h = row_hash_state(seeds, positions).unsqueeze(-1)
    return _fmix32(_absorb_mixed_key(h, _column_mix(int(columns.shape[0]), h.device)) ^ 16)


def gumbel_noise(seeds: Tensor, positions: Tensor, num_columns: int) -> Tensor:
    """Return the Gumbel noise for one draw per position, ``[..., cols]``.

    Args:
        seeds: ``[rows]`` per-request seeds.
        positions: ``[rows]`` or ``[rows, draws]`` draw identifiers.
        num_columns: Candidate count, i.e. the width of the logits.

    Returns:
        float64 noise shaped ``positions.shape + (num_columns,)``.
    """
    h = row_hash_state(seeds, positions).unsqueeze(-1)
    hashed = _fmix32(_absorb_mixed_key(h, _column_mix(num_columns, h.device)) ^ 16)
    # float64 throughout: the double log is numerically unstable in float32 and
    # a single flipped draw changes every frame that follows it.
    uniform = hashed.to(torch.float64) / _UINT32_MAX
    # Clamp both ends. uniform == 1 gives +inf Gumbel, which turns a -inf
    # masked entry into NaN; the upper cap is the hash spacing, so a saturated
    # bucket ties with its neighbour instead of always winning.
    gumbel = uniform.log().clamp_(min=torch.finfo(torch.float64).min, max=-(2.0**-32)).neg_()
    return gumbel.log().neg_()


def draw_from_gumbel(logits: Tensor, gumbel: Tensor, *, top_k: int = AR_TOP_K) -> Tensor:
    """Argmax of ``top_k``-masked logits plus precomputed Gumbel noise.

    Args:
        logits: ``[rows, vocab]`` unnormalized scores.
        gumbel: ``[rows, vocab]`` float64 noise from :func:`gumbel_noise`.
        top_k: Candidates retained per row before the draw.

    Returns:
        ``[rows]`` sampled column indices.
    """
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    k = min(int(top_k), values.shape[-1])
    threshold = torch.topk(values, k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    return (values.to(torch.float64) + gumbel).argmax(dim=-1)


def sample_topk_seeded(
    logits: Tensor,
    seeds: Tensor,
    positions: Tensor,
    *,
    top_k: int = AR_TOP_K,
) -> Tensor:
    """Draw one column per row from its own top-k, under its own RNG stream.

    Args:
        logits: ``[rows, vocab]`` unnormalized scores.
        seeds: ``[rows]`` per-request sampling seeds.
        positions: ``[rows]`` draw identifiers, unique per (frame, codebook).
        top_k: Candidates retained per row before the draw.

    Returns:
        ``[rows]`` sampled column indices.
    """
    gumbel = gumbel_noise(seeds, positions, int(logits.shape[-1]))
    return draw_from_gumbel(logits, gumbel, top_k=top_k)


def apply_c0_cfg(cond: Tensor, uncond: Tensor, *, scale: float = AR_CFG_SCALE) -> Tensor:
    """Guide ``cond`` by ``uncond``, restricted to ``cond``'s own top candidates.

    The mask is a second, separate restriction from the top-k the sampler
    applies afterwards. Dropping it widens the distribution the model draws
    from and audibly loosens how closely the output follows the caption.
    """
    guided = uncond + (cond - uncond) * scale
    k = min(AR_CFG_TOP_K, cond.shape[-1])
    threshold = torch.topk(cond, k, dim=-1).values[..., -1, None]
    return guided.masked_fill(cond < threshold, -float("inf"))


def apply_depth_cfg(cond: Tensor, uncond: Tensor, *, scale: float = AR_CFG_SCALE) -> Tensor:
    """Guide the depth heads. Unlike ``c0`` these are not additionally masked."""
    return uncond + (cond - uncond) * scale


__all__ = [
    "apply_c0_cfg",
    "apply_depth_cfg",
    "draw_from_gumbel",
    "gumbel_noise",
    "murmur_hash32",
    "row_hash_state",
    "sample_topk_seeded",
]
