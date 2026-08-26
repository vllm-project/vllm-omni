# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""Shared split reduction machinery for the fused GroupNorm operators.

Both :mod:`~.fused_group_norm_silu` and :mod:`~.fused_adaptive_group_norm_silu`
reduce mean/variance over the same axis -- every channel of a group across every
spatial position -- and differ only in the epilogue they apply afterwards. This
module owns the reduction so the two operators share one implementation.

Why a split at all
------------------
One program per ``(batch, group)`` pair means ``B * num_groups`` CTAs. Every
GroupNorm in the diffusion stack uses ``num_groups=32`` and the VAE decodes at
``B=1``, so the largest activations in the pipeline -- hundreds of MB apiece --
were being streamed by 32 CTAs on a GPU with 80-132 SMs. The reduction per group
is ``group_size * spatial_size`` elements, which at decode resolution is a few
million; well past the point where a single CTA is the right shape for the job.

The fix is a classic split reduction: partition the spatial axis, have each CTA
reduce its own slice into a workspace tensor, then let a second pass combine the
partials and apply the normalization. This is the same structure TorchInductor
uses for its split reductions, and deliberately *not* the lock/atomic scheme from
the Triton layernorm tutorial -- each slot in the workspace is written by exactly
one program, so the result is deterministic run to run.

The input is still read exactly twice (once for statistics, once to normalize),
which is the property that makes the fused operator worth having in the first
place.

What the split does to the numerics
-----------------------------------
Splitting adds a level to the reduction tree, so results are not bit-identical
to the unsplit path -- on ordinary fp32 input the two differ by at most ~2e-6.

It is not a loss of accuracy, though, which is worth stating because the
intuition runs the other way. Measured on an A10G against a float64 reference,
on the deliberately pathological ``10000 +- 0.1`` input (mean 1e5 times the
standard deviation), maximum output error was 0.031 unsplit, 0.021 at split=4
and 0.011 at split=16. More programs means each one folds fewer blocks into its
running Welford state, so the sequential drift that accumulates along that chain
is shorter. The floor at ~0.01 is fp32 storage of ``x`` itself and no reduction
order can get under it.
"""

import functools
import os

import torch
from vllm.triton_utils import tl, triton

# Spatial chunks handed to a program are a multiple of this many elements. It
# matches the largest autotuned ``BLOCK_SIZE`` so that every chunk boundary is
# also a block boundary: programs other than the last one then issue only full,
# aligned, maximally vectorized loads. It doubles as the floor on how little work
# justifies launching another CTA.
SPLIT_ALIGN = 4096

# Target CTA waves across the device. The split is sized to fill roughly this
# many CTAs per SM.
#
# 2 is measured, not assumed, but the measurement is that it barely matters:
# sweeping 1, 2, 4, 8 and 16 across the whole HunyuanImage3 decode ladder on an
# A10G moved every shape by less than 2%, with 2 the most frequent winner. What
# matters is getting off 32 CTAs at all; the exact multiple does not. (Published
# split-reduction results elsewhere report a much sharper optimum, so the knob
# stays exposed rather than baked in -- see
# ``benchmarks/kernels/group_norm_silu_benchmarks.py --sweep-waves``.)
#
# Setting it to 0 disables the split entirely, restoring the one-CTA-per-group
# behaviour -- an escape hatch if a shape ever regresses, and what the benchmark
# and the split-invariance test use as their baseline.
_SPLIT_WAVES_ENV = "VLLM_OMNI_GROUP_NORM_SPLIT_WAVES"
_DEFAULT_SPLIT_WAVES = 2

# Autotune space shared by the statistics kernel and both apply kernels.
#
# Chosen from a full sweep of BLOCK_SIZE x num_warps x num_stages on one A10G,
# bf16, 32 groups (see benchmarks/kernels/group_norm_silu_benchmarks.py):
#
#   * BLOCK_SIZE dominates, and the right value tracks the slice each program
#     reduces. On the 1024^2 VAE activation, 4096 runs at 1.63 ms and 1024 at
#     1.95 ms; on a 32x32 DiT activation (1024 spatial positions) that reverses,
#     1024 at 0.108 ms against 0.131 ms for 4096, because three quarters of a
#     4096-wide block is masked off. ``_prune_oversized_blocks`` below is what
#     keeps each shape on the right rung.
#   * ``num_warps=16`` -- what this operator used to hardcode -- was never the
#     winner, and cost up to 11% at the narrower block sizes. 4 and 8 tie within
#     noise, so both are kept: the balance between them is the kind of thing
#     that moves with the memory system, and this space also has to serve H100
#     and L20.
#   * ``num_stages=2`` wins at BLOCK_SIZE=4096 and ``num_stages=1`` at 1024
#     (0.108 ms vs 0.124 ms there), so both stay. 4 was never best.
#
# 512 is deliberately absent: it lost on every shape measured, badly on the
# large ones (2.26 ms vs 1.63 ms at 1024^2).
SPLIT_REDUCTION_CONFIGS = [
    triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=stages)
    for block in (1024, 2048, 4096)
    for warps in (4, 8)
    for stages in (1, 2)
]

# Autotune key. ``split_chunk`` rather than the ``SPLIT`` constexpr because it is
# a plain runtime argument and it -- not ``spatial_size`` -- is what determines
# how much work a single program does. Without it a change in batch size would
# silently reuse a config tuned for a different per-CTA workload.
SPLIT_REDUCTION_KEY = ["spatial_size", "C", "split_chunk"]


def _prune_oversized_blocks(configs, named_args, **kwargs):
    """Drop block sizes wider than the slice a program actually reduces.

    A block wider than its slice buys nothing but masked-off lanes, and it costs
    real time: 4096 is 21% slower than 1024 on a 1024-position activation. This
    also makes autotuning cheaper on exactly the small shapes where the tuning
    cost is largest relative to the work.

    If the slice is narrower than every configured block -- possible for tiny
    inputs -- the narrowest configs are kept rather than returning nothing.
    """
    span = min(named_args.get("spatial_size", 1 << 30), named_args.get("split_chunk", 1 << 30))
    cap = 1 << max(0, int(span) - 1).bit_length()
    kept = [c for c in configs if c.kwargs["BLOCK_SIZE"] <= cap]
    if kept:
        return kept
    narrowest = min(c.kwargs["BLOCK_SIZE"] for c in configs)
    return [c for c in configs if c.kwargs["BLOCK_SIZE"] == narrowest]


SPLIT_REDUCTION_PRUNE = {"early_config_prune": _prune_oversized_blocks}


def _split_waves() -> int:
    """Read the CTA-waves target. Uncached so tests and sweeps can retune it.

    ``0`` means "do not split". A negative or unparsable value is treated as
    unset rather than as a request, so a typo degrades to the default instead of
    silently turning the optimization off.
    """
    raw = os.environ.get(_SPLIT_WAVES_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_SPLIT_WAVES
    try:
        waves = int(raw)
    except ValueError:
        return _DEFAULT_SPLIT_WAVES
    return waves if waves >= 0 else _DEFAULT_SPLIT_WAVES


@functools.cache
def _multi_processor_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=1024)
def _split_for(spatial_size: int, groups: int, num_sms: int, waves: int) -> tuple[int, int]:
    """Pure arithmetic half of :func:`pick_split`, cached on its inputs."""
    if waves <= 0:
        return 1, spatial_size
    # Enough CTAs to cover the device ``waves`` times over, spread across the
    # groups we already have.
    split = -(-(waves * num_sms) // groups)
    # Never hand a program less than one aligned chunk.
    split = min(split, -(-spatial_size // SPLIT_ALIGN))
    if split <= 1:
        return 1, spatial_size

    # Round the chunk up to the alignment, then recompute how many chunks that
    # actually leaves. Recomputing is what guarantees no program gets an empty
    # slice, which in turn lets the combine assume every partial has n > 0.
    chunk = -(-spatial_size // split)
    chunk = -(-chunk // SPLIT_ALIGN) * SPLIT_ALIGN
    split = -(-spatial_size // chunk)
    if split <= 1:
        return 1, spatial_size
    return split, chunk


def pick_split(spatial_size: int, batch: int, num_groups: int, device: torch.device) -> tuple[int, int]:
    """Choose ``(split, split_chunk)`` for one GroupNorm launch.

    ``split`` is how many programs cooperate on each ``(batch, group)`` pair and
    ``split_chunk`` is how many spatial positions each of them covers. ``split ==
    1`` means "no split": one program per group, one kernel launch, no workspace,
    exactly the behaviour before this path existed.

    The result is a pure function of the shape and the device's SM count, so a
    given tensor on a given GPU always takes the same path. Different GPU models
    pick different splits and so sum the fp32 partials in a different order,
    which moves results by ~1e-7 relative -- the same order of device-to-device
    variation eager PyTorch already has.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return 1, spatial_size
    index = device.index if device.index is not None else torch.accelerator.current_device_index()
    return _split_for(spatial_size, batch * num_groups, _multi_processor_count(index), _split_waves())


@triton.jit
def welford_group_range(
    x_ptr,
    n_idx,
    g_idx,
    C,
    spatial_size,
    group_size,
    lo,
    hi,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Welford statistics over spatial ``[lo, hi)`` of every channel in a group.

    Returns the running ``(n, mean, m2)`` triple. Each block is reduced in
    registers (block mean, then centered M2) so the loaded values are reused
    instead of re-read, and blocks are merged with the Chan et al. parallel
    formula. That keeps both the mean and the variance accurate for inputs like
    ``10000 +- 0.1``, where the naive ``E[x^2] - E[x]^2`` cancels catastrophically.
    """
    n_total = tl.zeros([1], dtype=tl.float32)
    mean_total = tl.zeros([1], dtype=tl.float32)
    m2_total = tl.zeros([1], dtype=tl.float32)

    for c_offset in range(group_size):
        c_idx = g_idx * group_size + c_offset
        base = n_idx * C * spatial_size + c_idx * spatial_size

        for s_start in tl.range(lo, hi, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hi

            x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
            x_val = x_val.to(tl.float32)

            # block-level reduction in registers (no extra memory traffic)
            n = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            bsum = tl.sum(x_val, axis=0)
            bmean = bsum / n
            bm2 = tl.sum(tl.where(mask, (x_val - bmean) * (x_val - bmean), 0.0), axis=0)

            # Chan et al. merge into the running (n, mean, m2)
            delta = bmean - mean_total
            new_n = n_total + n
            mean_total = mean_total + delta * (n / new_n)
            m2_total = m2_total + bm2 + delta * delta * (n_total * n / new_n)
            n_total = new_n

    return n_total, mean_total, m2_total


@triton.jit
def welford_combine(
    ws_ptr,
    bg,
    SPLIT: tl.constexpr,
    SPLIT_POW2: tl.constexpr,
):
    """Merge the ``SPLIT`` partial ``(n, mean, m2)`` triples for one group.

    This is the Chan et al. parallel form, evaluated across all partials at once
    rather than folded in pairwise: the merged M2 is the sum of the partial M2s
    plus the between-partial spread ``n_i * (mean_i - mean)^2``. What it must not
    be is ``E[x^2] - E[x]^2`` reconstructed from the partials -- that is the
    cancellation the per-program Welford exists to avoid, and rebuilding it here
    would undo all of it.

    A shifted-mean variant (accumulating ``n_i * (mean_i - mean_0)`` instead of
    ``n_i * mean_i``) was measured and is not better: with ``SPLIT`` on the order
    of ten and ``tl.sum`` reducing as a tree, the weighted average is already
    accurate to a few ULP, and the residual error is dominated by ``mean_i``
    itself being an fp32 value -- at an offset of 1e4 that is ~1e-3 of
    quantization no arithmetic here can recover. The straightforward form is
    kept because it is the one a reader can check.
    """
    idx = tl.arange(0, SPLIT_POW2)
    live = idx < SPLIT
    row = ws_ptr + bg * (3 * SPLIT)

    # Padding lanes load 0, which contributes nothing to any of the sums below:
    # n_i = 0 zeroes both weighted terms and m2_i = 0 zeroes the last one.
    n_i = tl.load(row + 0 * SPLIT + idx, mask=live, other=0.0)
    mu_i = tl.load(row + 1 * SPLIT + idx, mask=live, other=0.0)
    m2_i = tl.load(row + 2 * SPLIT + idx, mask=live, other=0.0)

    n_total = tl.sum(n_i, axis=0)
    mean = tl.sum(n_i * mu_i, axis=0) / n_total
    delta = mu_i - mean
    m2_total = tl.sum(m2_i + n_i * delta * delta, axis=0)

    # Broadcast to [1] so this returns the same shape as welford_group_range and
    # both branches of the caller stay interchangeable.
    zero = tl.zeros([1], dtype=tl.float32)
    return n_total + zero, mean + zero, m2_total + zero


@triton.autotune(configs=SPLIT_REDUCTION_CONFIGS, key=SPLIT_REDUCTION_KEY, prune_configs_by=SPLIT_REDUCTION_PRUNE)
@triton.jit
def group_norm_partial_stats_kernel(
    x_ptr,
    ws_ptr,
    # Shape info; x is contiguous (B, C, spatial_size)
    C,
    spatial_size,
    split_chunk,
    num_groups: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Pass 1 of the split reduction: per ``(batch, group, slice)`` statistics.

    Grid is ``(B * num_groups * SPLIT,)``. Consecutive programs cover adjacent
    spatial slices of the same group, which keeps their loads neighbours in L2.
    Each writes one unique ``(n, mean, m2)`` slot, so no atomics and no locks.
    ``ws`` is laid out ``(B * num_groups, 3, SPLIT)``.
    """
    pid = tl.program_id(0)
    bg = pid // SPLIT
    s = pid % SPLIT

    group_size = C // num_groups
    n_idx = bg // num_groups
    g_idx = bg % num_groups

    lo = s * split_chunk
    hi = tl.minimum(lo + split_chunk, spatial_size)

    n_total, mean_total, m2_total = welford_group_range(
        x_ptr, n_idx, g_idx, C, spatial_size, group_size, lo, hi, BLOCK_SIZE, num_stages
    )

    # ``tl.arange(0, 1)`` is the zero vector; adding it carries the [1] shape of
    # the accumulators onto the pointer without moving it.
    lane = tl.arange(0, 1)
    slot = ws_ptr + bg * (3 * SPLIT) + s + lane
    tl.store(slot + 0 * SPLIT, n_total)
    tl.store(slot + 1 * SPLIT, mean_total)
    tl.store(slot + 2 * SPLIT, m2_total)


def launch_partial_stats(
    x_flat: torch.Tensor,
    batch: int,
    channels: int,
    spatial_size: int,
    num_groups: int,
    split: int,
    split_chunk: int,
) -> torch.Tensor:
    """Allocate the workspace and run pass 1. Only called when ``split > 1``."""
    # Every slot is written by exactly one program, so uninitialized is fine.
    ws = torch.empty((batch * num_groups, 3, split), dtype=torch.float32, device=x_flat.device)
    group_norm_partial_stats_kernel[(batch * num_groups * split,)](
        x_flat,
        ws,
        channels,
        spatial_size,
        split_chunk,
        num_groups=num_groups,
        SPLIT=split,
    )
    return ws


__all__ = [
    "SPLIT_ALIGN",
    "SPLIT_REDUCTION_CONFIGS",
    "SPLIT_REDUCTION_KEY",
    "SPLIT_REDUCTION_PRUNE",
    "group_norm_partial_stats_kernel",
    "launch_partial_stats",
    "pick_split",
    "welford_combine",
    "welford_group_range",
]
