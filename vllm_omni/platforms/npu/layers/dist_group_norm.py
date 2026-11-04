# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803  # Triton constexpr parameters follow kernel naming conventions.
#
"""NPU Triton distributed GroupNorm for channel-first and channel-last tensors.

Key design points:

* Three stages: collect per (batch, spatial-block, group) partial stats -> combine
  (locally + once across ranks) -> apply affine/activation.  All reductions use
  single-source ``tl.sum`` only.
* Numerically stable single pass: the collect uses the shifted-data ("assumed
  mean") trick -- subtract a per-channel shift ``x0`` (the block's first row) and
  accumulate ``sum(x-x0)`` / ``sum((x-x0)^2)`` so the squared accumulation runs
  over ``O(sigma)`` values; per-channel / per-group / cross-block / cross-rank
  stats are combined with the exact parallel (Chan et al. / Welford) formula.
  This matches the two-pass fp32 reference and, because it normalizes in fp32, is
  ~2x more accurate than the model's bf16-normalize ``_forward_torch``.
* Tiles are accumulated *elementwise* into a 2D buffer with a single cross-row
  reduction at the end (per-tile axis-0 reduction is very slow on NPU).
* Backend: the collect (strided spatial reduction) is ~10x faster in SIMT, while
  the apply (contiguous elementwise) is ~4x faster in SIMD; so we run a hybrid --
  collect on SIMT (``force_simt_only=True``), apply/reduce on SIMD.
* Cross-rank: one small ``all_gather_into_tensor`` of packed (mean, m2, count),
  no CPU gloo count sync (vs the model's gloo all_reduce + ``.item()``).

Public entry point: :func:`dist_group_norm`.  The op contains an HCCL collective
and runs eagerly (the VAE encoder is not inside the LLM ACLGraph), so it is a
plain function rather than a ``direct_register_custom_op`` custom op.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

try:
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
except Exception:  # pragma: no cover - triton_utils optional at import time
    get_vectorcore_num = None

_DEFAULT_VECTORCORE = 40

# Hybrid backend: collect (strided spatial reduction) -> SIMT; apply/reduce
# (contiguous elementwise / tiny combine) -> SIMD (the Ascend Triton default).
_COLLECT_KW = {"force_simt_only": True}
_REDUCE_KW: dict = {}
_APPLY_KW: dict = {}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _vectorcore_num() -> int:
    if get_vectorcore_num is not None:
        try:
            n = int(get_vectorcore_num())
            if n > 0:
                return n
        except Exception:
            pass
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
        n = int(props.get("num_vectorcore", -1))
        if n > 0:
            return n
    except Exception:
        pass
    return _DEFAULT_VECTORCORE


@triton.jit
def _collect_welford_kernel(
    x_ptr,
    mean_partial_ptr,  # (N, S, G) fp32
    m2_partial_ptr,  # (N, S, G) fp32  (sum of squared deviations)
    weight_partial_ptr,  # (N, S) fp32     (element count per group/block)
    channels,
    spatial,
    spatial_block,
    num_sblocks,
    num_groups,
    BLOCK_CHANNELS: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL_TILE: tl.constexpr,
):
    pid_cblock = tl.program_id(0)
    pid_sblock = tl.program_id(1)
    pid_batch = tl.program_id(2)

    chan = pid_cblock * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    chan_mask = chan < channels
    x_base = x_ptr + pid_batch.to(tl.int64) * spatial * channels

    sblock_start = pid_sblock * spatial_block
    sblock_end = tl.minimum(sblock_start + spatial_block, spatial)
    block_rows = sblock_end - sblock_start
    num_tiles = tl.cdiv(block_rows, SPATIAL_TILE)

    # Shifted-data ("assumed mean") shift = block's first row, per channel.
    x0_offs = sblock_start.to(tl.int64) * channels + chan
    x0 = tl.load(x_base + x0_offs, mask=chan_mask, other=0.0).to(tl.float32)

    acc_sum = tl.zeros((SPATIAL_TILE, BLOCK_CHANNELS), tl.float32)
    acc_sumsq = tl.zeros((SPATIAL_TILE, BLOCK_CHANNELS), tl.float32)
    for t in tl.range(0, num_tiles):
        rows = sblock_start + t * SPATIAL_TILE + tl.arange(0, SPATIAL_TILE)
        row_valid = rows < sblock_end
        mask = row_valid[:, None] & chan_mask[None, :]
        offs = rows.to(tl.int64)[:, None] * channels + chan[None, :]
        x = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        xc = (x - x0[None, :]) * row_valid[:, None].to(tl.float32)
        acc_sum += xc
        acc_sumsq += xc * xc

    ch_sum_d = tl.sum(acc_sum, axis=0)  # Sum(x - x0)
    ch_sumsq_d = tl.sum(acc_sumsq, axis=0)  # Sum((x - x0)^2)
    cnt = block_rows.to(tl.float32)
    ch_mean = x0 + ch_sum_d / cnt
    ch_m2 = ch_sumsq_d - ch_sum_d * (ch_sum_d / cnt)  # shift-invariant, stable

    m2_r = tl.reshape(ch_m2, (GROUPS_PER_BLOCK, CHANNELS_PER_GROUP))
    mean_r = tl.reshape(ch_mean, (GROUPS_PER_BLOCK, CHANNELS_PER_GROUP))
    gcount = cnt * CHANNELS_PER_GROUP
    gmean = tl.sum(mean_r, axis=1) / CHANNELS_PER_GROUP
    gdev = mean_r - gmean[:, None]
    gm2 = tl.sum(m2_r, axis=1) + cnt * tl.sum(gdev * gdev, axis=1)

    group_idx = pid_cblock * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
    group_mask = group_idx < num_groups
    out = (pid_batch * num_sblocks + pid_sblock) * num_groups + group_idx
    tl.store(mean_partial_ptr + out, gmean, mask=group_mask)
    tl.store(m2_partial_ptr + out, gm2, mask=group_mask)
    if pid_cblock == 0:
        tl.store(weight_partial_ptr + pid_batch * num_sblocks + pid_sblock, gcount)


@triton.jit
def _reduce_welford_kernel(
    mean_partial_ptr,
    m2_partial_ptr,
    weight_partial_ptr,
    num_groups,
    num_blocks,
    eps,
    mean_out_ptr,
    rstd_out_ptr,
    m2_out_ptr,
    weight_out_ptr,
    BLOCK_TILE: tl.constexpr,
    FINALIZE: tl.constexpr,
):
    group = tl.program_id(0)
    batch = tl.program_id(1)

    blk = tl.arange(0, BLOCK_TILE)
    mask = blk < num_blocks
    idx = (batch * num_blocks + blk) * num_groups + group
    mean_b = tl.load(mean_partial_ptr + idx, mask=mask, other=0.0)
    m2_b = tl.load(m2_partial_ptr + idx, mask=mask, other=0.0)
    w_b = tl.load(weight_partial_ptr + batch * num_blocks + blk, mask=mask, other=0.0)

    total = tl.sum(w_b)
    safe = tl.where(total > 0.0, total, 1.0)
    gmean = tl.sum(w_b * mean_b) / safe
    dev = mean_b - gmean
    gm2 = tl.sum(m2_b) + tl.sum(w_b * dev * dev)

    out = batch * num_groups + group
    tl.store(mean_out_ptr + out, gmean)
    if FINALIZE:
        tl.store(rstd_out_ptr + out, 1.0 / tl.sqrt(gm2 / safe + eps))
    else:
        tl.store(m2_out_ptr + out, gm2)
        if group == 0:
            tl.store(weight_out_ptr + batch, total)


@triton.jit
def _reduce_cross_rank_kernel(
    gathered_ptr,  # (world, N, 2G+1) contiguous: [mean(G) | m2(G) | weight(1)]
    world,
    n,
    num_groups,
    eps,
    mean_out_ptr,
    rstd_out_ptr,
    WORLD_TILE: tl.constexpr,
):
    group = tl.program_id(0)
    batch = tl.program_id(1)
    stride_w = n * (2 * num_groups + 1)

    w = tl.arange(0, WORLD_TILE)
    mask = w < world
    base = w * stride_w + batch * (2 * num_groups + 1)
    mean_b = tl.load(gathered_ptr + base + group, mask=mask, other=0.0)
    m2_b = tl.load(gathered_ptr + base + num_groups + group, mask=mask, other=0.0)
    w_b = tl.load(gathered_ptr + base + 2 * num_groups, mask=mask, other=0.0)

    total = tl.sum(w_b)
    safe = tl.where(total > 0.0, total, 1.0)
    gmean = tl.sum(w_b * mean_b) / safe
    dev = mean_b - gmean
    gm2 = tl.sum(m2_b) + tl.sum(w_b * dev * dev)

    out = batch * num_groups + group
    tl.store(mean_out_ptr + out, gmean)
    tl.store(rstd_out_ptr + out, 1.0 / tl.sqrt(gm2 / safe + eps))


@triton.jit
def _apply_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    channels,
    spatial,
    num_groups,
    channels_per_group,
    num_ablocks,
    total_work,
    y_ptr,
    CHANNELS_P2: tl.constexpr,
    SPATIAL_BLOCK: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
    ACT_SWISH: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    chan = tl.arange(0, CHANNELS_P2)
    chan_mask = chan < channels
    group = chan // channels_per_group
    group_mask = group < num_groups

    for work in tl.range(pid, total_work, nprog):
        pid_batch = work // num_ablocks
        pid_sblock = work % num_ablocks

        mean = tl.load(mean_ptr + pid_batch * num_groups + group, mask=group_mask, other=0.0)
        rstd = tl.load(rstd_ptr + pid_batch * num_groups + group, mask=group_mask, other=0.0)
        if HAS_AFFINE:
            gamma = tl.load(gamma_ptr + chan, mask=chan_mask, other=0.0).to(tl.float32)
            beta = tl.load(beta_ptr + chan, mask=chan_mask, other=0.0).to(tl.float32)
            scale = rstd * gamma
            shift = beta - scale * mean
        else:
            scale = rstd
            shift = -scale * mean

        rows = pid_sblock * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
        mask = (rows < spatial)[:, None] & chan_mask[None, :]
        offs = pid_batch.to(tl.int64) * spatial * channels + rows.to(tl.int64)[:, None] * channels + chan[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = scale[None, :] * x + shift[None, :]
        if ACT_SWISH:
            y = y * tl.sigmoid(y)
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def _pick_collect_tiling(spatial, channels, num_groups, batch):
    cpg = channels // num_groups
    # ~256B-wide channel block, snapped to a whole number of groups, capped at C.
    target = max(cpg, (128 // cpg) * cpg)
    block_channels = min(channels, target)
    block_channels = max(block_channels, cpg)
    # Ascend Triton requires the upper bound of tl.arange to be a power of two.
    # The reduction tile requires a power-of-two channel block.
    if not _is_power_of_two(block_channels):
        raise ValueError(
            f"unsupported non-power-of-two block_channels={block_channels} "
            f"for channels={channels}, num_groups={num_groups}"
        )
    groups_per_block = block_channels // cpg
    num_cblocks = triton.cdiv(channels, block_channels)

    # Keep the two (SPATIAL_TILE, BLOCK_CHANNELS) fp32 accumulators within UB.
    spatial_tile = max(8, 8192 // block_channels)
    cores = _vectorcore_num()
    target_blocks = 2 * cores
    num_sblocks = max(1, target_blocks // max(1, num_cblocks * batch))
    num_sblocks = min(num_sblocks, max(1, triton.cdiv(spatial, spatial_tile)))
    spatial_block = triton.cdiv(spatial, num_sblocks)
    num_sblocks = triton.cdiv(spatial, spatial_block)
    return spatial_tile, block_channels, groups_per_block, num_cblocks, spatial_block, num_sblocks


def _collect(x, num_groups):
    n, h, w, channels = x.shape
    spatial = h * w
    cpg = channels // num_groups
    spatial_tile, bc, gpb, num_cblocks, spatial_block, num_sblocks = _pick_collect_tiling(
        spatial, channels, num_groups, n
    )
    mean_p = torch.empty((n, num_sblocks, num_groups), dtype=torch.float32, device=x.device)
    m2_p = torch.empty_like(mean_p)
    w_p = torch.empty((n, num_sblocks), dtype=torch.float32, device=x.device)
    _collect_welford_kernel[(num_cblocks, num_sblocks, n)](
        x,
        mean_p,
        m2_p,
        w_p,
        channels,
        spatial,
        spatial_block,
        num_sblocks,
        num_groups,
        BLOCK_CHANNELS=bc,
        GROUPS_PER_BLOCK=gpb,
        CHANNELS_PER_GROUP=cpg,
        SPATIAL_TILE=spatial_tile,
        **_COLLECT_KW,
    )
    return mean_p, m2_p, w_p


def _reduce(mean_p, m2_p, w_p, eps, finalize):
    n, num_blocks, num_groups = mean_p.shape
    mean = torch.empty((n, num_groups), dtype=torch.float32, device=mean_p.device)
    rstd = torch.empty_like(mean) if finalize else None
    m2 = None if finalize else torch.empty_like(mean)
    weight = None if finalize else torch.empty((n,), dtype=torch.float32, device=mean_p.device)
    _reduce_welford_kernel[(num_groups, n)](
        mean_p,
        m2_p,
        w_p,
        num_groups,
        num_blocks,
        eps,
        mean,
        rstd,
        m2,
        weight,
        BLOCK_TILE=triton.next_power_of_2(num_blocks),
        FINALIZE=finalize,
        **_REDUCE_KW,
    )
    return (mean, rstd) if finalize else (mean, m2, weight)


def _reduce_cross_rank(gathered, num_groups, eps):
    world, n, _ = gathered.shape
    mean = torch.empty((n, num_groups), dtype=torch.float32, device=gathered.device)
    rstd = torch.empty_like(mean)
    _reduce_cross_rank_kernel[(num_groups, n)](
        gathered,
        world,
        n,
        num_groups,
        eps,
        mean,
        rstd,
        WORLD_TILE=triton.next_power_of_2(world),
        **_REDUCE_KW,
    )
    return mean, rstd


def _apply(x, mean, rstd, weight, bias, activation):
    n, h, w, channels = x.shape
    spatial = h * w
    num_groups = mean.shape[-1]
    cpg = channels // num_groups
    # Intentionally return the same dtype as x. This differs from the reference
    # torch path's fp32 affine promotion, but avoids extra writes before the next
    # autocast DistConv2d and matches the production VAE validation setup.
    y = torch.empty_like(x)
    channels_p2 = triton.next_power_of_2(channels)
    apply_block = max(1, min(triton.next_power_of_2(spatial), 16384 // channels_p2))
    num_ablocks = triton.cdiv(spatial, apply_block)
    total_work = n * num_ablocks
    num_programs = max(1, min(total_work, _vectorcore_num()))
    _apply_norm_kernel[(num_programs,)](
        x,
        weight,
        bias,
        mean,
        rstd,
        channels,
        spatial,
        num_groups,
        cpg,
        num_ablocks,
        total_work,
        y,
        CHANNELS_P2=channels_p2,
        SPATIAL_BLOCK=apply_block,
        HAS_AFFINE=weight is not None,
        ACT_SWISH=(activation == "swish"),
        **_APPLY_KW,
    )
    return y


@triton.jit
def _collect_cf(
    X, MP, M2P, WP, L: tl.constexpr, G: tl.constexpr, NB: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr
):
    offsets = tl.arange(0, B)
    for work in tl.range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        block = work % NB
        group = work // NB % G
        batch = work // (NB * G)
        i = block * B + offsets
        valid = i < L
        count = tl.minimum(B, L - block * B).to(tl.float32)
        base = (batch * G + group) * L
        shift = tl.load(X + base + block * B).to(tl.float32)
        x = tl.load(X + base + i, mask=valid, other=0).to(tl.float32)
        d = tl.where(valid, x - shift, 0.0)
        sd = tl.sum(d, 0)
        ss = tl.sum(d * d, 0)
        mean = shift + sd / count
        m2 = ss - sd * (sd / count)
        out = (batch * NB + block) * G + group
        tl.store(MP + out, mean)
        tl.store(M2P + out, m2)
        if group == 0:
            tl.store(WP + batch * NB + block, count)


@triton.jit
def _apply_cf(
    X,
    W,
    Bias,
    Mean,
    Rstd,
    Y,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    NB: tl.constexpr,
    B: tl.constexpr,
    ACT: tl.constexpr,
):
    offsets = tl.arange(0, B)
    nsb = tl.cdiv(S, B)
    worktotal = tl.cdiv(TOTAL, C * S) * C * nsb
    for work in tl.range(tl.program_id(0), worktotal, tl.num_programs(0)):
        block = work % nsb
        channel = work // nsb % C
        batch = work // (nsb * C)
        group = channel // (C // G)
        mean = tl.load(Mean + batch * G + group)
        rstd = tl.load(Rstd + batch * G + group)
        gamma = tl.load(W + channel).to(tl.float32)
        beta = tl.load(Bias + channel).to(tl.float32)
        i = block * B + offsets
        mask = i < S
        pos = (batch * C + channel) * S + i
        x = tl.load(X + pos, mask=mask, other=0).to(tl.float32)
        y = (x - mean) * rstd * gamma + beta
        if ACT:
            y = y * tl.sigmoid(y)
        tl.store(Y + pos, y, mask=mask)


def _group_norm_ncthw(
    x: torch.Tensor,
    groups: int,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-06,
    activation: str = "swish",
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if activation not in ("swish", "identity"):
        raise ValueError("activation must be swish or identity")
    assert x.is_contiguous() and x.dtype in (torch.float32, torch.float16, torch.bfloat16) and x.ndim in (4, 5)
    n, c = x.shape[:2]
    s = x.numel() // (n * c)
    group_elements = c // groups * s
    block = 4096
    nb = triton.cdiv(group_elements, block)
    mp = torch.empty((n, nb, groups), device=x.device, dtype=torch.float32)
    m2p = torch.empty_like(mp)
    wp = torch.empty((n, nb), device=x.device, dtype=torch.float32)
    cores = _vectorcore_num() * 2
    _collect_cf[min(n * groups * nb, cores),](x, mp, m2p, wp, group_elements, groups, nb, n * groups * nb, B=block)
    mean, rstd = _reduce(mp, m2p, wp, eps, finalize=True)
    y = torch.empty_like(x, dtype=output_dtype or x.dtype)
    total = x.numel()
    nab = triton.cdiv(total, block)
    _apply_cf[min(nab, cores),](
        x,
        w,
        b,
        mean,
        rstd,
        y,
        total,
        c,
        s,
        groups,
        nab,
        B=block,
        ACT=activation == "swish",
    )
    return y


def _distributed_group_norm_ncthw(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    group: torch.distributed.ProcessGroup,
    activation: str = "swish",
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """GroupNorm over equal-height shards with stable FP32 global statistics."""
    if activation not in ("swish", "identity"):
        raise ValueError(f"Unsupported GroupNorm activation: {activation}")
    if x.ndim != 5 or x.shape[1] % num_groups or not x.numel():
        raise ValueError("Expected a nonempty NCTHW tensor with channels divisible by groups")

    world = dist.get_world_size(group)

    x = x.contiguous()
    n, c = x.shape[:2]
    sp = x.numel() // (n * c)
    g = num_groups
    group_elements = c // g * sp
    block = 4096
    nb = triton.cdiv(group_elements, block)
    cores = _vectorcore_num() * 2
    mp = torch.empty((n, nb, g), device=x.device, dtype=torch.float32)
    m2 = torch.empty_like(mp)
    wp = torch.empty((n, nb), device=x.device, dtype=torch.float32)
    _collect_cf[min(n * g * nb, cores),](x, mp, m2, wp, group_elements, g, nb, n * g * nb, B=block)
    mean, m2, count = _reduce(mp, m2, wp, eps, finalize=False)
    packed = torch.cat((mean, m2, count.unsqueeze(1)), dim=1)
    parts = torch.empty((world, n, 2 * g + 1), device=x.device, dtype=torch.float32)
    dist.all_gather_into_tensor(parts, packed, group=group)
    mean, rstd = _reduce_cross_rank(parts, g, eps)
    y = torch.empty_like(x, dtype=output_dtype or x.dtype)
    total = x.numel()
    nab = triton.cdiv(total, block)
    _apply_cf[min(nab, cores),](
        x, weight, bias, mean, rstd, y, total, c, sp, g, nab, B=block, ACT=activation == "swish"
    )
    return y


def dist_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    activation: str = "identity",
    device_group: dist.ProcessGroup | None = None,
    local_only: bool = False,
    layout: str = "NHWC",
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused local/distributed GroupNorm for explicit NCTHW or NHWC layout.

    Args:
        x: ``(N, H, W, C)`` NHWC, contiguous; bf16/fp16/fp32 activations.
        num_groups: number of groups; ``C % num_groups == 0`` required.
        weight: ``(C,)`` affine gamma (fp32 under VAE autocast), or None.
        bias: ``(C,)`` affine beta (fp32 under VAE autocast), or None.
        eps: variance epsilon.
        activation: ``"identity"`` or ``"swish"`` (fused into the norm).
        layout: explicit physical tensor layout: rank-4 NHWC or rank-5 NCTHW.
            NHWC remains the default for compatibility; no implicit layout copy.
        output_dtype: optional output precision; NCTHW fuses its conversion.
        device_group: process group to reduce stats over (the caller-owned spatial group);
            ``None`` -> default group; a single-rank group -> no collective.

    Returns a tensor with the same shape/dtype as ``x``.
    """
    if layout == "NCTHW":
        if x.ndim != 5:
            raise ValueError("NCTHW GroupNorm expects a rank-5 tensor")
        if weight is None:
            weight = x.new_ones(x.shape[1])
        if bias is None:
            bias = x.new_zeros(x.shape[1])

        if local_only or not dist.is_initialized():
            return _group_norm_ncthw(x, num_groups, weight, bias, eps, activation=activation, output_dtype=output_dtype)
        return _distributed_group_norm_ncthw(
            x,
            num_groups,
            weight,
            bias,
            eps,
            device_group,
            activation=activation,
            output_dtype=output_dtype,
        )
    if layout != "NHWC":
        raise ValueError(f"Unsupported GroupNorm layout: {layout}; expected NHWC or NCTHW")
    if x.ndim != 4:
        raise ValueError(f"expected NHWC rank-4 input, got {x.ndim}D")
    channels = x.shape[-1]
    if channels % num_groups != 0:
        raise ValueError(f"channels={channels} not divisible by num_groups={num_groups}")
    if activation not in ("identity", "swish"):
        raise ValueError(f"unsupported activation {activation!r}")
    x = x.contiguous()

    mean_p, m2_p, w_p = _collect(x, num_groups)

    world = dist.get_world_size(device_group) if dist.is_initialized() and not local_only else 1
    if world > 1:
        mean, m2, weight_local = _reduce(mean_p, m2_p, w_p, eps, finalize=False)
        n, g = mean.shape
        packed = torch.cat((mean, m2, weight_local.unsqueeze(1)), dim=1)  # (N, 2G+1)
        gathered = torch.empty((world, n, 2 * g + 1), dtype=packed.dtype, device=packed.device)
        dist.all_gather_into_tensor(gathered, packed, group=device_group)
        mean, rstd = _reduce_cross_rank(gathered, g, eps)
    else:
        mean, rstd = _reduce(mean_p, m2_p, w_p, eps, finalize=True)

    y = _apply(x, mean, rstd, weight, bias, activation)
    return y.to(output_dtype) if output_dtype is not None else y
