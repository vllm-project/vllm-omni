# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Routing kernels adapted from SGLang (Apache-2.0), commit
# 367e3700cfb6a0b03b2fa41a4524febf18ec1f15,
# python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/kernels.py.
# Recipe reference: 704808ed27cef61e210100e7581c115db3ee8401.
# Modified to integrate GQA and protected-prefix routing in vLLM-Omni.
"""SubBlock pooled scoring and approximate top-k selection."""

import math

import torch
from vllm.triton_utils import tl, triton

from .abstract import BlockSelection, BlockSelector, validate_protected_kv_prefix


@torch.library.custom_op("vllm_omni::select_subblocks", mutates_args=())
def select_subblocks(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    sparsity: float,
    protected_prefix: int,
    block_q: int,
    block_kv: int,
) -> torch.Tensor:
    return SubBlockTopK._route(query, key, scale, sparsity, protected_prefix, (block_q, block_kv))


@select_subblocks.register_fake
def _select_subblocks_fake(query, key, scale, sparsity, protected_prefix, block_q, block_kv):
    prefix_blocks = (protected_prefix + block_kv - 1) // block_kv
    key_blocks = (key.shape[1] + block_kv - 1) // block_kv
    return torch.empty(
        (
            query.shape[0],
            query.shape[2],
            (query.shape[1] + block_q - 1) // block_q,
            prefix_blocks + SubBlockTopK.block_budget(key_blocks - prefix_blocks, sparsity),
        ),
        dtype=torch.int32,
        device=query.device,
    )


class SubBlockTopK(BlockSelector):
    """Keep protected blocks plus a rounded budget of unprotected candidates.

    target_sparsity applies only to the unprotected blocks. Prefix protection
    never consumes that candidate budget; rounding and protection both reduce
    the overall achieved sparsity. A prefix protects every intersecting block,
    including its non-prefix tokens, and an entirely protected sequence is dense.
    """

    @staticmethod
    def block_budget(num_blocks, sparsity):
        """Budget for unprotected blocks: ceil retained fraction, align to eight, cap at N."""
        if num_blocks < 0 or not 0 <= sparsity < 1:
            raise ValueError("SubBlock requires a nonnegative candidate count and 0 <= sparsity < 1")
        if num_blocks == 0:
            return 0
        return min(num_blocks, math.ceil(max(1, math.ceil((1 - sparsity) * num_blocks)) / 8) * 8)

    @staticmethod
    def _validate_geometry(block_size):
        """The pooled-score Triton tiles must contain whole logical blocks."""
        if len(block_size) != 2 or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 16
            or size % 16
            or size & (size - 1)
            or size // 16 > tile
            for size, tile in zip(block_size, (block_m, block_n))
        ):
            raise ValueError(
                "The SubBlock scorer requires power-of-two block dimensions from 16 to 2048 "
                "(16-token pooling cells dividing its 128-cell score tiles)"
            )

    @classmethod
    def _route(cls, query, key, scale, sparsity, protected_prefix, block_size):
        block_q, block_kv = block_size
        rows = (query.shape[0], query.shape[2], math.ceil(query.shape[1] / block_q))
        key_blocks = math.ceil(key.shape[1] / block_kv)
        prefix_blocks = math.ceil(protected_prefix / block_kv)
        kept = prefix_blocks + cls.block_budget(key_blocks - prefix_blocks, sparsity)
        # These patterns are known without pooling, scoring or top-k selection.
        if kept == key_blocks:
            indices = torch.arange(kept, dtype=torch.int32, device=query.device)
            return indices.expand(*rows, kept).contiguous()
        scores = cls._compute_scores(query, key, scale, block_size)
        return cls._select_blocks(scores, kept, prefix_blocks)

    @staticmethod
    def _compute_scores(query, key, scale, block_size):
        block_q, block_kv = block_size
        n_q, n_k = block_q // 16, block_kv // 16
        b, sq, hq, d = query.shape
        _, sk, hk, _ = key.shape
        gq, gk = math.ceil(sq / block_q), math.ceil(sk / block_kv)
        qp = torch.empty(b * hq, gq * n_q, d, device=query.device, dtype=query.dtype)
        kp = torch.empty(b * hk, gk * n_k, d, device=key.device, dtype=key.dtype)
        _fused_pool(query, gq * n_q, 16, qp, scale=scale * math.log2(math.e))
        _fused_pool(key, gk * n_k, 16, kp)
        scores = torch.empty(b * hq, gq, gk, device=query.device, dtype=torch.float32)
        _fused_scores(
            qp,
            kp,
            scores,
            n_k=n_k,
            n_valid=math.ceil(sk / 16),
            n_q=n_q,
            m_valid=math.ceil(sq / 16),
            heads_per_kv=hq // hk,
        )
        return scores.view(b, hq, gq, gk)

    @staticmethod
    def _select_blocks(scores, kept, prefix_blocks):
        # Reserve protected blocks structurally, outside approximate top-k.
        rows, key_blocks = scores.shape[:-1], scores.shape[-1]
        remaining = kept - prefix_blocks
        candidates = scores[..., prefix_blocks:].contiguous().view(-1, key_blocks - prefix_blocks)
        chosen = (_fused_topk(candidates, remaining) + prefix_blocks).view(*rows, remaining)
        if not prefix_blocks:
            return chosen
        prefix = torch.arange(prefix_blocks, dtype=torch.int32, device=scores.device)
        return torch.cat((prefix.expand(*rows, prefix_blocks), chosen), dim=-1)

    @classmethod
    def normalize_config(cls, options):
        if options.keys() - {"target_sparsity"}:
            raise ValueError("block_topk config only supports target_sparsity")
        sparsity = options.get("target_sparsity", 0.75)
        if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)) or not 0 <= sparsity < 1:
            raise ValueError("Require 0 <= target_sparsity < 1")
        return {"target_sparsity": float(sparsity)}

    def __init__(self, options):
        options = self.normalize_config(options)
        self.sparsity = options["target_sparsity"]

    def prepare(self, block_size, head_size, device):
        self._validate_geometry(block_size)
        if device.type != "cuda" or head_size <= 0:
            raise ValueError("SubBlock top-k requires CUDA and a positive head dimension")
        self.block_size = block_size

    def validate_request(self, query, key, protected_prefix):
        if key.shape[1] == 0:
            raise ValueError("SubBlock requires nonempty keys")
        if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype:
            raise ValueError("SubBlock top-k requires matching FP16 or BF16 Q/K")

        validate_protected_kv_prefix(protected_prefix, key.shape[1])

    def select(self, query, key, scale, protected_prefix):
        self.validate_request(query, key, protected_prefix)
        indices = select_subblocks(query, key, scale, self.sparsity, protected_prefix, *self.block_size)
        counts = torch.full(indices.shape[:-1], indices.shape[-1], dtype=torch.int32, device=indices.device)
        return BlockSelection(indices, counts)


@triton.jit
def _pool_kernel(
    input_ptr,
    output_ptr,
    stride_xb,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_yl,
    stride_yn,
    sequence_length,
    heads,
    subblock_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    scale,
):
    """Mean-pool token sub-blocks in the input dtype, masking token and head tails."""
    cell = tl.program_id(0)
    batch_head = tl.program_id(1)
    b = batch_head // heads
    h = batch_head % heads
    offs_t = cell * subblock_size + tl.arange(0, subblock_size)
    offs_d = tl.arange(0, block_d)
    mask = offs_t < sequence_length
    x = tl.load(
        input_ptr + b * stride_xb + offs_t[:, None] * stride_xt + h * stride_xh + offs_d[None, :] * stride_xd,
        mask=mask[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    cnt = tl.sum(mask.to(tl.float32), axis=0)
    acc = tl.sum(x, axis=0) / tl.maximum(cnt, 1.0) * scale
    tl.store(
        output_ptr + batch_head * stride_yl + cell * stride_yn + offs_d,
        acc.to(output_ptr.dtype.element_ty),
        mask=offs_d < head_dim,
    )


def _fused_pool(x, n_cells, sub, out, scale=1.0):
    """Pool [B, sequence_length, heads, head_dim] into [B*heads, cells, head_dim]."""
    B, sequence_length, heads, head_dim = x.shape
    _pool_kernel[(n_cells, B * heads)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        sequence_length,
        heads,
        subblock_size=sub,
        head_dim=head_dim,
        block_d=max(16, triton.next_power_of_2(head_dim)),
        scale=scale,
        num_warps=1,
    )
    return out


_NEG = tl.constexpr(-1.0e30)
_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _score_kernel(
    query_ptr,
    key_ptr,
    output_ptr,
    stride_qm,
    stride_ql,
    stride_kn,
    stride_kl,
    stride_om,
    stride_on,
    stride_ol,
    pooled_q_len,
    valid_q_cells,
    valid_k_cells,
    key_blocks,
    query_blocks,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    key_subblocks: tl.constexpr,
    query_subblocks: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    heads_per_kv: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)

    q = tl.load(
        query_ptr + pid_l * stride_ql + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=(offs_m[:, None] < pooled_q_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    k = tl.load(
        key_ptr + (pid_l // heads_per_kv) * stride_kl + offs_n[:, None] * stride_kn + offs_d[None, :],
        mask=(offs_n[:, None] < valid_k_cells) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)  # [block_m, block_n]

    # a sub-cell past the last real one must not contribute to its group's log-sum-exp
    acc = tl.where(offs_n[None, :] < valid_k_cells, acc, _NEG)
    # same on the query side: with NQ > 1 the last query block can own sub-cells that
    # are entirely padding, and those pool to zero -- an exp2(0) = 1 term that would
    # otherwise be folded into the block's score.
    acc = tl.where(offs_m[:, None] < valid_q_cells, acc, _NEG)

    acc = tl.reshape(acc, (block_m, block_n // key_subblocks, key_subblocks))
    m = tl.max(acc, axis=2)
    s = tl.sum(tl.exp2(acc - m[:, :, None]), axis=2)
    lse = m + tl.log2(s)
    lse = tl.where(m > _NEG / 2, lse, _NEG)  # whole group was padding

    # Fold the query_subblocks query sub-cells of a query block together. Log-sum-exp is
    # associative, so reducing key_subblocks then NQ is the same one log-sum-exp over all
    # NQ*key_subblocks sub-block pairs -- and two stages keeps both reductions on an axis
    # that is already contiguous in registers.
    if query_subblocks > 1:
        lse = tl.reshape(lse, (block_m // query_subblocks, query_subblocks, block_n // key_subblocks))
        m2 = tl.max(lse, axis=1)
        s2 = tl.sum(tl.exp2(lse - m2[:, None, :]), axis=1)
        lse = tl.where(m2 > _NEG / 2, m2 + tl.log2(s2), _NEG)

    # exp2/log2 internally (they map to the hardware instructions), then back to natural
    # log units so the fused and reference backends return the same numbers, not just the
    # same ranking. One multiply in registers.
    out = lse * _LN2
    out = out.to(output_ptr.dtype.element_ty)

    offs_o = pid_n * (block_n // key_subblocks) + tl.arange(0, block_n // key_subblocks)
    offs_q = pid_m * (block_m // query_subblocks) + tl.arange(0, block_m // query_subblocks)
    tl.store(
        output_ptr + pid_l * stride_ol + offs_q[:, None] * stride_om + offs_o[None, :] * stride_on,
        out,
        mask=(offs_q[:, None] < query_blocks) & (offs_o[None, :] < key_blocks),
    )


block_m = block_n = 128  # score tile; must hold whole blocks, so a multiple of n_q and n_k


def _fused_scores(qp, kp, out, *, n_k, n_valid, n_q, m_valid, heads_per_kv=1):
    """Compute FP32 block scores from FP16 or BF16 pooled query/key sub-blocks."""
    L, pooled_q_len, head_dim = qp.shape
    N = kp.shape[1]
    Mout, Nout = out.shape[1], out.shape[2]
    grid = (triton.cdiv(pooled_q_len, block_m), triton.cdiv(N, block_n), L)
    _score_kernel[grid](
        qp,
        kp,
        out,
        qp.stride(1),
        qp.stride(0),
        kp.stride(1),
        kp.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(0),
        pooled_q_len,
        m_valid,
        n_valid,
        Nout,
        Mout,
        block_m=block_m,
        block_n=block_n,
        key_subblocks=n_k,
        query_subblocks=n_q,
        head_dim=head_dim,
        block_d=max(16, triton.next_power_of_2(head_dim)),
        heads_per_kv=heads_per_kv,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _topk_kernel(scores_ptr, output_ptr, columns, topk, block_size: tl.constexpr, iterations: tl.constexpr):
    """Select blocks using bounded threshold search and compact their indices."""
    row = tl.program_id(0)
    offs = tl.arange(0, block_size)
    m = offs < columns
    s = tl.load(scores_ptr + row * columns + offs, mask=m, other=-float("inf")).to(tl.float32)
    lo = tl.min(tl.where(m, s, float("inf")))
    hi = tl.max(tl.where(m, s, -float("inf"))) + 1.0
    clo = tl.sum(m.to(tl.int32), axis=0).to(tl.float32)
    chi = 0.0
    for _ in tl.static_range(iterations):
        den = clo - chi
        t = (clo - topk) / tl.where(den > 0.5, den, 1.0)
        t = tl.minimum(tl.maximum(t, 0.05), 0.95)  # keep the step inside the bracket
        mid = lo + (hi - lo) * t
        cnt = tl.sum(((s >= mid) & m).to(tl.int32), axis=0).to(tl.float32)
        take = cnt >= topk
        lo = tl.where(take, mid, lo)
        clo = tl.where(take, cnt, clo)
        hi = tl.where(take, hi, mid)
        chi = tl.where(take, chi, cnt)
    sel = (s >= lo) & m
    # Nonfinite scores have no reliable ranking (finite inputs can overflow
    # during pooling). Use the first k candidates for the entire affected row.
    # Also cover finite-score threshold arithmetic that cannot fill the budget.
    nonfinite = tl.sum((m & ((s != s) | (tl.abs(s) == float("inf")))).to(tl.int32), axis=0) > 0
    insufficient = tl.sum(sel.to(tl.int32), axis=0) < topk
    sel = tl.where(nonfinite | insufficient, m & (offs < topk), sel)
    pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
    tl.store(output_ptr + row * topk + pos, offs.to(tl.int32), mask=sel & (pos < topk))


def _topk_iters(columns, k):
    """Use more threshold-search iterations for higher sparsity."""
    L = math.log2(max(columns, 1) / max(k, 1))
    return 16 if L <= 2.5 else 24 if L <= 3.5 else 32


def _fused_topk(scores2d, k):
    """Return ascending int32 indices; prefix-sum compaction preserves column order."""
    rows, columns = scores2d.shape
    out = torch.empty(rows, k, dtype=torch.int32, device=scores2d.device)
    _topk_kernel[(rows,)](
        scores2d,
        out,
        columns,
        k,
        block_size=triton.next_power_of_2(columns),
        iterations=_topk_iters(columns, k),
        num_warps=4 if columns >= 1024 else 2,  # short rows do not fill four warps
    )
    return out
