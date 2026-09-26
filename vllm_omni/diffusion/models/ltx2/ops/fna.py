# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""TileLang FNA for the LTX-2.5 DiffVAE stage-5 schedule.

This intentionally reuses NATTEN's token permutation.  It only replaces the
middle BF16 attention kernel for Q tiles (4, 4, 4), KV tiles (4, 4, 8),
head_dim=64, stride=dilation=1, and scale=1.0.
"""

import functools
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from vllm.tilelang_utils import T, tilelang

Q_TILE = (4, 4, 4)
KV_TILE = (4, 4, 8)
Q_TOKENS = math.prod(Q_TILE)
KV_TOKENS = math.prod(KV_TILE)
HEAD_DIM = 64
HEADS = 4
LOG2_E = 1.4426950408889634
KV_BUCKET_LIMITS = (32, 48, 60, 75)
PATTERN_TABLE_SIZE = 16
KV_TILE_ID_BITS = 15
SHAPE_CACHE_SIZE = 16


@dataclass(frozen=True)
class Problem:
    frames: int
    height: int
    width: int
    window: tuple[int, int, int]

    def __post_init__(self) -> None:
        if min(self.frames, self.height, self.width) <= 0:
            raise ValueError(f"invalid FNA problem: {self}")
        if len(self.window) != 3 or any(x <= 0 for x in self.window):
            raise ValueError(f"invalid FNA window: {self.window}")
        if any(w > n for w, n in zip(self.window, self.shape)):
            raise ValueError(f"window {self.window} exceeds shape {self.shape}")
        if math.prod(self.kv_grid) > 1 << KV_TILE_ID_BITS:
            raise ValueError(f"KV tile grid exceeds {KV_TILE_ID_BITS}-bit metadata encoding: {self.kv_grid}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.frames, self.height, self.width

    @property
    def q_padded(self) -> tuple[int, int, int]:
        return tuple(_round_up(n, tile) for n, tile in zip(self.shape, Q_TILE))

    @property
    def kv_padded(self) -> tuple[int, int, int]:
        return tuple(_round_up(n, tile) for n, tile in zip(self.shape, KV_TILE))

    @property
    def q_grid(self) -> tuple[int, int, int]:
        return tuple(n // tile for n, tile in zip(self.q_padded, Q_TILE))

    @property
    def kv_grid(self) -> tuple[int, int, int]:
        return tuple(n // tile for n, tile in zip(self.kv_padded, KV_TILE))

    @property
    def q_length(self) -> int:
        return math.prod(self.q_padded)

    @property
    def kv_length(self) -> int:
        return math.prod(self.kv_padded)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _window_start(index: int, window: int, length: int) -> int:
    return min(max(index - window // 2, 0), length - window)


@functools.lru_cache(maxsize=32)
def _dimension_pattern_table(
    length: int,
    q_tile_size: int,
    kv_tile_size: int,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map (Q tile, KV tile) pairs to a small table of packed 1D masks."""
    q_tiles = _round_up(length, q_tile_size) // q_tile_size
    kv_tiles = _round_up(length, kv_tile_size) // kv_tile_size
    zero = (0,) * q_tile_size
    patterns: list[tuple[int, ...]] = [zero]
    pattern_ids = {zero: 0}
    ids = torch.zeros((q_tiles, kv_tiles), dtype=torch.int64)

    for q_tile in range(q_tiles):
        for kv_tile in range(kv_tiles):
            rows: list[int] = []
            for q_inner in range(q_tile_size):
                q = q_tile * q_tile_size + q_inner
                bits = 0
                if q < length:
                    start = _window_start(q, window, length)
                    for kv_inner in range(kv_tile_size):
                        kv = kv_tile * kv_tile_size + kv_inner
                        if kv < length and start <= kv < start + window:
                            bits |= 1 << kv_inner
                rows.append(bits)
            signature = tuple(rows)
            pattern_id = pattern_ids.get(signature)
            if pattern_id is None:
                pattern_id = len(patterns)
                pattern_ids[signature] = pattern_id
                patterns.append(signature)
            ids[q_tile, kv_tile] = pattern_id

    if len(patterns) > PATTERN_TABLE_SIZE:
        raise AssertionError(f"4-bit pattern id overflow for length={length}: {len(patterns)}")
    packed = []
    for pattern in patterns:
        value = sum(bits << (q * kv_tile_size) for q, bits in enumerate(pattern))
        # Store uint32 bit patterns in an int32 tensor.  Arithmetic right shift
        # followed by `& 1` still extracts the desired bit when bit 31 is set.
        packed.append(value if value < (1 << 31) else value - (1 << 32))
    # The four-bit pattern ids address at most 16 entries.  Pad to that fixed
    # size so T/H/W-specific table lengths cannot specialize the GPU kernel.
    packed.extend([0] * (PATTERN_TABLE_SIZE - len(packed)))
    return ids, torch.tensor(packed, dtype=torch.int32)


@functools.lru_cache(maxsize=SHAPE_CACHE_SIZE)
def build_kv_tile_map(problem: Problem) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed KV tile ids and valid counts for each permuted Q tile."""
    q_grid_t, q_grid_h, q_grid_w = problem.q_grid
    kv_grid_t, kv_grid_h, _ = problem.kv_grid
    rows: list[list[int]] = []

    # Token permutation flattens outer tile modes as (W, H, T), with T
    # changing fastest.  Keep KV tiles in the same increasing linear order.
    for q_w_tile in range(q_grid_w):
        for q_h_tile in range(q_grid_h):
            for q_t_tile in range(q_grid_t):
                ranges: list[range] = []
                for q_tile_idx, q_tile_size, kv_tile_size, window, length in zip(
                    (q_t_tile, q_h_tile, q_w_tile),
                    Q_TILE,
                    KV_TILE,
                    problem.window,
                    problem.shape,
                ):
                    first_q = q_tile_idx * q_tile_size
                    last_q = min(first_q + q_tile_size, length) - 1
                    first_k = _window_start(first_q, window, length)
                    last_k = _window_start(last_q, window, length) + window - 1
                    ranges.append(range(first_k // kv_tile_size, last_k // kv_tile_size + 1))

                t_tiles, h_tiles, w_tiles = ranges
                row = [
                    (kv_w * kv_grid_h + kv_h) * kv_grid_t + kv_t
                    for kv_w in w_tiles
                    for kv_h in h_tiles
                    for kv_t in t_tiles
                ]
                if not row:
                    raise AssertionError("every Q tile must overlap at least one KV tile")
                rows.append(row)

    max_tiles = max(map(len, rows))
    # Duplicate a valid tile in padding slots.  The kernel masks those slots
    # with `counts`; avoiding -1 keeps every indirect global load in bounds.
    indices = torch.tensor(
        [row + [row[0]] * (max_tiles - len(row)) for row in rows],
        dtype=torch.int32,
    )
    counts = torch.tensor([len(row) for row in rows], dtype=torch.int32)
    return indices, counts


@functools.lru_cache(maxsize=SHAPE_CACHE_SIZE)
def build_kv_tile_metadata(
    problem: Problem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack KV tile id and three separable mask pattern ids into int32."""
    indices, counts = build_kv_tile_map(problem)
    q_grid_t, q_grid_h, _ = problem.q_grid
    kv_grid_t, kv_grid_h, _ = problem.kv_grid
    t_ids, t_patterns = _dimension_pattern_table(problem.frames, Q_TILE[0], KV_TILE[0], problem.window[0])
    h_ids, h_patterns = _dimension_pattern_table(problem.height, Q_TILE[1], KV_TILE[1], problem.window[1])
    w_ids, w_patterns = _dimension_pattern_table(problem.width, Q_TILE[2], KV_TILE[2], problem.window[2])

    q_blocks = torch.arange(indices.shape[0], dtype=torch.int64)[:, None]
    q_t = q_blocks % q_grid_t
    q_h = (q_blocks // q_grid_t) % q_grid_h
    q_w = q_blocks // (q_grid_t * q_grid_h)
    kv = indices.to(torch.int64)
    kv_t = kv % kv_grid_t
    kv_h = (kv // kv_grid_t) % kv_grid_h
    kv_w = kv // (kv_grid_t * kv_grid_h)
    t_pattern_id = t_ids[q_t, kv_t]
    h_pattern_id = h_ids[q_h, kv_h]
    w_pattern_id = w_ids[q_w, kv_w]
    metadata = kv | (t_pattern_id << 15) | (h_pattern_id << 19) | (w_pattern_id << 23)
    active = torch.arange(indices.shape[1])[None, :] < counts[:, None]
    metadata = torch.where(active, metadata, 0).to(torch.int32)
    return metadata, t_patterns, h_patterns, w_patterns


@functools.lru_cache(maxsize=SHAPE_CACHE_SIZE)
def build_kv_tile_buckets(
    problem: Problem,
) -> tuple[tuple[torch.Tensor, torch.Tensor, int], ...]:
    """Group Q tiles by an upper bound on their useful KV tile count."""
    metadata, *_ = build_kv_tile_metadata(problem)
    _, counts = build_kv_tile_map(problem)
    maximum = int(counts.max())
    if maximum > KV_BUCKET_LIMITS[-1]:
        raise ValueError(
            f"FNA neighborhood needs {maximum} KV tiles, exceeding the supported ceiling {KV_BUCKET_LIMITS[-1]}"
        )
    buckets = []
    lower = 0
    for limit in KV_BUCKET_LIMITS:
        q_tile_ids = torch.nonzero((counts > lower) & (counts <= limit), as_tuple=False).flatten()
        if q_tile_ids.numel():
            # Small grids can have fewer columns than the first bucket limit.
            # Zero metadata encodes an all-masked tile, preserving the fixed ABI.
            bucket_metadata = torch.zeros((q_tile_ids.numel(), limit), dtype=torch.int32)
            columns = min(limit, metadata.shape[1])
            bucket_metadata[:, :columns] = metadata[q_tile_ids, :columns]
            buckets.append(
                (
                    q_tile_ids.to(torch.int32),
                    bucket_metadata,
                    limit,
                )
            )
        lower = limit
        if limit >= maximum:
            break
    return tuple(buckets)


@functools.lru_cache(maxsize=32)
def compile_kernel(
    max_kv_tiles: int,
    *,
    target: str = "auto",
):
    """Compile one T/H/W-dynamic kernel for a fixed KV-count ceiling."""
    if max_kv_tiles not in KV_BUCKET_LIMITS:
        raise ValueError(f"unsupported KV bucket ceiling: {max_kv_tiles}")

    q_length = T.dynamic("q_length")
    kv_length = T.dynamic("kv_length")
    q_tile_count = T.dynamic("q_tile_count")

    # Batch is fixed to one by the public guard.  Excluding it from the
    # kernel ABI keeps every physical stride static even when token lengths
    # vary, preserving vectorized global/shared-memory copies.
    q_shape = [q_length, HEADS, HEAD_DIM]
    kv_shape = [kv_length, HEADS, HEAD_DIM]
    q_tile_id_shape = [q_tile_count]
    map_shape = [q_tile_count, max_kv_tiles]
    pattern_shape = [PATTERN_TABLE_SIZE]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        query: T.Tensor(q_shape, dtype),
        key: T.Tensor(kv_shape, dtype),
        value: T.Tensor(kv_shape, dtype),
        q_tile_ids: T.Tensor(q_tile_id_shape, T.int32),
        kv_tile_metadata: T.Tensor(map_shape, T.int32),
        t_mask_patterns: T.Tensor(pattern_shape, T.int32),
        h_mask_patterns: T.Tensor(pattern_shape, T.int32),
        w_mask_patterns: T.Tensor(pattern_shape, T.int32),
        output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(q_tile_count, HEADS, threads=128) as (bx, by):
            q_shared = T.alloc_shared([Q_TOKENS, HEAD_DIM], dtype)
            k_shared = T.alloc_shared([KV_TOKENS, HEAD_DIM], dtype)
            v_shared = T.alloc_shared([KV_TOKENS, HEAD_DIM], dtype)
            o_shared = T.alloc_shared([Q_TOKENS, HEAD_DIM], dtype)
            scores = T.alloc_fragment([Q_TOKENS, KV_TOKENS], accum_dtype)
            probs = T.alloc_fragment([Q_TOKENS, KV_TOKENS], dtype)
            acc_o = T.alloc_fragment([Q_TOKENS, HEAD_DIM], accum_dtype)
            row_max = T.alloc_fragment([Q_TOKENS], accum_dtype)
            row_max_prev = T.alloc_fragment([Q_TOKENS], accum_dtype)
            row_scale = T.alloc_fragment([Q_TOKENS], accum_dtype)
            row_sum = T.alloc_fragment([Q_TOKENS], accum_dtype)
            row_denom = T.alloc_fragment([Q_TOKENS], accum_dtype)

            q_block = q_tile_ids[bx]
            # CPU metadata only emits padded Q/KV tiles that are fully backed
            # by the permuted tensors.  Expose those invariants so dynamic
            # lengths do not add per-copy boundary predicates.
            T.assume(q_block >= 0)
            T.assume((q_block + 1) * Q_TOKENS <= q_length)
            T.copy(
                query[
                    q_block * Q_TOKENS : (q_block + 1) * Q_TOKENS,
                    by,
                    :,
                ],
                q_shared,
            )
            T.fill(acc_o, 0)
            T.fill(row_denom, 0)
            T.fill(row_max, -T.infinity(accum_dtype))

            for block_iter in T.Pipelined(max_kv_tiles, num_stages=1):
                metadata = kv_tile_metadata[bx, block_iter]
                kv_block = metadata & 0x7FFF
                T.assume((kv_block + 1) * KV_TOKENS <= kv_length)
                t_mask = t_mask_patterns[(metadata >> 15) & 0xF]
                h_mask = h_mask_patterns[(metadata >> 19) & 0xF]
                w_mask = w_mask_patterns[(metadata >> 23) & 0xF]
                T.copy(
                    key[
                        kv_block * KV_TOKENS : (kv_block + 1) * KV_TOKENS,
                        by,
                        :,
                    ],
                    k_shared,
                    prefer_instruction="cp_async",
                )

                for i, j in T.Parallel(Q_TOKENS, KV_TOKENS):
                    q_t = i % Q_TILE[0]
                    q_h = (i // Q_TILE[0]) % Q_TILE[1]
                    q_w = i // (Q_TILE[0] * Q_TILE[1])
                    k_t = j % KV_TILE[0]
                    k_h = (j // KV_TILE[0]) % KV_TILE[1]
                    k_w = j // (KV_TILE[0] * KV_TILE[1])
                    valid = (
                        (((t_mask >> (q_t * KV_TILE[0] + k_t)) & 1) != 0)
                        & (((h_mask >> (q_h * KV_TILE[1] + k_h)) & 1) != 0)
                        & (((w_mask >> (q_w * KV_TILE[2] + k_w)) & 1) != 0)
                    )
                    scores[i, j] = T.if_then_else(valid, 0, -T.infinity(accum_dtype))

                T.gemm(
                    q_shared,
                    k_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(row_max, row_max_prev)
                T.fill(row_max, -T.infinity(accum_dtype))
                T.reduce_max(scores, row_max, dim=1, clear=False)
                # A KV tile is selected for the union of all 64 queries in
                # this Q tile.  It can therefore be fully masked for one
                # individual query.  Preserve that row's previous online
                # softmax state instead of evaluating (-inf) - (-inf).
                T.copy(row_max, row_sum)
                for i in T.Parallel(Q_TOKENS):
                    has_score = row_sum[i] > -T.infinity(accum_dtype)
                    merged_max = T.max(row_sum[i], row_max_prev[i])
                    row_max[i] = T.if_then_else(has_score, merged_max, row_max_prev[i])
                    row_scale[i] = T.if_then_else(
                        has_score,
                        T.exp2((row_max_prev[i] - merged_max) * LOG2_E),
                        1,
                    )
                    # Match Hopper NATTEN's two separately rounded FP32
                    # products before exp2.
                    row_sum[i] = T.if_then_else(
                        has_score,
                        merged_max * LOG2_E,
                        -T.infinity(accum_dtype),
                    )
                for i, j in T.Parallel(Q_TOKENS, KV_TOKENS):
                    scores[i, j] = T.if_then_else(
                        row_sum[i] > -T.infinity(accum_dtype),
                        T.exp2(scores[i, j] * LOG2_E - row_sum[i]),
                        0,
                    )
                T.reduce_sum(scores, row_sum, dim=1)
                for i in T.Parallel(Q_TOKENS):
                    row_denom[i] = row_denom[i] * row_scale[i] + row_sum[i]
                T.copy(scores, probs)
                for i, j in T.Parallel(Q_TOKENS, HEAD_DIM):
                    acc_o[i, j] *= row_scale[i]

                T.copy(
                    value[
                        kv_block * KV_TOKENS : (kv_block + 1) * KV_TOKENS,
                        by,
                        :,
                    ],
                    v_shared,
                    prefer_instruction="cp_async",
                )
                T.gemm(probs, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Match NATTEN's epilogue: compute one reciprocal per query row,
            # then multiply every output accumulator by that shared alpha.
            for i in T.Parallel(Q_TOKENS):
                row_scale[i] = T.if_then_else(
                    row_denom[i] > 0,
                    T.call_extern("float32", "__frcp_rn", row_denom[i]),
                    0,
                )
            for i, j in T.Parallel(Q_TOKENS, HEAD_DIM):
                acc_o[i, j] *= row_scale[i]
            T.copy(acc_o, o_shared)
            T.copy(
                o_shared,
                output[
                    q_block * Q_TOKENS : (q_block + 1) * Q_TOKENS,
                    by,
                    :,
                ],
            )

    return tilelang.compile(
        main,
        out_idx=None,
        target="auto" if target == "auto" else {"kind": "cuda", "arch": target},
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            # Keeping all work in the 128-thread consumer warpgroup avoids a
            # separate producer warpgroup.  The generated 240-register
            # consumer can then admit two CTAs per SM instead of one.
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )


_DEVICE_MAP_CACHE: OrderedDict[tuple[Problem, torch.device], tuple[object, ...]] = OrderedDict()


def _fna3d_tilelang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    window: tuple[int, int, int],
) -> torch.Tensor:
    problem = Problem(*shape, window=window)
    if query.shape[1] != problem.q_length or key.shape[1] != problem.kv_length:
        raise ValueError(f"permuted lengths do not match {problem}: Q={query.shape[1]}, KV={key.shape[1]}")
    cache_key = (problem, query.device)
    device_inputs = _DEVICE_MAP_CACHE.get(cache_key)
    if device_inputs is None:
        _, t_patterns, h_patterns, w_patterns = build_kv_tile_metadata(problem)
        pattern_tables = tuple(
            tensor.to(query.device, non_blocking=True) for tensor in (t_patterns, h_patterns, w_patterns)
        )
        buckets = tuple(
            (
                q_tile_ids.to(query.device, non_blocking=True),
                metadata.to(query.device, non_blocking=True),
                limit,
            )
            for q_tile_ids, metadata, limit in build_kv_tile_buckets(problem)
        )
        device_inputs = (pattern_tables, buckets)
        _DEVICE_MAP_CACHE[cache_key] = device_inputs
        if len(_DEVICE_MAP_CACHE) > SHAPE_CACHE_SIZE:
            _DEVICE_MAP_CACHE.popitem(last=False)
    else:
        _DEVICE_MAP_CACHE.move_to_end(cache_key)

    pattern_tables, buckets = device_inputs
    output = torch.empty_like(query)
    for q_tile_ids, metadata, limit in buckets:
        from tilelang.backend.target import determine_target

        target = str(determine_target("auto", return_object=True).attrs["arch"])
        kernel = compile_kernel(limit, target=target)
        for batch in range(query.shape[0]):
            kernel(query[batch], key[batch], value[batch], q_tile_ids, metadata, *pattern_tables, output[batch])
    return output


def fna3d_tilelang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    window: tuple[int, int, int],
) -> torch.Tensor:
    """Run native stage-5 FNA, or raise for unsupported inputs/launch failures.

    Q and KV use the documented token-tiled layout. Batch members share spatial
    metadata; compilation is cached by target architecture and KV-count bucket.
    """
    if not (
        query.is_cuda
        and query.dtype is torch.bfloat16
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[0] > 0
        and query.shape[-2:] == (HEADS, HEAD_DIM)
        and query.numel() > 0
        and query.is_contiguous()
        and key.device == value.device == query.device
        and key.dtype is value.dtype is query.dtype
        and key.is_contiguous()
        and value.is_contiguous()
        and value.shape == key.shape
        and key.shape[0] == query.shape[0]
        and key.shape[-2:] == query.shape[-2:]
        and len(shape) == 3
        and tuple(window) == (11, 11, 11)
    ):
        raise ValueError(
            "LTX DiffVAE FNA requires contiguous CUDA BF16 [B, tokens, 4, 64] Q/K/V and an 11x11x11 window."
        )
    if torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value)):
        raise RuntimeError("LTX DiffVAE FNA is inference-only.")
    with torch.accelerator.device_index(query.device.index):
        return _fna3d_tilelang(query, key, value, shape=tuple(shape), window=tuple(window))


__all__ = ["fna3d_tilelang"]
