# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental NumPy-compatible sampler for the Ascend Talker graph."""

import numpy as np
import torch

# Triton constexpr parameters follow the kernel convention.
# ruff: noqa: N803
from vllm.triton_utils import tl, triton


@triton.jit
def resolve_codec_tie(scores, counts, ordinals, targets, output, depth_exceeded, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, WIDTH)
    values = tl.load(scores + row * WIDTH + lanes)
    keys = values
    count = tl.load(counts + row)
    indices = lanes
    ordinal = tl.load(ordinals + row)
    target = tl.load(targets + row)
    left, right, depth = 0, count - 1, 12
    same_score = (lanes < count) & (keys == target)
    tie_size = tl.sum(same_score.to(tl.int32), 0)
    result = tl.where(tie_size == 1, tl.sum(tl.where(same_score, indices, 0), 0), -1)
    failed = False
    while left <= right and result < 0:
        if depth < 0:
            failed = True
            left = count
        elif right - left > 15:
            mid = (left + right) // 2
            for pair in tl.static_range(3):
                a = mid if pair == 1 else left
                b = right if pair == 1 else mid
                ai = tl.sum(tl.where(lanes == a, indices, 0), 0)
                bi = tl.sum(tl.where(lanes == b, indices, 0), 0)
                av = tl.sum(tl.where(lanes == a, keys, 0.0), 0)
                bv = tl.sum(tl.where(lanes == b, keys, 0.0), 0)
                if bv < av:
                    indices = tl.where(lanes == a, bi, tl.where(lanes == b, ai, indices))
                    keys = tl.where(lanes == a, bv, tl.where(lanes == b, av, keys))
            mid_index = tl.sum(tl.where(lanes == mid, indices, 0), 0)
            bound = right - 1
            bound_index = tl.sum(tl.where(lanes == bound, indices, 0), 0)
            pivot = tl.sum(tl.where(lanes == mid, keys, 0.0), 0)
            bound_key = tl.sum(tl.where(lanes == bound, keys, 0.0), 0)
            indices = tl.where(lanes == mid, bound_index, tl.where(lanes == bound, mid_index, indices))
            keys = tl.where(lanes == mid, bound_key, tl.where(lanes == bound, pivot, keys))
            i, j = left, bound
            partition_done = False
            while not partition_done:
                i = tl.min(tl.where((lanes > i) & (lanes <= right) & (keys >= pivot), lanes, WIDTH), 0)
                j = tl.max(tl.where((lanes < j) & (lanes >= left) & (keys <= pivot), lanes, -1), 0)
                if i < j:
                    ai = tl.sum(tl.where(lanes == i, indices, 0), 0)
                    bi = tl.sum(tl.where(lanes == j, indices, 0), 0)
                    av = tl.sum(tl.where(lanes == i, keys, 0.0), 0)
                    bv = tl.sum(tl.where(lanes == j, keys, 0.0), 0)
                    indices = tl.where(lanes == i, bi, tl.where(lanes == j, ai, indices))
                    keys = tl.where(lanes == i, bv, tl.where(lanes == j, av, keys))
                else:
                    partition_done = True
            ii = tl.sum(tl.where(lanes == i, indices, 0), 0)
            ik = tl.sum(tl.where(lanes == i, keys, 0.0), 0)
            indices = tl.where(lanes == bound, ii, indices)
            indices = tl.where(lanes == i, mid_index, indices)
            keys = tl.where(lanes == bound, ik, keys)
            keys = tl.where(lanes == i, pivot, keys)
            depth -= 1
            if ordinal == i:
                result = mid_index
                left = count
            elif ordinal < i:
                right = i - 1
            else:
                left = i + 1
        else:
            in_leaf = (lanes >= left) & (lanes <= right)
            # The final insertion sort is stable for equal scores. Select
            # within the remaining tie group directly from its current order.
            lower = tl.sum((in_leaf & (keys < target)).to(tl.int32), 0)
            tie_rank = ordinal - left - lower
            matches = in_leaf & (keys == target)
            rank = tl.cumsum(matches.to(tl.int32), 0)
            selected = matches & (rank == tie_rank + 1)
            result = tl.sum(tl.where(selected, indices, 0), 0)
            failed = tl.sum(selected.to(tl.int32), 0) != 1
            left = count
    tl.store(output + row, result)
    tl.store(depth_exceeded + row, failed)


@triton.jit
def prepare_scores(
    raw,
    history,
    mask,
    ratio,
    out,
    abnormal,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
    TEMPERATURE_RECIPROCAL: tl.constexpr,
    EOS: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    offset = row * VOCAB + col
    logits = tl.load(raw + offset, col < VOCAB, other=0.0)
    count = tl.full((BLOCK,), 0, tl.int32)
    for hist_i in tl.static_range(16):
        token = tl.load(history + row * 16 + hist_i).to(tl.int32)
        count += (col == token).to(tl.int32)
    bits = logits.to(tl.int32, bitcast=True)
    mantissa = (bits >> 16) & 127
    exp_bits = bits & -8388608
    exponent = (bits >> 23) & 255
    negative = (bits >> 31) & 1
    table_index = count * 256 + mantissa + negative * 128
    multiplier = tl.load(ratio + table_index)
    exact = multiplier * exp_bits.to(tl.float32, bitcast=True)
    special = ((bits & 65535) != 0) | (exponent == 255) | ((exponent == 0) & (logits != 0.0))
    penalized = tl.where(count == 0, logits, exact)
    score = penalized * TEMPERATURE_RECIPROCAL
    suppress = tl.load(mask + row)
    score = tl.where((col == EOS) & suppress, -float("inf"), score)
    tl.store(out + offset, score, col < VOCAB)
    tl.store(abnormal + row * tl.cdiv(VOCAB, BLOCK) + block, tl.sum((special & (col < VOCAB)).to(tl.int32), 0) > 0)


def exact_penalty_tables(penalty):
    # BF16 mantissa and exponent scaling let the same rounded ratio table
    # work for all normal BF16 exponents. For unusual exponent classes,
    # keep the arithmetic path and set a replay guard.
    positive = np.empty((17, 128), dtype=np.float32)
    negative = np.empty_like(positive)
    for count in range(17):
        alpha = penalty**count
        for mantissa in range(128):
            base = 1.0 + mantissa / 128
            positive[count, mantissa] = base / alpha
            negative[count, mantissa] = base * alpha
    return positive, negative


@triton.jit
def draw_sorted(
    values, uniform, ordinal_out, target_out, boundaries, CAP: tl.constexpr, K: tl.constexpr, DROP_PROB: tl.constexpr
):
    row = tl.program_id(0)
    lanes = tl.arange(0, CAP)
    # Input width CAP+1 includes the first omitted value, used by the caller.
    scores = tl.load(values + row * (CAP + 1) + (CAP - 1 - lanes))
    cutoff = tl.load(values + row * (CAP + 1) + K - 1)
    largest = tl.load(values + row * (CAP + 1))
    valid = scores >= cutoff
    weights = tl.where(valid, tl.exp(scores - largest), 0.0)
    p = weights / tl.sum(weights, 0)
    cdf = tl.cumsum(p, 0)
    keep = valid & ((cdf > DROP_PROB) | (lanes == CAP - 1))
    p = tl.where(keep, p, 0.0)
    second = tl.cumsum(p / tl.sum(p, 0), 0)
    u = tl.load(uniform + row)
    slot = tl.minimum(tl.sum((second <= u).to(tl.int32), 0), CAP - 1)
    target = tl.sum(tl.where(lanes == slot, scores, 0.0), 0)
    candidate_count = tl.sum(valid.to(tl.int32), 0)
    edge = (tl.abs(cdf - DROP_PROB) <= 2e-6) | (tl.abs(second - u) <= 2e-6)
    guard = tl.sum(edge.to(tl.int32), 0) > 0
    tl.store(ordinal_out + row, slot - (CAP - candidate_count))
    tl.store(target_out + row, target)
    tl.store(boundaries + row, guard)


def make_device_sampler(eos, temp, k, p, penalty, device, cap=128):
    temp_reciprocal = float(np.float32(1 / np.float32(temp)))
    pos, neg = exact_penalty_tables(penalty)
    ratio = torch.from_numpy(np.concatenate((pos, neg), axis=1)).to(device).flatten()

    def sample(logits, history, masked, uniform):
        rows, vocab = logits.shape
        blocks = triton.cdiv(vocab, 512)
        scores = torch.empty_like(logits)
        abnormal = torch.empty((rows, blocks), dtype=torch.bool, device=logits.device)
        prepare_scores[(rows, blocks)](
            logits, history, masked, ratio, scores, abnormal, vocab, 512, temp_reciprocal, eos
        )
        top_values, ids = scores.topk(cap + 1)
        threshold = top_values[:, k - 1 : k]
        overflow = top_values[:, -1] >= threshold[:, 0]
        values, ids = top_values[:, :-1], ids[:, :-1]
        valid = values >= threshold
        num_candidates = valid.int().sum(dim=-1).int()
        target = torch.empty(rows, dtype=logits.dtype, device=logits.device)
        ordinal = torch.empty(rows, dtype=torch.int32, device=logits.device)
        close = torch.empty(rows, dtype=torch.bool, device=logits.device)
        draw_sorted[(rows,)](top_values, uniform, ordinal, target, close, cap, k, 1 - p)
        id_order = ids.float().masked_fill(~valid, 65536).argsort(dim=-1)
        input_ordered = values.gather(1, id_order).contiguous()
        candidate_ids = ids.gather(1, id_order)
        index = torch.empty_like(num_candidates)
        flag = torch.empty_like(masked)
        resolve_codec_tie[(logits.shape[0],)](input_ordered, num_candidates, ordinal, target, index, flag, cap)
        chosen = candidate_ids.gather(1, index.clamp(0, cap - 1).long()[:, None])[:, 0]
        return chosen, overflow | flag | close | abnormal.any(dim=-1)

    return sample
