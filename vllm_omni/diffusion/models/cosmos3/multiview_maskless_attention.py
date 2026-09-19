# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three-pass Phase 2.2 attention. Planning is host-side; kernels are opaque to GEN."""

from __future__ import annotations

import logging
from functools import lru_cache
from itertools import accumulate

import torch
import torch.nn.functional as F

from .multiview_flex_attention import DEFAULT_MAX_UND_TOKENS, MultiviewLayout

MERGE_CHUNK_SIZE = 8192
_INT32_LIMIT = 2**31
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_maskless_runtime() -> int:
    """Resolve and retain one FA implementation per serving worker."""
    from vllm_omni.diffusion.attention.backends.utils.fa import resolve_vllm_flash_attn_version

    fa_version = resolve_vllm_flash_attn_version()
    logger.info(
        "Cosmos3 maskless: GPU=%s FA=%s merge=local-fp32 PyTorch=%s CUDA=%s",
        torch.cuda.get_device_name(),
        fa_version,
        torch.__version__,
        torch.version.cuda,
    )
    return fa_version


def validate_indexing(lengths: list[int], heads: int, head_dim: int) -> None:
    """Check before allocating indices or converting cumulative offsets to int32."""
    if heads <= 0 or head_dim <= 0 or any(length < 0 for length in lengths):
        raise ValueError("Maskless attention requires nonnegative lengths and positive head geometry.")
    if sum(lengths) >= _INT32_LIMIT or any(length * heads * head_dim >= _INT32_LIMIT for length in lengths):
        raise ValueError(
            "Maskless attention exceeds int32 indexing (length × local heads × head_dim or cumulative offsets). "
            "Reduce frames/resolution/views or increase supported TP/Ulysses head partitioning."
        )


def _branch(
    queries: list[torch.Tensor],
    keys: list[torch.Tensor],
    total: int,
    device: torch.device,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> list[torch.Tensor]:
    q_lengths, k_lengths = [x.numel() for x in queries], [x.numel() for x in keys]
    validate_indexing(q_lengths, query_heads, head_dim)
    validate_indexing(k_lengths, kv_heads, head_dim)
    empty = torch.empty(0, dtype=torch.int64, device="cpu")
    qi = torch.cat(queries) if queries else empty
    ki = torch.cat(keys) if keys else empty
    inverse = torch.full((total,), -1, dtype=torch.int64, device="cpu")
    inverse[qi] = torch.arange(qi.numel(), device="cpu")
    # Maxima stay tensor data even when caption lengths change between requests.
    maxima = torch.tensor([max(q_lengths, default=0), max(k_lengths, default=0)], dtype=torch.int64, device="cpu")
    return [
        qi.to(device),
        ki.to(device),
        torch.tensor([0, *accumulate(q_lengths)], dtype=torch.int32, device=device),
        torch.tensor([0, *accumulate(k_lengths)], dtype=torch.int32, device=device),
        inverse.to(device),
        maxima,
    ]


def build_maskless_plan(
    layout: MultiviewLayout,
    num_und: int,
    device: torch.device,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> list[torch.Tensor]:
    """Build same-view, target-only instant and caption partitions in packed order.

    Sensor identity participates in grouping, so one camera plus LiDAR is two
    groups. Shared captions repeat keys per group, bounding query indexing.
    """
    if layout.backend != "maskless":
        raise ValueError("A maskless plan requires backend='maskless'.")
    lengths = layout.caption_lengths or (num_und,)
    if sum(lengths) != num_und or any(n < 0 or n > DEFAULT_MAX_UND_TOKENS for n in lengths):
        raise ValueError("Maskless caption lengths must cover compact UND and each be at most 4098 tokens.")
    if num_und > layout.max_und_tokens:
        raise ValueError("Maskless captions exceed the checkpoint's UND admission capacity.")
    cameras = sorted(
        {item.view_offset + view for item in layout.items if not item.is_lidar for view in range(item.num_views)}
    )
    if layout.caption_lengths and len(lengths) != len(cameras):
        raise ValueError("Per-view captions must match the camera view groups.")
    validate_indexing([layout.gen_tokens], 1, 1)
    anchor = next((item.seconds_per_frame for item in layout.items if not item.is_lidar), None)
    if anchor is None:
        raise ValueError("Maskless multiview requires a camera period as the instant anchor.")
    # Preflight combined control/target view sizes before allocating token maps.
    group_lengths: dict[tuple[bool, int], int] = {}
    for item in layout.items:
        for view in range(item.num_views):
            group = (item.is_lidar, item.view_offset + view)
            group_lengths[group] = group_lengths.get(group, 0) + item.num_tokens // item.num_views
    validate_indexing(list(group_lengths.values()), query_heads, head_dim)
    validate_indexing(list(group_lengths.values()), kv_heads, head_dim)
    groups: dict[tuple[bool, int], list[torch.Tensor]] = {}
    instants: dict[int, list[torch.Tensor]] = {}
    start = 0
    for item in layout.items:
        frames = item.token_shape[0] // item.num_views
        spatial = item.token_shape[1] * item.token_shape[2]
        validate_indexing([frames * spatial], query_heads, head_dim)
        # Explicit float64, including the midpoint tie-breaking epsilon.
        ids = (
            torch.floor(
                (torch.arange(frames, dtype=torch.float64, device="cpu") + 0.5) * item.seconds_per_frame / anchor + 1e-6
            )
            .to(torch.int64)
            .tolist()
        )
        for view in range(item.num_views):
            indices = torch.arange(start + view * frames * spatial, start + (view + 1) * frames * spatial, device="cpu")
            groups.setdefault((item.is_lidar, item.view_offset + view), []).append(indices)
            if not item.is_control and layout.attention_scope == "decomposed":
                for frame, instant in enumerate(ids):
                    instants.setdefault(instant, []).append(indices[frame * spatial : (frame + 1) * spatial])
        start += item.num_tokens
    views = [torch.cat(parts) for parts in groups.values()]
    instant_groups = [torch.cat(parts) for _, parts in sorted(instants.items())] if len(groups) > 1 else []
    offsets = [0, *accumulate(lengths)]
    caption_q, caption_k = [], []
    for (lidar, view), queries in zip(groups, views, strict=True):
        if lidar and not layout.lidar_attends_captions:
            continue
        first, last = (
            (0, num_und)
            if lidar or not layout.caption_lengths
            else (offsets[cameras.index(view)], offsets[cameras.index(view) + 1])
        )
        if first == last:
            continue  # No zero-key sequences reach FlashAttention.
        caption_q.append(queries)
        caption_k.append(torch.arange(first, last, device="cpu"))
    result = []
    for queries, keys in ((views, views), (instant_groups, instant_groups), (caption_q, caption_k)):
        branch = _branch(queries, keys, start, device, query_heads, kv_heads, head_dim)
        # Indices/offsets can change length; maxima always has shape [2]. Its
        # prompt-dependent values remain opaque tensor data with a static shape.
        for tensor in branch[:-1]:
            if tensor.numel() > 1:
                torch._dynamo.mark_dynamic(tensor, 0)
        result.extend(branch)
    return result


def normalize_varlen_lse(lse: torch.Tensor, tokens: int, query_heads: int) -> torch.Tensor:
    """vLLM FA2/FA3/FA4 varlen contract is [heads, total_q], including square shapes."""
    if lse.ndim != 2 or lse.shape != (query_heads, tokens):
        raise ValueError(f"Expected FlashAttention LSE [heads,tokens]={query_heads, tokens}, got {tuple(lse.shape)}.")
    return lse.transpose(0, 1).unsqueeze(0).contiguous()


def make_merge_scratch(query_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
    return [torch.empty(1, MERGE_CHUNK_SIZE, query_heads, head_dim, dtype=dtype, device=device) for _ in range(3)] + [
        torch.empty(1, MERGE_CHUNK_SIZE, query_heads, dtype=torch.float32, device=device) for _ in range(3)
    ]


def _merge_attention_outputs(outputs: list[torch.Tensor], lse_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Merge branch contexts, including duplicate keys, in inference mode.

    LSE is natural-log, FP32, and shaped [batch, tokens, heads]. Preserve the
    sequential sigmoid/logsigmoid recurrence used by NATTEN 0.21.6, with all
    weighting and accumulation in FP32 and only the final output cast back.
    The caller supplies a valid first branch for every row, including padding;
    absent later contributions have zero output and minimum-FP32 LSE.
    """
    if not outputs or len(outputs) != len(lse_tensors):
        raise ValueError("Maskless merge requires matching nonempty output and LSE lists.")
    if len(outputs) == 1:
        return outputs[0]
    output = outputs[0].float()
    lse = lse_tensors[0].float()
    for index in range(1, len(outputs)):
        next_lse = lse_tensors[index].float()
        weight = torch.sigmoid(next_lse - lse).unsqueeze(-1)
        output = output - weight * (output - outputs[index].float())
        if index + 1 < len(outputs):
            lse = lse - F.logsigmoid(lse - next_lse)
    return output.to(outputs[0].dtype)


# Fixed-size scratch chunks keep prompt lengths out of the merge specialization.
# Fullgraph prevents a silent fallback to eager elementwise GPU launches.
_compiled_merge_attention_outputs = torch.compile(_merge_attention_outputs, fullgraph=True)


@torch.library.custom_op("vllm_omni::cosmos3_maskless_attention", mutates_args=("scratch",))
def maskless_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    plan: list[torch.Tensor],
    scratch: list[torch.Tensor],
    fa_version: int,
) -> torch.Tensor:
    from vllm_omni.diffusion.attention.backends.utils.fa import vllm_flash_attn_varlen_with_lse

    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Maskless multiview attention requires B == 1.")
    # Every branch's inverse map spans exactly the planned packed GEN stream.
    # Check before gathering: singleton slices can otherwise broadcast silently.
    planned_tokens = plan[4].numel()
    if q.shape[1] != planned_tokens:
        raise ValueError(
            "Cosmos3 maskless packed GEN length does not match the request plan: "
            f"attention={q.shape[1]}, plan={planned_tokens}."
        )
    if k.shape != v.shape or k_und.shape != v_und.shape or k.shape[:2] != q.shape[:2]:
        raise ValueError("Maskless Q/K/V geometry mismatch.")
    if k_und.shape[0] != 1 or k_und.shape[2:] != k.shape[2:] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Maskless GEN/UND head geometry mismatch.")
    if q.shape[2] % k.shape[2]:
        raise ValueError("Maskless attention requires an integral GQA ratio.")
    # Preserve the caller's tensor mode: HSDP/offload use no_grad() and need
    # outputs with version counters. Only attention/merge temporaries require
    # inference mode; allocating here avoids an extra output clone.
    result = torch.empty_like(q)
    with torch.inference_mode():
        branches = []
        for branch in range(3):
            qi, ki, cuq, cuk, inverse, maxima = plan[branch * 6 : (branch + 1) * 6]
            maxq, maxk = maxima.tolist()
            if not maxq:
                continue
            # Actual heads here are already partitioned by TP and Ulysses.
            validate_indexing([maxq], q.shape[2], q.shape[3])
            validate_indexing([maxk], k.shape[2], k.shape[3])
            keys, values = (k_und[0], v_und[0]) if branch == 2 else (k[0], v[0])
            out, lse = vllm_flash_attn_varlen_with_lse(
                q[0].index_select(0, qi),
                keys.index_select(0, ki),
                values.index_select(0, ki),
                cu_seqlens_q=cuq,
                cu_seqlens_k=cuk,
                max_seqlen_q=maxq,
                max_seqlen_k=maxk,
                causal=False,
                fa_version=fa_version,
                fa_version_is_resolved=True,
            )
            branches.append((out, normalize_varlen_lse(lse, qi.numel(), q.shape[2])[0], inverse))
        for start in range(0, q.shape[1], MERGE_CHUNK_SIZE):
            count = min(MERGE_CHUNK_SIZE, q.shape[1] - start)
            outputs, lses = [], []
            for branch, (out, lse, inverse) in enumerate(branches):
                output, weights = scratch[branch], scratch[branch + 3]
                index = inverse[start : start + count]
                valid = index >= 0
                gather_index = index.clamp_min(0)
                # Gather directly into scratch, then zero excluded rows without
                # chunk-sized gather/multiply temporaries or NaN * 0 leakage.
                torch.index_select(out, 0, gather_index, out=output[0, :count])
                output[0, :count].masked_fill_(~valid[:, None, None], 0)
                weights[0, :count] = torch.where(valid[:, None], lse[gather_index], torch.finfo(weights.dtype).min)
                if count < MERGE_CHUNK_SIZE:
                    output[:, count:].zero_()
                    # Finite dummy rows; padding never enters attention.
                    weights[:, count:].fill_(0 if branch == 0 else torch.finfo(weights.dtype).min)
                outputs.append(output)
                lses.append(weights)
            merged = _compiled_merge_attention_outputs(outputs, lses)
            result[:, start : start + count] = merged[:, :count]
        return result


@maskless_attention_op.register_fake
def _maskless_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    plan: list[torch.Tensor],
    scratch: list[torch.Tensor],
    fa_version: int,
) -> torch.Tensor:
    return torch.empty_like(q)
