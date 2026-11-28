# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibrated centered FP8: encode new rows once, retain historical encodings."""

import os

import torch
from vllm.triton_utils import tl, triton

from .fp8_hadamard import rotate


@triton.jit
def transform(value, signs, dim: tl.constexpr, block: tl.constexpr):
    d = tl.arange(0, dim)
    value *= tl.load(signs + d)[None, :]
    for step in tl.static_range(7):
        bit = 1 << step
        partner = tl.gather(value, tl.broadcast_to((d ^ bit)[None, :], (block, dim)), axis=1)
        value = tl.where((d[None, :] & bit) == 0, value + partner, partner - value)
    return (value * dim**-0.5).to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["rows", "layer"])
def encode_qkv(
    q,
    k,
    v,
    qo,
    ko,
    vo,
    past,
    centers,
    scales,
    signs,
    clips,
    rows,
    heads: tl.constexpr,
    dim: tl.constexpr,
    layer,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    audit: tl.constexpr,
    block: tl.constexpr,
    text_capacity: tl.constexpr = 0,
    text_valid_rows=None,
):
    r = tl.program_id(0) * block + tl.arange(0, block)
    h, side = tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, dim)
    if tl.program_id(2) == 0:
        value = tl.load(q + r[:, None] * qs0 + h * qs1 + d[None, :], (r < rows)[:, None], 0)
    elif tl.program_id(2) == 1:
        value = tl.load(k + r[:, None] * ks0 + h * ks1 + d[None, :], (r < rows)[:, None], 0)
    else:
        value = tl.load(v + r[:, None] * vs0 + h * vs1 + d[None, :], (r < rows)[:, None], 0)
    value = value.to(tl.float32)
    if tl.program_id(2) > 0:
        center = tl.load(centers + ((layer * 2 + side - 1) * heads + h) * dim + d)
        value -= center[None, :]
    value = transform(value, signs, dim, block)
    scale = tl.load(scales + (layer * 3 + side) * heads + h)
    normalized = value / scale
    if audit:
        count = tl.sum(tl.sum(((tl.abs(normalized) > 448) & (r < rows)[:, None]).to(tl.int32), 1), 0)
        tl.atomic_add(clips + (layer * 3 + side) * heads + h, count)
    value = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
    if tl.program_id(2) == 0:
        encoded = value.to(tl.float8e4nv)
        tl.store(qo + (r[:, None] * heads + h) * dim + d[None, :], encoded, (r < rows)[:, None])
    else:
        offset = tl.load(past).to(tl.int64)
        target = r
        active = r < rows
        if text_capacity:
            real = tl.load(text_valid_rows)
            target = tl.where(r < text_capacity, r, r - text_capacity + real)
            active &= (r < real) | (r >= text_capacity)
        if tl.program_id(2) == 1:
            tl.store(ko + ((target[:, None] + offset) * heads + h) * dim + d[None, :], value, active[:, None])
        else:
            tl.store(vo + ((target[:, None] + offset) * heads + h) * dim + d[None, :], value, active[:, None])


@triton.jit(do_not_specialize=["rows", "layer"])
def encode_pages(
    k,
    v,
    master,
    shadow,
    slots,
    retain,
    centers,
    scales,
    signs,
    clips,
    rows,
    physical: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    layer,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    bootstrap: tl.constexpr,
    audit: tl.constexpr,
    block: tl.constexpr,
    logical_offset,
    capacity: tl.constexpr,
    workspace_mode: tl.constexpr,
):
    r = tl.program_id(0) * block + tl.arange(0, block)
    h = tl.program_id(1)
    side = tl.program_id(2) % 2
    li = tl.program_id(2).to(tl.int64) // 2 if bootstrap else layer.to(tl.int64)
    d = tl.arange(0, dim)
    slot = tl.load(slots + r, r < rows, 0).to(tl.int64)
    dest = ((li * 2 + side) * physical + slot[:, None]) * heads * dim + h * dim + d[None, :]
    if bootstrap:
        value = tl.load(master + dest, (r < rows)[:, None], 0)
    else:
        source = tl.load(retain + r, r < rows, 0).to(tl.int64)
        if tl.program_id(2) == 0:
            value = tl.load(k + source[:, None] * ks0 + h * ks1 + d[None, :], (r < rows)[:, None], 0)
        else:
            value = tl.load(v + source[:, None] * vs0 + h * vs1 + d[None, :], (r < rows)[:, None], 0)
        tl.store(master + dest, value, (r < rows)[:, None])
    center = tl.load(centers + ((li * 2 + side) * heads + h) * dim + d)
    value = transform(value.to(tl.float32) - center[None, :], signs, dim, block)
    scale = tl.load(scales + (li * 3 + side + 1) * heads + h)
    normalized = value / scale
    if audit:
        count = tl.sum(tl.sum(((tl.abs(normalized) > 448) & (r < rows)[:, None]).to(tl.int32), 1), 0)
        tl.atomic_add(clips + (li * 3 + side + 1) * heads + h, count)
    if workspace_mode:
        begin = tl.load(logical_offset).to(tl.int64)
        dest = ((li * 2 + side) * capacity + begin + r[:, None]) * heads * dim + h * dim + d[None, :]
    tl.store(shadow + dest, tl.minimum(tl.maximum(normalized, -448.0), 448.0), (r < rows)[:, None])


@triton.jit(do_not_specialize=["rows", "layer"])
def copy_pages(
    shadow,
    workspace,
    slots,
    offset,
    rows,
    physical: tl.constexpr,
    cap: tl.constexpr,
    hd: tl.constexpr,
    layer=-1,
    block: tl.constexpr = 4096,
):
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    s = tl.program_id(1).to(tl.int64)
    side = tl.where(layer < 0, s, layer.to(tl.int64) * 2 + s)
    r, c = x // hd, x % hd
    slot = tl.load(slots + r, r < rows, 0).to(tl.int64)
    value = tl.load(shadow + (side * physical + slot) * hd + c, r < rows, 0.0)
    begin = tl.load(offset).to(tl.int64)
    tl.store(workspace + (side * cap + begin + r) * hd + c, value, r < rows)


@triton.jit
def move_prefix(
    work,
    scratch,
    origin: tl.constexpr,
    target: tl.constexpr,
    rows: tl.constexpr,
    cap: tl.constexpr,
    hd: tl.constexpr,
    save: tl.constexpr,
    block: tl.constexpr,
):
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    side = tl.program_id(1).to(tl.int64)
    if save:
        value = tl.load(work + (side * cap + origin) * hd + x, x < rows * hd, 0.0)
        tl.store(scratch + side * rows * hd + x, value, x < rows * hd)
    else:
        value = tl.load(scratch + side * rows * hd + x, x < rows * hd, 0.0)
        tl.store(work + (side * cap + target) * hd + x, value, x < rows * hd)


def install(pool, profile, layout=None):
    """Bind one immutable calibration profile for the pool's lifetime."""
    device = pool.storage.device
    pool._realization_fp8_calibration = {key: profile[key].to(device) for key in ("centers", "scales", "signs")}
    pool._realization_fp8_layout = layout or os.environ.get("DUO_FP8_CACHE_LAYOUT", "shadow")
    if pool._realization_fp8_layout == "shadow":
        pool._realization_fp8_shadow = torch.empty_like(pool.storage, dtype=torch.float8_e4m3fn)
    pool._realization_fp8_generations = [0] * pool.storage.shape[2]
    pool._realization_fp8_allocation = 0
    pool._realization_fp8_active = True
    pool._realization_fp8_clips = torch.zeros(
        (pool.storage.shape[0], 3, pool.storage.shape[-2]), device=device, dtype=torch.int64
    )
    pool._realization_fp8_audit = False


def write_layer(pool, tx, layer, k, v):
    p = pool._realization_fp8_calibration
    rows = tx.slots.numel()
    workspace = pool._realization_fp8_layout == "workspace"
    if workspace and tx.workspace_offset is None:
        # Prefix domains may use the general attention plan; stage their KV later.
        for side, value in enumerate((k, v)):
            pool.storage[layer, side].flatten(0, 1).index_copy_(0, tx.slots, value.index_select(0, tx.retain))
        return
    target = pool._realization_native_attention["buffer"] if workspace else pool._realization_fp8_shadow
    encode_pages[(triton.cdiv(rows, 4), k.shape[1], 2)](
        k,
        v,
        pool.storage,
        target,
        tx.slots,
        tx.retain,
        p["centers"],
        p["scales"],
        p["signs"],
        pool._realization_fp8_clips,
        rows,
        pool.storage.shape[2] * pool.page_size,
        *k.shape[1:],
        layer,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        False,
        pool._realization_fp8_audit,
        4,
        tx.workspace_offset if workspace else pool._realization_fp8_clips,
        target.shape[2] if workspace else 0,
        workspace,
        num_warps=4,
    )


def span_slots(pool, spans):
    return torch.tensor(
        [
            s.block_ids[i // pool.page_size] * pool.page_size + i % pool.page_size
            for s in spans
            for i in range(s.group.end - s.group.start)
        ],
        device=pool.storage.device,
        dtype=torch.long,
    )


def bootstrap(pool, cache):
    if pool._realization_fp8_layout == "workspace":
        return
    p = pool._realization_fp8_calibration
    slots = span_slots(pool, cache.spans)
    encode_pages[(triton.cdiv(slots.numel(), 4), pool.storage.shape[-2], pool.storage.shape[0] * 2)](
        pool.storage,
        pool.storage,
        pool.storage,
        pool._realization_fp8_shadow,
        slots,
        slots,
        p["centers"],
        p["scales"],
        p["signs"],
        pool._realization_fp8_clips,
        slots.numel(),
        pool.storage.shape[2] * pool.page_size,
        *pool.storage.shape[-2:],
        0,
        0,
        0,
        0,
        0,
        True,
        pool._realization_fp8_audit,
        4,
        pool._realization_fp8_clips,
        0,
        False,
        num_warps=4,
    )


def span_key(pool, span):
    return (
        span.group.kind,
        span.group.block,
        span.group.end - span.group.start,
        tuple((i, pool._realization_fp8_generations[i]) for i in span.block_ids),
    )


def mark_staged(runtime, pool, cache):
    runtime.state["staged_spans"] = cache.spans
    runtime.state["staged_keys"] = tuple(span_key(pool, s) for s in cache.spans)


def stage_history(runtime, pool, cache, state):
    """Append newly published rows; move only the prefix when clean spans expire.

    Physical page generations distinguish recycled pages after cancel or eviction.
    A full FP8 copy is needed only at initialization, a non-linear cache switch,
    or workspace wrap. Retained history is never centered or quantized again.
    """
    force_compact = state.pop("force_compact", False)
    if state["cache"] is cache and state["spans"] == runtime.spans and not force_compact:
        return
    spans = runtime.spans
    keys = tuple(span_key(pool, s) for s in spans)
    old = state.get("staged_spans", ())
    oldkeys = state.get("staged_keys", ())
    op = sum(s.group.kind != "clean" for s in old)
    np = sum(s.group.kind != "clean" for s in spans)
    prefix = sum(s.group.end - s.group.start for s in spans[:np])
    oldbody, body = oldkeys[op:], keys[np:]
    cut = oldbody.index(body[0]) if body and body[0] in oldbody else len(oldbody)
    retained = oldbody[cut:]
    linear = (
        bool(oldkeys) and state["cache"] is not None and oldkeys[:op] == keys[:np] and body[: len(retained)] == retained
    )
    drop = sum(s.group.end - s.group.start for s in old[op : op + cut])
    origin = state.get("base", 0)
    base = origin + drop if linear else 0
    wrap = force_compact or base + runtime.past_count + runtime.rows > runtime.plan.capacity
    if wrap:
        base = 0
    compact = pool._realization_fp8_layout == "workspace"
    if linear and wrap and compact:
        retained_rows = sum(s.group.end - s.group.start for s in old[op + cut :])
        compact_workspace(state, prefix, retained_rows, origin, drop)
    append = spans[np + len(retained) :] if linear and (not wrap or compact) else spans
    rows = sum(s.group.end - s.group.start for s in append)
    layers, _, cap, h, d = state["buffer"].shape
    if not linear:
        state["prefix_scratch"] = torch.empty(
            (layers, 2, prefix, h, d), device=pool.storage.device, dtype=torch.float8_e4m3fn
        )
        state["append_offset"] = torch.zeros((), device=pool.storage.device, dtype=torch.int32)
        if compact:
            # Compile the byte mover during initial staging, before live wrap.
            move_workspace_rows[(1, 1)](state["buffer"].view(torch.uint8), 0, 0, 0, cap, h * d, 32768, num_warps=8)
    if linear and not wrap and drop and prefix:
        scratch = state["prefix_scratch"]
        grid = (triton.cdiv(prefix * h * d, 4096), layers * 2)
        move_prefix[grid](state["buffer"], scratch, origin, base, prefix, cap, h * d, True, 4096)
        move_prefix[grid](state["buffer"], scratch, origin, base, prefix, cap, h * d, False, 4096)
        state["prefix_moves"] = state.get("prefix_moves", 0) + 1
    if rows:
        slots = runtime.plan.slots[runtime.media_group, runtime.past_count - rows : runtime.past_count]
        offset = state["append_offset"]
        offset.fill_(base + runtime.past_count - rows)
        if compact:
            p = pool._realization_fp8_calibration
            encode_pages[(triton.cdiv(rows, 4), h, layers * 2)](
                pool.storage,
                pool.storage,
                pool.storage,
                state["buffer"],
                slots,
                slots,
                p["centers"],
                p["scales"],
                p["signs"],
                pool._realization_fp8_clips,
                rows,
                pool.storage.shape[2] * pool.page_size,
                h,
                d,
                0,
                0,
                0,
                0,
                0,
                True,
                pool._realization_fp8_audit,
                4,
                offset,
                cap,
                True,
                num_warps=4,
            )
        else:
            copy_pages[(triton.cdiv(rows * h * d, 4096), layers * 2)](
                pool._realization_fp8_shadow,
                state["buffer"],
                slots,
                offset,
                rows,
                pool.storage.shape[2] * pool.page_size,
                cap,
                h * d,
            )
    name = "incremental_rows" if linear and not wrap else "packed_rows"
    state[name] = state.get(name, 0) + rows
    state["compactions"] = state.get("compactions", 0) + int(wrap)
    state["base"], state["cache"], state["spans"] = base, cache, spans
    state["staged_spans"], state["staged_keys"] = spans, keys
    state["packs"] += int(not linear or wrap)


@triton.jit(do_not_specialize=["source", "target", "rows"])
def move_workspace_rows(work, source, target, rows, cap: tl.constexpr, hd: tl.constexpr, block: tl.constexpr):
    x = tl.program_id(0) * block + tl.arange(0, block)
    side = tl.program_id(1).to(tl.int64)
    value = tl.load(work + (side * cap + source) * hd + x, x < rows * hd, 0)
    tl.store(work + (side * cap + target) * hd + x, value, x < rows * hd)


def compact_workspace(state, prefix, retained, origin, drop):
    """Move exact FP8 bytes left in disjoint passes over all layers.

    A pass never spans more than the source/destination distance, so no CTA
    overwrites another CTA's unread source. Stream ordering separates passes.
    """
    if os.environ.get("DUO_FP8_COMPACTION", "inplace") == "layer_scratch":
        return compact_workspace_scratch(state, prefix, retained, origin, drop)
    work = state["buffer"].view(torch.uint8)
    layers, _, cap, h, d = work.shape
    if prefix and origin >= prefix:
        move_workspace_rows[(triton.cdiv(prefix * h * d, 32768), layers * 2)](
            work, origin, 0, prefix, cap, h * d, 32768, num_warps=8
        )
    elif prefix and origin:
        scratch = state["prefix_scratch"].view(torch.uint8)
        grid = (triton.cdiv(prefix * h * d, 4096), layers * 2)
        move_prefix[grid](work, scratch, origin, 0, prefix, cap, h * d, True, 4096)
        move_prefix[grid](work, scratch, origin, 0, prefix, cap, h * d, False, 4096)
    distance = origin + drop
    if distance:
        for start in range(0, retained, distance):
            rows = min(distance, retained - start)
            move_workspace_rows[(triton.cdiv(rows * h * d, 32768), layers * 2)](
                work, distance + prefix + start, prefix + start, rows, cap, h * d, 32768, num_warps=8
            )


def compact_workspace_scratch(state, prefix, retained, origin, drop):
    """Move already encoded rows using one layer of scratch, preserving bytes."""
    work = state["buffer"]
    if "compact_scratch" not in state:
        state["compact_scratch"] = torch.empty_like(work[0])
    scratch = state["compact_scratch"]
    for layer in range(work.shape[0]):
        scratch[:, :prefix].copy_(work[layer, :, origin : origin + prefix])
        scratch[:, prefix : prefix + retained].copy_(
            work[layer, :, origin + prefix + drop : origin + prefix + drop + retained]
        )
        work[layer, :, : prefix + retained].copy_(scratch[:, : prefix + retained])


def forward(runtime, q, k, v, layer, scale):
    p = runtime.calibration
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    ko, vo = runtime.state["buffer"][layer]
    encode_qkv[(triton.cdiv(q.shape[0], 4), q.shape[1], 3)](
        q,
        k,
        v,
        q8,
        ko,
        vo,
        runtime.past_rows,
        p["centers"],
        p["scales"],
        p["signs"],
        runtime.clip_counts,
        q.shape[0],
        *q.shape[1:],
        layer,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        runtime.audit,
        4,
        text_capacity=runtime.text_rows if runtime.plan.text_valid_rows is not None else 0,
        text_valid_rows=runtime.plan.text_valid_rows,
        num_warps=4,
    )
    output = torch.empty_like(q)
    qs, ks, vs = p["scales"][layer]
    for i, (_, start, end) in enumerate(runtime.plan.launches):
        if i == 0 and runtime.reuse_text:
            output[start:end].copy_(runtime.text_output[layer])
            continue
        if i == 1 and runtime.fp8_fa3_library is not None:
            from .fp8_fa3 import library_forward

            y = library_forward(
                runtime.fp8_fa3_library,
                q8[start:end],
                ko,
                vo,
                runtime.cuq[i],
                runtime.cuk[i],
                qs[None],
                ks[None],
                vs[None],
                scale,
                runtime.scheduler_metadata[i],
                runtime.plan.capacity,
            )
        else:
            y = runtime.function(
                q8[start:end],
                ko,
                vo,
                end - start,
                runtime.cuq[i],
                runtime.plan.capacity,
                cu_seqlens_k=runtime.cuk[i],
                softmax_scale=scale,
                causal=False,
                num_splits=runtime.text_splits if i == 0 else runtime.media_splits,
                fa_version=3,
                q_descale=qs[None],
                k_descale=ks[None],
                v_descale=vs[None],
                scheduler_metadata=runtime.scheduler_metadata[i],
            )
        rotate(y, p["signs"], center=p["centers"][layer, 1], inverse=True, out=output[start:end])
        if i == 0 and runtime.store_text:
            runtime.text_output[layer].copy_(output[start:end])
    return output
