# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolated staging for clean/next joint compute before cache publication."""

from dataclasses import replace

import torch
from vllm.triton_utils import tl, triton

from .attention import RealizationPagedAttentionPlan
from .native_attention import RealizationNativeAttention, _layout, append_current
from .paged_cache import RealizationGraphWriteBinding, RealizationPagedCache


class JointAttentionPair:
    def __init__(self, pool, capacity, history_blocks):
        self.pool, self.capacity, self.history_blocks = pool, capacity, history_blocks
        self.clean_plan = self.next_plan = None

    def prepare(self, clean, noisy, cache, tx):
        pool = self.pool
        pool.validate(cache)
        if tx is not pool._pending or not tx.pending or tx.prior is not cache:
            raise ValueError("Joint attention requires the live clean reservation")
        if self.clean_plan is None:
            self.clean_plan = RealizationPagedAttentionPlan(clean.groups, capacity=self.capacity, device=clean.device)
        state = getattr(pool, "_realization_native_attention", None)
        if state is not None and state.get("base", 0) + cache.length + tx.retain.numel() + noisy.length > self.capacity:
            # Rebuild from the acknowledged physical cache before any layer reads it.
            if getattr(pool, "_realization_fp8_layout", None) == "workspace":
                state["force_compact"] = True
            else:
                state["cache"] = None
        self.clean_plan.update(clean.groups, cache, pool)
        a = self.clean_plan.native_runtime
        if a is None:
            raise ValueError("Joint execution requires native TP4/TP8 attention")
        if max(g.block for g in clean.groups) + 1 != max(g.block for g in noisy.groups):
            raise ValueError("Joint clean and noisy chunks must be consecutive")
        if torch.any(clean.time_ids != 0):
            raise ValueError("Joint clean chunk requires time zero ids")
        self.count = clean.length
        first = max(g.block for g in noisy.groups) - self.history_blocks
        kept = tuple(s for s in tx.candidate.spans if s.group.kind != "clean" or s.group.block >= first)
        cursor, spans = 0, []
        for s in kept:
            n = s.group.end - s.group.start
            spans.append(replace(s, group=replace(s.group, start=cursor, end=cursor + n)))
            cursor += n
        self.private_next = RealizationPagedCache(pool.owner, pool.epoch, tuple(spans))
        self.drop = cache.length + tx.retain.numel() - cursor
        if self.drop < 0 or a.state.get("base", 0) + self.drop + cursor + noisy.length > self.capacity:
            raise ValueError("Joint staging needs compaction or unsupported history geometry")
        self.prefix = sum(s.group.end - s.group.start for s in cache.spans if s.group.kind in ("reference", "shared"))
        self.old_base = a.state.get("base", 0)
        self.new_base = self.old_base + self.drop
        if not hasattr(self, "write_offset"):
            self.write_offset = torch.zeros((), device=clean.device, dtype=torch.int32)
            self.old_origin = torch.zeros_like(self.write_offset)
            self.new_origin = torch.zeros_like(self.write_offset)
            self.prefix_scratch = torch.empty(
                (2, self.prefix, *pool.storage.shape[-2:]),
                device=clean.device,
                dtype=pool.storage.dtype if a.fp8_precondition else a.state["buffer"].dtype,
            )
        self.write_offset.fill_(self.old_base + cache.length)
        self.old_origin.fill_(self.old_base)
        self.new_origin.fill_(self.new_base)
        if self.next_plan is None:
            self.next_plan = RealizationPagedAttentionPlan(noisy.groups, capacity=self.capacity, device=noisy.device)
            self.next_plan._native_decided = True
            self.next_plan.native_runtime = RealizationNativeAttention(self.next_plan, a.version, a.function)
        p = self.next_plan
        b = p.native_runtime
        b.update_slot_metadata(noisy.groups, self.private_next, pool)
        p.metadata_updates += 1
        p._ready = True
        state = a.state
        saved = (state["cache"], state["spans"], state.get("base", 0))
        state.update(cache=self.private_next, spans=_layout(p, noisy.groups, self.private_next)[3], base=self.new_base)
        b.joint_private = True
        try:
            b.prepare(noisy.groups, self.private_next, pool)
        finally:
            b.joint_private = False
            state.update(cache=saved[0], spans=saved[1], base=saved[2])
        self.binding = (
            RealizationGraphWriteBinding(tx.slots.clone(), tx.retain.clone())
            if not hasattr(self, "binding")
            else self.binding
        )
        self.binding.workspace_offset = self.write_offset
        self.binding.slots.copy_(tx.slots)
        self.binding.retain.copy_(tx.retain)
        self.tx = tx
        a.reuse_text = False
        b.reuse_text = False
        b.store_text = True

    def attend(self, q, k, v, layer, scale):
        a, b = self.clean_plan.native_runtime, self.next_plan.native_runtime
        cut = self.count
        first = a.forward(q[:cut], k[:cut], v[:cut], layer, scale)
        self.pool.write_layer(self.binding, layer, k[:cut], v[:cut])
        if a.fp8_precondition and not a.fp8_incremental:
            second = b.forward(q[cut:], k[cut:], v[cut:], layer, scale)
            return torch.cat((first, second))
        workspace = a.state["buffer"][layer]
        # Same topology during warmup and rolling eviction; zero shift is exact.
        # Small prefix copy needs a temporary because source/destination overlap.
        hd = k.shape[1] * k.shape[2]
        grid = (triton.cdiv(self.prefix * hd, 4096), 2)
        copy_prefix[grid](workspace, self.prefix_scratch, self.old_origin, self.capacity, self.prefix, hd, True, 4096)
        copy_prefix[grid](workspace, self.prefix_scratch, self.new_origin, self.capacity, self.prefix, hd, False, 4096)
        if a.fp8_incremental and self.pool._realization_fp8_layout == "workspace":
            # write_layer fused BF16 publication with the logical FP8 append.
            pass
        elif a.fp8_incremental:
            from .fp8_incremental import copy_pages

            copy_pages[(triton.cdiv(self.binding.slots.numel() * hd, 4096), 2)](
                self.pool._realization_fp8_shadow,
                a.state["buffer"],
                self.binding.slots,
                self.write_offset,
                self.binding.slots.numel(),
                self.pool.storage.shape[2] * self.pool.page_size,
                self.capacity,
                hd,
                layer,
            )
        else:
            clean_k = k[:cut].index_select(0, self.binding.retain)
            clean_v = v[:cut].index_select(0, self.binding.retain)
            h, d = clean_k.shape[1:]
            append_current[(triton.cdiv(clean_k.numel(), 4096),)](
                clean_k,
                clean_v,
                workspace[0],
                workspace[1],
                self.write_offset,
                clean_k.shape[0],
                h,
                d,
                clean_k.stride(0),
                clean_k.stride(1),
                clean_v.stride(0),
                clean_v.stride(1),
                4096,
                k_scale=a.kv_scales[layer, 0] if a.fp8_attention else None,
                v_scale=a.kv_scales[layer, 1] if a.fp8_attention else None,
                fp8=a.fp8_attention,
                num_warps=8,
            )
        second = b.forward(q[cut:], k[cut:], v[cut:], layer, scale)
        return torch.cat((first, second))

    def stage_published(self, cache):
        """Only call after all-rank finite ack, publish and exact eviction."""
        self.pool.validate(cache)
        if cache.spans != self.private_next.spans:
            raise ValueError("Published cache differs from joint private projection")
        state = self.clean_plan.native_runtime.state
        if self.clean_plan.native_runtime.fp8_incremental:
            from .fp8_incremental import mark_staged

            mark_staged(self.clean_plan.native_runtime, self.pool, cache)
        state.update(
            cache=cache,
            spans=self.next_plan.native_runtime.spans,
            base=(
                0
                if (
                    self.clean_plan.native_runtime.fp8_precondition
                    and not self.clean_plan.native_runtime.fp8_incremental
                )
                else self.new_base
            ),
        )


@triton.jit
def copy_prefix(
    work, temp, origin, cap: tl.constexpr, rows: tl.constexpr, hd: tl.constexpr, save: tl.constexpr, b: tl.constexpr
):
    x = tl.program_id(0) * b + tl.arange(0, b)
    side = tl.program_id(1)
    base = tl.load(origin)
    off = side.to(tl.int64) * cap * hd + base * hd + x
    scratch = side * rows * hd + x
    if save:
        val = tl.load(work + off, x < rows * hd, 0.0)
        tl.store(temp + scratch, val, x < rows * hd)
    else:
        val = tl.load(temp + scratch, x < rows * hd, 0.0)
        tl.store(work + off, val, x < rows * hd)
