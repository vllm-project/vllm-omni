# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native dense attention for TP4/TP8, with immutable-history staging.

The serialized Realization graph runner owns the staging stream. KV publication and
physical page leases remain in RealizationPagedPool; this workspace only copies data.
"""

import os

import torch
from vllm.triton_utils import tl, triton

from .contracts import visible


@triton.jit
def expand_physical_pages(pages, starts, counts, slots, page_size: tl.constexpr, block: tl.constexpr):
    page = tl.program_id(0)
    lane = tl.arange(0, block)
    physical = tl.load(pages + page)
    begin = tl.load(starts + page)
    count = tl.load(counts + page)
    tl.store(slots + begin + lane, physical.to(tl.int64) * page_size + lane, lane < count)


@triton.jit(do_not_specialize=["past_rows"])
def gather_history(
    storage,
    workspace,
    slots,
    group: tl.constexpr,
    slot_cap: tl.constexpr,
    past_rows,
    physical_rows: tl.constexpr,
    capacity: tl.constexpr,
    hd: tl.constexpr,
    B: tl.constexpr,  # noqa: N803
    scales=None,
    head_dim: tl.constexpr = 128,
    fp8: tl.constexpr = False,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    # A 512-page TP4 pool has more than 2**31 elements across 50 layers.
    # Widen before computing either flattened layer offset.
    side = tl.program_id(1).to(tl.int64)
    layer = tl.program_id(2).to(tl.int64)
    row = x // hd
    channel = x % hd
    slot = tl.load(slots + group * slot_cap + row, row < past_rows, 0)
    value = tl.load(storage + (layer * 2 + side) * physical_rows * hd + slot * hd + channel, row < past_rows, 0)
    if fp8:
        h = hd // head_dim
        descale = tl.load(scales + (layer * 2 + side) * h + channel // head_dim)
        value = tl.minimum(tl.maximum(value.to(tl.float32) / descale, -448.0), 448.0)
    tl.store(workspace + (layer * 2 + side) * capacity * hd + x, value, row < past_rows)


@triton.jit(do_not_specialize=["n"])
def append_current(
    k,
    v,
    ko,
    vo,
    past_rows,
    n,
    h: tl.constexpr,
    d: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    B: tl.constexpr,  # noqa: N803
    k_scale=None,
    v_scale=None,
    fp8: tl.constexpr = False,
    text_capacity: tl.constexpr = 0,
    text_valid_rows=None,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    row = x // (h * d)
    head = x // d % h
    dim = x % d
    offset = tl.load(past_rows)
    key = tl.load(k + row * ks0 + head * ks1 + dim, row < n, 0)
    value = tl.load(v + row * vs0 + head * vs1 + dim, row < n, 0)
    if fp8:
        key = tl.minimum(tl.maximum(key.to(tl.float32) / tl.load(k_scale + head), -448.0), 448.0)
        value = tl.minimum(tl.maximum(value.to(tl.float32) / tl.load(v_scale + head), -448.0), 448.0)
    target = row
    active = row < n
    if text_capacity:
        real = tl.load(text_valid_rows)
        target = tl.where(row < text_capacity, row, row - text_capacity + real)
        active &= (row < real) | (row >= text_capacity)
    tl.store(ko + (offset + target) * h * d + head * d + dim, key, active)
    tl.store(vo + (offset + target) * h * d + head * d + dim, value, active)


def _layout(plan, groups, cache):
    if len(plan.launches) != 2:
        return None
    (_, a, b), (g1, c, end) = plan.launches
    if a != 0 or b != c or groups[0].kind != "text":
        return None
    q0, q1 = groups[0], groups[g1]
    past0 = tuple(s for s in cache.spans if plan.sees_past(q0, s.group))
    past1 = tuple(s for s in cache.spans if plan.sees_past(q1, s.group))
    current0 = [i for g in groups if visible(q0, g) for i in range(g.start, g.end)]
    current1 = [i for g in groups if visible(q1, g) for i in range(g.start, g.end)]
    if past0 != past1 or current0 != list(range(b)) or current1 != list(range(end)):
        return None
    n = sum(s.group.end - s.group.start for s in past1)
    if n + end > plan.capacity:
        raise ValueError("Native Realization attention workspace capacity exceeded")
    return b, end, g1, past1, n


def maybe_create(plan, groups, cache, pool):
    # Realization MHA56 / TP4 or TP8; keep other geometries on the paged path.
    if pool.storage.device.type != "cuda" or pool.storage.shape[-1] != 128 or pool.storage.shape[-2] not in (7, 14):
        return None
    if pool.storage.dtype not in (torch.bfloat16, torch.float16) or _layout(plan, groups, cache) is None:
        return None
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported
    except ImportError:
        return None
    requested = os.environ.get("DUO_NATIVE_FA_VERSION")
    if requested not in (None, "3", "4"):
        raise ValueError("DUO_NATIVE_FA_VERSION must be 3 or 4")
    policy = os.environ.get("DUO_ATTENTION_POLICY", "legacy")
    candidates = (int(requested),) if requested else ((3,) if policy in ("fa3_auto", "fa3_unified") else (4, 3))
    version = next((v for v in candidates if is_fa_version_supported(v)), None)
    if version is None and policy in ("fa3_auto", "fa3_unified"):
        raise RuntimeError("Requested official FA3 policy is unavailable on this device")
    return None if version is None else RealizationNativeAttention(plan, version, flash_attn_varlen_func)


class RealizationNativeAttention:
    def __init__(self, plan, version, function):
        self.policy = os.environ.get("DUO_ATTENTION_POLICY", "legacy")
        if self.policy not in ("legacy", "fa3_auto", "fa3_unified"):
            raise ValueError("Realization attention policy must be legacy, fa3_auto or fa3_unified")
        if self.policy in ("fa3_auto", "fa3_unified") and version != 3:
            raise ValueError(f"{self.policy} policy requires official FA3")
        self.fp8_attention = os.environ.get("DUO_FP8_ATTENTION", "0") == "1"
        self.fp8_precondition = self.fp8_attention and os.environ.get("DUO_FP8_PRECONDITION", "0") != "0"
        self.fp8_incremental = self.fp8_attention and os.environ.get("DUO_FP8_PRECONDITION") == "incremental"
        self.fp8_fa3_library = None
        fp8_library = os.environ.get("DUO_FP8_FA3_LIBRARY")
        if self.fp8_incremental and fp8_library:
            if self.policy not in ("fa3_auto", "fa3_unified"):
                raise ValueError("FP8 FA3 library requires official FA3 scheduling")
            from .fp8_fa3 import load_library

            self.fp8_fa3_library = load_library(fp8_library)
        self.plan, self.version, self.function = plan, version, function
        self.metadata_version = -1
        self.state = None
        self.store_text = False
        self.reuse_text = False
        if self.policy in ("fa3_auto", "fa3_unified"):
            # Start from official FA3 automatic scheduling. Unified mode adds
            # a shape/hardware partition budget during prepare; neither mode
            # reads legacy tuning flags. Device metadata survives graph replay.
            # Legacy tuning flags deliberately do not affect this policy.
            self.text_backend = "fa3"
            self.text_splits = self.media_splits = 0
            self.media_partitions = 1
            self.media_query_split = 0
            self.broadcast_partition_q = False
            self.merge_warps = 4
            return
        self.text_backend = os.environ.get("DUO_TEXT_BACKEND", "fa3")
        if self.text_backend not in ("fa3", "flashinfer_fa2"):
            raise ValueError("Invalid Realization text attention backend")
        self.text_splits = int(os.environ.get("DUO_TEXT_SPLITS", "8"))
        self.media_splits = int(os.environ.get("DUO_MEDIA_SPLITS", "2"))
        self.media_partitions = int(os.environ.get("DUO_MEDIA_PARTITIONS", "1"))
        self.broadcast_partition_q = os.environ.get("DUO_BROADCAST_PARTITION_Q", "0") == "1"
        if self.broadcast_partition_q and (self.media_partitions == 1 or version != 3):
            raise ValueError("Broadcast partition Q requires partitioned FA3")
        self.merge_warps = int(os.environ.get("DUO_MERGE_WARPS", "4"))
        if self.merge_warps not in (1, 2, 4, 8):
            raise ValueError("Invalid Realization partition merge warp count")
        self.media_query_split = int(os.environ.get("DUO_MEDIA_QUERY_SPLIT", "0"))
        if self.media_query_split not in (0, 128) or (self.media_query_split and self.media_partitions == 1):
            raise ValueError("Realization query-tail split requires partitioned FA3 and 0/128")
        if self.media_partitions not in (1, 2, 4, 8, 16) or (self.media_partitions > 1 and version != 3):
            raise ValueError("Realization media partitions require FA3 and 1/2/4/8/16 partitions")
        if self.text_splits not in (1, 2, 4, 8, 16, 32, 64) or self.media_splits not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("Invalid native Realization dense attention split count")

    def update_slot_metadata(self, groups, cache, pool):
        """Compile immutable historical slots once per live cache snapshot.

        The native layout proof guarantees identical historical visibility for
        every domain. Copy that shared map on device; only current negative
        indices differ. The caller validates the page lease on every update.
        """
        p = self.plan
        layout = _layout(p, groups, cache)
        if layout is None:
            raise RuntimeError("Native Realization metadata visibility changed")
        text_rows, rows, _, spans, n = layout
        saved = getattr(pool, "_realization_native_slot_metadata", None)
        if saved is None or saved[0] is not cache or saved[1] != spans:
            # O(pages) host metadata; expand exact partial-page slots on GPU.
            historical = torch.empty(n, device=p.slots.device, dtype=torch.long)
            if n:
                entries, cursor = [], 0
                for span in spans:
                    size = span.group.end - span.group.start
                    entries.extend(
                        (page, cursor + i * pool.page_size, min(pool.page_size, size - i * pool.page_size))
                        for i, page in enumerate(span.block_ids)
                    )
                    cursor += size
                page_map = torch.tensor(tuple(zip(*entries)), device=p.slots.device, dtype=torch.int32)
                expand_physical_pages[(len(entries),)](
                    page_map[0],
                    page_map[1],
                    page_map[2],
                    historical,
                    pool.page_size,
                    triton.next_power_of_2(pool.page_size),
                    num_warps=1,
                )
            saved = (cache, spans, historical)
            pool._realization_native_slot_metadata = saved
        if not hasattr(self, "_current_slot_indices"):
            self._current_slot_indices = -torch.arange(1, rows + 1, device=p.slots.device, dtype=torch.long)
        lengths = []
        for i, group in enumerate(groups):
            count = text_rows if group.kind == "text" else rows
            p.slots[i, :n].copy_(saved[2])
            p.slots[i, n : n + count].copy_(self._current_slot_indices[:count])
            lengths.append(n + count)
        p.lengths.copy_(torch.tensor(lengths, dtype=torch.int32))

    def prepare(self, groups, cache, pool):
        if self.plan.text_valid_rows is not None and (self.fp8_attention and not self.fp8_incremental):
            raise ValueError("Text graph buckets support dense BF16 or incremental FP8 FA3")
        p = self.plan
        if self.metadata_version != p.metadata_updates:
            layout = _layout(p, groups, cache)
            if layout is None:
                raise RuntimeError("Captured native Realization attention visibility changed; recreate graph slot")
            self.text_rows, self.rows, self.media_group, self.spans, self.past_count = layout
        if not hasattr(self, "text_output"):
            self.text_output = torch.empty(
                (pool.storage.shape[0], self.text_rows, *pool.storage.shape[-2:]),
                device=pool.storage.device,
                dtype=pool.storage.dtype,
            )
        n = self.past_count
        if self.fp8_attention:
            if self.fp8_incremental:
                self.calibration = pool._realization_fp8_calibration
                self.kv_scales = self.calibration["scales"][:, 1:]
                self.clip_counts = pool._realization_fp8_clips
                self.audit = pool._realization_fp8_audit
            elif self.fp8_precondition:
                from .fp8_precondition import prepare

                prepare(self, pool)
            else:
                self.kv_scales = pool._realization_fp8_scales
            self.text_backend = "fa3"
        if self.policy == "fa3_unified" and not hasattr(self, "unified_geometry"):
            from .fa3_policy import query_partition, split_budget

            count = self.rows - self.text_rows
            main, tail = query_partition(count)
            from vllm.platforms import current_platform

            sms = current_platform.num_compute_units(pool.storage.device.index)
            self.media_partitions = split_budget(main, pool.storage.shape[-2], sms)
            # Single-row text is a bandwidth-oriented decode workload.
            if self.text_rows == 1 and not self.fp8_attention:
                self.text_backend = "flashinfer_fa2"
            self.broadcast_partition_q = self.media_partitions > 1
            self.unified_geometry = (main, tail)
        if self.fp8_precondition:
            if self.policy == "fa3_unified":
                from vllm.platforms import current_platform

                from .fa3_policy import fp8_split_budget

                sms = current_platform.num_compute_units(pool.storage.device.index)
                self.media_splits = fp8_split_budget(self.rows - self.text_rows, pool.storage.shape[-2], sms)
            self.media_partitions = 1
            self.broadcast_partition_q = False
        if self.fp8_fa3_library is not None:
            self.media_splits = 1
        if self.fp8_attention and not self.fp8_precondition and not hasattr(self, "fp8_partition_scales"):
            self.fp8_partition_scales = (
                self.kv_scales[:, :, None, :].expand(-1, -1, self.media_partitions, -1).contiguous()
            )
        stream = torch.cuda.current_stream(pool.storage.device).cuda_stream
        if not hasattr(pool, "_realization_native_attention"):
            layers, _, _, _, h, d = pool.storage.shape
            pool._realization_native_attention = dict(
                buffer=None
                if self.fp8_precondition and not self.fp8_incremental
                else torch.empty(
                    (layers, 2, p.capacity, h, d),
                    device=p.slots.device,
                    dtype=torch.float8_e4m3fn if self.fp8_attention else pool.storage.dtype,
                ),
                capacity=p.capacity,
                cache=None,
                spans=None,
                packs=0,
                base=0,
                stream=stream,
            )
        state = pool._realization_native_attention
        if state["capacity"] != p.capacity or state["stream"] != stream:
            raise RuntimeError("Native Realization attention staging capacity or serialized stream changed")
        self.state = state
        if self.fp8_incremental and not getattr(self, "joint_private", False):
            from .fp8_incremental import stage_history

            stage_history(self, pool, cache, state)
        needs_pack = state["cache"] is not cache or state["spans"] != self.spans
        if state.get("base", 0) + n + self.rows > p.capacity:
            needs_pack = True
        if (needs_pack and not self.fp8_incremental) or (self.fp8_precondition and not self.fp8_incremental):
            state["base"] = 0
        base = state.get("base", 0)
        if not hasattr(self, "past_rows"):
            self.past_rows = torch.zeros((), device=p.slots.device, dtype=torch.int32)
            self.cuq = [
                torch.tensor([0, count], device=p.slots.device, dtype=torch.int32)
                for count in (self.text_rows, self.rows - self.text_rows)
            ]
            self.cuk = [torch.zeros(2, device=p.slots.device, dtype=torch.int32) for _ in range(2)]
        if (
            self.metadata_version != p.metadata_updates
            or getattr(self, "origin", None) != base
            or p.text_valid_rows is not None
        ):
            self.past_rows.fill_(base + n)
            for target, count in zip(self.cuk, (self.text_rows, self.rows), strict=True):
                target.copy_(torch.tensor([base, base + n + count], dtype=torch.int32))
                if p.text_valid_rows is not None:
                    target[1].sub_(self.text_rows - p.text_valid_rows[0])
            if self.policy in ("fa3_auto", "fa3_unified"):
                self._prepare_auto_schedule(pool)
            if self.text_backend == "flashinfer_fa2" and self.text_rows == 1:
                import flashinfer

                if not hasattr(self, "text_wrapper"):
                    self.text_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=p.slots.device)
                    self.text_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                        self.text_workspace,
                        kv_layout="NHD",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuq[0],
                        kv_indptr_buf=self.cuk[0],
                        backend="fa2",
                    )
                self.text_wrapper.plan(
                    torch.tensor([0, self.text_rows], dtype=torch.int32),
                    torch.tensor([base, base + n + self.text_rows], dtype=torch.int32),
                    pool.storage.shape[-2],
                    pool.storage.shape[-2],
                    pool.storage.shape[-1],
                    causal=False,
                    q_data_type=pool.storage.dtype,
                    kv_data_type=pool.storage.dtype,
                    sm_scale=pool.storage.shape[-1] ** -0.5,
                    max_sequence_kv=p.capacity,
                )
            if self.media_partitions > 1:
                count = self.rows - self.text_rows
                self.media_main_rows = (
                    self.unified_geometry[0]
                    if self.policy == "fa3_unified"
                    else (128 if self.media_query_split and count > 128 else count)
                )
                if self.fp8_attention:
                    self.media_main_rows = count
                self.media_tail_rows = count - self.media_main_rows
                if self.media_tail_rows and not hasattr(self, "tail_cuq"):
                    self.tail_cuq = torch.tensor([0, self.media_tail_rows], device=p.slots.device, dtype=torch.int32)
                if self.media_tail_rows and p.text_valid_rows is None:
                    import flashinfer

                    if not hasattr(self, "tail_wrapper"):
                        self.tail_cuq = torch.tensor(
                            [0, self.media_tail_rows], device=p.slots.device, dtype=torch.int32
                        )
                        self.tail_workspace = torch.empty(128 * 1024 * 1024, device=p.slots.device, dtype=torch.uint8)
                        self.tail_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                            self.tail_workspace,
                            kv_layout="NHD",
                            use_cuda_graph=True,
                            qo_indptr_buf=self.tail_cuq,
                            kv_indptr_buf=self.cuk[1],
                            backend="fa2",
                        )
                    self.tail_wrapper.plan(
                        torch.tensor([0, self.media_tail_rows], dtype=torch.int32),
                        torch.tensor([base, base + n + self.rows], dtype=torch.int32),
                        pool.storage.shape[-2],
                        pool.storage.shape[-2],
                        pool.storage.shape[-1],
                        causal=False,
                        q_data_type=pool.storage.dtype,
                        kv_data_type=pool.storage.dtype,
                        sm_scale=pool.storage.shape[-1] ** -0.5,
                        max_sequence_kv=p.capacity,
                    )
                if not hasattr(self, "partition_cuq"):
                    self.partition_cuq = (
                        torch.arange(self.media_partitions + 1, dtype=torch.int32, device=p.slots.device)
                        * self.media_main_rows
                    )
                    self.partition_cuk = torch.empty_like(self.partition_cuq)
                bounds = torch.arange(self.media_partitions + 1, device=p.slots.device, dtype=torch.int32)
                self.partition_cuk.copy_(base + bounds * (self.cuk[1][1] - base) // self.media_partitions)
            self.metadata_version = p.metadata_updates
            self.origin = base
        if needs_pack:
            hd = pool.storage.shape[-2] * pool.storage.shape[-1]
            if n and not self.fp8_precondition:
                gather_history[(triton.cdiv(n * hd, 4096), 2, pool.storage.shape[0])](
                    pool.storage,
                    state["buffer"],
                    p.slots,
                    self.media_group,
                    p.capacity,
                    n,
                    pool.storage.shape[2] * pool.storage.shape[3],
                    p.capacity,
                    hd,
                    4096,
                    scales=self.kv_scales if self.fp8_attention else None,
                    head_dim=pool.storage.shape[-1],
                    fp8=self.fp8_attention,
                    num_warps=8,
                )
            state["cache"], state["spans"] = cache, self.spans
            state["packs"] += 1

    def _prepare_auto_schedule(self, pool):
        """Plan once per snapshot; preserve metadata addresses for graph replay.

        Capacity bounds the kernel topology. Official FA3's GPU planner reads
        the actual domain lengths to choose dynamic splits. This is independent
        of resolution and call duration, and never reads a GPU scalar on CPU.
        """
        from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata

        if not hasattr(self, "scheduler_metadata"):
            self.scheduler_metadata = [None, None]
        h, d = pool.storage.shape[-2:]
        for i, current in enumerate((self.text_rows, self.rows)):
            length = self.cuk[i][1:] - self.cuk[i][:1]
            if i == 1 and self.fp8_fa3_library is not None:
                from .fp8_fa3 import library_metadata

                metadata = library_metadata(
                    self.fp8_fa3_library, self.rows - self.text_rows, self.plan.capacity, h, d, length, self.cuq[i]
                )
            else:
                metadata = get_scheduler_metadata(
                    1,
                    self.text_rows if i == 0 else self.rows - self.text_rows,
                    self.plan.capacity,
                    h,
                    h,
                    d,
                    length,
                    qkv_dtype=torch.float8_e4m3fn if self.fp8_attention else pool.storage.dtype,
                    cu_seqlens_q=self.cuq[i],
                    causal=False,
                    num_splits=self.text_splits if i == 0 else self.media_splits,
                )
            saved = self.scheduler_metadata[i]
            if saved is None:
                self.scheduler_metadata[i] = metadata
            else:
                if saved.shape != metadata.shape or saved.dtype != metadata.dtype:
                    raise RuntimeError("FA3 scheduler topology changed; recreate graph slot")
                saved.copy_(metadata)

    def forward(self, q, k, v, layer, scale):
        p = self.plan
        if self.state is None or k.shape[0] != self.rows:
            raise ValueError("Native Realization attention requires prepared matching geometry")
        if self.fp8_attention:
            from .fp8_attention import fp8_forward

            return fp8_forward(self, q, k, v, layer, scale)
        output = torch.empty_like(q)
        h, d = q.shape[1:]
        ko, vo = self.state["buffer"][layer, 0], self.state["buffer"][layer, 1]
        append_current[(triton.cdiv(k.shape[0] * h * d, 4096),)](
            k,
            v,
            ko,
            vo,
            self.past_rows,
            k.shape[0],
            h,
            d,
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            4096,
            text_capacity=self.text_rows if p.text_valid_rows is not None else 0,
            text_valid_rows=p.text_valid_rows,
            num_warps=8,
        )
        for i, (_, start, end) in enumerate(p.launches):
            if i == 0 and self.reuse_text:
                output[start:end].copy_(self.text_output[layer])
                continue
            if i == 0 and hasattr(self, "text_wrapper"):
                self.text_wrapper.run(q[start:end], ko, vo, out=output[start:end])
                if self.store_text:
                    self.text_output[layer].copy_(output[start:end])
                continue
            if i == 1 and self.media_partitions > 1:
                from .attention_partition import merge_partitions

                count = self.media_main_rows
                main_end = start + count
                max_k = (p.capacity + self.media_partitions - 1) // self.media_partitions
                if self.broadcast_partition_q:
                    from .attention_partition import broadcast_partition_attention

                    partial, lse = broadcast_partition_attention(
                        q[start:main_end], ko, vo, self.partition_cuk, max_k, scale, self.media_partitions
                    )
                    lse_part, lse_head, lse_row = lse.stride()
                else:
                    repeated = q[start:main_end].repeat(self.media_partitions, 1, 1)
                    partial, lse = self.function(
                        repeated,
                        ko,
                        vo,
                        count,
                        self.partition_cuq,
                        max_k,
                        cu_seqlens_k=self.partition_cuk,
                        softmax_scale=scale,
                        causal=False,
                        num_splits=1,
                        fa_version=3,
                        return_softmax_lse=True,
                    )
                    lse_part, lse_head, lse_row = 0, lse.stride(0), lse.stride(1)
                merge_partitions[(count, h)](
                    partial,
                    lse,
                    output[start:main_end],
                    count,
                    h,
                    d,
                    self.media_partitions,
                    lse_head,
                    lse_row,
                    triton.next_power_of_2(self.media_partitions),
                    triton.next_power_of_2(d),
                    lse_part=lse_part,
                    num_warps=self.merge_warps,
                )
                if self.media_tail_rows:
                    if p.text_valid_rows is None:
                        self.tail_wrapper.run(q[main_end:end], ko, vo, out=output[main_end:end])
                    else:
                        self.function(
                            q[main_end:end],
                            ko,
                            vo,
                            self.media_tail_rows,
                            self.tail_cuq,
                            p.capacity,
                            cu_seqlens_k=self.cuk[1],
                            causal=False,
                            softmax_scale=scale,
                            num_splits=self.media_splits,
                            fa_version=3,
                            out=output[main_end:end],
                        )
                continue
            self.function(
                q[start:end],
                ko,
                vo,
                end - start,
                self.cuq[i],
                p.capacity,
                cu_seqlens_k=self.cuk[i],
                softmax_scale=scale,
                causal=False,
                num_splits=8 if self.version == 4 else (self.text_splits if i == 0 else self.media_splits),
                fa_version=self.version,
                out=output[start:end],
                scheduler_metadata=self.scheduler_metadata[i] if self.policy in ("fa3_auto", "fa3_unified") else None,
            )
            if i == 0 and self.store_text:
                self.text_output[layer].copy_(output[start:end])
        return output
