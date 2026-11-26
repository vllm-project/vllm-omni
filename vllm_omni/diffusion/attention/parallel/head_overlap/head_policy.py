# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Learn an uneven head split from native denoising layers and agree once per layout."""

import math
import statistics
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from weakref import WeakSet

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.cuda_graph import kwargs_signature

from .ce_plan import HeadPlan, prepare_plan
from .query_tail import candidate, eligible


def canonical_strides(value):
    strides = list(value.stride())
    # Batch-one attention never addresses another batch. Prefix-cache clone()
    # can preserve this unused pitch whereas graph input clones canonicalize it.
    # Keep all sequence/head strides, including fused V's threefold row pitch.
    if len(value.shape) == 4 and value.shape[0] == 1:
        strides[0] = 0
    return tuple(strides)


def tensor_spec(value):
    return tuple(value.shape), canonical_strides(value), str(value.dtype), value.device.type


def metadata_spec(value):
    """Describe host semantics and tensor layouts without reading device data."""
    if hasattr(value, "shape"):
        return tensor_spec(value), True
    if isinstance(value, (int, float, bool, str, type(None))):
        return value, True
    if isinstance(value, dict):
        rows = [(key, metadata_spec(item)) for key, item in sorted(value.items())]
        return tuple((key, item[0]) for key, item in rows), all(item[1] for _, item in rows)
    if isinstance(value, (tuple, list)):
        rows = [metadata_spec(item) for item in value]
        return tuple(item[0] for item in rows), all(item[1] for item in rows)
    if is_dataclass(value):
        rows = [(item.name, metadata_spec(getattr(value, item.name))) for item in dataclass_fields(value)]
        return (type(value).__qualname__, tuple((name, item[0]) for name, item in rows)), all(
            item[1] for _, item in rows
        )
    return ("opaque", type(value).__module__, type(value).__qualname__), False


def qualified_cycles(choices):
    """Pair complete rotations, discarding a cycle if any rank/choice starves.

    Filtering each choice independently would compare different layer
    neighborhoods and reintroduce position/frequency bias.
    """
    count = min(len(rows) for rows in choices)
    discard = 2 if count >= 10 else 1
    valid = [index for index in range(discard, count) if all(row[1] for rows in choices for row in rows[index])]
    return [[max(row[0] for row in rows[index]) for index in valid] for rows in choices]


def choose(samples, *, minimum=6, relative_margin=0.015, absolute_margin_us=5.0):
    """Use position-paired observations and an explicit noise margin."""
    baseline = samples[0]
    if len(baseline) < minimum:
        return 0
    best, saving = 0, 0.0
    center = statistics.median(baseline)
    for index, values in enumerate(samples[1:], 1):
        count = min(len(baseline), len(values))
        if count < minimum:
            continue
        differences = [a - b for a, b in zip(baseline[:count], values[:count])]
        gain = statistics.median(differences)
        mad = statistics.median(abs(value - gain) for value in differences)
        margin = max(absolute_margin_us, center * relative_margin, 2 * mad)
        if gain > margin and gain > saving and sum(d > 0 for d in differences) >= math.ceil(0.8 * count):
            best, saving = index, gain
    return best


def candidate_index(index):
    # Repeat the original two-choice rotation in both directions.
    cycle, position = divmod(index, 2)
    return position ^ (cycle % 2) ^ ((cycle // 2) % 2)


logger = init_logger(__name__)
_OWNERS = WeakSet()


def clear():
    for policy in list(_OWNERS):
        if policy.layouts:
            torch.accelerator.synchronize()
        policy.layouts.clear()
        policy.pending.clear()
        policy.component_layouts.clear()
        policy.comm = None


_PREPARING_GRAPH = ContextVar("ulysses_preparing_graph", default=False)


@contextmanager
def frozen_graph_policy(original_context):
    """Warmup and capture can only consume a previously measured decision."""
    token = _PREPARING_GRAPH.set(True)
    try:
        with original_context():
            yield
    finally:
        _PREPARING_GRAPH.reset(token)


@dataclass
class Trial:
    widths: tuple
    tail_rows: int
    example: tuple = ()
    observations: list = field(default_factory=list)
    timers: list = field(default_factory=list)
    plan: HeadPlan | None = None
    seen: int = 0
    selected: bool | None = None


class HeadPolicy:
    """Owned by one DiT component and shared by its eligible Ulysses layers."""

    def __init__(self, group, sms):
        _OWNERS.add(self)
        self.group = group
        self.sms = sms
        self.layouts = {}
        self.pending = set()
        self.component_layouts = {}
        self.component_key = None
        self.discovery = None
        self.enabled_for_call = False
        self.generation = 0
        self.comm = None

    def begin_component(self, module, args, kwargs):
        # Component inputs precede sequence sharding. All SP peers enter a new
        # shape's discovery pass even if a local core is ineligible.
        self.component_key = kwargs_signature((args, kwargs))
        keys = self.component_layouts.get(self.component_key)
        frozen = _PREPARING_GRAPH.get() or torch.cuda.is_current_stream_capturing()
        discover = keys is None and not frozen and len(self.component_layouts) < 16
        self.discovery = [] if discover else None
        self.enabled_for_call = discover or any(
            self.layouts[key].selected or (not frozen and key in self.pending) for key in keys or ()
        )

    def try_run(self, strategy, layer, q, k, v, md):
        """Return None to execute the normal attention path, including native CE."""
        if q.ndim != 4 or q.shape[0] != 1 or not q.is_cuda or q.dtype != torch.bfloat16:
            return None
        world = strategy._sp_group.ulysses_world_size
        if (
            world < 2
            or q.shape[1] == 0
            or q.shape[2] % world
            or q.shape != k.shape
            or q.shape != v.shape
            or any(
                x.dtype != q.dtype
                or x.device != q.device
                or x.stride(3) != 1
                or x.stride(2) != x.shape[3]
                or x.stride(1) < x.shape[2] * x.shape[3]
                for x in (q, k, v)
            )
        ):
            return None
        md = layer._with_kv_cache_dtype(md)
        if not eligible(layer._run_local_attention, md, q.shape[1] * world):
            return None
        joint = None
        if md is not None and any(x is not None for x in (md.joint_query, md.joint_key, md.joint_value)):
            joint = md.joint_query, md.joint_key, md.joint_value
            if (
                any(
                    x is None
                    or x.ndim != 4
                    or x.shape[0] != 1
                    or x.shape[2:] != q.shape[2:]
                    or x.dtype != q.dtype
                    or x.device != q.device
                    or x.stride(3) != 1
                    or x.stride(2) != q.shape[3]
                    for x in joint
                )
                or joint[1].shape != joint[2].shape
                or md.joint_strategy not in ("front", "rear")
            ):
                return None
        # Each owner is bound to one device, process group and installed BF16 FA3 backend.
        # A cached signature describes every participating Q/K/V layout, including the prefix KV.
        signature = (world, tuple(tensor_spec(x) for x in (q, k, v)), metadata_spec(md)[0])
        trial = self.layouts.get(signature)
        frozen_only = _PREPARING_GRAPH.get() or torch.cuda.is_current_stream_capturing()
        if self.discovery is not None and not frozen_only:
            self.discovery.append(signature)
        if trial is None:
            if self.discovery is None or frozen_only or len(self.layouts) >= 16:
                return None
            choices, tail = candidate(q.shape[2] // world, q.shape[1] * world, world, self.sms)
            trial = self.layouts[signature] = Trial(choices[-1], tail)
            if len(choices) == 1:
                trial.selected = False
            else:
                # Only this first layer's example survives to the component boundary.
                trial.example = ((q, k, v), joint, strategy._ulysses_pg, layer._run_local_attention, md)
            return None
        if self.discovery is not None or signature not in self.component_layouts.get(self.component_key, ()):
            return None
        if trial.selected is False or trial.plan is None or (frozen_only and trial.selected is None):
            return None

        kernel = layer._run_local_attention

        def split():
            metadata = None if md is None else replace(md, joint_query=None, joint_key=None, joint_value=None)
            trial.plan.bind((q, k, v), joint, kernel, metadata)
            # The plan writes to a reusable symmetric slot. Attention's caller
            # owns its result and may retain it across later layers and requests.
            return trial.plan.run().clone()

        if trial.selected:
            return split()
        i = trial.seen
        trial.seen += 1
        if i < 6 or i - 6 >= len(trial.timers):
            return None
        index = i - 6
        start, end = trial.timers[index]
        choice = candidate_index(index)
        start.record()
        if choice:
            result = split()
        else:
            qq, kk, vv, native_md, ctx = strategy.pre_attention(q, k, v, md)
            result = strategy.post_attention(kernel(qq, kk, vv, native_md), ctx)
        end.record()
        trial.observations.append((choice, start, end, not start.query()))
        return result

    def finish_component(self, module, inputs, output):
        # Hooks execute once per real component call, outside individual attention layers.
        self.enabled_for_call = False
        if _PREPARING_GRAPH.get() or torch.cuda.is_current_stream_capturing():
            return
        if self.discovery is not None:
            order, self.discovery = self.discovery, None
            # Include ordered visits: a different order would issue unmatched
            # native/split collectives even if the set of layouts agreed.
            info = [
                (key, self.layouts[key].widths, self.layouts[key].tail_rows) for key in order if key in self.layouts
            ]
            peers = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(peers, info, group=self.group)
            agreed = all(peer == info for peer in peers)
            keys = tuple(dict.fromkeys(key for key, _, _ in info)) if agreed else ()
            self.component_layouts[self.component_key] = keys
            for key in dict.fromkeys(order):
                trial = self.layouts.get(key)
                if trial is None or not trial.example:
                    continue
                inputs, joint, pg, kernel, md = trial.example
                trial.example = ()
                if not agreed:
                    trial.selected = False
                    continue
                if self.comm is None:
                    self.comm = torch.cuda.Stream(device=inputs[0].device)
                trial.plan = prepare_plan(
                    inputs,
                    joint,
                    pg,
                    trial.widths,
                    self.comm,
                    kernel,
                    md,
                    joint_strategy="front" if md is None else md.joint_strategy,
                    query_tail_rows=trial.tail_rows,
                )
                trial.timers = [
                    (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(24)
                ]
                for start, end in trial.timers:
                    start.record()
                    end.record()
                self.pending.add(key)
            return
        self.resolve_pending(self.component_key)

    def resolve_pending(self, component_key):
        """Read completed measurements without waiting on the GPU."""
        keys = [key for key in self.component_layouts.get(component_key, ()) if key in self.pending]
        if not keys:
            return
        reports = {}
        for key in keys:
            observations = self.layouts[key].observations
            if len(observations) == 24 and all(end.query() for _, _, end, _ in observations):
                reports[key] = [(c, start.elapsed_time(end) * 1000, queued) for c, start, end, queued in observations]
            else:
                reports[key] = None
        peers = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(peers, reports, group=self.group)
        for key in keys:
            trial = self.layouts[key]
            if any(peer[key] is None for peer in peers):
                continue
            choices = [[], []]
            for values in zip(*(peer[key] for peer in peers), strict=True):
                choice = values[0][0]
                if any(value[0] != choice for value in values):
                    raise RuntimeError("Ulysses ranks sampled different head partitions")
                choices[choice].append([value[1:] for value in values])
            samples = qualified_cycles(choices)
            trial.selected = bool(choose(samples))
            self.generation += int(trial.selected)
            logger.info(
                "Ulysses head overlap: Q=%s heads=%s local_query_tail=%d qualified_pairs=%d",
                key[1][0][0],
                trial.widths if trial.selected else (sum(trial.widths),),
                trial.tail_rows if trial.selected else 0,
                len(samples[0]),
            )
            self.pending.discard(key)
            trial.observations.clear()
            trial.timers.clear()
            if not trial.selected:
                trial.plan = None


def configure_head_overlap(pipeline):
    """Attach shared state to a component's normal attention strategies at model load."""
    from vllm_omni.diffusion.attention.backends.utils.fa import dense_flash_attn_func_for
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_cuda() or torch.version.hip is not None:
        return
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    backend = getattr(dense_flash_attn_func_for(torch.bfloat16), "__module__", None)
    group = get_sp_group().cpu_group
    signature = (props.name, props.major, props.minor, props.multi_processor_count, backend)
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, signature, group=group)
    if any(peer != signature for peer in peers) or props.major != 9 or backend != "fa3_fwd_interface":
        logger.info("Automatic Ulysses head overlap uses native attention on this GPU/backend")
        return
    for name in getattr(pipeline, "_dit_modules", ("transformer", "transformer_2")):
        module = getattr(pipeline, name, None)
        if module is None:
            continue
        policy = HeadPolicy(group, props.multi_processor_count)
        for layer in module.modules():
            strategy = getattr(layer, "parallel_strategy", None)
            if getattr(strategy, "auto_head_overlap", False):
                strategy.head_policy = policy
        module._ulysses_head_policy = policy
        module.register_forward_pre_hook(policy.begin_component, with_kwargs=True)
        module.register_forward_hook(policy.finish_component)
