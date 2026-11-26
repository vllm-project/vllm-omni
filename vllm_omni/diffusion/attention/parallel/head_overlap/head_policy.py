# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Learn an uneven head split from native denoising layers and agree once per layout."""

import math
import statistics
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from weakref import WeakSet

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.cuda_graph import kwargs_signature

from ..ulysses_ce_overlap import CEOverlapUlysses, PreparationError, prepare_local
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


# Experiment defaults for a 32-layer denoiser: leave six visits untimed, then
# sample twelve position-balanced pairs. The 80% win rate and median/MAD
# margins prevent a single fast pair from enabling a split.
_UNTIMED = 6
_SAMPLES = 24

logger = init_logger(__name__)
_OWNERS = WeakSet()


def clear():
    for policy in list(_OWNERS):
        if policy.layouts:
            torch.accelerator.synchronize()
        policy.layouts.clear()
        policy.pending.clear()
        policy.component_layouts.clear()
        policy.routes.clear()
        policy.discovered.clear()
        policy.visits.clear()
        policy.slots.clear()


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
    signature: tuple = ()
    example: tuple = ()
    observations: dict = field(default_factory=dict)
    timers: list = field(default_factory=list)
    plan: CEOverlapUlysses | None = None
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
        self.discovered = {}
        self.enabled_for_call = False
        self.generation = 0
        self.routes = {}
        self.visits = {}
        self.slots = {}

    def begin_component(self, module, args, kwargs):
        # Component inputs precede sequence sharding. All SP peers enter a new
        # shape's discovery pass even if a local core is ineligible.
        self.component_key = kwargs_signature((args, kwargs))
        keys = self.component_layouts.get(self.component_key)
        frozen = _PREPARING_GRAPH.get() or torch.cuda.is_current_stream_capturing()
        discover = keys is None and not frozen and len(self.component_layouts) < 16
        self.discovered.clear()
        self.discovery = [] if discover else None
        self.routes = self.visits.get(self.component_key, {})
        self.enabled_for_call = discover or any(
            self.layouts[key].selected or (not frozen and key in self.pending) for key in keys or ()
        )

        self.slots = {}
        if self.pending and not frozen:
            for name, trial in self.routes.items():
                if trial.selected is None and trial.seen < _UNTIMED + _SAMPLES:
                    index = trial.seen - _UNTIMED
                    trial.seen += 1
                    if index >= 0:
                        self.slots[name] = index

    def try_run(self, strategy, layer, q, k, v, md):
        """Return None to execute the normal attention path, including native CE."""
        name = strategy.head_layer_name
        trial = self.routes.get(name)
        if self.discovery is None and (trial is None or trial.plan is None):
            return None
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
        joint = ()
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
        placement = md.joint_strategy if joint else None
        grid = md.video_layout.latent_grid if md is not None and md.video_layout else None
        if self.discovery is not None:
            signature = (
                world,
                tuple(tensor_spec(x) for x in (q, k, v)),
                tuple(tensor_spec(x) for x in joint),
                placement,
                grid,
                layer.softmax_scale,
                layer.borrow_ulysses_output,
            )
            trial = self.layouts.get(signature) or self.discovered.get(signature)
            if trial is None and len(self.layouts) + len(self.discovered) < 16:
                choices, tail = candidate(q.shape[2] // world, q.shape[1] * world, world, self.sms)
                trial = self.discovered[signature] = Trial(choices[-1], tail, signature)
                if len(choices) == 1:
                    trial.selected = False
                else:
                    trial.example = (q.device, strategy._ulysses_pg)
            if trial is not None:
                self.discovery.append((name, signature))
            return None
        # All admitted tensors have batch one, dense heads, BF16 and the same
        # device. Validate the variable geometry without rebuilding/hashing the
        # discovery signature or recursively walking the metadata every layer.
        _, input_specs, joint_specs, expected_placement, expected_grid, _, _ = trial.signature
        if (
            q.shape != input_specs[0][0]
            or placement != expected_placement
            or grid != expected_grid
            or any(x.stride(1) != spec[1][1] for x, spec in zip((q, k, v), input_specs))
            or any(x.shape != spec[0] or x.stride(1) != spec[1][1] for x, spec in zip(joint, joint_specs))
        ):
            return None

        kernel = layer._run_local_attention

        def split():
            metadata = replace(md, joint_query=None, joint_key=None, joint_value=None) if joint else md
            out = trial.plan.run(
                q, k, v, lambda a, b, c: kernel(a, b, c, metadata), joint_kv=joint[1:] if trial.plan.prefix else ()
            )
            return out if layer.borrow_ulysses_output else out.clone()

        if trial.selected:
            return split()
        index = self.slots.pop(name, None)
        if index is None:
            return None
        start, end = trial.timers[index]
        choice = candidate_index(index)
        start.record()
        if choice:
            result = split()
        else:
            qq, kk, vv, native_md, ctx = strategy.pre_attention(q, k, v, md)
            result = strategy.post_attention(kernel(qq, kk, vv, native_md), ctx)
        end.record()
        trial.observations[index] = start, end, not start.query()
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
            trials = self.layouts | self.discovered
            self.discovered.clear()
            info = [(name, key, trials[key].widths, trials[key].tail_rows) for name, key in order]
            peers = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(peers, info, group=self.group)
            agreed = all(peer == info for peer in peers)
            # Only uniquely identifiable layer visits get a sampling slot. All
            # ranks derive the entire next component schedule from this table.
            names = Counter(name for name, _ in order)
            visits = {name: key for name, key in order if agreed and names[name] == 1}
            self.visits[self.component_key] = {name: trials[key] for name, key in visits.items()}
            keys = self.component_layouts[self.component_key] = tuple(dict.fromkeys(visits.values()))
            for key in keys:
                trial = self.layouts.setdefault(key, trials[key])
                if not trial.example:
                    continue
                device, pg = trial.example
                trial.example = ()
                try:
                    trial.plan, trial.timers = prepare_local(lambda: self._prepare(key, trial, device, pg), self.group)
                except PreparationError as exc:
                    logger.warning("Keeping native Ulysses: %s", exc)
                    trial.selected = False
                    continue
                trial.plan.connect()
                self.pending.add(key)
            return
        self.resolve_pending(self.component_key)

    @staticmethod
    def _prepare(key, trial, device, pg):
        _, qkv, joint, placement, *_ = key
        _, rows, heads, dim = qkv[0][0]
        plan = CEOverlapUlysses(pg, device=device)
        plan._configure(
            rows,
            heads,
            dim,
            trial.widths,
            torch.bfloat16,
            prefix=joint[1][0][1] if joint else 0,
            front=placement == "front",
            tail=trial.tail_rows,
            defer_rendezvous=True,
        )
        timers = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(_SAMPLES)]
        for event in (*plan.ev_attn, *plan.ev_in_done.values(), *plan.ev_out_done.values(), plan.ev_inputs):
            event.record()
        for start, end in timers:
            start.record()
            end.record()
        return plan, timers

    def resolve_pending(self, component_key):
        """Read completed measurements without waiting on the GPU."""
        keys = [key for key in self.component_layouts.get(component_key, ()) if key in self.pending]
        if not keys:
            return
        reports = {}
        for key in keys:
            observations = self.layouts[key].observations
            if self.layouts[key].seen >= _UNTIMED + _SAMPLES and len(observations) < _SAMPLES:
                reports[key] = []  # A discovered layer changed layout or skipped a visit.
            elif len(observations) == _SAMPLES and all(end.query() for _, end, _ in observations.values()):
                reports[key] = [
                    (start.elapsed_time(end) * 1000, queued)
                    for start, end, queued in (observations[i] for i in range(_SAMPLES))
                ]
            else:
                reports[key] = None
        peers = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(peers, reports, group=self.group)
        for key in keys:
            trial = self.layouts[key]
            rows = [peer[key] for peer in peers]
            choices = [[], []]
            if not any(row == [] for row in rows):
                if any(row is None for row in rows):
                    continue
                # Every peer reports the same preassigned slot order.
                for i, values in enumerate(zip(*rows, strict=True)):
                    choices[candidate_index(i)].append(values)
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
        for layer_name, layer in module.named_modules():
            strategy = getattr(layer, "parallel_strategy", None)
            if getattr(strategy, "auto_head_overlap", False):
                strategy.head_layer_name = layer_name
                strategy.head_policy = policy
        module._ulysses_head_policy = policy
        module.register_forward_pre_hook(policy.begin_component, with_kwargs=True)
        module.register_forward_hook(policy.finish_component)
