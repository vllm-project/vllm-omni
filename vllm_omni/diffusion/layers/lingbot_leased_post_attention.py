# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Default-off SP8 projection/post-attention graph with explicit output leases.

The reverse All-to-All producer writes ``lease.projection_input`` directly.
Only self.o, residual, camera affine and norm3 are captured. Graph outputs are
borrowed until the context exits; they must never enter a request/cache owner.
Conditioning staging is retained and charged. No shared format tags are changed.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


def _signature(value):
    return (tuple(value.shape), tuple(value.stride()), value.storage_offset(), str(value.dtype), str(value.device))


def _identity(values):
    return tuple((id(value), value.data_ptr(), value._version) for value in values)


def _parameters(block):
    return tuple(
        value for module in (block.self_attn.o, block.norm3) for value in (*module.parameters(), *module.buffers())
    )


def _callable_identity(value):
    # Accessing a bound method creates a new wrapper each time. Its function
    # and receiver identify the actual provider without rejecting that wrapper.
    return (id(getattr(value, "__self__", None)), id(getattr(value, "__func__", value)))


def _provider_owners(block):
    projection, norm = block.self_attn.o, block.norm3
    method = getattr(projection, "quant_method", None)
    return (
        compute,
        projection.forward,
        norm.forward,
        getattr(norm, "_forward_method", None),
        getattr(norm, "forward_native", None),
        getattr(norm, "forward_npu", None),
        getattr(method, "apply", None),
        getattr(method, "_gemm_impl", None),
        torch.nn.functional.linear,
        torch.nn.functional.layer_norm,
        getattr(projection, "_compiled_call_impl", None),
        getattr(norm, "_compiled_call_impl", None),
        *(
            value
            for module in (projection, norm)
            for name in ("_forward_pre_hooks", "_forward_hooks")
            for value in getattr(module, name).values()
        ),
    )


def _runtime_contract(block):
    """Values/providers baked into capture, separate from weight ownership."""
    projection, norm = block.self_attn.o, block.norm3
    method = getattr(projection, "quant_method", None)
    constants = tuple(
        (name, getattr(projection, name, None))
        for name in ("tp_size", "input_is_parallel", "reduce_results", "skip_bias_add", "return_bias")
    )
    constants += (
        ("norm.eps", norm.eps),
        ("norm.normalized_shape", tuple(norm.normalized_shape)),
        ("norm.elementwise_affine", norm.elementwise_affine),
    )
    providers = tuple(_callable_identity(value) for value in _provider_owners(block))
    # Forward hooks also alter the mathematical provider. Capturing after a
    # hook changes must not replay the old operation silently.
    hooks = tuple(
        tuple((key, _callable_identity(value)) for key, value in getattr(module, name).items())
        for module in (projection, norm)
        for name in ("_forward_pre_hooks", "_forward_hooks")
    )
    return (
        id(projection),
        id(norm),
        id(method),
        type(projection),
        type(norm),
        type(method),
        constants,
        providers,
        hooks,
    )


def compute(block, projection_input, hidden, gate, camera_scale, camera_shift):
    """Use the original producer and retain every native BF16 boundary."""
    projected = block.self_attn.o(projection_input)
    if not isinstance(projected, torch.Tensor):
        raise ValueError("Leased projection requires the native tensor return contract")
    frames = gate.shape[1]
    grid = hidden.unflatten(1, (frames, hidden.shape[1] // frames))
    projected_grid = projected.unflatten(1, (frames, hidden.shape[1] // frames))
    post_hidden = (grid + projected_grid * gate).flatten(1, 2).to(hidden.dtype)
    post_hidden = ((1 + camera_scale) * post_hidden + camera_shift).to(hidden.dtype)
    return projected, post_hidden, block.norm3(post_hidden)


class _NPURuntime:
    def eligible(self, inputs):
        return all(value.device.type == "npu" for value in inputs) and not torch.npu.is_current_stream_capturing()

    def current_stream(self, device):
        return torch.npu.current_stream(device)

    def empty(self, value):
        return torch.empty(value.shape, dtype=value.dtype, device=value.device)

    def record_owner(self, value, stream):
        value.record_stream(stream)

    def record_event(self, stream):
        event = torch.npu.Event()
        event.record(stream)
        return event

    def wait_event(self, stream, event):
        stream.wait_event(event)

    def capture(self, operation, inputs, stream):
        # Each slot has a separate pool. A pool shared by alternating graphs
        # could recycle an output while the other slot still has consumers.
        capture_stream = torch.npu.Stream(device=inputs[0].device)
        capture_stream.wait_stream(stream)
        with torch.npu.stream(capture_stream), torch.inference_mode():
            operation(*inputs)  # Prime native linear/norm outside capture.
        capture_stream.synchronize()
        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.inference_mode(), torch.npu.graph(graph, pool=pool, stream=capture_stream):
            outputs = operation(*inputs)
        capture_stream.synchronize()
        stream.wait_stream(capture_stream)
        # Keep the explicit capture stream and pool token with the graph.
        return graph, outputs, (pool, capture_stream)


@dataclass
class _Slot:
    inputs: tuple
    graph: object = None
    outputs: tuple | None = None
    graph_owners: object = None
    completion_events: list = field(default_factory=list)
    active: bool = False
    generation: int = 0


class Lease:
    """A short-lived producer/consumer capability, invalid after release."""

    def __init__(self, controller, slot, stream):
        self.controller, self.slot, self.stream = controller, slot, stream
        self.generation = slot.generation
        self.closed = False
        self.projection_written = False
        self.finished = False
        self.ready = None
        self.consumers = [stream]

    def _check(self):
        if self.closed or not self.slot.active or self.slot.generation != self.generation:
            raise RuntimeError("Leased graph output is outside its consumer lifetime")
        if self.controller.failed:
            raise RuntimeError("Leased graph is poisoned; restart all eight workers")

    @property
    def projection_input(self):
        self._check()
        if self.finished:
            raise RuntimeError("Projection producer cannot write after graph submission")
        return self.slot.inputs[0]

    def mark_projection_written(self):
        self._check()
        if self.controller.runtime.current_stream(self.slot.inputs[0].device) != self.stream:
            raise RuntimeError("Projection producer must use the acquired stream")
        if self.projection_written or self.finished:
            raise RuntimeError("Projection producer must submit exactly once per lease")
        self.projection_written = True
        self.controller.stats["producer_writes"] += 1

    def stage_projection(self, value):
        """Charged compatibility arm; model integration uses the direct producer."""
        target = self.projection_input
        if (tuple(value.shape), value.dtype, value.device) != (tuple(target.shape), target.dtype, target.device):
            raise ValueError("Projection staging shape/dtype/device differs")
        self.controller.runtime.record_owner(value, self.stream)
        target.copy_(value)
        self.controller.stats["projection_copy_bytes"] += value.numel() * value.element_size()
        self.mark_projection_written()

    def finish(self):
        self._check()
        if self.controller.runtime.current_stream(self.slot.inputs[0].device) != self.stream:
            raise RuntimeError("Leased graph must submit on its acquired producer stream")
        if not self.projection_written or self.finished:
            raise RuntimeError("Complete projection producer once before leased graph execution")
        controller, slot = self.controller, self.slot
        controller._check_owners()
        if slot.graph is None:
            slot.graph, slot.outputs, slot.graph_owners = controller.runtime.capture(
                lambda *args: compute(controller.block, *args), slot.inputs, self.stream
            )
            if len(slot.outputs) != 3 or any(
                tuple(value.shape) != tuple(slot.inputs[0].shape)
                or value.dtype != torch.bfloat16
                or value.device != slot.inputs[0].device
                for value in slot.outputs
            ):
                raise RuntimeError("Captured projection/post-attention output contract changed")
            input_storages = {value.untyped_storage().data_ptr() for value in slot.inputs}
            if any(value.untyped_storage().data_ptr() in input_storages for value in slot.outputs):
                raise RuntimeError("Graph output aliases a producer or conditioning owner")
            controller.stats["captures"] += 1
        # Capture defines persistent outputs; replay is required even for the
        # first lease to materialize this producer's values before consumption.
        slot.graph.replay()
        controller.stats["replays"] += 1
        self.finished = True
        for value in slot.outputs:
            controller.runtime.record_owner(value, self.stream)
        self.ready = controller.runtime.record_event(self.stream)
        return self.outputs

    @property
    def outputs(self):
        self._check()
        if not self.finished:
            raise RuntimeError("Graph outputs require completed producer submission")
        return self.slot.outputs

    def consume(self, stream=None):
        """Wait on this result only; add each stream that consumes a view."""
        self._check()
        if not self.finished:
            raise RuntimeError("Consumer must follow leased graph submission")
        stream = stream or self.controller.runtime.current_stream(self.slot.inputs[0].device)
        if all(stream != existing for existing in self.consumers):
            self.controller.runtime.wait_event(stream, self.ready)
            self.consumers.append(stream)
            for value in self.outputs:
                self.controller.runtime.record_owner(value, stream)
        return self.outputs


class LeasedPostAttention:
    def __init__(self, block, *, max_signatures=4, runtime=None, failure_reporter=None):
        if isinstance(max_signatures, bool) or not isinstance(max_signatures, int) or max_signatures < 1:
            raise ValueError("Leased graph signature limit must be a positive integer")
        self.block = block
        self.runtime = runtime or _NPURuntime()
        self.max_signatures = max_signatures
        self.owners = _parameters(block)
        self.identities = _identity(self.owners)
        self.provider_owners = (
            block.self_attn.o,
            block.norm3,
            getattr(block.self_attn.o, "quant_method", None),
            *_provider_owners(block),
        )
        self.runtime_contract = _runtime_contract(block)
        self.arenas = {}
        self.failed = False
        self.quarantine = []
        self.failure_reporter = failure_reporter
        self.lock = threading.RLock()
        self.stats = dict(
            captures=0,
            replays=0,
            producer_writes=0,
            projection_copy_bytes=0,
            conditioning_copy_bytes=0,
            output_clone_bytes=0,
            consumer_fences=0,
            slot_reuse_waits=0,
            fallbacks=0,
        )

    def _check_owners(self):
        if _identity(_parameters(self.block)) != self.identities:
            raise RuntimeError("Leased graph parameter owner changed; restart all eight workers")
        if _runtime_contract(self.block) != self.runtime_contract:
            raise RuntimeError("Leased graph runtime constant or provider changed; restart all eight workers")

    def eligible(self, hidden, gate, camera_scale, camera_shift):
        inputs = (hidden, gate, camera_scale, camera_shift)
        if not self.runtime.eligible(inputs):
            return False
        if hidden.dtype != torch.bfloat16 or hidden.ndim != 3 or hidden.shape[0] != 1:
            return False
        if any(value.device != hidden.device for value in inputs):
            return False
        frames = gate.shape[1] if gate.ndim == 4 else 0
        if (
            not frames
            or hidden.shape[1] % frames
            or tuple(gate.shape) != (1, frames, 1, hidden.shape[-1])
            or gate.dtype != torch.float32
        ):
            return False
        if (
            tuple(camera_scale.shape) != tuple(hidden.shape)
            or camera_shift.shape != camera_scale.shape
            or camera_scale.dtype != torch.bfloat16
            or camera_shift.dtype != torch.bfloat16
        ):
            return False
        return True

    def _fatal(self, error, lease, inputs):
        self.failed = True
        owners = (self, lease, inputs, self.owners, self.provider_owners)
        self.quarantine.append(owners)
        if self.failure_reporter is not None:
            self.failure_reporter("leased_post_attention", str(error), owners)
        else:
            from .lingbot_sp8_fatal import report_failure

            report_failure("leased_post_attention", str(error), owners)

    @contextmanager
    def acquire(self, hidden, gate, camera_scale, camera_shift):
        inputs = (hidden, gate, camera_scale, camera_shift)
        lease = None
        # Unsupported inputs select native before any slot write/capture.
        with self.lock:
            if self.failed:
                raise RuntimeError("Leased graph is poisoned; restart all eight workers")
            self._check_owners()
            key = tuple(_signature(value) for value in inputs)
            eligible = self.eligible(*inputs)
            if eligible and key not in self.arenas and len(self.arenas) >= self.max_signatures:
                eligible = False
            if eligible:
                if key not in self.arenas:
                    # Each slot owns projection input, all conditioning and its
                    # graph output arena. No request/cache tensor is borrowed.
                    template = (hidden, *inputs)
                    self.arenas[key] = tuple(
                        _Slot(tuple(self.runtime.empty(value) for value in template)) for _ in range(2)
                    )
                slots = self.arenas[key]
                free = [slot for slot in slots if not slot.active]
                if not free:
                    raise RuntimeError("Both leased graph slots still have consumers")
                slot = min(free, key=lambda value: value.generation)
                slot.active = True
                slot.generation += 1
                stream = self.runtime.current_stream(hidden.device)
                lease = Lease(self, slot, stream)
            else:
                self.stats["fallbacks"] += 1
        if lease is None:
            yield None
            return
        try:
            for event in slot.completion_events:
                self.runtime.wait_event(stream, event)
                self.stats["slot_reuse_waits"] += 1
            slot.completion_events = []
            for target, value in zip(slot.inputs[1:], inputs, strict=True):
                self.runtime.record_owner(value, stream)
                self.runtime.record_owner(target, stream)
                target.copy_(value)
                self.stats["conditioning_copy_bytes"] += value.numel() * value.element_size()
            self.runtime.record_owner(slot.inputs[0], stream)
            yield lease
            if not lease.finished:
                raise RuntimeError("Leased producer context exited without graph submission")
            # No device-wide or per-block synchronize. Consumers have enqueued
            # their final accesses before these events; reuse waits on them.
            slot.completion_events = [self.runtime.record_event(consumer) for consumer in lease.consumers]
            self.stats["consumer_fences"] += len(slot.completion_events)
            lease.closed = True
            with self.lock:
                slot.active = False
        except BaseException as error:
            self._fatal(error, lease, inputs)
            raise


def install(transformer, enabled=False, *, max_signatures=4):
    if not enabled:
        return
    for block in transformer.blocks:
        if hasattr(block, "_lingbot_leased_post_attention"):
            raise ValueError("Leased post-attention graph is already installed")
        projection = block.self_attn.o
        if (
            block.self_attn.ulysses_world_size != 8
            or getattr(projection, "tp_size", 1) != 1
            or getattr(projection, "_lingbot_quant_mode", None) is not None
            or getattr(projection, "return_bias", False)
            or getattr(projection, "skip_bias_add", False)
            or tuple(projection.weight.shape) != (5120, 5120)
            or projection.weight.dtype != torch.bfloat16
            or type(projection.quant_method).__name__ != "UnquantizedLinearMethod"
        ):
            raise ValueError("Leased post-attention requires SP8 TP1 native BF16 self.o")
        block._lingbot_leased_post_attention = LeasedPostAttention(block, max_signatures=max_signatures)


@contextmanager
def acquire(block, hidden, gate_msa, camera_scale, camera_shift):
    controller = getattr(block, "_lingbot_leased_post_attention", None)
    if controller is None or tuple(hidden.shape) != (1, 585, 5120):
        yield None
        return
    with controller.acquire(hidden, gate_msa, camera_scale, camera_shift) as lease:
        yield lease
