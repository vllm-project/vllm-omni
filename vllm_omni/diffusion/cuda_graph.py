# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph replay of fixed-shape keyword-only forwards.

Written for TaoMate-H3's audio teacher, which runs nine short, launch-bound
DiT forwards per five-second request; replaying a captured graph removes the
Python launch path. ``XPUGraphedForward`` (``xpu_graph.py``) derives from this
class and overrides only the device hooks below; pick the class for the
current platform with ``current_omni_platform.get_graphed_forward_cls()``.

``GraphedForward`` wraps a keyword-only ``forward``. Calls are keyed
by the *structure* of their kwargs (tensor shapes/dtypes and every non-tensor
value); the first call of a key captures a graph on static copies of the
tensors and later calls copy the live tensors into those buffers and replay.
Values that only change per request (prompt embeddings, RoPE table, row
positions) are therefore ordinary inputs and never force a re-capture; only
a new document *shape* does (a prompt with another token count, the first
request's longer audio, a reference tail present or not).

Graph capture forbids host synchronization inside the captured region, so the
wrapper (a) enters the caller's ``capture_context`` around the capture
forwards (TaoMate uses it to bypass its host-keyed AdaLN projection cache),
and (b) validates the very first capture with
``torch.cuda.set_sync_debug_mode("error")`` before capturing, so a
synchronizing operator fails loudly on every rank at the same program point
instead of leaving peers waiting in a collective. Any failure disables the
wrapper for the rest of the process (agreed across the sequence-parallel
group when one is given) and the caller falls back to eager execution.
"""

from __future__ import annotations

import dataclasses
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_PRIMITIVES = (int, float, bool, str, bytes, type(None))


def kwargs_signature(value: Any) -> Any:
    """A hashable description of ``value`` in which tensors count by shape and dtype only."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), value.device.type)
    if isinstance(value, dict):
        return ("dict", tuple((str(key), kwargs_signature(value[key])) for key in sorted(value, key=str)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(kwargs_signature(item) for item in value))
    if isinstance(value, _PRIMITIVES):
        return ("value", type(value).__name__, value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple((field.name, kwargs_signature(getattr(value, field.name))) for field in dataclasses.fields(value)),
        )
    return ("repr", type(value).__qualname__, repr(value))


def _holds_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_holds_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_tensor(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(_holds_tensor(getattr(value, field.name)) for field in dataclasses.fields(value))
    return False


def clone_static(value: Any) -> Any:
    """Deep-copy the tensors of a kwargs tree into fresh contiguous buffers.

    Tensors are reached through dicts, lists and tuples only. Any other object
    (a layout dataclass, a device, a dtype) is passed through unchanged and
    must not hold a tensor: it would keep its capture-time address without
    being refreshed on replay, so such objects are rejected.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict):
        return {key: clone_static(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_static(item) for item in value)
    if isinstance(value, list):
        return [clone_static(item) for item in value]
    if isinstance(value, (*_PRIMITIVES, torch.device, torch.dtype, torch.Size)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and not _holds_tensor(value):
        return value
    raise TypeError(
        "graph kwargs may hold tensors inside dicts, lists and tuples, primitives, torch.device/dtype/Size and "
        f"tensor-free dataclasses only, got {type(value)!r}"
    )


def copy_into(static: Any, current: Any) -> int:
    """Copy the tensors of ``current`` into the matching buffers of ``static``.

    Returns the number of tensors copied. The trees must share one signature.
    """
    if isinstance(static, torch.Tensor):
        if not isinstance(current, torch.Tensor) or static.shape != current.shape or static.dtype != current.dtype:
            raise ValueError("graph input differs in shape or dtype from the captured buffer")
        if static.data_ptr() != current.data_ptr() or static.stride() != current.stride():
            static.copy_(current)
        return 1
    if isinstance(static, dict):
        if not isinstance(current, dict) or static.keys() != current.keys():
            raise ValueError("graph input keys differ from the captured kwargs")
        return sum(copy_into(static[key], current[key]) for key in static)
    if isinstance(static, (list, tuple)):
        if not isinstance(current, (list, tuple)) or len(static) != len(current):
            raise ValueError("graph input sequence differs from the captured kwargs")
        return sum(copy_into(s, c) for s, c in zip(static, current, strict=True))
    return 0


@dataclass
class _Entry:
    graph: Any
    static_kwargs: dict[str, Any]
    outputs: Any
    # Objects whose tensors the captured kernels read by address although
    # they are not among the kwargs (for example an embedding plan installed
    # through a context variable). The entry owns a reference so the memory
    # cannot be freed and reused while the graph is resident.
    keep_alive: tuple[Any, ...] = ()
    # Pinned entries (captured up front for a configured shape range) are
    # never evicted by later, unpinned shapes.
    pinned: bool = False


@contextmanager
def _sync_debug_mode(mode: str) -> Iterator[None]:
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(mode)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


class GraphedForward:
    """Capture-and-replay wrapper around a keyword-only ``nn.Module`` forward.

    The device-specific steps are the hooks ``_graphs_supported``,
    ``_warmup_stream`` and ``_capture_graph`` (plus ``_sync_probe``); a
    platform subclass overrides those and inherits keying, static buffers,
    replay, LRU eviction and rank agreement.
    """

    # Used in log messages.
    graph_kind = "CUDA"
    # Validate the first capture with torch.cuda.set_sync_debug_mode("error").
    _sync_probe = True

    def __init__(
        self,
        module: nn.Module,
        *,
        device: torch.device,
        max_entries: int = 16,
        warmup_iters: int = 2,
        group: torch.distributed.ProcessGroup | None = None,
        name: str = "taomate_h3",
        enabled: bool = True,
        capture_context: Callable[[], AbstractContextManager] | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.module = module
        self.device = torch.device(device)
        self.max_entries = int(max_entries)
        self.warmup_iters = int(warmup_iters)
        self.group = group
        self.name = name
        # Entered around the probe, warm-up and capture forwards.
        self.capture_context = capture_context or nullcontext
        self.enabled = bool(enabled) and self._graphs_supported()
        self._entries: OrderedDict[Any, _Entry] = OrderedDict()
        self._pool: Any = None
        self._probe_pending = self._sync_probe
        self._pinned_overflow_warned = False
        self.captures = 0
        self.replays = 0
        self.evictions = 0
        self.eager_calls = 0

    # -- public --------------------------------------------------------------

    def __call__(
        self, *, variant: Any = None, keep_alive: tuple[Any, ...] = (), pin: bool = False, **kwargs: Any
    ) -> Any:
        """Run ``module(**kwargs)``, by replay when a graph of this shape exists.

        ``variant`` distinguishes module states the kwargs do not show (for
        example whether the LoRA hooks are enabled, or which embedding plan is
        installed); ``keep_alive`` holds every object outside ``kwargs`` whose
        tensors the forward reads, so a resident graph keeps them allocated;
        ``pin`` exempts a newly captured graph from LRU eviction. Returned
        tensors are clones, so the caller may hold them across the next call.
        """
        if not self.enabled or current_omni_platform.is_graph_capturing():
            # Inside an outer capture, which records these kernels anyway.
            self.eager_calls += 1
            return self.module(**kwargs)
        key = (kwargs_signature(variant), kwargs_signature(kwargs))
        entry = self._entries.get(key)
        if entry is None:
            entry = self._capture(key, kwargs, keep_alive, variant, pin=pin)
            if entry is None:
                self.eager_calls += 1
                return self.module(**kwargs)
        else:
            entry.pinned = entry.pinned or pin
            copy_into(entry.static_kwargs, kwargs)
            self._entries.move_to_end(key)
        entry.graph.replay()
        self.replays += 1
        return _clone_outputs(entry.outputs)

    @property
    def num_graphs(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        return {
            "graphs": len(self._entries),
            "pinned": sum(1 for entry in self._entries.values() if entry.pinned),
            "captures": self.captures,
            "replays": self.replays,
            "evictions": self.evictions,
            "eager_calls": self.eager_calls,
        }

    def clear(self) -> None:
        """Release captured graphs before their external weight buffers are freed."""
        self._entries.clear()

    def disable(self, reason: str) -> None:
        if self.enabled:
            logger.warning("%s %s graphs disabled: %s", self.name, self.graph_kind, reason)
        self.enabled = False
        self.clear()

    # -- capture -------------------------------------------------------------

    def _capture(
        self,
        key: Any,
        kwargs: dict[str, Any],
        keep_alive: tuple[Any, ...] = (),
        variant: Any = None,
        *,
        pin: bool = False,
    ) -> _Entry | None:
        failure: str | None = None
        static: dict[str, Any] = {}
        graph: Any = None
        outputs: Any = None
        try:
            static = clone_static(kwargs)
            with self.capture_context():
                if self._probe_pending:
                    # A synchronizing operator raises here on every rank at
                    # the same point, before any capture starts.
                    with _sync_debug_mode("error"):
                        self.module(**static)
                    self._probe_pending = False
                # The first capture warms lazy kernel/library state for the
                # whole process; later shapes need a single warm forward.
                warmups = self.warmup_iters if self.captures == 0 else min(self.warmup_iters, 1)
                with self._warmup_stream():
                    for _ in range(warmups):
                        self.module(**static)
                graph, outputs = self._capture_graph(lambda: self.module(**static))
        except Exception as exc:  # noqa: BLE001 - any capture failure means eager from now on
            failure = f"{type(exc).__name__}: {exc}"
            logger.warning("%s %s graph capture failed:\n%s", self.name, self.graph_kind, traceback.format_exc())
        ok = self._agree(failure is None)
        if not ok:
            del graph, outputs, static
            self.disable(failure or "a peer rank failed to capture")
            return None
        if len(self._entries) >= self.max_entries:
            victim = next((k for k, e in self._entries.items() if not e.pinned), None)
            if victim is None:
                if not self._pinned_overflow_warned:
                    self._pinned_overflow_warned = True
                    logger.warning(
                        "%s: all %d resident %s graphs are pinned; growing past max_entries",
                        self.name,
                        len(self._entries),
                        self.graph_kind,
                    )
            else:
                del self._entries[victim]
                self.evictions += 1
        entry = _Entry(graph=graph, static_kwargs=static, outputs=outputs, keep_alive=tuple(keep_alive), pinned=pin)
        self._entries[key] = entry
        self.captures += 1
        logger.info(
            "%s %s graph captured for %s (%d resident)", self.name, self.graph_kind, variant, len(self._entries)
        )
        return entry

    # -- device hooks --------------------------------------------------------

    def _graphs_supported(self) -> bool:
        return self.device.type == "cuda"

    @contextmanager
    def _warmup_stream(self) -> Iterator[None]:
        """Run the warm-up forwards on a side stream, ordered after the current work."""
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            yield
        torch.cuda.current_stream(self.device).wait_stream(side)

    def _capture_graph(self, body: Callable[[], Any]) -> tuple[Any, Any]:
        """Capture ``body`` into a graph; return the graph and the outputs it writes."""
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            outputs = body()
        if self._pool is None:
            self._pool = graph.pool()
        return graph, outputs

    def _agree(self, ok: bool) -> bool:
        group = self.group
        if group is None or torch.distributed.get_world_size(group) <= 1:
            return ok
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=self.device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
        return bool(int(flag.item()))


def _clone_outputs(outputs: Any) -> Any:
    if isinstance(outputs, torch.Tensor):
        return outputs.clone()
    if isinstance(outputs, tuple):
        return tuple(_clone_outputs(item) for item in outputs)
    if isinstance(outputs, list):
        return [_clone_outputs(item) for item in outputs]
    if isinstance(outputs, dict):
        return {key: _clone_outputs(item) for key, item in outputs.items()}
    return outputs


__all__ = ["GraphedForward", "clone_static", "copy_into", "kwargs_signature"]
