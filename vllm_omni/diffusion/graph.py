# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device graph capture and replay for diffusion denoise steps.

A denoise loop calls the transformer once per step with *identical* tensor
shapes; only tensor contents change. After the first step the call is therefore
a fixed sequence of kernels over fixed buffers, which is exactly what a device
graph records: capture once, then replay and pay no per-kernel launch cost.

How much that is worth depends on how launch-bound the step already is. For a
HunyuanImage3 1024x1024 denoise (regional torch.compile + FLASH_ATTN, one XPU
card) the device is busy 82.6% of the denoise span, with 13.5% of the span lost
to 13.9k sub-millisecond gaps between kernels. So a graph is worth roughly 10%
here -- not the 3.8x that the launch-bound AR decode loop gets from the same
mechanism. Expect single digits to low teens.

CUDA spells graphs ``torch.cuda.CUDAGraph``, XPU spells them
``torch.xpu.XPUGraph``. vLLM's XPU model runner monkey-patches the former onto
the latter, but only in a process that constructs an AR/generation model runner
-- the diffusion worker is a separate process, so this module dispatches on the
device type rather than relying on that alias being installed.

Safety model: every tensor argument is copied into a capture-owned static
buffer before each replay, so the runner never depends on the caller handing
back the same tensor objects. Non-tensor arguments must compare equal to their
capture-time values; if any of them differs the call falls back to eager,
because the graph froze whatever branch they selected. Anything the callee
reads from outside its arguments (module attributes, forward context) is the
caller's responsibility -- see :meth:`DenoiseGraphRunner.__call__`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def write_through(dst: torch.Tensor | None, src: torch.Tensor) -> torch.Tensor:
    """Write ``src`` into ``dst``'s storage when they match, otherwise return ``src``.

    For state that a graph body reads off a module rather than off its arguments:
    rebinding such a tensor would leave the replay pointing at a freed
    allocation, so the buffer has to keep the same storage once its shape has
    settled. The caller must invalidate its graphs whenever this returns ``src``,
    because that is the case where the storage did move.
    """
    if dst is not None and dst.shape == src.shape and dst.dtype == src.dtype and dst.device == src.device:
        dst.copy_(src)
        return dst
    return src


def graphs_supported(device: torch.device) -> tuple[bool, str]:
    """Whether this device can capture graphs, and why not when it cannot."""
    if device.type == "cuda":
        return True, ""
    if device.type == "xpu":
        # torch.xpu.XPUGraph landed in torch 2.11.0.dev.
        if not hasattr(torch.xpu, "XPUGraph") or not hasattr(torch.xpu, "graph"):
            return False, f"torch {torch.__version__} has no torch.xpu.XPUGraph"
        return True, ""
    return False, f"device type {device.type!r} has no graph support"


def _new_graph(device: torch.device) -> Any:
    if device.type == "xpu":
        return torch.xpu.XPUGraph()
    return torch.cuda.CUDAGraph()


def _graph_context(graph: Any, device: torch.device, pool: Any) -> Any:
    if device.type == "xpu":
        return torch.xpu.graph(graph, pool=pool)
    return torch.cuda.graph(graph, pool=pool)


def _graph_pool_handle(device: torch.device) -> Any:
    """A shared memory pool, so N graphs do not reserve N private arenas."""
    try:
        if device.type == "xpu":
            return torch.xpu.graph_pool_handle()
        return torch.cuda.graph_pool_handle()
    except Exception:
        # A private pool per graph still works, it just costs more memory.
        return None


def _is_capturing(device: torch.device) -> bool:
    """True while a capture is already in flight; graphs must not nest."""
    mod = torch.xpu if device.type == "xpu" else torch.cuda
    fn = getattr(mod, "is_current_stream_capturing", None)
    return bool(fn()) if fn is not None else False


def is_capturing(device: torch.device) -> bool:
    """Whether a device graph is recording on ``device``'s current stream.

    For host-side checks that read a device tensor (``.item()``, ``bool()``):
    such a read synchronizes, which a capture rejects.
    """
    return device.type in ("cuda", "xpu") and _is_capturing(device)


def _synchronize(device: torch.device) -> None:
    torch.accelerator.synchronize(device)


def _structural_key(value: Any) -> Any:
    """A hashable summary of one argument: shape/dtype for tensors, value else.

    Two calls with the same key are replay-compatible *provided* the non-tensor
    parts also compare equal, which :meth:`DenoiseGraphRunner._constants_match`
    checks separately -- the key only has to be cheap and hashable.
    """
    if isinstance(value, torch.Tensor):
        return ("t", tuple(value.shape), value.dtype, value.device.type, value.device.index)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("v", value)
    if isinstance(value, Mapping):
        return ("m", tuple((k, _structural_key(v)) for k, v in sorted(value.items())))
    if isinstance(value, Sequence):
        return ("s", type(value).__name__, tuple(_structural_key(v) for v in value))
    # Opaque objects (caches, generators, modules). Keyed by identity, so a new
    # object is a cache miss rather than a silently wrong replay.
    return ("o", type(value).__name__, id(value))


def _clone_static(value: Any) -> Any:
    """Capture-owned copy of an argument, preserving container structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous()
    if isinstance(value, Mapping):
        return {k: _clone_static(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cloned = [_clone_static(v) for v in value]
        return type(value)(cloned) if isinstance(value, list) else tuple(cloned)
    return value


def _copy_into_static(static: Any, fresh: Any) -> None:
    """Refresh the capture-owned buffers in place from this call's arguments."""
    if isinstance(static, torch.Tensor):
        static.copy_(fresh, non_blocking=True)
        return
    if isinstance(static, Mapping):
        for k, v in static.items():
            _copy_into_static(v, fresh[k])
        return
    if isinstance(static, (list, tuple)):
        for v, f in zip(static, fresh):
            _copy_into_static(v, f)


def _non_tensor_parts(value: Any) -> Any:
    """Strip tensors out, leaving the parts that must compare equal."""
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, Mapping):
        return {k: _non_tensor_parts(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_non_tensor_parts(v) for v in value]
    return value


@dataclass
class _GraphEntry:
    graph: Any
    static_in: dict[str, Any]
    static_out: torch.Tensor
    constants: dict[str, Any]


@dataclass
class DenoiseGraphRunner:
    """Replay ``fn(**kwargs) -> Tensor`` from a per-shape device graph.

    ``fn`` must be a pure function of its tensor arguments: same shapes in, same
    kernels out, no host synchronisation (no ``.item()``, ``.cpu()``, data
    dependent control flow) and no reliance on state that changes between steps.
    A denoise step satisfies this once the first step is past, provided the
    caller keeps everything the callee reads off its arguments pinned to the
    same buffers for the life of the graph.

    Graphs are cached per structural key with a small LRU, so a new image size
    costs one extra eager step rather than a wrong result. Any capture failure
    disables the runner for the rest of the process and falls back to eager.
    """

    fn: Callable[..., torch.Tensor]
    name: str
    max_graphs: int = 2
    enabled: bool = True
    _cache: OrderedDict[Any, _GraphEntry] = field(default_factory=OrderedDict, repr=False)
    _stats: dict[str, int] = field(default_factory=lambda: dict.fromkeys(("calls", "hits", "captures", "eager"), 0))

    def stats_snapshot(self) -> dict[str, int]:
        return {**self._stats, "cache_size": len(self._cache)}

    def invalidate(self, reason: str) -> None:
        """Drop the captured graphs, but keep capturing new ones.

        For when a buffer the callee reads off its arguments got reallocated: the
        recorded kernels still point at the old addresses, so those graphs are
        dead, but a fresh capture against the new buffers is fine.
        """
        if self._cache:
            logger.debug("%s: dropping %d captured denoise graph(s) (%s)", self.name, len(self._cache), reason)
            self._cache.clear()

    def disable(self, reason: str) -> None:
        if self.enabled:
            logger.warning("%s: disabling denoise graphs (%s)", self.name, reason)
        self.enabled = False
        self._cache.clear()

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        self._stats["calls"] += 1
        device = _first_tensor_device(kwargs)
        if not self.enabled or device is None:
            if device is None:
                # Silent here would be indistinguishable from never being called.
                logger.warning_once("%s: no device tensor among the arguments; running eager", self.name)
            self._stats["eager"] += 1
            return self.fn(**kwargs)
        ok, reason = graphs_supported(device)
        if not ok:
            self.disable(reason)
            self._stats["eager"] += 1
            return self.fn(**kwargs)
        if _is_capturing(device):
            # Already inside someone else's capture; recording a nested graph is
            # not allowed, and the outer capture covers these kernels anyway.
            logger.warning_once("%s: an outer capture is in flight; running eager", self.name)
            self._stats["eager"] += 1
            return self.fn(**kwargs)

        key = tuple(sorted((k, _structural_key(v)) for k, v in kwargs.items()))
        entry = self._cache.get(key)
        if entry is not None and not self._constants_match(entry, kwargs):
            # Same shapes, different flags: the graph froze the other branch.
            logger.warning_once("%s: non-tensor arguments changed for a cached shape; running eager", self.name)
            del self._cache[key]
            entry = None
            self._stats["eager"] += 1
            return self.fn(**kwargs)

        if entry is None:
            entry = self._capture(kwargs, device)
            if entry is None:
                self._stats["eager"] += 1
                return self.fn(**kwargs)
            self._cache[key] = entry
            while len(self._cache) > self.max_graphs:
                self._cache.popitem(last=False)
            self._stats["captures"] += 1
            # Capture records kernels without running them, so the static output
            # holds nothing until the first replay.
            entry.graph.replay()
            return entry.static_out

        self._cache.move_to_end(key)
        self._stats["hits"] += 1
        for name, static in entry.static_in.items():
            _copy_into_static(static, kwargs[name])
        entry.graph.replay()
        # The caller gets the graph's own output buffer. It stays valid until the
        # next replay, and every consumer is enqueued on this stream before then.
        return entry.static_out

    def _constants_match(self, entry: _GraphEntry, kwargs: dict[str, Any]) -> bool:
        """Backstop for the structural key, which normally separates these already.

        The key carries plain values verbatim, so a changed flag is usually just a
        miss. What it cannot catch is an opaque argument keyed by ``id()`` that a
        freed-and-reallocated object now collides with; comparing by ``==`` does.
        """
        if set(kwargs) != set(entry.constants):
            return False
        return all(_non_tensor_parts(v) == entry.constants[k] for k, v in kwargs.items())

    def _capture(self, kwargs: dict[str, Any], device: torch.device) -> _GraphEntry | None:
        static_in = {k: _clone_static(v) for k, v in kwargs.items()}
        try:
            # Warm run outside capture: kernel autotuning, lazy module init and
            # any torch.compile guard/codegen must not happen under capture.
            with torch.no_grad():
                warm = self.fn(**static_in)
            if not isinstance(warm, torch.Tensor):
                self.disable(f"callee returned {type(warm).__name__}, expected a Tensor")
                return None
            _synchronize(device)
            graph = _new_graph(device)
            with torch.no_grad(), _graph_context(graph, device, _graph_pool_handle(device)):
                static_out = self.fn(**static_in)
        except Exception:
            logger.warning(
                "%s: graph capture failed, continuing eager for the rest of the process",
                self.name,
                exc_info=True,
            )
            self.enabled = False
            return None
        shapes = {k: tuple(v.shape) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        logger.info("%s: captured denoise graph, input shapes %s", self.name, shapes)
        return _GraphEntry(
            graph=graph,
            static_in=static_in,
            static_out=static_out,
            constants={k: _non_tensor_parts(v) for k, v in kwargs.items()},
        )


def _first_tensor_device(kwargs: Mapping[str, Any]) -> torch.device | None:
    for value in kwargs.values():
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, (list, tuple)):
            for inner in value:
                if isinstance(inner, torch.Tensor):
                    return inner.device
    return None


def build_denoise_graph_runner(
    od_config: Any,
    fn: Callable[..., torch.Tensor],
    name: str,
    *,
    refusal: str | None = None,
) -> DenoiseGraphRunner | None:
    """Return a runner if ``od_config`` opts in and nothing makes a step unsafe to replay.

    ``refusal`` lets the pipeline add its own reason to stay eager.
    """
    requested = bool(getattr(od_config, "diffusion_use_graph", False))
    # Stated either way: a silent None is indistinguishable from "engaged and did not pay".
    logger.info("%s: diffusion_use_graph=%s", name, requested)
    if not requested:
        return None
    cache_backend = getattr(od_config, "cache_backend", None) or "none"
    parallel_config = getattr(od_config, "parallel_config", None)
    if refusal is None and cache_backend != "none":
        refusal = f"cache_backend={cache_backend!r} decides per step whether the transformer runs at all"
    if refusal is None and (
        getattr(od_config, "enable_cpu_offload", False) or getattr(od_config, "enable_layerwise_offload", False)
    ):
        refusal = "CPU offload moves the weights that a captured graph reads"
    if refusal is None and getattr(parallel_config, "pipeline_parallel_size", 1) > 1:
        refusal = "pipeline parallelism hands intermediate tensors between stages"
    if refusal is not None:
        logger.warning("%s: diffusion_use_graph ignored because %s", name, refusal)
        return None
    return DenoiseGraphRunner(fn=fn, name=name)


def warn_if_denoise_graph_unsupported(od_config: Any, pipeline: Any) -> None:
    """Say so when ``diffusion_use_graph`` is set on a pipeline that never builds a runner.

    Pipelines opt in by assigning ``self._denoise_graph_runner`` (``None`` when refused, which
    :func:`build_denoise_graph_runner` already reports); anything else would ignore the flag silently.
    """
    if getattr(od_config, "diffusion_use_graph", False) and not hasattr(pipeline, "_denoise_graph_runner"):
        logger.warning(
            "diffusion_use_graph=True is ignored: %s does not support denoise graphs; running eager.",
            type(pipeline).__name__,
        )
