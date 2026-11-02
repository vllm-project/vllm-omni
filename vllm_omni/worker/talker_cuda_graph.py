# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger

logger = init_logger(__name__)
NARROW_STATS: dict[str, int] = {}


def _key_fingerprint(state_key: Any) -> str:
    """Compact, log-safe rendering of the fused-graph cache key components."""
    if not isinstance(state_key, tuple):
        return str(state_key)[:120]
    parts = []
    for item in state_key:
        text = repr(item)
        parts.append(text if len(text) <= 60 else text[:57] + "...")
    return "|".join(parts)


_SEQ_BOUND_CEIL = 4096


def _graph_seq_bound(value: int) -> int:
    """Upper bound used for the captured attention geometry.

    FlashAttention only needs an upper bound, so per-token power-of-two buckets
    bought nothing but captures: one capture per (rows, bucket) pair, each
    150-700 ms while requests are already queued. One bound (the Talker's 4096
    context) leaves rows x 1 graphs, so a 64-request c4 run stops paying for
    the sweep.
    """
    return max(1 << max(0, int(value) - 1).bit_length(), _SEQ_BOUND_CEIL)


def _decline(reason: str) -> None:
    key = "decline:" + reason
    NARROW_STATS[key] = NARROW_STATS.get(key, 0) + 1
    if NARROW_STATS[key] == 1:
        logger.info("[kstep-cuda-narrow] declined reason=%s", reason)
    return None


@dataclass
class FlashAttentionKVTransaction:
    caches: dict[str, torch.Tensor]
    slots: dict[str, torch.Tensor]

    def snapshot(self) -> dict[str, torch.Tensor]:
        saved = {}
        for name, cache in self.caches.items():
            if cache.ndim != 4:
                raise ValueError("CUDA K-step KV transaction requires [blocks, heads, tokens, 2*head_dim]")
            slots = self.slots[name].reshape(-1).long()
            slots = torch.unique(slots[slots >= 0])
            if slots.numel() and bool((slots >= cache.shape[0] * cache.shape[2]).any()):
                raise ValueError("CUDA K-step KV slot exceeds the bound cache")
            saved[name + ":slots"] = slots
            saved[name] = cache[slots // cache.shape[2], :, slots % cache.shape[2], :].clone()
        return saved

    def restore(self, saved: dict[str, torch.Tensor]) -> None:
        for name, cache in self.caches.items():
            slots = saved[name + ":slots"]
            expected = (slots.numel(), cache.shape[1], cache.shape[3])
            if tuple(saved[name].shape) != expected:
                raise ValueError("CUDA K-step KV snapshot does not match the bound cache")
        for name, cache in self.caches.items():
            slots = saved[name + ":slots"]
            cache[slots // cache.shape[2], :, slots % cache.shape[2], :] = saved[name]


@dataclass
class _CapturedCall:
    graph: Any
    output: Any
    replays: int = 0
    keepalive: Any = None


@dataclass
class CUDANarrowStep:
    rows: int
    frames: int
    inputs_embeds: torch.Tensor
    positions: torch.Tensor
    step_slots: torch.Tensor
    step_positions: torch.Tensor
    step_seq_lens: torch.Tensor
    metadata: dict[str, Any]
    layer_steps: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    context: Any = None
    saved_metadata: Any = None
    saved_slots: Any = None
    saved_mode: Any = None
    spans: list[tuple[int, int]] = field(default_factory=list)
    saved_descriptor: Any = None
    saved_runtime_mode: Any = None
    graphs: OrderedDict[Hashable, _CapturedCall] = field(default_factory=OrderedDict)
    capture_failures: set = field(default_factory=set)
    capture_failure_reasons: dict = field(default_factory=dict)
    capture_failure_eager: int = 0
    frame_zero_prepared: bool = True
    eager_calls: int = 0
    captures: int = 0
    replays: int = 0

    def select(self, frame: int) -> None:
        if not 0 <= frame < self.frames:
            raise IndexError(f"frame {frame} outside [0, {self.frames})")
        self.positions.copy_(self.step_positions[:, frame])
        for name, metadata in self.metadata.items():
            slots, seq_lens = self.layer_steps.get(name, (self.step_slots, self.step_seq_lens))
            metadata.slot_mapping.copy_(slots[:, frame])
            metadata.seq_lens.copy_(seq_lens[:, frame])

    bind_frame = select

    def run_model(
        self,
        forward: Callable[[], Any],
        *,
        graph_key: Hashable = "forward",
        capture: bool = True,
        snapshot: Callable[[], Any] | None = None,
        restore: Callable[[Any], None] | None = None,
    ) -> Any:
        """Replay a raw, device-only forward over this object's persistent buffers.

        Host hooks, sampling and post-forward bookkeeping belong in the caller,
        outside this callback. The returned storage is overwritten by the next
        replay and must be cloned before retaining multiple frame outputs.
        """
        if not capture or self.inputs_embeds.device.type != "cuda" or os.environ.get("KSTEP_CUDA_GRAPH", "1") == "0":
            self.eager_calls += 1
            return forward()
        captured = self.graphs.get(graph_key)
        if captured is None:
            if snapshot is None or restore is None:
                raise RuntimeError("CUDA graph capture requires snapshot/restore of mutable state including KV slots")
            initial = snapshot()
            stream = torch.cuda.Stream(device=self.inputs_embeds.device)
            current = torch.cuda.current_stream(self.inputs_embeds.device)
            try:
                stream.wait_stream(current)
                try:
                    with torch.cuda.stream(stream):
                        forward()
                finally:
                    current.wait_stream(stream)
                restore(initial)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = forward()
            finally:
                restore(initial)
            captured = self.graphs[graph_key] = _CapturedCall(graph, output, keepalive=(forward, snapshot, restore))
            self.captures += 1
        self.graphs.move_to_end(graph_key)
        if len(self.graphs) > 4:
            self.graphs.popitem(last=False)
        captured.graph.replay()
        captured.replays += 1
        self.replays += 1
        if captured.replays == 1 or captured.replays % 200 == 0:
            logger.info(
                "[kstep-cuda-graph] rows=%d frames=%d captures=%d replays=%d eager=%d",
                self.rows,
                self.frames,
                self.captures,
                self.replays,
                self.eager_calls,
            )
        return captured.output

    def run_fused(
        self,
        forward_frames: Callable[[], Any],
        *,
        state_key: Hashable,
        snapshot: Callable[[], Any],
        restore: Callable[[Any], None],
        generators: tuple = (),
    ) -> Any:
        """Replay all K frames with device-only sampling and feedback.

        ``state_key`` must identify every captured state/output allocation and
        sampling branch. ``snapshot`` and ``restore`` must cover ALL mutable
        device state advanced by the callback (including RNG and EOS/history).
        They run outside capture; restore must copy into, not replace, storage.
        The callback must bind each frame, update embeddings, forward, sample,
        and store frame outputs without host reads or request-dictionary writes.
        """
        if self.inputs_embeds.device.type != "cuda" or os.environ.get("KSTEP_CUDA_GRAPH", "1") == "0":
            self.eager_calls += 1
            return forward_frames()
        key = ("fused", state_key)
        captured = self.graphs.get(key)
        if captured is None and key in self.capture_failures:
            # Failed captures are not retried (retrying re-runs the failing
            # work every step); the eager fallback is permanent, so log it.
            self.capture_failure_eager += 1
            if self.capture_failure_eager == 1 or self.capture_failure_eager % 200 == 0:
                logger.info(
                    "[kstep-cuda-graph] capture-failure fallback rows=%d frames=%d "
                    "failed_keys=%d eager_from_failure=%d eager=%d reason=%s",
                    self.rows,
                    self.frames,
                    len(self.capture_failures),
                    self.capture_failure_eager,
                    self.eager_calls,
                    self.capture_failure_reasons.get(key, "unknown"),
                )
            self.eager_calls += 1
            return forward_frames()
        if captured is None:
            capture_started = time.perf_counter()
            initial = snapshot()
            capture_error = None
            try:
                stream = torch.cuda.Stream(device=self.inputs_embeds.device)
                current = torch.cuda.current_stream(self.inputs_embeds.device)
                stream.wait_stream(current)
                try:
                    with torch.cuda.stream(stream):
                        forward_frames()
                finally:
                    current.wait_stream(stream)
            except Exception as exc:
                capture_error = exc
            finally:
                restore(initial)
            if capture_error is None:
                try:
                    graph = torch.cuda.CUDAGraph()
                    # Register generators before entering the capture window.
                    for generator in generators:
                        if generator is not None:
                            graph.register_generator_state(generator)
                    with torch.cuda.graph(graph):
                        output = forward_frames()
                except Exception as exc:
                    capture_error = exc
                finally:
                    restore(initial)
            if capture_error is not None:
                self.capture_failures.add(key)
                self.capture_failure_reasons[key] = f"{type(capture_error).__name__}: {capture_error}"
                logger.warning(
                    "[kstep-cuda-graph] capture failed rows=%d frames=%d (%s: %s); running eager frames for this shape",
                    self.rows,
                    self.frames,
                    type(capture_error).__name__,
                    capture_error,
                )
                self.eager_calls += 1
                return forward_frames()
            captured = self.graphs[key] = _CapturedCall(graph, output, keepalive=(forward_frames, snapshot, restore))
            self.captures += 1
            logger.info(
                "[kstep-cuda-graph] capture#%d rows=%d frames=%d cost_ms=%.1f cached=%d key=%s",
                self.captures,
                self.rows,
                self.frames,
                (time.perf_counter() - capture_started) * 1e3,
                len(self.graphs),
                _key_fingerprint(state_key),
            )
        self.graphs.move_to_end(key)
        if len(self.graphs) > 4:
            _, dropped = self.graphs.popitem(last=False)
            # Silent eviction turns into an invisible re-capture mid-run, which
            # looks like a slow step rather than a hit-rate collapse.
            logger.info(
                "[kstep-cuda-graph] evicted graph graph_replays=%d total_replays=%d cached=%d",
                dropped.replays,
                self.replays,
                len(self.graphs),
            )
        captured.graph.replay()
        captured.replays += 1
        self.replays += 1
        if captured.replays == 1 or captured.replays % 50 == 0:
            logger.info(
                "[kstep-cuda-graph] rows=%d frames=%d captures=%d replays=%d eager=%d",
                self.rows,
                self.frames,
                self.captures,
                self.replays,
                self.eager_calls,
            )
        return captured.output

    def restore(self) -> None:
        context = self.context
        if context is None:
            return
        saved_metadata = self.saved_metadata
        saved_slots = self.saved_slots
        saved_mode = self.saved_mode
        saved_descriptor = self.saved_descriptor
        try:
            context.attn_metadata = saved_metadata
            context.slot_mapping = saved_slots
            context.cudagraph_runtime_mode = saved_mode
            if hasattr(context, "batch_descriptor"):
                context.batch_descriptor = saved_descriptor
        finally:
            self.context = None
            self.saved_metadata = None
            self.saved_slots = None
            self.saved_mode = None
            self.saved_descriptor = None
            self.saved_runtime_mode = None


def _sampling_owner(prepare: Any, model: Any) -> Any:
    """Object that actually owns prepare_cuda_kstep (the Talker, not its wrapper).

    The omni wrapper only re-exposes the Talker's bound method, so the sampling
    generators and sampling config live on the method's ``__self__``.
    """
    owner = getattr(prepare, "__self__", None)
    return owner if owner is not None else model


def run_cuda_fused_step(
    narrow: CUDANarrowStep,
    *,
    model: Any,
    model_kwargs: dict[str, Any],
    intermediate_tensors: Any,
) -> tuple[Any, torch.Tensor] | None:
    if intermediate_tensors is not None:
        _decline("fused:model-interface:intermediate-tensors")
        return None
    # FULL cudagraph mode wraps the module in CUDAGraphWrapper (and friends),
    # which do not expose the Talker's k-step interfaces.
    unwrap = getattr(model, "unwrap", None)
    if callable(unwrap):
        model = unwrap()
    missing = [
        name
        for name in ("tts_model", "prepare_cuda_kstep", "cuda_kstep_frame", "commit_cuda_kstep")
        if not hasattr(model, name)
    ]
    if missing:
        _decline(f"fused:model-interface:missing={','.join(missing)}:type={type(model).__name__}")
        return None
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    # The decline checks are pure interface validation over engine-lifetime
    # state; caching them keeps the launch path sub-millisecond. One data_ptr
    # probe guards against a KV-pool re-allocation.
    meta_key = tuple(narrow.metadata)
    fast = getattr(model, "_cuda_fused_step_cache", None)
    # fast tuple: (meta_key, probe_tensor, probe_ptr_at_build, caches, fingerprint)
    if fast is not None and fast[0] == meta_key and fast[1].data_ptr() == fast[2]:
        caches, cache_fingerprint = fast[3], fast[4]
    else:
        caches = {}
        for name in narrow.metadata:
            layer = narrow.context.no_compile_layers.get(name)
            if layer is None or getattr(layer, "attn_backend", None) is not FlashAttentionBackend:
                backend = getattr(layer, "attn_backend", None) if layer is not None else None
                _decline(f"fused:layer-interface:{name}:backend={type(backend).__name__}")
                return None
            cache = getattr(layer, "kv_cache", None)
            if not isinstance(cache, torch.Tensor) or cache.ndim != 4 or not cache.is_cuda:
                shape = tuple(cache.shape) if isinstance(cache, torch.Tensor) else None
                _decline(
                    f"fused:kv-layout:{name}:ndim={getattr(cache, 'ndim', None)}"
                    f":cuda={getattr(cache, 'is_cuda', None)}:shape={shape}"
                )
                return None
            caches[name] = cache
        cache_fingerprint = tuple(
            (name, cache.data_ptr(), tuple(cache.shape), cache.stride()) for name, cache in caches.items()
        )
        probe = next(iter(caches.values()))
        # Guard against a KV-pool re-allocation between steps: one data_ptr
        # probe of the first layer's cache replaces re-walking every layer.
        model._cuda_fused_step_cache = (
            meta_key,
            probe,
            probe.data_ptr(),
            {k: v for k, v in caches.items()},
            cache_fingerprint,
        )
    kwargs = dict(model_kwargs)
    kwargs["request_token_spans"] = [(row, row + 1) for row in range(narrow.rows)]
    prepare = getattr(model, "prepare_cuda_kstep", None)
    if not callable(prepare):
        _decline("fused:model-interface:no-prepare")
        return None
    workspace = prepare(narrow.inputs_embeds, narrow.frames, **kwargs)
    if workspace is None:
        _decline("fused:kstep-workspace")
        return None
    # The sampling state is bound on the object that actually owns the method
    # (the Talker); the wrapper only delegates the callable. Reading it off the
    # wrapper would silently yield an empty generator tuple, so the RNG op
    # inside the capture would fail as unregistered.
    owner = _sampling_owner(prepare, model)
    initial = getattr(narrow, "_fused_initial_embeds", None)
    if initial is None:
        initial = narrow.inputs_embeds.clone()
        narrow._fused_initial_embeds = initial
    initial.copy_(narrow.inputs_embeds)
    transaction = FlashAttentionKVTransaction(caches, {name: slots for name, (slots, _) in narrow.layer_steps.items()})

    generators = tuple(g for g in getattr(owner, "_cuda_kstep_generators", ()) if g is not None)

    def snapshot():
        return (
            workspace.snapshot(),
            transaction.snapshot(),
            narrow.inputs_embeds.clone(),
            tuple(g.get_state() for g in generators),
        )

    def restore(saved):
        workspace.restore(saved[0])
        transaction.restore(saved[1])
        narrow.inputs_embeds.copy_(saved[2])
        for generator, state in zip(generators, saved[3]):
            generator.set_state(state)

    def forward_frames():
        narrow.inputs_embeds.copy_(initial)
        stops = []
        hidden_frames = []
        for frame in range(narrow.frames):
            narrow.select(frame)
            hidden = model.tts_model(
                input_ids=None,
                positions=narrow.positions,
                intermediate_tensors=None,
                inputs_embeds=narrow.inputs_embeds,
            )
            hidden_frames.append(hidden.clone())
            feedback, stop = model.cuda_kstep_frame(hidden, frame)
            narrow.inputs_embeds.copy_(feedback)
            stops.append(stop)
        return (
            torch.stack(hidden_frames, dim=1).reshape(-1, hidden.shape[-1]),
            torch.stack(stops, dim=1),
        )

    state_key = (
        id(model),
        workspace.storage_key(),
        initial.data_ptr(),
        tuple(id(g) for g in generators),
        getattr(owner, "_cuda_kstep_sampling", ()),
        cache_fingerprint,
    )
    hidden, stops = narrow.run_fused(
        forward_frames, state_key=state_key, snapshot=snapshot, restore=restore, generators=generators
    )
    output = model.commit_cuda_kstep(hidden.clone())
    return output, stops.clone()


def begin_cuda_narrow_step(
    *,
    forward_context: Any,
    frames: int,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    spans: list[tuple[int, int]],
    graph_owner: Any = None,
) -> CUDANarrowStep | None:
    if os.environ.get("KSTEP_CUDA_NARROW", "1") == "0":
        return _decline("disabled")
    rows = len(spans)
    metadata = forward_context.attn_metadata
    if not rows or frames < 1 or not isinstance(metadata, dict) or positions.ndim != 1:
        return _decline("layout")
    if spans != [(i * frames, (i + 1) * frames) for i in range(rows)]:
        return _decline("nonuniform-spans")
    if inputs_embeds is None or positions.numel() < rows * frames or inputs_embeds.shape[0] < rows * frames:
        return _decline("short-input")
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    if not metadata or any(not isinstance(m, FlashAttentionMetadata) for m in metadata.values()):
        return _decline("attention-backend")
    for group, m in metadata.items():
        if m.max_dcp_context_kv_len:
            return _decline(f"attention-layout:dcp:{group}")
        if getattr(m, "mm_prefix_query_range_tensor", getattr(m, "mm_prefix_range_tensor", None)) is not None:
            return _decline(f"attention-layout:mm-prefix:{group}")
        if m.rswa_prefix_lens is not None:
            return _decline(f"attention-layout:rswa:{group}")
        if getattr(m, "use_cascade", False):
            return _decline(f"attention-layout:cascade:{group}")
        if getattr(m, "local_attn_metadata", None) is not None:
            return _decline(f"attention-layout:local:{group}")
        if not isinstance(m.causal, bool) or not m.causal:
            return _decline(f"attention-layout:causal:{group}")
        if m.seq_lens.numel() < rows:
            return _decline(f"attention-layout:seq-len:{group}")
        if m.slot_mapping.numel() < rows * frames:
            return _decline(f"attention-layout:slots:{group}")
    source = next(iter(metadata.values()))
    # FlashAttention accepts an upper bound; a power-of-two bucket avoids one
    # graph per generated token while keeping launch geometry valid.
    max_seq_len = _graph_seq_bound(source.max_seq_len)
    key = (
        rows,
        frames,
        inputs_embeds.device,
        inputs_embeds.dtype,
        tuple(inputs_embeds.shape[1:]),
        positions.dtype,
        max_seq_len,
        tuple(
            (
                name,
                tuple(m.block_table.shape),
                m.block_table.dtype,
                m.seq_lens.dtype,
                m.slot_mapping.dtype,
                m.max_num_splits,
                _graph_seq_bound(m.max_seq_len),
                getattr(m, "use_cascade", False),
                getattr(m, "common_prefix_len", 0),
                repr(getattr(m, "local_attn_metadata", None)),
            )
            for name, m in metadata.items()
        ),
    )
    owner = graph_owner if graph_owner is not None else forward_context
    cache = getattr(owner, "_talker_cuda_narrow_cache", None)
    if cache is None:
        cache = OrderedDict()
        owner._talker_cuda_narrow_cache = cache
    narrow = cache.get(key)
    if narrow is None:
        narrow_metadata = {}
        query_start = torch.arange(rows + 1, dtype=torch.int32, device=positions.device)
        for name, original in metadata.items():
            item = copy.copy(original)
            item.num_actual_tokens = rows
            item.max_query_len = 1
            item.max_seq_len = _graph_seq_bound(original.max_seq_len)
            item.query_start_loc = query_start
            item.seq_lens = torch.empty_like(original.seq_lens[:rows])
            item.slot_mapping = torch.empty_like(original.slot_mapping[:rows])
            item.block_table = torch.empty_like(original.block_table[:rows])
            item.num_decode_reqs = rows
            item.num_decode_tokens = rows
            item.num_prefill_reqs = item.num_prefill_tokens = 0
            item.scheduler_metadata = None
            narrow_metadata[name] = item
        narrow = CUDANarrowStep(
            rows=rows,
            frames=frames,
            inputs_embeds=torch.empty_like(inputs_embeds[:rows]),
            positions=torch.empty_like(positions[:rows]),
            step_slots=torch.empty_like(source.slot_mapping[: rows * frames].reshape(rows, frames)),
            step_positions=torch.empty_like(positions[: rows * frames].reshape(rows, frames)),
            step_seq_lens=torch.empty((rows, frames), dtype=source.seq_lens.dtype, device=positions.device),
            metadata=narrow_metadata,
            spans=[(i, i + 1) for i in range(rows)],
        )
        first_name = next(iter(metadata))
        for name, original in metadata.items():
            narrow.layer_steps[name] = (
                narrow.step_slots
                if name == first_name
                else torch.empty_like(original.slot_mapping[: rows * frames].reshape(rows, frames)),
                narrow.step_seq_lens
                if name == first_name
                else torch.empty((rows, frames), dtype=original.seq_lens.dtype, device=original.seq_lens.device),
            )
        cache[key] = narrow
        # The key space is rows x seq-bound (up to ~8 x 3 here). A 16-entry cap
        # thrashed: an evicted narrow takes its captured graphs with it, so the
        # next step of that shape paid a 150-700 ms re-capture. 227 captures in
        # one c4/c8 run came from exactly that.
        if len(cache) > 64:
            # Drop a bucket that captured nothing first: evicting one that owns
            # graphs makes the next step of that shape re-capture silently.
            victim = next((name for name, item in cache.items() if not item.graphs), None)
            if victim is None:
                victim = next(iter(cache))
                logger.info(
                    "[kstep-cuda-narrow] evicting bucket with captures=%d replays=%d",
                    cache[victim].captures,
                    cache[victim].replays,
                )
            del cache[victim]
    else:
        cache.move_to_end(key)
    if narrow.context is not None:
        raise RuntimeError("CUDA narrow step must be restored before reusing its buffers")
    narrow.context = forward_context
    narrow.saved_metadata = metadata
    narrow.saved_slots = forward_context.slot_mapping
    narrow.saved_mode = forward_context.cudagraph_runtime_mode
    narrow.saved_descriptor = getattr(forward_context, "batch_descriptor", None)
    narrow.saved_runtime_mode = narrow.saved_mode
    try:
        narrow.step_slots.copy_(source.slot_mapping[: rows * frames].reshape(rows, frames))
        narrow.step_positions.copy_(positions[: rows * frames].reshape(rows, frames))
        narrow.step_seq_lens.copy_(
            source.seq_lens[:rows, None]
            - frames
            + torch.arange(
                1,
                frames + 1,
                dtype=source.seq_lens.dtype,
                device=positions.device,
            )
        )
        for name, item in narrow.metadata.items():
            original = metadata[name]
            slots, seq_lens = narrow.layer_steps[name]
            slots.copy_(original.slot_mapping[: rows * frames].reshape(rows, frames))
            seq_lens.copy_(
                original.seq_lens[:rows, None]
                - frames
                + torch.arange(1, frames + 1, dtype=seq_lens.dtype, device=seq_lens.device)
            )
            item.block_table.copy_(original.block_table[:rows])
        narrow.inputs_embeds.copy_(inputs_embeds[::frames][:rows])
        narrow.select(0)
        forward_context.attn_metadata = narrow.metadata
        forward_context.slot_mapping = {name: item.slot_mapping for name, item in narrow.metadata.items()}
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    except BaseException:
        narrow.restore()
        raise
    NARROW_STATS["engaged"] = NARROW_STATS.get("engaged", 0) + 1
    if NARROW_STATS["engaged"] == 1:
        logger.info(
            "[kstep-cuda-narrow] engaged rows=%d frames=%d device=%s; graph_replays=%d",
            rows,
            frames,
            inputs_embeds.device,
            narrow.replays,
        )
    return narrow
