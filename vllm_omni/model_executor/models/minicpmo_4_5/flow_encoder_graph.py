# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unified Code2Wav encoder graphs: shared CUDA/NPU arena and exact-shape fallback.

The CosyVoice2 upsample-conformer encoder (embedding, ``forward_chunk``,
``encoder_proj``) is ~320 small kernels whose launches, not their GPU time, set
a serving step's latency; a graph replays them with one launch. One graph per
exact ``(rows, tokens, conformer cache frames)``, replayed only for calls of
exactly that shape, so it runs the eager kernels and changes no output. Nothing
is padded: rounding 3 rows up to a 4-row graph changes the M of every GEMM, and
with it cuBLAS's kernel and FP32 rounding, which the TF32 CFM and HiFT amplified
to 0.81 dB. The streaming trim caps the cache at ``prompt + 100`` frames, so one
prompt's streams only pass through the lengths ``reachable_conformer_cache_frames``
lists. All graphs take prefix views of one storage per static tensor and share
one private pool; a replay's results stay valid until the next replay.
"""

from __future__ import annotations

import math
import weakref
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from vllm.logger import init_logger

from .cuda_graph_wrapper import _graph_api, _graph_execution_context
from .encoder_graph import NPUEncoderGraphRunners

logger = init_logger(__name__)


def reachable_conformer_cache_frames(*, start: int, prompt_len: int, suffix: int, growths: Iterable[int]) -> list[int]:
    """Conformer attention-cache lengths a stream passes through from ``start``, for chunks of any mix of ``growths``.

    Past ``prompt_len + suffix`` frames the streaming trim keeps that many (``_trim_streaming_cache``).
    """
    steady = int(prompt_len) + int(suffix)
    steps = sorted({int(g) for g in growths if int(g) > 0})
    seen, frontier = {int(start)}, [int(start)]
    while frontier:
        frames = frontier.pop()
        for following in (min(frames + growth, steady) for growth in steps):
            if following not in seen:
                seen.add(following)
                frontier.append(following)
    return sorted(seen)


def _precision_state() -> tuple[object, ...]:
    """The matmul/conv precision flags a captured graph's kernels were chosen under."""
    return (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        torch.get_float32_matmul_precision(),
    )


def _new_capture_stream(device):
    # torch.cuda.Stream() draws from a round-robin pool: retaining the Python
    # object does not prevent another graph from capturing on the same stream.
    # PyTorch reset() clears cuBLAS workspaces for the entire capture stream,
    # including workspaces still referenced by other graphs. Own a CUDA stream
    # outside that pool so retiring one slot cannot invalidate a live peer.
    from cuda.bindings import runtime

    from vllm_omni.platforms import current_omni_platform

    previous_device = current_omni_platform.current_device()
    current_omni_platform.set_device(device)
    try:
        error, handle = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot create Code2Wav capture stream: {error}")
        stream = torch.cuda.ExternalStream(int(handle), device=device)
    finally:
        current_omni_platform.set_device(torch.device(device.type, previous_device))
    weakref.finalize(stream, runtime.cudaStreamDestroy, handle)
    return stream


class FlowEncoderGraphs:
    """Shared-arena continuation graphs and exact-shape CUDA/NPU fallback.

    ``encode_fn(tokens, cnn_cache=, att_cache=)`` supplies the continuation
    body. ``chunk_forward`` additionally accepts ``last_chunk`` for generic
    calls. Startup capture freezes admission; unseen shapes then run eager.

    Caches use the encoder layouts: ``cnn_cache`` ``(rows, channels, width)``,
    ``att_cache`` ``(depth, rows, heads, frames, 2 * head_dim)``; a chunk of
    ``tokens`` tokens emits ``upsample * (tokens - lookahead)`` frames.
    ``held_tensors`` returns tensors the graphs read in place (the RelPos
    tables): kept alive, and a replaced one disables the graphs.
    """

    def __init__(
        self,
        encode_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        rows: Iterable[int] = (),
        token_widths: Iterable[int] = (),
        lookahead: int = 0,
        upsample: int = 2,
        held_tensors: Callable[[], Sequence[torch.Tensor]] = tuple,
        max_graphs: int = 8,
        capture_after: int = 2,
        eager_min_batch: int | None = None,
        chunk_forward: Callable | None = None,
    ) -> None:
        self.encode_fn = encode_fn
        self.rows = tuple(sorted({int(r) for r in rows if int(r) > 0}))
        self.lookahead, self.upsample = int(lookahead), int(upsample)
        self.token_widths = tuple(sorted({int(w) for w in token_widths if int(w) > self.lookahead}))
        self._held_tensors = held_tensors
        self._held: tuple[torch.Tensor, ...] = ()
        self._precision: tuple[Any, ...] | None = None
        # (rows, tokens, cache frames) -> (graph, static views by name)
        self.graphs: dict[tuple[int, int, int], tuple[Any, dict[str, torch.Tensor]]] = {}
        self.replays = 0
        self.enabled = True
        self._storage: dict[str, torch.Tensor] = {}
        self._pool: Any = None
        self._capture_stream = None
        self._caller_stream = None
        self._amp = None
        self.forward = chunk_forward or encode_fn
        self.max_graphs = int(max_graphs)
        if self.max_graphs < 0:
            raise ValueError("encoder graph capacity must be >= 0")
        self.capture_after = int(capture_after)
        if self.capture_after < 2:
            raise ValueError("encoder graph capture_after must be >= 2")
        self.eager_min_batch = None if eager_min_batch is None else int(eager_min_batch)
        if self.eager_min_batch is not None and self.eager_min_batch < 1:
            raise ValueError("encoder graph eager_min_batch must be >= 1")
        # Startup precapture turns this off so serving only replays those graphs.
        self.capture_on_request = True
        self.exact_graphs = {}
        self.seen = OrderedDict()
        # Exact-shape fallback entries own their pools and capture streams.
        self._slots = []
        self._failed = False
        self.stats = Counter()
        # NPU replay state. CUDA fallback entries live in ``exact_graphs``.
        self._npu_runners = NPUEncoderGraphRunners(
            max_graphs=self.max_graphs,
            component_name="MiniCPM-o Code2Wav encoder",
            disable_config_hint="set enable_code2wav_encoder_graph=false in the stage connector extra",
        )
        self._npu_pe = {}

    def output_frames(self, tokens: int) -> int:
        return self.upsample * (int(tokens) - self.lookahead)

    def keys_for(self, *, start: int, prompt_len: int, suffix: int) -> list[tuple[int, int, int]]:
        """``(rows, tokens, cache frames)`` of every continuation chunk a prompt's streams can produce."""
        growths = [self.output_frames(width) for width in self.token_widths]
        frames = reachable_conformer_cache_frames(start=start, prompt_len=prompt_len, suffix=suffix, growths=growths)
        return [(rows, width, length) for rows in self.rows for width in self.token_widths for length in frames]

    def _body(self, views: dict[str, torch.Tensor]) -> None:
        outputs = self.encode_fn(views["tokens"], cnn_cache=views["cnn"], att_cache=views["att"])
        for name, value in zip(("hidden", "new_cnn", "new_att"), outputs, strict=True):
            target = views[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError(
                    f"flow encoder {name} output {tuple(value.shape)}/{value.dtype} does not match "
                    f"the captured layout {tuple(target.shape)}/{target.dtype}"
                )
            target.copy_(value)

    def _record(self, views: dict[str, torch.Tensor]) -> Any:
        """Warm up on a side stream, then capture into the shared private pool."""
        device = views["att"].device
        api = _graph_api(device)
        current = api.current_stream(device)
        if self._capture_stream is None:
            self._capture_stream = (
                api.Stream(device=device)
                if device.type == "npu" or torch.version.hip is not None
                else _new_capture_stream(device)
            )
        side = self._capture_stream
        side.wait_stream(current)
        amp, dtype = torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type)
        with (
            api.stream(side),
            _graph_execution_context(device),
            torch.autocast(device.type, enabled=amp, dtype=dtype, cache_enabled=False),
        ):
            for _ in range(2):
                self._body(views)
        current.wait_stream(side)
        api.synchronize(device)
        if self._pool is None:
            self._pool = api.graph_pool_handle()
        graph = api.NPUGraph() if device.type == "npu" else api.CUDAGraph()
        with (
            _graph_execution_context(device),
            torch.autocast(device.type, enabled=amp, dtype=dtype, cache_enabled=False),
            api.graph(graph, pool=self._pool, stream=side),
        ):
            self._body(views)
        current.wait_stream(side)
        return graph

    def capture(
        self,
        keys: Iterable[tuple[int, int, int]],
        *,
        cnn_shape: tuple[int, int],
        att_layout: tuple[int, int, int],
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        cnn_dtype: torch.dtype | None = None,
    ) -> int:
        """Capture ``keys``, all or none; ``cnn_shape`` ``(channels, width)``, ``att_layout`` ``(depth, heads, 2D)``."""
        if self.graphs:
            raise RuntimeError("flow encoder graphs are already captured")
        depth, heads, att_width = att_layout

        def shapes(key: tuple[int, int, int]) -> dict[str, tuple[int, ...]]:
            rows, width, frames = key
            out = self.output_frames(width)
            return {
                "tokens": (rows, width),
                "cnn": (rows, *cnn_shape),
                "att": (depth, rows, heads, frames, att_width),
                "hidden": (rows, out, hidden_dim),
                "new_cnn": (rows, *cnn_shape),
                "new_att": (depth, rows, heads, frames + out, att_width),
            }

        # Largest first, so the private pool grows once and later graphs reuse it.
        keys = sorted(
            {(int(r), int(w), int(f)) for r, w, f in keys},
            key=lambda key: (key[0] * (key[2] + self.output_frames(key[1])), key[0] * key[1]),
            reverse=True,
        )
        if not keys:
            return 0
        layouts = [shapes(key) for key in keys]
        # Autocast Conformer caches are mixed: CNN state can remain FP32
        # while attention state and projected hidden states are FP16.
        # Probe output metadata eagerly before recording any graph.
        dtypes = {"tokens": torch.long, "cnn": cnn_dtype or dtype, "att": dtype}
        with torch.no_grad(), _graph_execution_context(device):
            probe = {name: torch.zeros(layouts[0][name], dtype=value, device=device) for name, value in dtypes.items()}
            outputs = self.encode_fn(probe["tokens"], cnn_cache=probe["cnn"], att_cache=probe["att"])
            dtypes.update(
                (name, value.dtype) for name, value in zip(("hidden", "new_cnn", "new_att"), outputs, strict=True)
            )
            del outputs, probe
        # Ordinary tensors, not inference tensors: replays write them in and outside inference mode.
        with torch.inference_mode(False):
            for name in layouts[0]:
                numel = max(math.prod(layout[name]) for layout in layouts)
                self._storage[name] = torch.zeros(numel, dtype=dtypes[name], device=device)
        self._precision = _precision_state()
        if device.type in {"cuda", "npu"}:
            self._caller_stream = _graph_api(device).current_stream(device)
            self._amp = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
        try:
            for key, layout in zip(keys, layouts, strict=True):
                with torch.inference_mode(False):
                    views = {name: self._storage[name][: math.prod(s)].view(s) for name, s in layout.items()}
                self.graphs[key] = (self._record(views), views)
        except Exception:
            # A partial set would replay some shapes and not others.
            self.graphs.clear()
            self._storage.clear()
            self._pool = None
            self.enabled = False
            self._failed = True
            raise
        self._held = tuple(self._held_tensors())
        return len(self.graphs)

    def storage_bytes(self) -> int:
        """Bytes of the shared static inputs and results (not the private pool)."""
        return sum(t.numel() * t.element_size() for t in self._storage.values())

    def run(
        self,
        tokens: torch.Tensor,
        cnn_rows: Sequence[torch.Tensor],
        att_rows: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The continuation chunk of these streams (one cache row each) from its graph, or ``None`` (run eager).

        Returns the shared result views: copy what must outlive the next replay.
        """
        if self._failed:
            raise RuntimeError("Code2Wav encoder graph capture failed; restart the stage before retrying")
        if tokens.device.type in {"cuda", "npu"} and self._caller_stream is not None:
            stream = _graph_api(tokens.device).current_stream(tokens.device)
            stream_attr = "npu_stream" if tokens.device.type == "npu" else "cuda_stream"
            if getattr(stream, stream_attr) != getattr(self._caller_stream, stream_attr):
                return None
            if (
                torch.is_autocast_enabled(tokens.device.type),
                torch.get_autocast_dtype(tokens.device.type),
            ) != self._amp:
                return None
        rows = len(att_rows)
        # The batch cutoff applies to opportunistic exact-shape captures.
        # Explicitly precaptured shared-arena shapes remain eligible: they
        # avoid the separate stacked cache and per-call output clones.
        if not (self.enabled and rows and tokens.dim() == 2 and len(tokens) == len(cnn_rows) == rows):
            return None
        key = (rows, int(tokens.shape[1]), int(att_rows[0].shape[3]) if att_rows[0].dim() == 5 else -1)
        if (entry := self.graphs.get(key)) is None:
            return None
        graph, views = entry
        static_att, static_cnn = views["att"], views["cnn"]
        att_shape, cnn_shape = (static_att.shape[0], 1, *static_att.shape[2:]), (1, *static_cnn.shape[1:])
        if (tokens.dtype, tokens.device) != (views["tokens"].dtype, views["tokens"].device) or not all(
            (att.shape, att.dtype, att.device, cnn.shape, cnn.dtype, cnn.device)
            == (att_shape, static_att.dtype, static_att.device, cnn_shape, static_cnn.dtype, static_cnn.device)
            for att, cnn in zip(att_rows, cnn_rows, strict=True)
        ):
            return None
        if _precision_state() != self._precision or _graph_api(tokens.device).is_current_stream_capturing():
            return None
        if [id(t) for t in self._held_tensors()] != [id(t) for t in self._held]:  # ``_held`` keeps them alive
            self.enabled = False
            logger.warning("Flow encoder RelPos tables were replaced after capture; encoder graphs disabled")
            return None
        with torch.no_grad():  # a replay is not differentiable; serving runs in inference mode anyway
            torch.cat(list(cnn_rows), dim=0, out=static_cnn)
            torch.cat(list(att_rows), dim=1, out=static_att)
            views["tokens"].copy_(tokens)
            graph.replay()
        self.replays += 1
        return views["hidden"], views["new_cnn"], views["new_att"]

    def __call__(self, tokens, *, last_chunk, cnn_cache, att_cache, position_tables=(), shared_checked=False):
        if self._failed:
            raise RuntimeError("Code2Wav encoder graph capture failed; restart the stage before retrying")
        if (
            self.graphs
            and not shared_checked
            and not torch.is_grad_enabled()
            and not last_chunk
            and cnn_cache is not None
            and att_cache is not None
        ):
            result = self.run(tokens, cnn_cache.split(1, dim=0), att_cache.split(1, dim=1))
            if result is not None:
                return tuple(value.clone() for value in result)
        inputs = (tokens, cnn_cache, att_cache)
        self.stats["calls"] += 1

        def eager(reason):
            self.stats["eager"] += 1
            self.stats[reason] += 1
            self._log_stats()
            return self.forward(tokens, last_chunk=last_chunk, cnn_cache=cnn_cache, att_cache=att_cache)

        if tokens.device.type == "npu":
            return self._call_npu(
                tokens,
                last_chunk=last_chunk,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
                position_tables=position_tables,
                eager=eager,
            )

        if (
            not self.max_graphs
            or tokens.device.type != "cuda"
            # ROCm also reports CUDA tensor devices, but the private capture
            # streams below are created through NVIDIA cuda.bindings.
            or torch.version.hip is not None
            or torch.is_grad_enabled()
            or any(x is not None and x.device != tokens.device for x in inputs)
            or torch.cuda.is_current_stream_capturing()
        ):
            return eager("ineligible")
        shape = getattr(tokens, "shape", None)
        if (
            self.eager_min_batch is not None
            and shape is not None
            and len(shape) >= 2
            and int(shape[0]) >= self.eager_min_batch
        ):
            # Large batches make the attention-cache copy larger than the launches saved.
            return eager("batch")
        amp = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")
        key = self._exact_key(inputs, last_chunk, amp, dtype, position_tables)
        entry = self.exact_graphs.get(key)
        captured = entry is None
        if entry is None:
            if not self.capture_on_request or len(self.exact_graphs) >= self.max_graphs:
                return eager("capacity")
            count = self.seen.pop(key, 0) + 1
            self.seen[key] = count
            if len(self.seen) > self.max_graphs * 8:
                self.seen.popitem(last=False)
            if count < self.capture_after:
                return eager("admission")
            try:
                entry = self._install_capture(key, inputs, last_chunk, amp, dtype, position_tables)
            except Exception:
                self._failed = True
                logger.exception("Code2Wav encoder CUDA capture failed; stage restart required")
                raise
        graph, static_inputs, outputs, _tables = entry
        # Capture already cloned these exact inputs; only cache hits need copies.
        if not captured:
            for target, source in zip(static_inputs, inputs, strict=True):
                if target is not None:
                    target.copy_(source)
        graph.replay()
        self.stats["hits"] += 1
        # Subsequent replay must not overwrite another chunk/request's state.
        result = tuple(x.clone() for x in outputs)
        self._log_stats()
        return result

    def _call_npu(self, tokens, *, last_chunk, cnn_cache, att_cache, position_tables, eager):
        """Replay the chunk encoder with NPUGraph.

        The CUDA class records ``torch.cuda.CUDAGraph`` objects. On Ascend the
        same admission rule (repeat a shape ``capture_after`` times, then keep
        at most ``max_graphs`` exact signatures) goes through
        ``NPUExactGraphRunner``, which is what Code2Wav already uses for the
        CFM estimator. ``None`` caches are omitted from the captured inputs;
        their presence is part of the signature instead.
        """
        from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

        if (cnn_cache is None) != (att_cache is None):
            return eager("ineligible")
        autocast_enabled = False
        try:
            autocast_enabled = bool(torch.is_autocast_enabled("npu"))
        except (TypeError, RuntimeError):
            autocast_enabled = False
        if (
            not self.max_graphs
            or torch.is_grad_enabled()
            or autocast_enabled
            or not NPUExactGraphRunner.is_supported()
            or not hasattr(torch.npu, "graph_pool_handle")
            or NPUExactGraphRunner._stream_is_capturing()
            or (cnn_cache is not None and cnn_cache.device != tokens.device)
            or (att_cache is not None and att_cache.device != tokens.device)
        ):
            return eager("ineligible")

        caches_present = cnn_cache is not None
        graph_inputs = (tokens, cnn_cache, att_cache) if caches_present else (tokens,)
        stream = torch.npu.current_stream(tokens.device)
        stream_key = (tokens.device, stream.npu_stream)
        shape_key = (
            stream_key,
            bool(last_chunk),
            caches_present,
            tuple(
                None if value is None else (tuple(value.shape), str(value.dtype), str(value.device))
                for value in graph_inputs
            ),
            tuple((id(value), value.data_ptr(), value.shape, value.dtype, value.device) for value in position_tables),
        )
        if shape_key not in self._npu_pe:
            if not self.capture_on_request or self._npu_runners.captures >= self.max_graphs:
                return eager("capacity")
            count = self.seen.pop(shape_key, 0) + 1
            self.seen[shape_key] = count
            if len(self.seen) > self.max_graphs * 8:
                self.seen.popitem(last=False)
            if count < self.capture_after:
                return eager("admission")

        runner = self._npu_runners.get(stream_key)
        if runner is None:
            return eager("capacity")
        captures = runner.stats["captures"]

        def compute(*values):
            if caches_present:
                step_tokens, step_cnn, step_att = values
            else:
                step_tokens = values[0]
                step_cnn = step_att = None
            return self.forward(
                step_tokens,
                last_chunk=last_chunk,
                cnn_cache=step_cnn,
                att_cache=step_att,
            )

        try:
            result = runner.run(
                "conformer_chunk",
                graph_inputs,
                (bool(last_chunk), caches_present, shape_key[-1]),
                compute,
            )
        except Exception:
            self._failed = True
            logger.exception("Code2Wav encoder NPUGraph capture failed; stage restart required")
            raise
        if runner.stats["captures"] > captures:
            # Only captured graphs need to retain replaced positional tables.
            self._npu_pe[shape_key] = tuple(position_tables)
            self.seen.pop(shape_key, None)
            self.stats["captures"] += 1
            # NPUExactGraphRunner returns its eager warmup result on capture.
            self.stats["eager"] += 1
            logger.info(
                "Code2Wav encoder captured NPUGraph %d/%d",
                self.stats["captures"],
                self.max_graphs,
            )
        elif shape_key in self._npu_pe:
            self.stats["hits"] += 1
        else:
            self.stats["eager"] += 1
            self.stats["ineligible"] += 1
        self._log_stats()
        return result

    def _log_stats(self):
        if self.stats["calls"] % 256 == 0:
            logger.info(
                "Code2Wav encoder graph stats: calls=%d hits=%d (%.1f%%) "
                "eager=%d admission=%d capacity=%d ineligible=%d captures=%d evictions=%d resident=%d",
                self.stats["calls"],
                self.stats["hits"],
                100 * self.stats["hits"] / self.stats["calls"],
                self.stats["eager"],
                self.stats["admission"],
                self.stats["capacity"],
                self.stats["ineligible"],
                self.stats["captures"],
                self.stats["evictions"],
                len(self.exact_graphs) + len(self._npu_pe),
            )

    def capture_now(self, tokens, *, last_chunk, cnn_cache, att_cache, position_tables=()):
        """Capture this exact CUDA shape immediately, skipping admission.

        Returns whether a new graph was installed. A full cache or a batch at
        ``eager_min_batch`` or above is left eager. Failures latch the wrapper shut, just like request-time capture.
        """
        if self._failed:
            raise RuntimeError("Code2Wav encoder graph capture failed; restart the stage before retrying")
        if tokens.device.type != "cuda" or torch.version.hip is not None:
            return False
        shape = getattr(tokens, "shape", None)
        if (
            self.eager_min_batch is not None
            and shape is not None
            and len(shape) >= 2
            and int(shape[0]) >= self.eager_min_batch
        ):
            return False
        if len(self.exact_graphs) >= self.max_graphs:
            return False
        inputs = (tokens, cnn_cache, att_cache)
        amp = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")
        key = self._exact_key(inputs, last_chunk, amp, dtype, position_tables)
        if key in self.exact_graphs:
            return False
        try:
            self._install_capture(key, inputs, last_chunk, amp, dtype, position_tables)
        except Exception:
            self._failed = True
            logger.exception("Code2Wav encoder CUDA precapture failed; stage restart required")
            raise
        return True

    @staticmethod
    def _exact_key(inputs, last_chunk, amp, dtype, position_tables):
        return (
            torch.cuda.current_stream(inputs[0].device).cuda_stream,
            last_chunk,
            amp,
            dtype,
            _precision_state(),
            tuple(None if x is None else (x.shape, x.dtype, x.device) for x in inputs),
            tuple((id(x), x.data_ptr(), x.shape, x.dtype, x.device) for x in position_tables),
        )

    def _install_capture(self, key, inputs, last_chunk, amp, dtype, position_tables):
        slot = len(self._slots)
        self._slots.append(None)
        entry = self._capture(inputs, last_chunk, amp, dtype, position_tables, slot)
        self.exact_graphs[key] = entry
        self.seen.pop(key, None)
        self.stats["captures"] += 1
        logger.info("Code2Wav encoder captured CUDA graph %d/%d", len(self.exact_graphs), self.max_graphs)
        return entry

    def _ensure_pool(self, device, slot):
        existing = self._slots[slot]
        if existing is not None and existing[0] == device:
            return existing[1]
        pool = torch.cuda.graph_pool_handle()
        stream = _new_capture_stream(device)
        static = torch.zeros(1, device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                kept = static + 0
            stream.synchronize()
            sentinel = torch.cuda.CUDAGraph()
            sentinel._capture_stream_owner = stream
            with torch.cuda.graph(sentinel, pool=pool, stream=stream):
                kept = static + 0
        torch.cuda.current_stream(device).wait_stream(stream)
        self._slots[slot] = (device, pool, sentinel, static, kept, stream)
        return pool

    def _capture(self, inputs, last_chunk, amp, dtype, position_tables, slot):
        device = inputs[0].device
        pool = self._ensure_pool(device, slot)
        stream = self._slots[slot][5]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.autocast("cuda", enabled=amp, dtype=dtype, cache_enabled=False):
            static = tuple(None if x is None else x.clone() for x in inputs)

            def compute():
                return self.forward(static[0], last_chunk=last_chunk, cnn_cache=static[1], att_cache=static[2])

            # Prime kernels on the capture stream, with autocast weight caching
            # disabled so temporary casted weights cannot escape capture.
            for _ in range(2):
                compute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            graph._capture_stream_owner = stream
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                outputs = compute()
        torch.cuda.current_stream(device).wait_stream(stream)
        return graph, static, outputs, tuple(position_tables)
