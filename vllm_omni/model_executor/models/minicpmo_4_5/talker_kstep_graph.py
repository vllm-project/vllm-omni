# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs for the per-frame tail of the MiniCPM-o Talker K-step decode.

``gpu_talker_multiframe.decode_frames`` runs frames 1..K-1 of a K-step step as
one talker FULL-graph replay per frame plus an eager tail of ~40 small kernels:
the codec head, the EOS force/mask and windowed repetition penalty, vLLM's
``Sampler`` (temperature, Triton top-k/top-p, exponential race), the stop/emit
bookkeeping and the next frame's input embedding. On H100 that tail keeps the
GPU busy for ~0.15 ms but costs ~1.7 ms of host time per frame.

Opt-in (stage-1 ``hf_overrides: {talker_kstep_graph_sampling: true}``), this
module captures the tail once per step shape and replays it between the talker
graph replays (``R k`` = the runner's own replay of frame k)::

    first (in 1) -> R 1 -> mid (out 1, in 2) -> R 2 -> ... -> R K-1 -> last (out K-1)

``in k`` writes the id sampled at frame k-1 into frame k's ``inputs_embeds``
row; ``out k`` reads frame k's row of the talker graph's static output and
samples. The frame index lives on the device, so one ``mid`` graph serves every
frame. The talker forward itself stays vLLM's FULL graph: a graph cannot be
launched inside another capture, and capturing the forward here would bake in
this step's attention metadata.

Lossless (see ``notes/s1graph.md``): the captured ops are ``decode_frames``'s
ops in the same order on the same values. Frame k's controls are selected with
a device index instead of a Python one; the repetition penalty and the
min_tokens mask always run, with exact-identity values (penalty 1.0, an
all-false mask) on steps that have none. Seeded rows sample from per-row
Philox *slot* generators registered with the graphs: before the frames each
slot takes its request generator's (seed, offset), afterwards the request
generator takes the slot's back, so every frame consumes the Philox counters
the eager sampler would have. Unseeded rows draw from the default generator,
which every capture registers. The sampled ids are therefore bit-identical to
the eager K-step path; anything the graphs cannot reproduce declines to it.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

#: Stop ids a row may carry in the captured tail (-1 padded); wider rows decline.
STOP_WIDTH = 8
#: Captured step shapes kept (least recently used dropped); each holds three small graphs.
MAX_SHAPES = 32
#: Captures one (rows, frames) shape may take; more means its buffers keep moving.
MAX_CAPTURES_PER_SHAPE = 4
#: Steps of one (rows, frames) shape whose forward returned a fresh buffer before it stays eager.
MAX_UNSTABLE_PER_SHAPE = 2

_LOGGED: set[str] = set()

#: ``(fn, generators) -> graph`` with ``graph.replay()`` rerunning ``fn``.
GraphFactory = Callable[[Callable[[], None], list[torch.Generator]], Any]


def _log_once(kind: str, message: str, *args: Any) -> None:
    if kind in _LOGGED:
        return
    _LOGGED.add(kind)
    logger.info("[minicpmo] Talker K-step frame graphs: " + message, *args)


def _ptr(tensor: torch.Tensor | None) -> tuple[int, tuple[int, ...]] | None:
    return None if tensor is None else (tensor.data_ptr(), tuple(tensor.shape))


def _native_sampling(sampler: Any, md: Any) -> bool:
    """Whether vLLM's sampler takes its PyTorch path for this batch.

    That path (Triton top-k/top-p masks, softmax, an exponential race on the
    request or default generator) is capturable. FlashInfer's sampler reads the
    generator state on the host, so a replay would reuse frozen randoms.
    """
    if md.all_greedy:
        return True
    topk_topp = getattr(sampler, "topk_topp_sampler", None)
    forward = getattr(getattr(topk_topp, "forward", None), "__name__", "")
    if forward == "forward_native":
        return True
    if forward == "forward_cuda":
        # forward_cuda falls back to forward_native for these.
        return (
            bool(md.generators)
            or (md.top_k is None and md.top_p is None)
            or bool(getattr(topk_topp, "use_fp64_gumbel", False))
        )
    return False


def graph_decline(sampler: Any, md: Any, controls: Any) -> str | None:
    """Why this step's frames cannot replay captured tails (they run eagerly), or None.

    ``gpu_talker_multiframe.ineligible_reason`` already declined logprobs, bad
    words, allowed ids, custom logits processors and penalties other than the
    Talker's own; this only adds what a graph cannot reproduce.
    """
    if not _native_sampling(sampler, md):
        return "FlashInfer sampling without per-request generators (host-side RNG state)"
    for proc in md.logitsprocs.all:
        if getattr(proc, "min_p_count", 0) or getattr(proc, "biases", None):
            return f"active {type(proc).__name__} (a Python branch per step)"
    if int(controls.stop_ids.shape[1]) > STOP_WIDTH:
        return f"{int(controls.stop_ids.shape[1])} stop ids per row (captured width {STOP_WIDTH})"
    return None


def _forward_graph_mode() -> str | None:
    """The forward context's cudagraph mode name, or None outside a forward context."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
    except ImportError:  # pragma: no cover - vLLM without a forward context
        return None
    if not is_forward_context_available():
        return None
    mode = getattr(get_forward_context(), "cudagraph_runtime_mode", None)
    return getattr(mode, "name", None)


def _triton_sampler_buffers() -> list[Any]:
    """vLLM's cached Triton top-k/top-p scratch buffers, kept alive by each capture.

    A graph bakes their addresses in; if vLLM ever replaced or cleared its cache,
    these references keep the captured scratch memory from being reused.
    """
    try:
        from vllm.v1.sample.ops import topk_topp_triton as ops
    except ImportError:  # pragma: no cover - vLLM without the Triton sampler
        return []
    held: list[Any] = []
    for name in ("_TRITON_BUFFER_CACHE", "_TRITON_TABLE_CACHE", "_TRITON_SPLIT_CACHE"):
        cache = getattr(ops, name, None)
        if isinstance(cache, dict):
            held.extend(cache.values())
    return held


class FrameTail:
    """Static buffers of one step shape and the per-frame functions over them.

    Every function reads and writes only these buffers (plus the talker's
    ``inputs_embeds`` and output rows), so running it eagerly and replaying its
    capture give the same results. Rows are ``(B,)``, frames ``(K, B)``.
    """

    def __init__(
        self,
        *,
        batch: int,
        frames: int,
        vocab: int,
        window: int,
        eos: int,
        hidden: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> None:
        device = hidden.device
        long, flag = torch.long, torch.bool
        self.batch, self.frames, self.vocab, self.eos = batch, frames, vocab, int(eos)
        self.hidden = hidden
        self.inputs_embeds = inputs_embeds
        self.k = torch.ones(1, dtype=long, device=device)
        self.prev = torch.zeros(batch, dtype=long, device=device)
        self.alive = torch.ones(batch, dtype=flag, device=device)
        self.sampled = torch.zeros((batch, frames), dtype=long, device=device)
        self.emitted = torch.ones((batch, frames), dtype=flag, device=device)
        self.window = torch.full((batch, window), vocab, dtype=long, device=device)
        self.force_eos = torch.zeros((frames, batch), dtype=flag, device=device)
        self.mask_eos = torch.zeros((frames, batch), dtype=flag, device=device)
        self.allowed = torch.zeros((frames, batch), dtype=flag, device=device)
        self.min_tokens = torch.zeros((frames, batch), dtype=flag, device=device)
        self.min_tokens_stop = torch.zeros((batch, vocab), dtype=flag, device=device)
        self.stop_ids = torch.full((batch, STOP_WIDTH), -1, dtype=long, device=device)
        self.penalties = torch.ones(batch, dtype=torch.float32, device=device)
        self.rows = torch.zeros((frames, batch), dtype=long, device=device)
        # build_controls' constants, for this shape's EOS.
        self.eos_col = torch.arange(vocab, device=device) == self.eos
        self.forced_row = torch.where(
            self.eos_col,
            torch.zeros((), dtype=torch.float32, device=device),
            torch.full((), float("-inf"), dtype=torch.float32, device=device),
        )

    def load(self, first: torch.Tensor, controls: Any, rows: torch.Tensor) -> None:
        """Stage one step: frame 0's ids and the step's controls."""
        self.k.fill_(1)
        self.prev.copy_(first)
        self.alive.fill_(True)
        self.sampled[:, 0].copy_(first)
        self.window.copy_(controls.window)
        self.force_eos.copy_(controls.force_eos)
        self.mask_eos.copy_(controls.mask_eos)
        self.allowed.copy_(controls.allowed)
        self.min_tokens.copy_(controls.min_tokens)
        if controls.min_tokens_stop is None:
            self.min_tokens_stop.zero_()
        else:
            self.min_tokens_stop.copy_(controls.min_tokens_stop)
        width = int(controls.stop_ids.shape[1])
        self.stop_ids.fill_(-1)
        self.stop_ids[:, :width].copy_(controls.stop_ids)
        if controls.penalties is None:
            self.penalties.fill_(1.0)
        else:
            self.penalties.copy_(controls.penalties)
        self.rows.copy_(rows)

    def _at_frame(self, per_frame: torch.Tensor) -> torch.Tensor:
        return per_frame.index_select(0, self.k).squeeze(0)

    def frame_in(self, embed: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """``decode_frames`` up to the forward: emit flags, the window and frame k's input row."""
        prev = self.prev
        stopped = (prev.unsqueeze(1) == self.stop_ids).any(dim=1)
        alive = self.alive & ~stopped & self._at_frame(self.allowed)
        self.alive.copy_(alive)
        self.emitted.index_copy_(1, self.k, alive.unsqueeze(1))
        self.window.copy_(torch.cat([self.window[:, 1:], prev.unsqueeze(1)], dim=1))
        embeds = embed(prev)
        self.inputs_embeds.index_copy_(0, self._at_frame(self.rows), embeds.to(self.inputs_embeds.dtype))

    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """``gpu_talker_multiframe.adjust_logits`` at the device frame index."""
        neg_inf = float("-inf")
        logits = torch.where(self._at_frame(self.force_eos).unsqueeze(1), self.forced_row, logits)
        logits = torch.where(self._at_frame(self.mask_eos).unsqueeze(1) & self.eos_col, neg_inf, logits)
        num_reqs, vocab = logits.shape
        counts = torch.zeros((num_reqs, vocab + 1), dtype=torch.long, device=logits.device)
        counts.scatter_add_(1, self.window, torch.ones_like(self.window))
        alpha = torch.pow(self.penalties.to(logits.dtype).unsqueeze(1), counts[:, :vocab].to(logits.dtype))
        logits = torch.where(logits < 0, logits * alpha, logits / alpha)
        return logits.masked_fill(self.min_tokens_stop & self._at_frame(self.min_tokens).unsqueeze(1), neg_inf)

    def frame_out(
        self,
        hidden: torch.Tensor,
        codec_logits: Callable[[torch.Tensor], torch.Tensor],
        sample: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """``decode_frames`` after the forward: frame k's logits, sample and forced EOS."""
        logits = codec_logits(hidden.index_select(0, self._at_frame(self.rows)))
        logits = self.adjust_logits(logits)
        prev = torch.where(self._at_frame(self.force_eos), self.eos, sample(logits).to(torch.long))
        self.prev.copy_(prev)
        self.sampled.index_copy_(1, self.k, prev.unsqueeze(1))

    def next_frame(self) -> None:
        self.k.add_(1)


@dataclass
class _Shape:
    tail: FrameTail
    graphs: dict[str, Any] | None = None
    held: list[Any] = field(default_factory=list)
    replays: int = 0


@dataclass
class _Stats:
    graph_steps: int = 0
    eager_steps: int = 0
    declined: dict[str, int] = field(default_factory=dict)
    captures: int = 0
    capture_ms: float = 0.0
    unstable: int = 0


class TalkerKStepFrameGraphs:
    """Runs a K-step step's frames 1..K-1 with captured tails (see the module doc).

    ``__call__`` is ``gpu_talker_multiframe.maybe_run``'s ``kstep_decode_frames``
    hook: it returns ``decode_frames``'s ``(sampled, emitted)`` or None to let the
    eager loop run the step. The first step of a shape runs the tail functions
    eagerly (that is also the capture warm-up) and captures afterwards.
    """

    def __init__(self, *, max_shapes: int = MAX_SHAPES, graph_factory: GraphFactory | None = None) -> None:
        """``graph_factory(fn, generators)`` returns an object whose ``replay()``
        reruns ``fn`` (tests); by default tails are CUDA-graph captured, on CUDA only."""
        self.max_shapes = int(max_shapes)
        self._graph_factory = graph_factory
        self._shapes: OrderedDict[tuple[Any, ...], _Shape] = OrderedDict()
        self._captures_per_shape: dict[tuple[int, int], int] = {}
        self._unstable_per_shape: dict[tuple[int, int], int] = {}
        self._failed: set[tuple[Any, ...]] = set()
        self._slots: list[torch.Generator] = []
        self._pool: Any = None
        self._stream: Any = None
        self.stats = _Stats()
        self._stats_enabled = os.environ.get("VLLM_OMNI_TALKER_KSTEP_STATS") == "1"

    # -- public hook --------------------------------------------------------

    def __call__(
        self,
        *,
        first: torch.Tensor,
        frames: int,
        controls: Any,
        rows: torch.Tensor,
        inputs_embeds: torch.Tensor,
        hidden: torch.Tensor,
        replay: Callable[[], torch.Tensor],
        embed: Callable[[torch.Tensor], torch.Tensor],
        codec_logits: Callable[[torch.Tensor], torch.Tensor],
        sampler: Any,
        sampling_metadata: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        md = sampling_metadata
        decline = graph_decline(sampler, md, controls)
        if decline is None:
            mode = _forward_graph_mode()
            if mode is not None and mode != "FULL":
                decline = f"the talker forward runs {mode}, not one FULL graph"
        if (
            decline is None
            and first.device.type == "cuda"
            and not hasattr(torch.cuda.CUDAGraph, "register_generator_state")
        ):
            decline = "torch.cuda.CUDAGraph has no register_generator_state"
        batch = int(first.shape[0])
        vocab = int(controls.forced_row.shape[0])
        key = (
            batch,
            int(frames),
            vocab,
            int(controls.eos_token_id),
            _ptr(hidden),
            hidden.dtype,
            _ptr(inputs_embeds),
            inputs_embeds.dtype,
            bool(md.all_greedy),
            bool(md.all_random),
            _ptr(md.temperature),
            _ptr(md.top_k),
            _ptr(md.top_p),
            tuple(sorted(int(i) for i in md.generators)),
        )
        if decline is None and key in self._failed:
            decline = "capture failed for this step shape"
        if decline is not None:
            self._count_decline(decline)
            return None
        shape = self._shape(key, batch, frames, vocab, controls, hidden, inputs_embeds)
        if shape is None:
            self._count_decline("capture budget for this step shape is spent (its buffers keep moving)")
            return None
        tail = shape.tail
        seeded = {int(i): gen for i, gen in md.generators.items()}
        slots = {i: self._slot(i, first.device) for i in seeded}
        slot_md = replace(md, generators=slots)

        def sample(logits: torch.Tensor) -> torch.Tensor:
            return sampler(logits, slot_md).sampled_token_ids[:, 0]

        tail.load(first, controls, rows)
        for i, gen in seeded.items():
            slots[i].set_state(gen.get_state())
        try:
            if shape.graphs is not None:
                stable = self._replay(shape, frames, replay, embed, codec_logits, sample)
            else:
                stable = self._eager(tail, frames, replay, embed, codec_logits, sample)
        finally:
            # The request generators end where frames-1 eager samples leave them.
            for i, gen in seeded.items():
                gen.set_state(slots[i].get_state())
        if shape.graphs is not None:
            self.stats.graph_steps += 1
        else:
            self.stats.eager_steps += 1
        if not stable:
            self._unstable(key)
        elif shape.graphs is None and (first.device.type == "cuda" or self._graph_factory is not None):
            self._capture(key, shape, embed, codec_logits, sample, list(slots.values()))
        self._maybe_log_stats()
        return tail.sampled.clone(), tail.emitted.clone()

    # -- frames -------------------------------------------------------------

    @staticmethod
    def _same_output(out: torch.Tensor, tail: FrameTail) -> bool:
        return out.data_ptr() == tail.hidden.data_ptr() and out.shape == tail.hidden.shape

    def _eager(self, tail: FrameTail, frames: int, replay, embed, codec_logits, sample) -> bool:
        """The tail functions, eagerly; True if every forward returned the static output."""
        stable = True
        tail.frame_in(embed)
        for k in range(1, frames):
            out = replay()
            stable = stable and self._same_output(out, tail)
            tail.frame_out(out, codec_logits, sample)
            if k < frames - 1:
                tail.next_frame()
                tail.frame_in(embed)
        return stable

    def _replay(self, shape: _Shape, frames: int, replay, embed, codec_logits, sample) -> bool:
        tail, graphs = shape.tail, shape.graphs
        assert graphs is not None
        shape.replays += 1
        graphs["first"].replay()
        for k in range(1, frames):
            out = replay()
            if not self._same_output(out, tail):
                # The forward did not return the captured output buffer: finish
                # this step with the same functions, eagerly, on what it returned.
                tail.frame_out(out, codec_logits, sample)
                for _ in range(k + 1, frames):
                    tail.next_frame()
                    tail.frame_in(embed)
                    tail.frame_out(replay(), codec_logits, sample)
                return False
            graphs["mid" if k < frames - 1 else "last"].replay()
        return True

    # -- shapes, generators, capture ----------------------------------------

    def _shape(self, key, batch, frames, vocab, controls, hidden, inputs_embeds) -> _Shape | None:
        shape = self._shapes.get(key)
        if shape is not None:
            self._shapes.move_to_end(key)
            return shape
        size = (batch, int(frames))
        if (
            self._captures_per_shape.get(size, 0) >= MAX_CAPTURES_PER_SHAPE
            or self._unstable_per_shape.get(size, 0) >= MAX_UNSTABLE_PER_SHAPE
        ):
            return None
        tail = FrameTail(
            batch=batch,
            frames=int(frames),
            vocab=vocab,
            window=int(controls.window.shape[1]),
            eos=int(controls.eos_token_id),
            hidden=hidden,
            inputs_embeds=inputs_embeds,
        )
        shape = _Shape(tail=tail)
        self._shapes[key] = shape
        while len(self._shapes) > self.max_shapes:
            self._shapes.popitem(last=False)
        return shape

    def _slot(self, index: int, device: torch.device) -> torch.Generator:
        while len(self._slots) <= index:
            self._slots.append(torch.Generator(device=device))
        return self._slots[index]

    def _unstable(self, key: tuple[Any, ...]) -> None:
        self.stats.unstable += 1
        size = (int(key[0]), int(key[1]))
        self._unstable_per_shape[size] = self._unstable_per_shape.get(size, 0) + 1
        self._shapes.pop(key, None)
        self._failed.add(key)
        _log_once(
            "unstable",
            "the talker forward returned a new buffer inside a K-step step (rows %d); "
            "that shape runs eagerly from now on",
            key[0],
        )

    def _capture(self, key, shape: _Shape, embed, codec_logits, sample, slots: list[torch.Generator]) -> None:
        tail = shape.tail
        size = (tail.batch, tail.frames)
        self._captures_per_shape[size] = self._captures_per_shape.get(size, 0) + 1
        device = tail.hidden.device

        def mid() -> None:
            tail.frame_out(tail.hidden, codec_logits, sample)
            tail.next_frame()
            tail.frame_in(embed)

        steps: list[tuple[str, Callable[[], None], list[torch.Generator]]] = [
            ("first", lambda: tail.frame_in(embed), []),
            ("last", lambda: tail.frame_out(tail.hidden, codec_logits, sample), slots),
        ]
        if tail.frames > 2:
            steps.insert(1, ("mid", mid, slots))
        generators = list(slots)
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.accelerator.current_device_index()
            generators.append(torch.cuda.default_generators[index])
        else:
            generators.append(torch.default_generator)
        # Capturing records the RNG ops without running them; restore every
        # generator anyway so a capture can never shift what eager would draw.
        saved = [(gen, gen.get_state()) for gen in generators]
        reserved = torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
        start = time.perf_counter()
        try:
            if self._graph_factory is not None:
                graphs = {name: self._graph_factory(fn, gens) for name, fn, gens in steps}
            else:
                graphs = self._cuda_capture(steps, device)
        except Exception as exc:  # noqa: BLE001 - a failed capture keeps the eager path
            self._failed.add(key)
            self._shapes.pop(key, None)
            logger.warning(
                "[minicpmo] Talker K-step frame graphs: capture failed for rows %d, frames %d (%s: %s); "
                "this step shape keeps the eager tail",
                tail.batch,
                tail.frames,
                type(exc).__name__,
                exc,
            )
            return
        finally:
            for gen, state in saved:
                gen.set_state(state)
        elapsed = (time.perf_counter() - start) * 1e3
        shape.graphs = graphs
        shape.held = _triton_sampler_buffers()
        self.stats.captures += 1
        self.stats.capture_ms += elapsed
        grown = (torch.cuda.memory_reserved(device) - reserved) / 2**20 if device.type == "cuda" else 0.0
        logger.info(
            "[minicpmo] Talker K-step frame graphs captured: rows %d, frames %d, %d graphs in %.1f ms, "
            "pool reserved %+.1f MiB (%d shapes)",
            tail.batch,
            tail.frames,
            len(graphs),
            elapsed,
            grown,
            len(self._shapes),
        )

    def _cuda_capture(self, steps, device: torch.device) -> dict[str, Any]:
        """Capture each tail on a side stream into one private pool.

        Not ``torch.cuda.graph``: it synchronizes and empties the allocator cache.
        The pool is private because vLLM's shared pool counts the talker graph's
        (weak-ref) output as free, and a tail temporary there could overwrite it.
        """
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        self._stream.wait_stream(current)
        graphs: dict[str, Any] = {}
        with torch.cuda.stream(self._stream):
            for name, fn, generators in steps:
                graph = torch.cuda.CUDAGraph()
                for gen in generators:
                    graph.register_generator_state(gen)
                graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
                try:
                    fn()
                except BaseException:
                    try:
                        graph.capture_end()
                    except Exception:  # noqa: BLE001 - the capture error is the one to report
                        pass
                    raise
                graph.capture_end()
                graphs[name] = graph
        current.wait_stream(self._stream)
        return graphs

    # -- diagnostics ----------------------------------------------------------

    def _count_decline(self, reason: str) -> None:
        _log_once(reason, "eager tail: %s", reason)
        self.stats.declined[reason] = self.stats.declined.get(reason, 0) + 1
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        if not self._stats_enabled:
            return
        s = self.stats
        total = s.graph_steps + s.eager_steps + sum(s.declined.values())
        if total in (1, 100, 1000) or total % 10000 == 0:
            logger.info(
                "[minicpmo] Talker K-step frame graph stats: steps=%d graph=%d eager=%d declined=%s "
                "captures=%d (%.1f ms) unstable=%d shapes=%d",
                total,
                s.graph_steps,
                s.eager_steps,
                dict(s.declined),
                s.captures,
                s.capture_ms,
                s.unstable,
                len(self._shapes),
            )
