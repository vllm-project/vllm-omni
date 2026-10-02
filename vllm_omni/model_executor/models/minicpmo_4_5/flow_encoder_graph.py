# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape CUDA graphs of the Code2Wav flow encoder's streaming chunk.

Every Code2Wav step runs the CosyVoice2 upsample-conformer encoder
(``UpsampleConformerEncoderV2.forward_chunk``: pre-lookahead conv, 6 conformer
layers, 2x upsample, 4 more) between the token embedding and
``encoder_proj``: ~320 small eager kernels whose launches, not their GPU time,
set the step's latency in serving. A CUDA graph replays them with one launch.

The graphs change no output. One is captured per exact
``(rows, tokens, conformer attention-cache frames)`` and replays only calls of
exactly that shape, so it runs the kernels the eager call runs. Nothing is
padded: rounding 3 rows up to a 4-row graph (7cd254051) changes the M of every
GEMM, and with it cuBLAS's kernel and the FP32 rounding of every row; the TF32
CFM and HiFT amplified that to 0.81 dB at B=3. Exact shapes stay few because
the streaming trim caps the conformer cache at ``prompt + 100`` frames: one
prompt's streams only pass through the cache lengths
:func:`reachable_conformer_cache_frames` lists (300, 350 and 400 frames for the
6 s default voice and 28-token duplex chunks). Calls of any other shape, the
last chunk (lookahead padding) and the prompt's own encode run eager.

Graphs replay one at a time and every replay refills its inputs, so all of
them take prefix views of one storage per static input and per result, sized
for the largest key (like ``streaming_audio_encoder_graph``), and share one
private memory pool for their temporaries. A replay copies its results into
the shared result views: they stay valid until the next encoder replay.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

EncodeFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def reachable_conformer_cache_frames(
    *,
    start: int,
    prompt_len: int,
    suffix: int,
    growths: Iterable[int],
) -> list[int]:
    """Conformer attention-cache lengths a stream passes through, from its prompt state.

    A chunk adds ``growth`` frames (the encoder's output frames) to the cache;
    past ``prompt_len + suffix`` frames the streaming trim keeps the first
    ``prompt_len`` and the last ``suffix`` (``_trim_streaming_cache``). These
    are the cache lengths a continuation chunk's encoder call can see, for
    chunks of any mix of the ``growths``.
    """
    steady = int(prompt_len) + int(suffix)
    steps = sorted({int(g) for g in growths if int(g) > 0})
    seen = {int(start)}
    frontier = [int(start)]
    while frontier:
        frames = frontier.pop()
        for growth in steps:
            following = frames + growth
            if following > steady:
                following = steady
            if following not in seen:
                seen.add(following)
                frontier.append(following)
    return sorted(seen)


def _precision_state() -> tuple[Any, ...]:
    """The matmul/conv precision flags a captured graph's kernels were chosen under."""
    return (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        torch.get_float32_matmul_precision(),
    )


@dataclass
class _EncoderGraph:
    graph: Any
    tokens: torch.Tensor
    cnn: torch.Tensor
    att: torch.Tensor
    hidden: torch.Tensor
    new_cnn: torch.Tensor
    new_att: torch.Tensor


class FlowEncoderGraphs:
    """Exact-shape CUDA graphs of ``encode_fn`` (embedding, ``forward_chunk``, projection) for continuation chunks.

    ``encode_fn(tokens, cnn_cache, att_cache)`` is the eager continuation chunk
    (both caches present, ``last_chunk=False``). Caches use the encoder's
    layouts: ``cnn_cache`` ``(rows, channels, width)``, ``att_cache``
    ``(depth, rows, heads, frames, 2 * head_dim)``; a chunk of ``tokens``
    tokens emits ``upsample * (tokens - lookahead)`` frames.
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        *,
        rows: Iterable[int],
        token_widths: Iterable[int],
        lookahead: int,
        upsample: int,
        held_tensors: Callable[[], Sequence[torch.Tensor]] | None = None,
    ) -> None:
        self.encode_fn = encode_fn
        self.rows = tuple(sorted({int(r) for r in rows if int(r) > 0}))
        self.lookahead = int(lookahead)
        self.upsample = int(upsample)
        self.token_widths = tuple(sorted({int(w) for w in token_widths if int(w) > self.lookahead}))
        # Tensors the graphs read in place without owning them (the RelPos
        # tables): held so they stay alive, and checked so a replaced table
        # sends calls back to eager instead of replaying stale reads.
        self._held_tensors = held_tensors or (lambda: ())
        self._held: tuple[torch.Tensor, ...] = ()
        self._precision: tuple[Any, ...] | None = None
        self.graphs: dict[tuple[int, int, int], _EncoderGraph] = {}
        self.stats: Counter = Counter()
        self.enabled = True
        self._pool: Any = None
        self._storage: dict[str, torch.Tensor] = {}

    def output_frames(self, tokens: int) -> int:
        return self.upsample * (int(tokens) - self.lookahead)

    def keys_for(self, *, start: int, prompt_len: int, suffix: int) -> list[tuple[int, int, int]]:
        """``(rows, tokens, cache frames)`` of every continuation chunk a prompt's streams can produce."""
        frames = reachable_conformer_cache_frames(
            start=start,
            prompt_len=prompt_len,
            suffix=suffix,
            growths=[self.output_frames(width) for width in self.token_widths],
        )
        return [(rows, width, length) for rows in self.rows for width in self.token_widths for length in frames]

    # -- capture ---------------------------------------------------------

    def _views(
        self,
        key: tuple[int, int, int],
        *,
        cnn_shape: tuple[int, int],
        att_layout: tuple[int, int, int],
        hidden_dim: int,
    ) -> dict[str, tuple[int, ...]]:
        rows, width, frames = key
        depth, heads, att_width = att_layout
        out_frames = self.output_frames(width)
        return {
            "tokens": (rows, width),
            "cnn": (rows, *cnn_shape),
            "att": (depth, rows, heads, frames, att_width),
            "hidden": (rows, out_frames, hidden_dim),
            "new_cnn": (rows, *cnn_shape),
            "new_att": (depth, rows, heads, frames + out_frames, att_width),
        }

    def _view(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for size in shape:
            numel *= int(size)
        return self._storage[name][:numel].view(shape)

    def _body(self, entry: _EncoderGraph) -> None:
        hidden, new_cnn, new_att = self.encode_fn(entry.tokens, entry.cnn, entry.att)
        for name, value, target in (
            ("hidden", hidden, entry.hidden),
            ("cnn", new_cnn, entry.new_cnn),
            ("att", new_att, entry.new_att),
        ):
            if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
                raise RuntimeError(
                    f"flow encoder {name} output {tuple(value.shape)}/{value.dtype} does not match "
                    f"the captured layout {tuple(target.shape)}/{target.dtype}"
                )
            target.copy_(value)

    def _record(self, entry: _EncoderGraph) -> Any:
        """Warm up on a side stream, then capture ``_body`` into the shared private pool."""
        device = entry.att.device
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(2):
                self._body(entry)
        current.wait_stream(side)
        torch.accelerator.synchronize(device)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            self._body(entry)
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
    ) -> int:
        """Capture ``keys``; returns the number of graphs held afterwards.

        ``cnn_shape`` is ``(channels, width)``, ``att_layout``
        ``(depth, heads, 2 * head_dim)``. Larger keys go first, so the private
        pool grows once to the largest graph's temporaries and the others
        reuse that memory.
        """
        if self.graphs:
            raise RuntimeError("flow encoder graphs are already captured")
        keys = sorted(
            {(int(r), int(w), int(f)) for r, w, f in keys},
            key=lambda key: (key[0] * (key[2] + self.output_frames(key[1])), key[0] * key[1]),
            reverse=True,
        )
        if not keys:
            return 0
        shapes = [self._views(key, cnn_shape=cnn_shape, att_layout=att_layout, hidden_dim=hidden_dim) for key in keys]
        for name in shapes[0]:
            numel = 0
            for views in shapes:
                count = 1
                for size in views[name]:
                    count *= int(size)
                numel = max(numel, count)
            # Ordinary tensors, not inference tensors: replays write them in
            # place from inside and outside inference mode alike.
            with torch.inference_mode(False):
                self._storage[name] = torch.zeros(numel, dtype=torch.long if name == "tokens" else dtype, device=device)
        self._precision = _precision_state()
        try:
            for key, views in zip(keys, shapes, strict=True):
                with torch.inference_mode(False):
                    entry = _EncoderGraph(
                        graph=None, **{name: self._view(name, shape) for name, shape in views.items()}
                    )
                entry.graph = self._record(entry)
                self.graphs[key] = entry
        except Exception:
            # All or nothing: a partial set would replay some shapes and not others.
            self.graphs.clear()
            self._storage.clear()
            self._pool = None
            self.enabled = False
            raise
        self._held = tuple(self._held_tensors())
        return len(self.graphs)

    def storage_bytes(self) -> int:
        """Bytes of the shared static inputs and results (not the private pool)."""
        return sum(t.numel() * t.element_size() for t in self._storage.values())

    # -- replay ----------------------------------------------------------

    def _layout_matches(
        self,
        entry: _EncoderGraph,
        tokens: torch.Tensor,
        cnn_rows: Sequence[torch.Tensor],
        att_rows: Sequence[torch.Tensor],
    ) -> bool:
        if tokens.dtype != entry.tokens.dtype or tokens.device != entry.tokens.device:
            return False
        att_shape = (entry.att.shape[0], 1, *entry.att.shape[2:])
        cnn_shape = (1, *entry.cnn.shape[1:])
        for att, cnn in zip(att_rows, cnn_rows, strict=True):
            if tuple(att.shape) != att_shape or att.dtype != entry.att.dtype or att.device != entry.att.device:
                return False
            if tuple(cnn.shape) != cnn_shape or cnn.dtype != entry.cnn.dtype or cnn.device != entry.cnn.device:
                return False
        return True

    def _held_changed(self) -> bool:
        current = tuple(self._held_tensors())
        return len(current) != len(self._held) or any(a is not b for a, b in zip(current, self._held, strict=True))

    def run(
        self,
        tokens: torch.Tensor,
        cnn_rows: Sequence[torch.Tensor],
        att_rows: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The continuation chunk of ``rows`` streams from its graph, or ``None`` (run it eager).

        ``cnn_rows`` / ``att_rows`` are each stream's caches, one row each, in
        batch order. The results are the shared result views: copy what must
        outlive the next encoder replay.
        """
        if not self.enabled or not self.graphs:
            return None
        rows = len(att_rows)
        if tokens.dim() != 2 or int(tokens.shape[0]) != rows or len(cnn_rows) != rows or rows == 0:
            self.stats["eager"] += 1
            return None
        first = att_rows[0]
        if first.dim() != 5:
            self.stats["eager"] += 1
            return None
        entry = self.graphs.get((rows, int(tokens.shape[1]), int(first.shape[3])))
        if entry is None or not self._layout_matches(entry, tokens, cnn_rows, att_rows):
            self.stats["eager"] += 1
            return None
        if _precision_state() != self._precision or torch.cuda.is_current_stream_capturing():
            self.stats["eager_precision"] += 1
            return None
        if self._held_changed():
            self.enabled = False
            logger.warning("Flow encoder RelPos tables were replaced after capture; encoder graphs disabled")
            return None
        with torch.no_grad():  # a replay is not differentiable; serving runs in inference mode anyway
            torch.cat(list(cnn_rows), dim=0, out=entry.cnn)
            torch.cat(list(att_rows), dim=1, out=entry.att)
            entry.tokens.copy_(tokens)
            entry.graph.replay()
        self.stats["replays"] += 1
        return entry.hidden, entry.new_cnn, entry.new_att
