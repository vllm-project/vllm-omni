# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape CUDA graphs of the Code2Wav flow encoder's continuation chunk.

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
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from vllm.logger import init_logger

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


def _precision_state() -> tuple[Any, ...]:
    """The matmul/conv precision flags a captured graph's kernels were chosen under."""
    return (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        torch.get_float32_matmul_precision(),
    )


class FlowEncoderGraphs:
    """Exact-shape graphs of ``encode_fn(tokens, cnn_cache=, att_cache=)``, the eager continuation chunk.

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
        rows: Iterable[int],
        token_widths: Iterable[int],
        lookahead: int,
        upsample: int,
        held_tensors: Callable[[], Sequence[torch.Tensor]] = tuple,
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
        """Warm up on a side stream, then capture ``_body`` into the shared private pool."""
        device = views["att"].device
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(2):
                self._body(views)
        current.wait_stream(side)
        torch.accelerator.synchronize(device)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            self._body(views)
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
        # Ordinary tensors, not inference tensors: replays write them in and outside inference mode.
        with torch.inference_mode(False):
            for name in layouts[0]:
                numel = max(math.prod(layout[name]) for layout in layouts)
                self._storage[name] = torch.zeros(numel, dtype=torch.long if name == "tokens" else dtype, device=device)
        self._precision = _precision_state()
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
        rows = len(att_rows)
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
        if _precision_state() != self._precision or torch.cuda.is_current_stream_capturing():
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
