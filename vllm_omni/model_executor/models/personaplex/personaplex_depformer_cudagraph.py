# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay the PersonaPlex per-frame model steps from per-padded-B CUDA graphs.

To reduce launch-bound cost of the depformer's 16-step unrolled inner loop,
this module support full-graph capture and replay of the forward pass under
the unified duplex path. Shapes are static per padded batch, so a model-local
wrapper captures ``PersonaPlexDepformer.forward`` and replays it.

Opt-in via ``CUDAGraphDepformerWrapper.warmup`` and ``PersonaPlexConfig.depformer_cuda_graphs``
on the talker; capture failure falls back to eager. The Upperbound of capture sizes
are derived from `duplex_session.max_sessions`, not a YAML override.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
    _DepformerKVBuffers,
)

logger = init_logger(__name__)

__all__ = ["CUDAGraphDepformerWrapper", "DepformerCUDAGraphStats", "resolve_depformer_graph_settings"]


def _capture_sizes_up_to(max_num: int) -> tuple[int, ...]:
    """Powers of two up to `max_num`, inclusive."""
    sizes: list[int] = []
    size = 1
    while size <= max_num:
        sizes.append(size)
        size *= 2
    if sizes[-1] < max_num:
        sizes.append(max_num)
    return tuple(sizes)


def resolve_depformer_graph_settings(
    vllm_config: Any,
    *,
    enabled: bool,
    default_max_batch: int = 32,
    default_warmup_iters: int = 3,
) -> tuple[bool, tuple[int, ...], int]:
    model_config = getattr(vllm_config, "model_config", None)
    enforce_eager = bool(getattr(model_config, "enforce_eager", False))
    enabled = enabled and not enforce_eager
    compilation = getattr(vllm_config, "compilation_config", None)
    warmup = getattr(compilation, "cudagraph_num_of_warmups", None)
    warm_iters = default_warmup_iters if warmup is None else max(warmup, 0)
    # Assume one live duplex session contributes at most 1 row per tick
    # then duplex_max_sessions is the real batch ceiling
    max_batch = getattr(model_config, "duplex_max_sessions", None)
    max_batch = max(int(max_batch), 1) if max_batch else default_max_batch
    sizes = _capture_sizes_up_to(max_batch)
    return enabled, sizes, warm_iters


@dataclass
class DepformerCUDAGraphStats:
    calls: int = 0
    replays: int = 0
    eager: int = 0
    eager_outer_capture: int = 0
    eager_shape_mismatch: int = 0
    capture_failure: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "replays": self.replays,
            "eager": self.eager,
            "eager_outer_capture": self.eager_outer_capture,
            "eager_shape_mismatch": self.eager_shape_mismatch,
            "capture_failure": self.capture_failure,
        }


class _EagerReason(Enum):
    """Why a call fell back to eager instead of replaying a captured graph."""

    DISABLED = "disabled"
    NON_CUDA = "non_cuda"
    OUTER_CAPTURE = "outer_capture"
    NO_CAPTURE_SIZE = "no_capture_size"
    GRAPH_MISSING = "graph_missing"
    INPUT_MISMATCH = "input_mismatch"


@dataclass
class _DepformerCUDAGraph:
    graph: torch.cuda.CUDAGraph
    padded_b: int
    static_text_token: torch.Tensor
    static_transformer_out: torch.Tensor
    static_audio_tokens: torch.Tensor
    static_audio_provided: torch.Tensor
    static_out: torch.Tensor
    _warned: bool = field(default=False, repr=False)

    def matches(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        audio_provided: torch.Tensor | None,
    ) -> bool:
        if (audio_tokens is None) != (audio_provided is None):
            return False

        inputs = (
            (text_token, self.static_text_token),
            (transformer_out, self.static_transformer_out),
        )
        if audio_tokens is not None:
            inputs += ((audio_tokens, self.static_audio_tokens),)
        if audio_provided is not None:
            inputs += ((audio_provided, self.static_audio_provided),)

        actual_b = text_token.shape[0] if text_token.ndim > 0 else -1
        if actual_b < 0 or actual_b > self.padded_b:
            return False
        for tensor, static_tensor in inputs:
            if (
                tensor.device != static_tensor.device
                or tensor.dtype != static_tensor.dtype
                or tensor.ndim != static_tensor.ndim
                or tensor.shape[1:] != static_tensor.shape[1:]
                or tensor.shape[0] != actual_b
            ):
                return False
        return True

    def replay(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        audio_provided: torch.Tensor | None,
        actual_b: int,
    ) -> torch.Tensor:
        self.static_text_token.zero_()
        self.static_transformer_out.zero_()
        self.static_audio_tokens.zero_()
        self.static_audio_provided.zero_()

        self.static_text_token[:actual_b].copy_(text_token, non_blocking=True)
        self.static_transformer_out[:actual_b].copy_(transformer_out, non_blocking=True)
        if audio_tokens is not None:
            self.static_audio_tokens[:actual_b].copy_(audio_tokens, non_blocking=True)
        if audio_provided is not None:
            self.static_audio_provided[:actual_b].copy_(audio_provided, non_blocking=True)
        self.graph.replay()

        return self.static_out[:actual_b].clone()


class CUDAGraphDepformerWrapper:
    """Replay ``PersonaPlexDepformer.forward`` via per-padded-B CUDA graphs.

    The wrapper is the talker's optional dispatch target; graph hit, else ``depformer.forward``
    """

    def __init__(
        self,
        depformer: PersonaPlexDepformer,
        *,
        capture_sizes: Sequence[int],
        enabled: bool = True,
        warmup_iters: int = 3,
        num_steps: int | None = None,
    ) -> None:
        sizes = sorted({int(size) for size in capture_sizes if int(size) > 0})
        if not sizes:
            raise ValueError("capture_sizes must contain at least one positive integer")
        self.depformer = depformer
        self.capture_sizes: tuple[int, ...] = tuple(sizes)
        self.enabled = enabled
        self.warmup_iters = warmup_iters
        self._graphs: dict[int, _DepformerCUDAGraph] = {}
        self._stats = DepformerCUDAGraphStats()
        self._warmed_up = False
        self.num_steps = self.depformer.dep_q if num_steps is None else int(num_steps)
        if not 1 <= self.num_steps <= self.depformer.dep_q:
            raise ValueError(f"num_steps must be in [1, {self.depformer.dep_q}]; got {num_steps}")
        self._graph_kv_buffers: _DepformerKVBuffers | None = None
        if self.enabled:
            param = next(self.depformer.parameters())
            config = self.depformer.config
            # Size fixed scratch storage for the largest graph this wrapper can replay.
            self._graph_kv_buffers = _DepformerKVBuffers(
                num_layers=config.num_hidden_layers,
                max_batch=max(self.capture_sizes),
                num_heads=config.num_attention_heads,
                dep_q=config.dep_q,
                head_dim=config.head_dim,
                device=param.device,
                dtype=param.dtype,
            )

    @property
    def stats(self) -> DepformerCUDAGraphStats:
        return self._stats

    @property
    def is_ready(self) -> bool:
        return bool(self._graphs)

    def stats_snapshot(self) -> dict[str, int]:
        snap = self.stats.snapshot()
        snap["num_graphs"] = len(self._graphs)
        return snap

    def _select_padded_b(self, actual_b: int) -> int | None:
        return next((size for size in self.capture_sizes if size >= actual_b), None)

    def _synthetic_kwargs(self, padded_b: int, device: torch.device) -> dict[str, torch.Tensor]:
        dep_q = self.depformer.dep_q
        dtype = next(self.depformer.parameters()).dtype
        hidden = self.depformer.temporal_hidden_size
        return {
            "text_token": torch.zeros(padded_b, dtype=torch.long, device=device),
            "transformer_out": torch.zeros(padded_b, 1, hidden, dtype=dtype, device=device),
            "audio_tokens": torch.zeros(padded_b, dep_q, dtype=torch.long, device=device),
            "audio_provided": torch.zeros(padded_b, dep_q, dtype=torch.bool, device=device),
        }

    def _eager(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        audio_provided: torch.Tensor | None,
        *,
        graph_kv_buffers: _DepformerKVBuffers | None = None,
    ) -> torch.Tensor:
        out = self.depformer(
            text_token,
            transformer_out,
            audio_tokens,
            audio_provided,
            num_steps=self.num_steps,
            graph_kv_buffers=graph_kv_buffers,
        )
        if isinstance(out, tuple):
            return out[0]
        return out

    def _capture_model(self, padded_b: int, device: torch.device) -> _DepformerCUDAGraph | None:
        kwargs = self._synthetic_kwargs(padded_b, device)
        graph_kv_buffers = self._graph_kv_buffers
        if graph_kv_buffers is None:
            raise RuntimeError("Depformer graph KV buffers are not initialized")
        # Warmup and capture replay the same eager path so the recorded graph
        # is provably what `_eager()` would have computed for these inputs.
        try:
            pool = torch.cuda.graph_pool_handle()
            with torch.inference_mode(False), torch.no_grad():
                for _ in range(max(self.warmup_iters, 0)):
                    self._eager(**kwargs, graph_kv_buffers=graph_kv_buffers)
                torch.accelerator.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    static_out = self._eager(**kwargs, graph_kv_buffers=graph_kv_buffers)
                torch.accelerator.synchronize(device)
        except Exception as exc:
            logger.warning(
                "Failed to capture Depformer graph for padded batch size %d (%s); using eager execution",
                padded_b,
                exc,
            )
            self.stats.capture_failure += 1
            return None
        logger.info("Captured Depformer graph for padded batch size %d", padded_b)
        return _DepformerCUDAGraph(
            graph=graph,
            padded_b=padded_b,
            static_text_token=kwargs["text_token"],
            static_transformer_out=kwargs["transformer_out"],
            static_audio_tokens=kwargs["audio_tokens"],
            static_audio_provided=kwargs["audio_provided"],
            static_out=static_out,
        )

    def warmup(self, device: torch.device) -> None:
        """Capture all sizes. No-op when disabled or not CUDA."""
        if self._graphs or (self._warmed_up and self.enabled):
            return
        if not self.enabled:
            self._warmed_up = True
            return
        if device.type != "cuda" or not torch.cuda.is_available():
            logger.info(
                "CUDAGraphDepformerWrapper warmup skipped (enabled=%s, device=%s)",
                self.enabled,
                device,
            )
            return
        self._warmed_up = True
        for padded_b in self.capture_sizes:
            entry = self._capture_model(padded_b, device)
            if entry is not None:
                self._graphs[padded_b] = entry

    def _resolve_replay_entry(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        audio_provided: torch.Tensor | None,
    ) -> tuple[_DepformerCUDAGraph | None, _EagerReason | None]:
        """Pick the captured graph for this call, or why eager is required."""
        if not self.enabled:
            return None, _EagerReason.DISABLED
        if text_token.device.type != "cuda":
            return None, _EagerReason.NON_CUDA
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return None, _EagerReason.OUTER_CAPTURE

        padded_b = self._select_padded_b(text_token.shape[0])
        if padded_b is None:
            # fallback to eager mode if no suitable cudagraph size found
            return None, _EagerReason.NO_CAPTURE_SIZE
        entry = self._graphs.get(padded_b)
        if entry is None:
            return None, _EagerReason.GRAPH_MISSING
        if not entry.matches(text_token, transformer_out, audio_tokens, audio_provided):
            if not entry._warned:
                logger.warning(
                    "Depformer graph input mismatch for padded batch size %d; "
                    "using eager execution. This warning is only logged once.",
                    padded_b,
                )
                entry._warned = True
            return None, _EagerReason.INPUT_MISMATCH
        return entry, None

    def __call__(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor | None = None,
        audio_provided: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.stats.calls += 1
        entry, reason = self._resolve_replay_entry(text_token, transformer_out, audio_tokens, audio_provided)
        if entry is None:
            self.stats.eager += 1
            if reason is _EagerReason.OUTER_CAPTURE:
                self.stats.eager_outer_capture += 1
            elif reason in (_EagerReason.NO_CAPTURE_SIZE, _EagerReason.INPUT_MISMATCH):
                self.stats.eager_shape_mismatch += 1
            return self._eager(text_token, transformer_out, audio_tokens, audio_provided)

        self.stats.replays += 1
        return entry.replay(text_token, transformer_out, audio_tokens, audio_provided, int(text_token.shape[0]))
