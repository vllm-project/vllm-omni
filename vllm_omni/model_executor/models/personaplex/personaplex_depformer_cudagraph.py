# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay the PersonaPlex per-frame model steps from per-padded-B CUDA graphs.

To reduce launch-bound cost of the depformer's 16-step unrolled inner loop,
this module support full-graph capture and replay of the forward pass under
the unified duplex path. Shapes are static per padded batch, so a model-local
wrapper captures ``PersonaPlexDepformer.forward`` and replays it.

Opt-in via ``CUDAGraphDepformerWrapper.warmup`` and ``PersonaPlexConfig.depformer_cuda_graphs``
on the talker; capture failure falls back to eager.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
)

logger = init_logger(__name__)

__all__ = ["CUDAGraphDepformerWrapper", "DepformerCUDAGraphStats", "resolve_depformer_graph_settings"]

DEFAULT_DEPFORMER_CUDA_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16)


def resolve_depformer_graph_settings(
    vllm_config: Any,
    *,
    enabled: bool,
    default_sizes: Sequence[int] = DEFAULT_DEPFORMER_CUDA_SIZES,
    default_max_batch: int = 32,
    default_warmup_iters: int = 3,
) -> tuple[bool, tuple[int, ...], int, int]:
    model_config = getattr(vllm_config, "model_config", None)
    enforce_eager = bool(getattr(model_config, "enforce_eager", False))
    enabled = enabled and not enforce_eager
    compilation = getattr(vllm_config, "compilation_config", None)
    raw_sizes = getattr(compilation, "cudagraph_capture_sizes", None)
    warmup = getattr(compilation, "cudagraph_num_of_warmups", None)
    warm_iters = default_warmup_iters if warmup is None else max(warmup, 0)
    sizes = raw_sizes if raw_sizes else default_sizes
    scheduler = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = getattr(scheduler, "max_num_seqs", 0)
    max_batch = max(max(sizes), max_num_seqs, 1)
    if not enabled:
        max_batch = max(max_batch, default_max_batch)
    return enabled, tuple(sizes), max_batch, warm_iters


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


@dataclass
class _CaptureDepformerGraph:
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
    ) -> None:
        sizes = sorted({int(size) for size in capture_sizes if int(size) > 0})
        if not sizes:
            raise ValueError("capture_sizes must contain at least one positive integer")
        self.depformer = depformer
        self.capture_sizes: tuple[int, ...] = tuple(sizes)
        self.enabled = enabled
        self.warmup_iters = warmup_iters
        self._graphs: dict[int, _CaptureDepformerGraph] = {}
        self._stats = DepformerCUDAGraphStats()
        self._warmed_up = False

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

    def _synthetic_kwards(self, padded_b: int, device: torch.device) -> dict[str, torch.Tensor]:
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
    ) -> torch.Tensor:
        out = self.depformer(text_token, transformer_out, audio_tokens, audio_provided)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _capture_one(self, padded_b: int, device: torch.device) -> _CaptureDepformerGraph | None:
        if padded_b > self.depformer.max_graph_batch_size:
            logger.warning(
                "Cannot capture Depformer graph for padded batch size %d > max_graph_batch_size %d",
                padded_b,
                self.depformer.max_graph_batch_size,
            )
            self.stats.capture_failure += 1
            return None
        kwargs = self._synthetic_kwards(padded_b, device)
        try:
            with torch.inference_mode(False), torch.no_grad():
                for _ in range(max(self.warmup_iters, 0)):
                    self.depformer(
                        kwargs["text_token"],
                        kwargs["transformer_out"],
                        kwargs["audio_tokens"],
                        kwargs["audio_provided"],
                    )
                torch.accelerator.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                    static_out = self.depformer(
                        kwargs["text_token"],
                        kwargs["transformer_out"],
                        kwargs["audio_tokens"],
                        kwargs["audio_provided"],
                    )
                torch.accelerator.synchronize(device)
        except Exception as exc:
            logger.warning(
                "Failed to capture Depformer graph for padded batch size %d (%s); using eager execution",
                padded_b,
                exc,
            )
            self.stats.capture_failure += 1
            return None
        if isinstance(static_out, tuple):
            static_out = static_out[0]
        logger.info("Captured Depformer graph for padded batch size %d", padded_b)
        return _CaptureDepformerGraph(
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
            entry = self._capture_one(padded_b, device)
            if entry is not None:
                self._graphs[padded_b] = entry

    def __call__(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor | None = None,
        audio_provided: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.stats.calls += 1
        capturing = False
        if text_token.device.type == "cuda" and torch.cuda.is_available():
            capturing = torch.cuda.is_current_stream_capturing()
        if not self.enabled or text_token.device.type != "cuda" or capturing:
            self.stats.eager += 1
            if capturing:
                self.stats.eager_outer_capture += 1
            return self._eager(text_token, transformer_out, audio_tokens, audio_provided)

        actual_b = text_token.shape[0]
        padded_b = self._select_padded_b(actual_b)
        entry = self._graphs.get(padded_b) if padded_b is not None else None
        if entry is None:
            self.stats.eager += 1
            if padded_b is None:
                self.stats.eager_shape_mismatch += 1
            return self._eager(text_token, transformer_out, audio_tokens, audio_provided)
        if not entry.matches(text_token, transformer_out, audio_tokens, audio_provided):
            self.stats.eager += 1
            self.stats.eager_shape_mismatch += 1
            if not entry._warned:
                logger.warning(
                    "Depformer graph input mismatch for padded batch size %d; "
                    "using eager execution. This warning is only logged once.",
                    padded_b,
                )
                entry._warned = True
            return self._eager(text_token, transformer_out, audio_tokens, audio_provided)

        self.stats.replays += 1
        return entry.replay(text_token, transformer_out, audio_tokens, audio_provided, actual_b)
