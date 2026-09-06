# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#

"""Model-local waveform decode and semantic feedback for one audio token.

The kernel is intentionally stateless: causal Acoustic Decoder and Semantic
Encoder caches are supplied by the caller and returned in the result. Request
ownership, dynamic-batch cache packing, cleanup, and waveform serving belong to
the stateful inference module rather than this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class _ReplayableGraph(Protocol):
    def replay(self) -> None: ...


@dataclass(slots=True)
class VibeVoiceAudioTokenDecodeOutput:
    """Outputs needed by the next AR step and future request state."""

    audio: torch.Tensor
    semantic_latent: torch.Tensor
    next_embedding: torch.Tensor
    acoustic_cache: Any
    semantic_cache: Any


@dataclass(frozen=True, slots=True)
class VibeVoiceAudioTokenDecoder:
    """Immutable shape/config view for VibeVoice audio-token decoding."""

    latent_size: int
    semantic_size: int
    condition_size: int
    audio_channels: int
    samples_per_token: int

    def __post_init__(self) -> None:
        for name in (
            "latent_size",
            "semantic_size",
            "condition_size",
            "audio_channels",
            "samples_per_token",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"VibeVoice {name} must be positive.")

    @classmethod
    def from_model_config(cls, config: Any) -> VibeVoiceAudioTokenDecoder:
        decoder_config = config.audio_config.decoder_config
        upsampling_ratios = tuple(decoder_config.upsampling_ratios)
        if not upsampling_ratios:
            raise ValueError("VibeVoice Acoustic Decoder upsampling_ratios cannot be empty.")
        return cls(
            latent_size=int(config.audio_config.hidden_size),
            semantic_size=int(config.semantic_model_config.hidden_size),
            condition_size=int(config.hidden_size),
            audio_channels=int(decoder_config.channels),
            samples_per_token=math.prod(int(ratio) for ratio in upsampling_ratios),
        )

    def _validate_latent(self, audio_latent: torch.Tensor) -> int:
        if audio_latent.ndim != 3:
            raise ValueError(
                "VibeVoice audio_latent must have shape "
                f"(batch, 1, {self.latent_size}), got "
                f"{tuple(audio_latent.shape)}."
            )
        if audio_latent.shape[0] < 1:
            raise ValueError("VibeVoice audio_latent batch cannot be empty.")
        if audio_latent.shape[1:] != (1, self.latent_size):
            raise ValueError(
                "VibeVoice audio_latent must have shape "
                f"(batch, 1, {self.latent_size}), got "
                f"{tuple(audio_latent.shape)}."
            )
        if not audio_latent.is_floating_point():
            raise TypeError("VibeVoice audio_latent must be a floating-point tensor.")
        return audio_latent.shape[0]

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters(), None)
        if parameter is None:
            raise ValueError(f"VibeVoice decode module {type(module).__name__} has no parameters.")
        return parameter.device, parameter.dtype

    @staticmethod
    def _validate_factor(name: str, factor: torch.Tensor) -> None:
        if factor.numel() != 1 or not factor.is_floating_point():
            raise ValueError(f"VibeVoice {name} must be one floating-point scalar tensor.")

    @torch.inference_mode()
    def decode_audio_token(
        self,
        *,
        audio_tower: nn.Module,
        semantic_encoder: nn.Module,
        acoustic_projector: nn.Module,
        semantic_connector: nn.Module,
        latent_scaling_factor: torch.Tensor,
        latent_bias_factor: torch.Tensor,
        audio_latent: torch.Tensor,
        acoustic_cache: Any = None,
        semantic_cache: Any = None,
    ) -> VibeVoiceAudioTokenDecodeOutput:
        """Decode one latent token and build the next Qwen input embedding."""
        batch_size = self._validate_latent(audio_latent)
        self._validate_factor("latent_scaling_factor", latent_scaling_factor)
        self._validate_factor("latent_bias_factor", latent_bias_factor)

        tower_device, tower_dtype = self._module_device_dtype(audio_tower)
        audio_latent = audio_latent.to(device=tower_device, dtype=tower_dtype)
        decoder_latent = audio_latent / latent_scaling_factor.to(audio_latent) - latent_bias_factor.to(audio_latent)
        decoder_output = audio_tower.decode(
            decoder_latent,
            padding_cache=acoustic_cache,
            use_cache=True,
        )
        audio = getattr(decoder_output, "audio", None)
        next_acoustic_cache = getattr(decoder_output, "padding_cache", None)
        expected_audio_shape = (
            batch_size,
            self.audio_channels,
            self.samples_per_token,
        )
        if not isinstance(audio, torch.Tensor) or tuple(audio.shape) != expected_audio_shape:
            actual_shape = tuple(audio.shape) if isinstance(audio, torch.Tensor) else None
            raise ValueError(
                f"VibeVoice Acoustic Decoder output must have shape {expected_audio_shape}, got {actual_shape}."
            )
        if next_acoustic_cache is None:
            raise ValueError("VibeVoice Acoustic Decoder did not return a causal padding cache.")

        semantic_device, semantic_dtype = self._module_device_dtype(semantic_encoder)
        semantic_output = semantic_encoder(
            audio.to(device=semantic_device, dtype=semantic_dtype),
            padding_cache=semantic_cache,
            use_cache=True,
        )
        semantic_latent = getattr(semantic_output, "latents", None)
        next_semantic_cache = getattr(semantic_output, "padding_cache", None)
        expected_semantic_shape = (batch_size, 1, self.semantic_size)
        if not isinstance(semantic_latent, torch.Tensor) or tuple(semantic_latent.shape) != expected_semantic_shape:
            actual_shape = tuple(semantic_latent.shape) if isinstance(semantic_latent, torch.Tensor) else None
            raise ValueError(
                f"VibeVoice Semantic Encoder output must have shape {expected_semantic_shape}, got {actual_shape}."
            )
        if next_semantic_cache is None:
            raise ValueError("VibeVoice Semantic Encoder did not return a causal padding cache.")

        acoustic_device, acoustic_dtype = self._module_device_dtype(acoustic_projector)
        acoustic_embedding = acoustic_projector(audio_latent.to(device=acoustic_device, dtype=acoustic_dtype))
        semantic_connector_device, semantic_connector_dtype = self._module_device_dtype(semantic_connector)
        semantic_embedding = semantic_connector(
            semantic_latent.to(
                device=semantic_connector_device,
                dtype=semantic_connector_dtype,
            )
        ).to(acoustic_embedding)
        expected_embedding_shape = (batch_size, 1, self.condition_size)
        if tuple(acoustic_embedding.shape) != expected_embedding_shape:
            raise ValueError(
                "VibeVoice acoustic projector output must have shape "
                f"{expected_embedding_shape}, got "
                f"{tuple(acoustic_embedding.shape)}."
            )
        if tuple(semantic_embedding.shape) != expected_embedding_shape:
            raise ValueError(
                "VibeVoice semantic connector output must have shape "
                f"{expected_embedding_shape}, got "
                f"{tuple(semantic_embedding.shape)}."
            )

        return VibeVoiceAudioTokenDecodeOutput(
            audio=audio,
            semantic_latent=semantic_latent,
            next_embedding=acoustic_embedding + semantic_embedding,
            acoustic_cache=next_acoustic_cache,
            semantic_cache=next_semantic_cache,
        )


class _DecodeGraphEntry:
    """Static I/O buffers and the captured graph for one request's decode loop."""

    def __init__(self, latent_in: torch.Tensor) -> None:
        self.latent_in = latent_in
        self.graph: torch.cuda.CUDAGraph | None = None
        self.audio_out: torch.Tensor | None = None
        self.semantic_latent: torch.Tensor | None = None
        self.next_embedding: torch.Tensor | None = None

    def replay_outputs(self) -> tuple[_ReplayableGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fully initialized capture state or fail on internal corruption."""
        if self.graph is None or self.audio_out is None or self.semantic_latent is None or self.next_embedding is None:
            raise RuntimeError("VibeVoice decode graph entry is incomplete.")
        return self.graph, self.audio_out, self.semantic_latent, self.next_embedding


class VibeVoiceDecodeGraphExecutor:
    """Manual CUDA-graph replay of ``decode_audio_token`` for one request.

    The decode path is stateful: every
    causal Conv1d layer updates its padding cache in place, so the cache is
    both an input and an output of the graph. Capture/warmup consume cache
    state; a save/restore protocol (clone before, copy_ after) keeps the
    cache at the correct token boundary so replay advances it exactly like
    eager. Replay is bitwise identical to eager (same kernels, same order).

    The graph is per-request: it is captured against the request's own
    acoustic/semantic cache buffer addresses and stored on the acoustic cache
    object (``_vv_decode_graph``) so it lives exactly as long as the cache.
    Each request graph uses an independent CUDA graph memory pool. Sharing one
    pool across request graphs is unsafe because continuous batching can replay
    and destroy them in a different order from capture. Segment boundaries zero
    the caches in place (``_reset_conv_caches``), keeping addresses stable so
    the graph stays valid across segments.

    Replay output tensors are borrowed views of graph-owned static buffers and
    remain valid only until the next replay for the same graph entry. Callers
    that retain audio, semantic latents, or embeddings across replays must
    clone them first. Request caches are likewise mutable request-owned state.
    """

    def __init__(
        self,
        decoder: VibeVoiceAudioTokenDecoder,
        *,
        capture_failure_fatal: bool = False,
    ) -> None:
        self._decoder = decoder
        self._disabled = False
        self._capture_failure_fatal = capture_failure_fatal

    def decode(
        self,
        *,
        audio_tower: nn.Module,
        semantic_encoder: nn.Module,
        acoustic_projector: nn.Module,
        semantic_connector: nn.Module,
        latent_scaling_factor: torch.Tensor,
        latent_bias_factor: torch.Tensor,
        audio_latent: torch.Tensor,
        acoustic_cache: Any,
        semantic_cache: Any,
    ) -> VibeVoiceAudioTokenDecodeOutput | None:
        """Replay the captured decode graph, or None to fall back to eager."""
        if self._disabled:
            if self._capture_failure_fatal:
                raise RuntimeError("Required VibeVoice decode CUDA graph is disabled after a prior capture failure.")
            return None
        if not audio_latent.is_cuda or acoustic_cache is None or semantic_cache is None:
            return None
        entry = getattr(acoustic_cache, "_vv_decode_graph", None)
        if entry is None:
            try:
                entry = self._capture(
                    audio_tower=audio_tower,
                    semantic_encoder=semantic_encoder,
                    acoustic_projector=acoustic_projector,
                    semantic_connector=semantic_connector,
                    latent_scaling_factor=latent_scaling_factor,
                    latent_bias_factor=latent_bias_factor,
                    audio_latent=audio_latent,
                    acoustic_cache=acoustic_cache,
                    semantic_cache=semantic_cache,
                )
            except Exception as exc:
                self._disabled = True
                if self._capture_failure_fatal:
                    raise RuntimeError("Required VibeVoice decode CUDA-graph capture failed.") from exc
                logger.warning(
                    "VibeVoice decode CUDA-graph capture failed; falling back to eager decode permanently.",
                    exc_info=True,
                )
                return None
            acoustic_cache._vv_decode_graph = entry
            logger.info_once("Captured VibeVoice decode CUDA graph for a request cache.")
        entry.latent_in.copy_(audio_latent)
        graph, audio_out, semantic_latent, next_embedding = entry.replay_outputs()
        graph.replay()
        return VibeVoiceAudioTokenDecodeOutput(
            audio=audio_out,
            semantic_latent=semantic_latent,
            next_embedding=next_embedding,
            acoustic_cache=acoustic_cache,
            semantic_cache=semantic_cache,
        )

    def _capture(
        self,
        *,
        audio_tower: nn.Module,
        semantic_encoder: nn.Module,
        acoustic_projector: nn.Module,
        semantic_connector: nn.Module,
        latent_scaling_factor: torch.Tensor,
        latent_bias_factor: torch.Tensor,
        audio_latent: torch.Tensor,
        acoustic_cache: Any,
        semantic_cache: Any,
    ) -> _DecodeGraphEntry:
        device = audio_latent.device
        latent_in = audio_latent.clone()
        entry = _DecodeGraphEntry(latent_in)

        def run() -> VibeVoiceAudioTokenDecodeOutput:
            return self._decoder.decode_audio_token(
                audio_tower=audio_tower,
                semantic_encoder=semantic_encoder,
                acoustic_projector=acoustic_projector,
                semantic_connector=semantic_connector,
                latent_scaling_factor=latent_scaling_factor,
                latent_bias_factor=latent_bias_factor,
                audio_latent=latent_in,
                acoustic_cache=acoustic_cache,
                semantic_cache=semantic_cache,
            )

        with torch.inference_mode():
            snap = self._snapshot(acoustic_cache, semantic_cache)
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                run()
            torch.cuda.current_stream(device).wait_stream(side)
            self._restore(snap, acoustic_cache, semantic_cache)

            # The default creates a private pool for this graph. Do not pass a
            # shared pool: request graphs are replayed and destroyed in dynamic
            # continuous-batching order, which violates shared-pool ordering.
            graph = torch.cuda.CUDAGraph()
            snap2 = self._snapshot(acoustic_cache, semantic_cache)
            with torch.cuda.graph(graph):
                out = run()
            self._restore(snap2, acoustic_cache, semantic_cache)

        entry.graph = graph
        entry.audio_out = out.audio
        entry.semantic_latent = out.semantic_latent
        entry.next_embedding = out.next_embedding
        return entry

    @staticmethod
    def _snapshot(
        acoustic_cache: Any,
        semantic_cache: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        snaps: list[list[torch.Tensor]] = []
        for cache in (acoustic_cache, semantic_cache):
            layers = getattr(cache, "layers", None)
            snaps.append([layer.cache.clone() for layer in layers.values()] if isinstance(layers, dict) else [])
        return snaps[0], snaps[1]

    @staticmethod
    def _restore(
        snap: tuple[list[torch.Tensor], list[torch.Tensor]],
        acoustic_cache: Any,
        semantic_cache: Any,
    ) -> None:
        for cache, layers_snap in zip((acoustic_cache, semantic_cache), snap, strict=True):
            layers = getattr(cache, "layers", None)
            if not isinstance(layers, dict):
                continue
            for layer, saved in zip(layers.values(), layers_snap, strict=True):
                if getattr(layer, "is_initialized", False) and layer.cache is not None:
                    layer.cache.copy_(saved)


__all__ = [
    "VibeVoiceAudioTokenDecodeOutput",
    "VibeVoiceAudioTokenDecoder",
    "VibeVoiceDecodeGraphExecutor",
]
