# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph replay for F5-TTS DiT forward calls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _tp_world_size() -> int:
    """Tensor-parallel size, or 1 when the TP group is not initialized.

    Mirrors the magi2/sana_wm pattern: capture-time collectives inside
    ``RowParallelLinear`` make manual graph capture unsafe under TP > 1, so
    the wrapper must be able to detect it without assuming init order.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    try:
        from vllm.distributed.parallel_state import get_tp_group

        return int(get_tp_group().world_size)
    except (AssertionError, RuntimeError):
        return 1


@dataclass(frozen=True)
class _GraphKey:
    device_index: int
    audio_dtype: torch.dtype
    text_dtype: torch.dtype
    timestep_dtype: torch.dtype
    batch: int
    seq_len: int
    mel_dim: int


@dataclass
class _GraphState:
    noisy_audio: torch.Tensor
    cond_audio: torch.Tensor
    cond_text: torch.Tensor
    timestep: torch.Tensor
    drop_audio_mask: torch.Tensor
    drop_text_mask: torch.Tensor
    output: torch.Tensor
    # Assigned right after successful capture; ``None`` on the pre-capture
    # placeholder so a failed capture leaves no half-built CUDAGraph alive.
    graph: CUDAGraph | None = None


class F5TTSDiTCUDAGraphWrapper:
    """Exact-shape CUDA Graph cache for ``F5TTSDiTModel.forward``.

    Unsupported inputs fall back to eager execution.  This keeps the wrapper
    limited to the F5 DiT hot path; preprocessing, diffusion-loop control, and
    vocoder/post-processing stay eager.
    """

    def __init__(self, transformer: torch.nn.Module, *, max_graphs: int = 8) -> None:
        self.transformer = transformer
        self.max_graphs = int(max_graphs)
        self.graphs: dict[_GraphKey, _GraphState] = {}
        self.disabled_keys: set[_GraphKey] = set()

    @staticmethod
    def parse_bool(value: object, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, int):
            return bool(value)
        return default

    def _run_eager(
        self,
        *,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_mask: torch.Tensor | None = None,
        drop_text_mask: torch.Tensor | None = None,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return self.transformer(
                noisy_audio=noisy_audio,
                cond_audio=cond_audio,
                cond_text=cond_text,
                timestep=timestep,
                mask=mask,
                drop_audio_mask=drop_audio_mask,
                drop_text_mask=drop_text_mask,
                rotary_embedding=rotary_embedding,
            )

    @staticmethod
    def _masks_compatible(mask: torch.Tensor | None, batch: int, device: torch.device) -> bool:
        return mask is None or (mask.dtype == torch.bool and mask.shape == (batch,) and mask.device == device)

    def _can_graph(
        self,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None,
        drop_audio_mask: torch.Tensor | None,
        drop_text_mask: torch.Tensor | None,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> bool:
        if (
            not torch.cuda.is_available()
            or noisy_audio.device.type != "cuda"
            or self.transformer.training
            or torch.cuda.is_current_stream_capturing()
            or mask is not None
            or rotary_embedding is not None
            # RowParallelLinear issues TP collectives inside the forward —
            # manual graph capture without vLLM's capture-aware all-reduce
            # setup can diverge across ranks, so TP > 1 stays eager.
            or _tp_world_size() > 1
            or noisy_audio.ndim != 3
            or cond_audio.shape != noisy_audio.shape
            or cond_text.shape[:2] != noisy_audio.shape[:2]
            or timestep.ndim not in (0, 1)
            or (timestep.ndim == 1 and timestep.shape[0] != noisy_audio.shape[0])
        ):
            return False
        if not self._masks_compatible(drop_audio_mask, noisy_audio.shape[0], noisy_audio.device):
            return False
        if not self._masks_compatible(drop_text_mask, noisy_audio.shape[0], noisy_audio.device):
            return False
        return cond_audio.device == cond_text.device == timestep.device == noisy_audio.device

    def _make_key(
        self,
        noisy_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
    ) -> _GraphKey:
        # _can_graph() already guaranteed a CUDA tensor, whose device index is
        # always concrete; get_device() returns the ordinal without touching
        # torch.cuda global state.
        device_index = noisy_audio.get_device()
        return _GraphKey(
            int(device_index),
            noisy_audio.dtype,
            cond_text.dtype,
            timestep.dtype,
            int(noisy_audio.shape[0]),
            int(noisy_audio.shape[1]),
            int(noisy_audio.shape[2]),
        )

    def _copy_inputs(
        self,
        state: _GraphState,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
        drop_audio_mask: torch.Tensor | None,
        drop_text_mask: torch.Tensor | None,
    ) -> None:
        state.noisy_audio.copy_(noisy_audio)
        state.cond_audio.copy_(cond_audio)
        state.cond_text.copy_(cond_text)
        state.timestep.copy_(timestep.expand_as(state.timestep) if timestep.ndim == 0 else timestep)
        if drop_audio_mask is None:
            state.drop_audio_mask.fill_(False)
        else:
            state.drop_audio_mask.copy_(drop_audio_mask)
        if drop_text_mask is None:
            state.drop_text_mask.fill_(False)
        else:
            state.drop_text_mask.copy_(drop_text_mask)

    def _capture(
        self,
        key: _GraphKey,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
        drop_audio_mask: torch.Tensor | None,
        drop_text_mask: torch.Tensor | None,
    ) -> _GraphState | None:
        if len(self.graphs) >= self.max_graphs:
            logger.warning_once("F5-TTS DiT CUDA Graph max_graphs=%d reached; using eager", self.max_graphs)
            return None

        try:
            batch = noisy_audio.shape[0]
            device = noisy_audio.device
            state = _GraphState(
                noisy_audio=torch.empty_like(noisy_audio),
                cond_audio=torch.empty_like(cond_audio),
                cond_text=torch.empty_like(cond_text),
                timestep=torch.empty((batch,), device=device, dtype=timestep.dtype),
                drop_audio_mask=torch.empty((batch,), device=device, dtype=torch.bool),
                drop_text_mask=torch.empty((batch,), device=device, dtype=torch.bool),
                output=torch.empty_like(noisy_audio),
            )
            self._copy_inputs(state, noisy_audio, cond_audio, cond_text, timestep, drop_audio_mask, drop_text_mask)

            with torch.no_grad():
                # Side-stream warmup (MOSS-TTS codec/decoder pattern): run a
                # few eager iterations off the default stream so lazy model
                # buffers (e.g. RoPE) are initialized and allocator state stays
                # off the capture stream.
                stream = torch.cuda.Stream(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._run_eager(
                            noisy_audio=state.noisy_audio,
                            cond_audio=state.cond_audio,
                            cond_text=state.cond_text,
                            timestep=state.timestep,
                            drop_audio_mask=state.drop_audio_mask,
                            drop_text_mask=state.drop_text_mask,
                        )
                torch.cuda.current_stream(device).wait_stream(stream)
            torch.accelerator.synchronize(device)

            graph = CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(
                graph,
                pool=current_platform.get_global_graph_pool(),
                # Keep a failed capture from poisoning global CUDA state for
                # other capture users (MOSS-TTS / MiniMax-Music3 pattern).
                capture_error_mode="thread_local",
            ):
                static_output = self._run_eager(
                    noisy_audio=state.noisy_audio,
                    cond_audio=state.cond_audio,
                    cond_text=state.cond_text,
                    timestep=state.timestep,
                    drop_audio_mask=state.drop_audio_mask,
                    drop_text_mask=state.drop_text_mask,
                )
        except Exception:
            logger.warning("Failed to capture F5-TTS DiT CUDA Graph; using eager", exc_info=True)
            self.disabled_keys.add(key)
            return None

        state.graph = graph
        state.output = static_output
        self.graphs[key] = state
        return state

    @torch.inference_mode()
    def __call__(
        self,
        *,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_mask: torch.Tensor | None = None,
        drop_text_mask: torch.Tensor | None = None,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self._can_graph(
            noisy_audio, cond_audio, cond_text, timestep, mask, drop_audio_mask, drop_text_mask, rotary_embedding
        ):
            return self._run_eager(
                noisy_audio=noisy_audio,
                cond_audio=cond_audio,
                cond_text=cond_text,
                timestep=timestep,
                mask=mask,
                drop_audio_mask=drop_audio_mask,
                drop_text_mask=drop_text_mask,
                rotary_embedding=rotary_embedding,
            )

        key = self._make_key(noisy_audio, cond_text, timestep)
        state = None if key in self.disabled_keys else self.graphs.get(key)
        if state is None and key not in self.disabled_keys:
            state = self._capture(key, noisy_audio, cond_audio, cond_text, timestep, drop_audio_mask, drop_text_mask)
        if state is None:
            return self._run_eager(
                noisy_audio=noisy_audio,
                cond_audio=cond_audio,
                cond_text=cond_text,
                timestep=timestep,
                mask=mask,
                drop_audio_mask=drop_audio_mask,
                drop_text_mask=drop_text_mask,
                rotary_embedding=rotary_embedding,
            )

        try:
            self._copy_inputs(state, noisy_audio, cond_audio, cond_text, timestep, drop_audio_mask, drop_text_mask)
        except Exception:
            logger.warning("F5-TTS DiT CUDA Graph input copy failed; using eager", exc_info=True)
            self.disabled_keys.add(key)
            return self._run_eager(
                noisy_audio=noisy_audio,
                cond_audio=cond_audio,
                cond_text=cond_text,
                timestep=timestep,
                mask=mask,
                drop_audio_mask=drop_audio_mask,
                drop_text_mask=drop_text_mask,
                rotary_embedding=rotary_embedding,
            )
        state.graph.replay()
        # The static output buffer is reused by the next replay; clone so the
        # caller may hold the tensor across subsequent diffusion steps.
        return state.output.clone()
