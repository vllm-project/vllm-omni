# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Graph wrapper for the higgs_audio_v2 Stage-0 talker AR hot loop.

Mirrors ``qwen3_tts/cuda_graph_decoder_wrapper.py`` but targets the Higgs
DualFFN-aware decoder layers.  Capture is performed at fixed batch sizes
(default ``{1, 2, 4}``); on replay failure or any RuntimeError the wrapper
falls back to eager execution and emits a ``higgs_audio_v2``-named warning.

If the underlying module has not been warmed up yet (e.g. before its first
forward call), the wrapper returns ``WrapperState.NOT_CAPTURED`` so callers
can treat that as eager-fallback and proceed.
"""

from __future__ import annotations

import enum
from typing import Any

import torch
from vllm.logger import init_logger

__all__ = [
    "WrapperState",
    "HiggsAudioV2CUDAGraphWrapper",
]

logger = init_logger(__name__)


class WrapperState(enum.IntEnum):
    NOT_CAPTURED = 0
    CAPTURED = 1
    REPLAY_FAILED = 2


class HiggsAudioV2CUDAGraphWrapper:
    """Wraps a Stage-0 forward call with optional CUDA-graph replay.

    Parameters
    ----------
    talker:
        The Stage-0 talker module. Must expose ``forward(input_ids, positions, **kw)``.
    capture_batch_sizes:
        Batch sizes at which to capture graphs. Defaults to ``(1, 2, 4)`` to
        match the AC-8 positive test (replay must preserve AC-2 parity at
        batch sizes 1, 2, 4).
    enabled:
        Master kill switch. Capture is skipped when False; the wrapper still
        delegates to the underlying module so callers can keep using the same
        interface.
    """

    def __init__(
        self,
        talker: torch.nn.Module,
        *,
        capture_batch_sizes: tuple[int, ...] = (1, 2, 4),
        enabled: bool = True,
    ) -> None:
        self.talker = talker
        self.capture_batch_sizes = tuple(sorted(set(int(b) for b in capture_batch_sizes)))
        self.enabled = enabled

        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._static_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self._static_outputs: dict[int, torch.Tensor] = {}
        self._state: dict[int, WrapperState] = {b: WrapperState.NOT_CAPTURED for b in self.capture_batch_sizes}
        self._warmup_attempted = False

    # ------------------------------------------------------------------ capture
    def warmup(self, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Capture CUDA graphs at each configured batch size.

        Any capture failure for an individual batch size is recorded as
        ``WrapperState.REPLAY_FAILED`` and the wrapper falls back to eager for
        that size at replay time.
        """
        self._warmup_attempted = True
        if not self.enabled or not torch.cuda.is_available():
            return
        for batch_size in self.capture_batch_sizes:
            try:
                self._capture_one(batch_size, sample_inputs)
                self._state[batch_size] = WrapperState.CAPTURED
                logger.info(
                    "higgs_audio_v2: captured CUDA graph at batch_size=%d", batch_size
                )
            except Exception as exc:  # pragma: no cover - depends on talker forward
                logger.warning(
                    "higgs_audio_v2: CUDA-graph capture failed at batch_size=%d (%s); "
                    "falling back to eager for that size",
                    batch_size,
                    exc,
                )
                self._state[batch_size] = WrapperState.REPLAY_FAILED

    def _capture_one(self, batch_size: int, sample_inputs: dict[str, torch.Tensor]) -> None:
        device = sample_inputs[next(iter(sample_inputs))].device
        if device.type != "cuda":
            raise RuntimeError("CUDA-graph capture requires CUDA tensors")

        static_inputs: dict[str, torch.Tensor] = {}
        for name, tensor in sample_inputs.items():
            shape = list(tensor.shape)
            shape[0] = batch_size
            static_inputs[name] = torch.empty(shape, dtype=tensor.dtype, device=device)
            static_inputs[name].copy_(
                tensor[:batch_size].contiguous() if int(tensor.shape[0]) >= batch_size else tensor.expand(
                    batch_size, *tensor.shape[1:]
                )
            )

        torch.cuda.synchronize()
        for _ in range(2):  # warmup forward passes
            with torch.inference_mode():
                _ = self.talker(**static_inputs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.inference_mode():
                static_output = self.talker(**static_inputs)
        self._graphs[batch_size] = graph
        self._static_inputs[batch_size] = static_inputs
        self._static_outputs[batch_size] = static_output

    # ------------------------------------------------------------------ replay
    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.talker(**inputs)
        first = next(iter(inputs.values()))
        batch_size = int(first.shape[0])
        state = self._state.get(batch_size, WrapperState.NOT_CAPTURED)
        if state != WrapperState.CAPTURED:
            return self.talker(**inputs)
        try:
            static_inputs = self._static_inputs[batch_size]
            for name, tensor in inputs.items():
                static_inputs[name].copy_(tensor)
            self._graphs[batch_size].replay()
            return self._static_outputs[batch_size].clone()
        except Exception as exc:  # pragma: no cover - hardware path
            logger.warning(
                "higgs_audio_v2: CUDA-graph replay failed at batch_size=%d (%s); falling back to eager",
                batch_size,
                exc,
            )
            self._state[batch_size] = WrapperState.REPLAY_FAILED
            return self.talker(**inputs)

    # ------------------------------------------------------------------ introspection
    def is_captured(self, batch_size: int) -> bool:
        return self._state.get(batch_size, WrapperState.NOT_CAPTURED) == WrapperState.CAPTURED

    @property
    def warmup_attempted(self) -> bool:
        return self._warmup_attempted
