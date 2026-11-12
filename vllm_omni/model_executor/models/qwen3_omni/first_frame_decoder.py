# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""First codec frame -> PCM inside the Qwen3-Omni Talker process.

Code2Wav decodes each streaming chunk statelessly from its codes and left
context. With a one-frame first chunk (``codec_chunk_ramp`` starting at 1),
chunk 0 is a context-free decode of the stream's first frame, which the Talker
can run as soon as eager MTP completes that frame, removing the connector hop
and the Code2Wav step from time to first audio. The orchestrator then trims
Code2Wav's copy of those samples.

The decoder is a second ``Qwen3OmniMoeCode2Wav`` built and loaded exactly like
the Code2Wav stage's, and captured with the same one-frame input shape as that
stage's ``(batch=1, frames=1)`` graph. Optional cuDNN autotuning can select
different algorithms in the two processes and change floating-point rounding.
Each bucket gets its own graph pool because it replays on a side stream while
the Talker's own graphs may still run.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorWrapper
from vllm_omni.model_executor.models.common.talker_first_audio import supports_talker_first_audio
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_chunk_ramp

logger = init_logger(__name__)


def talker_first_audio_enabled(vllm_config: Any) -> bool:
    """Opt in only when chunk 0 carries the first frame in the same step."""
    extra = CodePredictorWrapper._stage_connector_extra_config(vllm_config)
    ramp = parse_chunk_ramp(extra)
    initial_frames = ramp[0] if ramp is not None else int(extra.get("initial_codec_chunk_frames") or 0)
    return (
        CodePredictorWrapper._parse_bool_config(extra.get("talker_first_audio"))
        and initial_frames == 1
        and os.environ.get("VLLM_OMNI_TALKER_FIRST_AUDIO", "1") == "1"
        and supports_talker_first_audio(vllm_config)
    )


class Qwen3OmniFirstFrameDecoder(nn.Module):
    def __init__(self, code2wav: nn.Module, sample_rate: int) -> None:
        super().__init__()
        self.code2wav = code2wav
        self.num_quantizers = int(code2wav.config.num_quantizers)
        self.sample_rate = int(sample_rate)
        self._graphs: dict[int, tuple[CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint's ``code2wav.*`` weights as the Code2Wav stage does."""
        loaded = self.code2wav.load_weights(weights)
        if hasattr(self.code2wav, "precompute_snake_caches"):
            self.code2wav.precompute_snake_caches()
        return loaded

    @torch.inference_mode()
    def capture(self, batch_sizes: Sequence[int] = (1, 2, 4, 8)) -> None:
        device = next(self.code2wav.parameters()).device
        if device.type != "cuda":
            return
        for batch_size in sorted(set(batch_sizes)):
            static_input = torch.zeros(batch_size, self.num_quantizers, 1, dtype=torch.long, device=device)
            self.code2wav(static_input)
            torch.accelerator.synchronize(device)
            graph = CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                static_output = self.code2wav(static_input)
            self._graphs[batch_size] = (graph, static_input, static_output)
        logger.info("Captured Qwen3-Omni Talker first-frame decoder graphs for batch sizes %s", sorted(self._graphs))

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` [n, num_quantizers] long on device -> float32 PCM [n, samples] (new tensor)."""
        if not self._graphs:
            return self.code2wav(codes.to(torch.long).unsqueeze(-1))[:, 0, :].float()
        outputs = []
        largest = max(self._graphs)
        for start in range(0, int(codes.shape[0]), largest):
            chunk = codes[start : start + largest]
            rows = int(chunk.shape[0])
            batch_size = min(size for size in self._graphs if size >= rows)
            graph, static_input, static_output = self._graphs[batch_size]
            static_input.zero_()
            static_input[:rows, :, 0].copy_(chunk)
            graph.replay()
            # A replay overwrites graph-owned storage; .float() aliases FP32 outputs.
            outputs.append(static_output[:rows, 0, :].to(dtype=torch.float32, copy=True))
        return torch.cat(outputs, dim=0)
