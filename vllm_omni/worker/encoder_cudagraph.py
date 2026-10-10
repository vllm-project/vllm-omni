# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encoder CUDA graph manager that replays a group only when one graph holds it."""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

logger = init_logger(__name__)


class SingleReplayEncoderCudaGraphManager(EncoderCudaGraphManager):
    """Replay an encoder group only when one captured graph holds all of it.

    The upstream manager splits a group across several replays and gives each
    item above the largest budget its own eager call. For a launch-bound vision
    tower that is usually slower than one eager call over the whole group, so a
    group that does not fit one graph goes back to the runner, which then
    encodes it with a single ``embed_multimodal`` call. Models opt in by
    setting ``encoder_cudagraph_single_replay``.
    """

    def capture(self, graph_pool: Any):
        super().capture(graph_pool)
        capture_audio = getattr(self.model, "capture_audio_encoder_cudagraph", None)
        if capture_audio is not None:
            capture_audio(graph_pool)

    def clear(self):
        super().clear()
        audio = getattr(self.model, "_audio_encoder_graphs", None)
        if audio is not None:
            audio.graphs.clear()

    def get_num_graphs_to_capture(self):
        audio = getattr(self.model, "_audio_encoder_graphs", None)
        return super().get_num_graphs_to_capture() + (len(audio.capture_shapes) if audio is not None else 0)

    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor] | None:  # type: ignore[override]
        specs = self._get_item_specs(mm_kwargs)
        replay_supported = getattr(self.model, "encoder_cudagraph_replay_supported", None)
        fits = replay_supported is None or replay_supported(mm_kwargs, self.max_frames_per_batch)
        if not self.use_dp:
            fits = (
                fits
                and len(specs) <= self.max_batch_size
                and all(
                    sum(spec.get_path_output_tokens(path) for spec in specs) <= max(budgets)
                    for path, budgets in self.path_token_budgets.items()
                )
            )
        if not fits:
            self.graph_misses += len(specs)
            if self.graph_misses <= 3 or self.graph_misses % 100 == 0:
                logger.info(
                    "Vision encoder graph fallback: tokens=%d hits=%d misses=%d",
                    sum(spec.output_tokens for spec in specs),
                    self.graph_hits,
                    self.graph_misses,
                )
            return None
        outputs = super().execute(mm_kwargs)
        if self.graph_hits and (self.graph_hits <= 3 or self.graph_hits % 64 == 0):
            logger.info("Vision encoder CUDA graphs: hits=%d misses=%d", self.graph_hits, self.graph_misses)
        return outputs


class AudioOnlyEncoderCudaGraphManager:
    """Runner capture lifecycle for Qwen3-Omni with vision disabled."""

    def __init__(self, model):
        self.model = model
        self.token_budgets = [1]

    def supports_modality(self, modality):
        # Audio execution stays in embed_multimodal so audio/video pairing and
        # output length bookkeeping continue to use the model's audio path.
        return False

    def get_num_graphs_to_capture(self):
        return len(self.model._audio_encoder_graphs.capture_shapes)

    def capture(self, graph_pool):
        self.model.capture_audio_encoder_cudagraph(graph_pool)

    def clear(self):
        self.model._audio_encoder_graphs.graphs.clear()
