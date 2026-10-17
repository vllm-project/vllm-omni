# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encoder CUDA graph manager that replays a group only when one graph holds it."""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_omni.model_executor.models.qwen3_omni.vision_encoder_cudagraph import MAX_GRAPH_FRAME_PATCHES

logger = init_logger(__name__)


class SingleReplayEncoderCudaGraphManager(EncoderCudaGraphManager):
    """Replay one captured graph for a vision group, repeating it per frame chunk.

    The upstream manager splits a group across several replays and gives each
    item above the largest budget its own eager call. For a launch-bound vision
    tower that is usually slower than one eager call over the whole group, so a
    group of separate items that does not fit one graph goes back to the
    runner. A video whose frames are independent can instead replay that same
    graph on successive frame chunks. Models opt in by setting
    ``encoder_cudagraph_single_replay``.
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
            audio.clear()

    def get_num_graphs_to_capture(self):
        audio = getattr(self.model, "_audio_encoder_graphs", None)
        extra = audio.num_graphs_to_capture() if audio is not None else 0
        return super().get_num_graphs_to_capture() + extra

    def _fits_one_replay(self, mm_kwargs: dict[str, Any], specs) -> bool:
        replay_supported = getattr(self.model, "encoder_cudagraph_replay_supported", None)
        fits = replay_supported is None or replay_supported(mm_kwargs, self.max_frames_per_batch)
        if self.use_dp:
            return fits
        return (
            fits
            and len(specs) <= self.max_batch_size
            and all(
                sum(spec.get_path_output_tokens(path) for spec in specs) <= max(budgets)
                for path, budgets in self.path_token_budgets.items()
            )
        )

    def _chunked_video_kwargs(self, mm_kwargs: dict[str, Any], specs):
        """Split a video batch into frame chunks that each fit the largest graph.

        Qwen3-Omni vision attention and RoPE are per frame, so concatenating
        chunk embeddings matches one forward. Images and frames that cannot be
        split this way return None and stay eager.
        """
        if self.use_dp or self.model.get_input_modality(mm_kwargs) != "video":
            return None
        # The runner sees the stage wrapper, which forwards the encoder protocol
        # but not the thinker's private grid helper.
        grid = mm_kwargs["video_grid_thw"]
        grid = grid if isinstance(grid, list) else grid.tolist()
        if len(grid) != len(specs):
            return None
        max_budget = max(max(budgets) for budgets in self.path_token_budgets.values())
        max_frames = self.max_frames_per_batch
        pixels = mm_kwargs["pixel_values_videos"]
        chunk_grids = []
        owners = []
        pixel_slices = []
        offset = 0
        for index, ((t, h, w), spec) in enumerate(zip(grid, specs)):
            t, h, w = int(t), int(h), int(w)
            if min(t, h, w) <= 0 or h * w > MAX_GRAPH_FRAME_PATCHES or spec.output_tokens % t or spec.input_size % t:
                return None
            frame_tokens = spec.output_tokens // t
            frame_pixels = spec.input_size // t
            if frame_tokens <= 0 or frame_tokens > max_budget:
                return None
            frames_per_replay = max_budget // frame_tokens
            if max_frames:
                frames_per_replay = min(frames_per_replay, max_frames)
            if frames_per_replay <= 0:
                return None
            for start in range(0, t, frames_per_replay):
                count = min(frames_per_replay, t - start)
                begin = offset + start * frame_pixels
                chunk_grids.append([count, h, w])
                owners.append(index)
                pixel_slices.append(pixels[begin : begin + count * frame_pixels])
            offset += spec.input_size
        if len(owners) == len(specs):
            return None
        # Replay one chunk at a time. Packing several chunks into one replay can
        # exceed the captured per-frame cu_seqlens capacity.
        chunk_kwargs = [
            {"pixel_values_videos": pixels_chunk, "video_grid_thw": [grid_chunk]}
            for pixels_chunk, grid_chunk in zip(pixel_slices, chunk_grids)
        ]
        return chunk_kwargs, owners, len(specs)

    def _record_vision_fallback(self, specs) -> None:
        self.graph_misses += len(specs)
        if self.graph_misses <= 3 or self.graph_misses % 100 == 0:
            logger.info(
                "Vision encoder graph fallback: tokens=%d hits=%d misses=%d",
                sum(spec.output_tokens for spec in specs),
                self.graph_hits,
                self.graph_misses,
            )

    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor] | None:  # type: ignore[override]
        specs = self._get_item_specs(mm_kwargs)
        if not self._fits_one_replay(mm_kwargs, specs):
            chunked = self._chunked_video_kwargs(mm_kwargs, specs)
            if chunked is None:
                self._record_vision_fallback(specs)
                return None
            chunk_kwargs, owners, item_count = chunked
            grouped: list[list[torch.Tensor]] = [[] for _ in range(item_count)]
            for kwargs, owner in zip(chunk_kwargs, owners):
                chunk_outputs = EncoderCudaGraphManager.execute(self, kwargs)
                if len(chunk_outputs) != 1:
                    self._record_vision_fallback(specs)
                    return None
                grouped[owner].append(chunk_outputs[0])
            outputs = [torch.cat(parts, dim=0) if len(parts) > 1 else parts[0] for parts in grouped]
        else:
            outputs = super().execute(mm_kwargs)
        if self.graph_hits and (self.graph_hits <= 3 or self.graph_hits % 64 == 0):
            logger.info("Vision encoder graphs: hits=%d misses=%d", self.graph_hits, self.graph_misses)
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
        return self.model._audio_encoder_graphs.num_graphs_to_capture()

    def capture(self, graph_pool):
        self.model.capture_audio_encoder_cudagraph(graph_pool)

    def clear(self):
        self.model._audio_encoder_graphs.clear()
