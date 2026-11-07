# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encoder CUDA graph manager that replays a group only when one graph holds it."""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

from vllm_omni.model_executor.models.qwen3_omni.vision_encoder_cudagraph import MAX_GRAPH_FRAME_PATCHES

logger = init_logger(__name__)


class SingleReplayEncoderCudaGraphManager(EncoderCudaGraphManager):
    """Replay an encoder group only when one captured graph holds all of it.

    The upstream manager splits a group across several replays and gives each
    item above the largest budget its own eager call. For a launch-bound vision
    tower that is usually slower than one eager call over the whole group, so a
    group that does not fit one graph goes back to the runner, which then
    encodes it with a single ``embed_multimodal`` call. A group that fits is
    replayed here once per path rather than through the upstream packing. A
    video whose frames are independent replays that same graph on successive
    frame chunks. Models opt in by setting ``encoder_cudagraph_single_replay``.
    """

    def __init__(self, vllm_config, device: torch.device, dtype: torch.dtype, model):
        super().__init__(vllm_config, device, dtype, model)
        # Upstream regenerates a path that has its own minimum budget from the
        # maximum of an explicit encoder_cudagraph_token_budgets only; a model
        # may derive that path's budgets from the whole list instead.
        user_budgets = list(vllm_config.compilation_config.encoder_cudagraph_token_budgets or [])
        path_budgets = getattr(model, "get_encoder_cudagraph_path_budgets", None)
        if user_budgets and path_budgets is not None:
            for path, budgets in self.path_token_budgets.items():
                derived = path_budgets(path, user_budgets)
                if derived is None:
                    continue
                if not derived:
                    raise ValueError(f"Encoder CUDA graph path {path!r} has no budget in {user_budgets}")
                self.path_token_budgets[path] = ([0] if budgets and budgets[0] == 0 else []) + derived
        # A model may cap one path's budgets below the shared maximum, which
        # explicit encoder_cudagraph_token_budgets otherwise raise for every path.
        path_max_budget = getattr(model, "get_encoder_cudagraph_path_max_budget", None)
        if path_max_budget is not None:
            for path, budgets in self.path_token_budgets.items():
                cap = path_max_budget(path)
                if cap is None:
                    continue
                capped = [budget for budget in budgets if budget <= cap]
                if not any(capped):
                    raise ValueError(f"Encoder CUDA graph path {path!r} has no budget at or below its cap {cap}")
                self.path_token_budgets[path] = capped
            logger.info("Encoder CUDA graph budgets after per-path caps: %s", self.path_token_budgets)

    def _fits_one_replay(self, mm_kwargs: dict[str, Any], specs: list[EncoderItemSpec]) -> bool:
        replay_supported = getattr(self.model, "encoder_cudagraph_replay_supported", None)
        if replay_supported is not None and not replay_supported(mm_kwargs, self.max_frames_per_batch):
            return False
        return len(specs) <= self.max_batch_size and all(
            sum(spec.get_path_output_tokens(path) for spec in specs) <= max(budgets)
            for path, budgets in self.path_token_budgets.items()
        )

    def _chunked_video_kwargs(self, mm_kwargs: dict[str, Any], specs: list[EncoderItemSpec]):
        """Split a video batch into frame chunks that each fit the largest graph.

        Qwen3-Omni vision attention and RoPE are per frame, so concatenating
        chunk embeddings matches one forward. Images and frames that cannot be
        split this way return None and stay eager.
        """
        if self.use_dp or self.model.get_input_modality(mm_kwargs) != "video":
            return None
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
        chunk_kwargs = [
            {"pixel_values_videos": pixels_chunk, "video_grid_thw": [grid_chunk]}
            for pixels_chunk, grid_chunk in zip(pixel_slices, chunk_grids)
        ]
        return chunk_kwargs, owners, len(specs)

    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor] | None:  # type: ignore[override]
        # Models may keep a group eager when a graph would not reproduce it.
        specs = self._get_item_specs(mm_kwargs)
        group_eligible = getattr(self.model, "encoder_cudagraph_group_eligible", None)
        if group_eligible is not None and not group_eligible(mm_kwargs):
            self.graph_misses += len(specs)
            return None
        if self.use_dp:
            return super().execute(mm_kwargs)
        if self._fits_one_replay(mm_kwargs, specs):
            return self._replay_group(mm_kwargs, specs)
        chunked = self._chunked_video_kwargs(mm_kwargs, specs)
        if chunked is None:
            self.graph_misses += len(specs)
            return None
        chunk_kwargs, owners, item_count = chunked
        grouped: list[list[torch.Tensor]] = [[] for _ in range(item_count)]
        for kwargs, owner in zip(chunk_kwargs, owners):
            chunk_specs = self._get_item_specs(kwargs)
            chunk_outputs = self._replay_group(kwargs, chunk_specs)
            if chunk_outputs is None or len(chunk_outputs) != 1:
                self.graph_misses += len(specs)
                return None
            grouped[owner].append(chunk_outputs[0])
        return [torch.cat(parts, dim=0) if len(parts) > 1 else parts[0] for parts in grouped]

    def _replay_group(self, mm_kwargs: dict[str, Any], specs: list[EncoderItemSpec]) -> list[torch.Tensor] | None:
        # Upstream packing may still split a group that one graph holds (vLLM
        # 0.32 defers an item whose own budget outweighs the current batch), so
        # replay the whole group once per path here.
        if not specs:
            return []
        indices = list(range(len(specs)))
        # Without capture axes the whole group in its original order already is
        # the selected batch, so skip slicing and re-concatenating every item.
        if self._capture_axes:
            batch_mm_kwargs, axis_keys = self._select_items(mm_kwargs, indices)
        else:
            batch_mm_kwargs, axis_keys = mm_kwargs, ()
        outputs: dict[str, torch.Tensor] = {}
        for path, budgets in self.path_token_budgets.items():
            tokens = sum(spec.get_path_output_tokens(path) for spec in specs)
            if tokens == 0:
                continue
            budget = self._find_smallest_fitting_budget_given_tokens(tokens, budgets)
            assert budget is not None  # execute() checked that the group fits
            output = self._run_budget_graph(batch_mm_kwargs, budget, path=path, axis_keys=axis_keys)
            if output is None:
                return None
            outputs[path] = output
        if not outputs:
            self.graph_misses += len(specs)
        dest: dict[int, torch.Tensor] = {}
        self.model.postprocess_encoder_output(
            outputs,
            indices,
            [spec.output_tokens for spec in specs],
            dest,
            clone=True,
            batch_mm_kwargs=batch_mm_kwargs,
        )
        total = self.graph_hits + self.graph_misses
        if total and total % self.log_stats_interval == 0:
            stats = self.get_cumulative_stats()
            logger.debug(
                "Encoder CUDA graph cumulative stats: hits=%d, misses=%d, hit_rate=%.1f%%",
                stats["graph_hits"],
                stats["graph_misses"],
                stats["hit_rate"] * 100,
            )
        return [dest[i] for i in indices]
