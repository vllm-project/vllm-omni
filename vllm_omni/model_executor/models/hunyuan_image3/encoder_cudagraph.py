# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import torch
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)


def _pad_cu_seqlens(buffer: torch.Tensor, source: torch.Tensor) -> None:
    capacity = buffer[-1].clone()
    buffer.copy_(source[-1].expand_as(buffer))
    buffer[: source.numel()].copy_(source)
    buffer[-1].copy_(capacity)


class _VisionEncoder(SupportsEncoderCudaGraph):
    def __init__(self, model):
        self.model = model
        self.min_tokens = model.config.vit_processor["max_num_patches"]

    def get_input_modality(self, mm_kwargs):
        return "image"

    def get_max_frames_per_video(self):
        return 1

    def get_encoder_cudagraph_budget_range(self, vllm_config):
        return self.min_tokens, max(self.min_tokens, vllm_config.scheduler_config.max_num_batched_tokens)

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        return {key: value[indices] for key, value in mm_kwargs.items()}

    def get_encoder_cudagraph_config(self):
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixels", "positions", "cu_seqlens", "indices", "max_seqlen"],
            out_hidden_size=self.model.config.hidden_size,
            padding_logics={"cu_seqlens": _pad_cu_seqlens},
        )

    def get_encoder_cudagraph_item_specs(self, mm_kwargs):
        batch, patches = mm_kwargs["mask"].shape
        return [EncoderItemSpec(patches, patches) for _ in range(batch)]

    def prepare_encoder_cudagraph_capture_inputs(
        self, token_budget, max_batch_size, max_frames_per_batch, device, dtype, path="default", axis_keys=None
    ):
        model = self.model.vision_model
        patches = self.min_tokens
        batch = min(max_batch_size, token_budget // patches)
        capacity = batch * patches
        return EncoderCudaGraphCaptureInputs(
            values={
                "pixels": torch.zeros(
                    capacity, model.embeddings.patch_embedding.in_features, device=device, dtype=dtype
                ),
                "positions": torch.zeros(capacity, model.embed_dim, device=device, dtype=dtype),
                "cu_seqlens": torch.tensor([0] * (batch + 1) + [capacity], device=device, dtype=torch.int32),
                "indices": torch.zeros(batch, patches, device=device, dtype=torch.long),
                "max_seqlen": torch.tensor(capacity, dtype=torch.int32),
            }
        )

    def prepare_encoder_cudagraph_replay_buffers(self, mm_kwargs, max_batch_size, max_frames_per_batch, path="default"):
        pixels, mask, shapes = (mm_kwargs[key] for key in ("pixels", "mask", "shapes"))
        mask = mask.bool()
        lengths = shapes.prod(-1).to(device=pixels.device, dtype=torch.int32)
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
        indices = mask.flatten().long().cumsum(0).reshape_as(mask) * mask
        return EncoderCudaGraphReplayBuffers(
            values={
                "pixels": pixels[mask].to(self.model.vision_model.embeddings.patch_embedding.weight.dtype),
                "positions": self.model.vision_model.embeddings.interpolate_pos_encoding(shapes),
                "cu_seqlens": cu_seqlens,
                "indices": indices,
            }
        )

    def encoder_cudagraph_forward(self, inputs, path="default"):
        hidden = self.model.vision_model.forward_packed(
            inputs["pixels"], inputs["positions"], inputs["cu_seqlens"], inputs["max_seqlen"]
        )
        hidden = torch.cat([hidden.new_zeros(1, hidden.shape[-1]), hidden])
        padded = hidden[inputs["indices"]]
        return self.model.vision_aligner(padded).flatten(0, 1)

    def encoder_eager_forward(self, mm_kwargs, path="default"):
        hidden = self.model.vision_model(mm_kwargs["pixels"], mm_kwargs["mask"], mm_kwargs["shapes"])
        return self.model.vision_aligner(hidden).flatten(0, 1)


class _BudgetedEncoderCudaGraphManager(EncoderCudaGraphManager):
    """Capture only budgets selected by vLLM's fixed-size SigLIP2 packing."""

    def __init__(self, vllm_config, device, dtype, model):
        super().__init__(vllm_config, device, dtype, model)
        self.token_budgets = sorted(set(self.token_budgets))
        self.path_token_budgets = {path: sorted(set(budgets)) for path, budgets in self.path_token_budgets.items()}

    def _capture_budgets(self, budgets: list[int]) -> set[int]:
        tokens = self.model.min_tokens
        max_items = min(self.max_batch_size, max(budgets) // tokens)
        if max_items == 0:
            return set()
        single_budget = self._find_smallest_fitting_budget_given_tokens(tokens, budgets)
        assert single_budget is not None
        selected = {single_budget}
        current_budget = single_budget
        for num_items in range(2, max_items + 1):
            merged_budget = self._find_smallest_fitting_budget_given_tokens(num_items * tokens, budgets)
            assert merged_budget is not None
            # Match vLLM 0.32's should_defer_item. Every SigLIP2 item has
            # the same padded size, so once a batch closes, later batches
            # repeat it; no larger batch can be selected.
            if current_budget + single_budget < merged_budget and single_budget >= current_budget:
                break
            selected.add(merged_budget)
            current_budget = merged_budget
        return selected

    def _capture_budget_graph(self, token_budget, path="default", axis_keys=()):
        if token_budget not in self._capture_budgets(self.path_token_budgets[path]):
            return
        super()._capture_budget_graph(token_budget, path=path, axis_keys=axis_keys)

    def _num_graphs_for_budgets(self, budgets):
        return len(self._capture_budgets(budgets))


class HunyuanImage3EncoderCudaGraphManager:
    """Replay SigLIP2 graphs and keep VAE encoding on the model's eager path."""

    def __init__(self, vllm_config, device, dtype, model):
        self.model = model
        self.vision = _BudgetedEncoderCudaGraphManager(vllm_config, device, dtype, _VisionEncoder(model))
        self.token_budgets = self.vision.token_budgets

    def supports_modality(self, modality):
        return modality == "image"

    def get_num_graphs_to_capture(self):
        return self.vision.get_num_graphs_to_capture()

    def capture(self, graph_pool):
        self.vision.capture(graph_pool)

    def clear(self):
        self.vision.clear()

    def is_captured(self):
        return self.vision.is_captured()

    def get_cumulative_stats(self):
        stats = self.vision.get_cumulative_stats()
        return {**stats, "encoders": {"vision": stats}}

    @torch.inference_mode()
    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        parsed = self.model._parse_and_validate_image_input(**mm_kwargs)
        if parsed is None:
            return []
        images = parsed["pixel_values"]
        vision_inputs = {
            "pixels": images["vit_pixel_values"],
            "mask": images["vit_pixel_attention_mask"],
            "shapes": images["vit_spatial_shapes"],
        }
        if vision_inputs["pixels"].shape[1] == self.vision.model.min_tokens:
            vision = torch.stack(self.vision.execute(vision_inputs))
        else:
            self.vision.graph_misses += len(vision_inputs["pixels"])
            vision = self.model._vit_encode(*vision_inputs.values())

        vae_tokens = self.model._encode_vae_tokens(images["vae_pixel_values"], images.get("vae_generator_seed"))
        return self.model._combine_image_embeddings(vision, vae_tokens)
