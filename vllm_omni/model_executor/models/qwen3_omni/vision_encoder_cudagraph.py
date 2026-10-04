# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni image encoder contract for the upstream CUDA graph manager."""

from collections.abc import Hashable
from typing import Any

import torch
from vllm.distributed import get_pp_group
from vllm.model_executor.models.utils import scatter_output_slices
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)


def _pad_cu_seqlens(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Unused attention sequences must be empty, not run backwards to zero.
    dst.copy_(src[-1].expand_as(dst))
    dst[: src.shape[0]].copy_(src)


class Qwen3OmniVisionEncoderCudaGraphMixin:
    """Image-only protocol methods; video and audio retain embed_multimodal.

    The capability attribute is set on eligible model instances after tower
    construction. It is deliberately absent from this mixin's class: the
    runtime protocol checks attribute presence, not the value of a false flag.
    """

    # The eager tower is launch-bound, so a group that needs several replays
    # or per-image eager calls runs as one eager call instead; see
    # SingleReplayEncoderCudaGraphManager.
    encoder_cudagraph_single_replay = True

    def _enable_image_encoder_cudagraph(self) -> None:
        # Only the first pipeline rank runs the encoder. Dynamic-scale FP8 ViT
        # attention advances a host-side amax slot per call, which a graph
        # would freeze.
        if (
            not self.multimodal_config.enable_mm_embeds
            and self.multimodal_config.mm_encoder_attn_dtype is None
            and self.multimodal_config.get_limit_per_prompt("image") > 0
            and get_pp_group().is_first_rank
            and self.visual.attn_backend
            in {
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.ROCM_AITER_FA,
                AttentionBackendEnum.TRITON_ATTN,
            }
        ):
            self.supports_encoder_cudagraph = True

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        keys = ["pixel_values", "rotary_pos_emb_cos", "rotary_pos_emb_sin", "cu_seqlens", "max_seqlen"]
        if self.visual.apply_vit_abs_pos_embed:
            keys.append("pos_embeds")
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=keys,
            out_hidden_size=self.visual_dim + self.multiscale_dim,
            padding_logics={"cu_seqlens": _pad_cu_seqlens},
        )

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def get_encoder_cudagraph_budget_range(self, vllm_config) -> tuple[int, int]:
        # Replay pays off while the eager tower is launch-bound: on an A800 a
        # 196-token image replays 32% faster, while 280-400 tokens replay
        # 4-10% slower and 560-784 tokens padded to the 1024 budget 12-51%
        # slower. Every budget also adds captured graph memory. Auto-inferred
        # budgets therefore stop at 256 tokens; larger images run eagerly
        # unless encoder_cudagraph_token_budgets asks for more.
        maximum = min(vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.model_config.max_model_len, 256)
        return min(64, maximum), maximum

    def _image_grid(self, mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid = mm_kwargs["image_grid_thw"]
        return grid if isinstance(grid, list) else grid.tolist()

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        merge = self.visual.spatial_merge_size
        result = []
        for t, h, w in self._image_grid(mm_kwargs):
            if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge:
                raise ValueError(f"Invalid Qwen3-Omni image grid: {(t, h, w)}")
            result.append(EncoderItemSpec(input_size=h * w, output_tokens=(h // merge) * (w // merge)))
        return result

    def select_encoder_cudagraph_items(self, mm_kwargs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        grid = self._image_grid(mm_kwargs)
        pixels = mm_kwargs["pixel_values"]
        offsets = [0]
        for spec in self.get_encoder_cudagraph_item_specs(mm_kwargs):
            offsets.append(offsets[-1] + spec.input_size)
        return {
            "pixel_values": torch.cat([pixels[offsets[i] : offsets[i + 1]] for i in indices])
            if indices
            else pixels[:0],
            "image_grid_thw": [grid[i] for i in indices],
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        if path != "default" or axis_keys:
            raise ValueError("Qwen3-Omni images use one encoder path without capture axes")
        visual = self.visual
        merge = visual.spatial_merge_size
        patches = token_budget * merge**2
        # Infer the RoPE layout from one real minimal grid. No large dummy
        # aspect ratio is needed, and no dummy grid indexes past the RoPE cache.
        small = visual.prepare_encoder_metadata([[1, merge, merge]])
        values = {
            "rotary_pos_emb_cos": torch.ones(
                (patches, *small["rotary_pos_emb_cos"].shape[1:]),
                device=device,
                dtype=small["rotary_pos_emb_cos"].dtype,
            ),
            "rotary_pos_emb_sin": torch.zeros(
                (patches, *small["rotary_pos_emb_sin"].shape[1:]),
                device=device,
                dtype=small["rotary_pos_emb_sin"].dtype,
            ),
            "cu_seqlens": torch.full((max_batch_size + 1,), patches, device=device, dtype=torch.int32),
            # This host scalar is baked into the captured attention launch.
            # Any single image in this graph has at most `patches` patches.
            "max_seqlen": torch.tensor(patches, dtype=torch.int32),
        }
        values["cu_seqlens"][0] = 0
        patch_embed = visual.patch_embed
        patch_width = patch_embed.proj.in_channels * patch_embed.temporal_patch_size * patch_embed.patch_size**2
        values["pixel_values"] = torch.zeros((patches, patch_width), device=device, dtype=dtype)
        if visual.apply_vit_abs_pos_embed:
            values["pos_embeds"] = torch.zeros((patches, visual.hidden_size), device=device, dtype=visual.dtype)
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        values = self.visual.prepare_encoder_metadata(self._image_grid(mm_kwargs))
        # Replaying a smaller request must not lower the scalar bound retained
        # by the capture. Tensor positions and sequence boundaries do change.
        values.pop("max_seqlen")
        values["pixel_values"] = mm_kwargs["pixel_values"]
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(self, inputs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        return self.visual(inputs["pixel_values"], None, encoder_metadata=inputs)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any], path: str = "default") -> torch.Tensor:
        return self.visual(mm_kwargs["pixel_values"], self._image_grid(mm_kwargs))

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        scatter_output_slices(outputs["default"], indices, per_item_out_tokens, dest, clone)
