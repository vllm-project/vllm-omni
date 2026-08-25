# SPDX-License-Identifier: Apache-2.0
"""Action-free Cosmos-Dreams-Transfer transformer."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.cosmos_dreams.conditioning_control import (
    build_shared_vision_mrope_position_ids,
    pack_pure_vision_tokens,
    unpack_pure_vision_tokens,
)
from vllm_omni.diffusion.models.cosmos_dreams.transformer_cosmos_dreams import (
    CosmosDreamsTransformer,
)


class CosmosDreamsTransferTransformer(CosmosDreamsTransformer):
    """Pure-vision Transfer variant with no action modules or weights."""

    def _validate_conditioning_config(self) -> None:
        self.manifest.require_control_video_conditioning()
        if self.action_gen:
            raise ValueError("Cosmos-Dreams-Transfer checkpoints must set action_gen=False.")

    def _prepare_conditioning_tokens(
        self,
        hidden_states: torch.Tensor,
        *,
        num_frames: int,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_latents is not None or action_domain_ids is not None:
            raise ValueError("Cosmos-Dreams-Transfer does not accept action conditioning.")
        return hidden_states.new_empty(1, num_frames, 0, self.hidden_size)

    def _pack_tokens(self, conditioning_tokens: torch.Tensor, vision_tokens: torch.Tensor) -> torch.Tensor:
        if conditioning_tokens.ndim != 4 or conditioning_tokens.shape[2] != 0:
            raise ValueError("Cosmos-Dreams-Transfer conditioning token sequence must be empty.")
        return pack_pure_vision_tokens(vision_tokens)

    def _unpack_tokens(
        self,
        hidden: torch.Tensor,
        *,
        num_frames: int,
        conditioning_tokens_per_frame: int,
        vision_tokens_per_frame: int,
    ) -> torch.Tensor:
        if conditioning_tokens_per_frame != 0:
            raise ValueError("Cosmos-Dreams-Transfer cannot unpack action conditioning tokens.")
        return unpack_pure_vision_tokens(
            hidden,
            num_frames=num_frames,
            vision_tokens_per_frame=vision_tokens_per_frame,
        )

    def _build_position_ids(
        self,
        *,
        frame_start: int,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        real_text_kv_len: int,
        fps: float,
        null_action_frame_indexes: tuple[int, ...],
    ) -> torch.Tensor:
        if null_action_frame_indexes:
            raise ValueError("Cosmos-Dreams-Transfer does not accept null action frame indexes.")
        return build_shared_vision_mrope_position_ids(
            frame_start=frame_start,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
            text_temporal_offset=real_text_kv_len,
            temporal_modality_margin=self.temporal_modality_margin,
            fps=fps,
            base_fps=self.base_fps,
            temporal_compression_factor=self.manifest.temporal_compression_factor,
            enable_fps_modulation=self.enable_fps_modulation,
        )


Cosmos3TransferInteractiveTransformer = CosmosDreamsTransferTransformer

__all__ = [
    "Cosmos3TransferInteractiveTransformer",
    "CosmosDreamsTransferTransformer",
]
