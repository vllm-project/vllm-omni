# SPDX-License-Identifier: Apache-2.0
"""Action-free Cosmos3-Nano-Sim-Transfer transformer."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import compute_mrope_position_ids_vision
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualTransformer,
)


class Cosmos3NanoSimTransferTransformer(Cosmos3NanoSimBimanualTransformer):
    """Pure-vision Transfer variant with no action modules or weights."""

    def _validate_conditioning_config(self) -> None:
        """Require a control-video artifact and a checkpoint without action modules."""
        self.manifest.require_control_video_conditioning()
        if self.action_gen:
            raise ValueError("Cosmos3-Nano-Sim-Transfer checkpoints must set action_gen=False.")

    def _prepare_conditioning_tokens(
        self,
        hidden_states: torch.Tensor,
        *,
        num_frames: int,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return an empty ``[1, T, 0, D]`` sequence; the base packing then keeps only vision tokens."""
        if action_latents is not None or action_domain_ids is not None:
            raise ValueError("Cosmos3-Nano-Sim-Transfer does not accept action conditioning.")
        return hidden_states.new_empty(1, num_frames, 0, self.hidden_size)

    def _build_position_ids(
        self,
        *,
        frame_start: int,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        real_text_kv_len: int,
        fps: float,
        null_action_frames: tuple[int, ...],
    ) -> torch.Tensor:
        """Build the vision-only mRoPE IDs that control and target RGB frames share."""
        if null_action_frames:
            raise ValueError("Cosmos3-Nano-Sim-Transfer does not accept null action frame indexes.")
        position_ids, _ = compute_mrope_position_ids_vision(
            num_frames,
            grid_h,
            grid_w,
            temporal_offset=real_text_kv_len + self.temporal_modality_margin,
            fps=fps,
            base_fps=self.base_fps,
            temporal_compression_factor=self.temporal_compression_factor,
            enable_fps_modulation=self.enable_fps_modulation,
            start_frame_offset=frame_start,
        )
        return position_ids


__all__ = [
    "Cosmos3NanoSimTransferTransformer",
]
