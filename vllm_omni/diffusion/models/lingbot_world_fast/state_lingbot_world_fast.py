# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lingbot World Fast pipeline persistent state."""

from __future__ import annotations

import logging
from enum import IntEnum

import torch

logger = logging.getLogger(__name__)


class CacheIndex(IntEnum):
    K = 0
    V = 1


class LingbotWorldFastState:
    """Pipeline persistent state across forward() calls.

    Lifecycle:
        - Created once in LingbotWorldFastPipeline.__init__()
        - Mutated every forward() call (frame append, KV cache grow)
        - reset() on new session / local_attn_size exceeded
    """

    def __init__(self) -> None:
        self.is_initialized = False
        self.reset()

    # ------------------------------------------------------------------
    # Reset / should_reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state."""
        self.current_start_frame: int = 0
        self.local_end_index: list[torch.Tensor] | None = None
        self.global_end_index: list[torch.Tensor] | None = None

        self.is_initialized: bool = False
        self.current_lat_f: int = 0
        self.session_id: str | None = None

        self.num_layers: int | None = None

        # Shape constants captured on the first call of a session and reused
        # on extension calls, where multi_modal_data["image"] is absent.
        self.h: int | None = None
        self.w: int | None = None
        self.lat_h: int | None = None
        self.lat_w: int | None = None
        self.frame_seqlen: int | None = None

        # Last few latents emitted by the diffusion loop on the previous call.
        # Prepended to pred_latent_chunks on extension so the Wan VAE decoder's
        # stacked temporal feat_maps are fully warmed before the first NEW
        # latent is decoded. The decoder's temporal receptive field spans
        # ~2 latents, so we cache the last 2.
        self.last_decoded_latent: torch.Tensor | None = None

    def init_state(
        self,
        device: torch.device,
        num_layers: int,
    ) -> None:
        self.num_layers = num_layers

        self.local_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]
        self.global_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]

        self.is_initialized = True

    def advance(self, delta: int):
        self.current_lat_f += delta
