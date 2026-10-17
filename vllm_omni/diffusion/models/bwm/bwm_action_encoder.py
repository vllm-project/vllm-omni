# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Action encoder for Boundless-World-Model (BWM).

Ported from the reference implementation
(https://github.com/boundless-large-model/boundless-world-model,
``wan_video_action/models/wan_video_action_encoder.py``), keeping only the
two branches the released ``adaln`` checkpoint uses:

* ``action_mlp1``: per-frame action embeddings appended to the
  cross-attention context (the DiT runs with the text pathway disabled, so
  these tokens are the *entire* conditioning context);
* ``action_mlp2``: actions grouped 4 pixel frames per latent frame and
  projected to a per-latent-frame modulation added to the timestep
  embedding (adaLN injection).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BWMActionEncoder(nn.Module):
    """Encode robot action trajectories for the BWM Wan2.2-TI2V-5B DiT."""

    def __init__(self, action_dim: int = 14, dim: int = 3072):
        super().__init__()
        self.action_dim = action_dim
        self.dim = dim
        self.action_mlp1 = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_mlp2 = nn.Sequential(
            nn.Linear(action_dim * 4, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an action trajectory.

        Args:
            action: ``(batch, frames, action_dim)`` normalized actions, where
                ``frames = 1 + 4 * (num_latent_frames - 1)`` (one action per
                pixel frame, aligned to the VAE temporal compression).

        Returns:
            ``(action_context_emb, action_mod_emb)``:
            per-frame context tokens ``(batch, frames, dim)`` and grouped
            per-latent-frame modulation ``(batch, num_latent_frames, dim)``.
        """
        action_context_emb = self.action_mlp1(action)
        # Group actions 4-per-latent-frame; the first frame is replicated to
        # complete the leading group (frame layout: 1 + 4 * (T_latent - 1)).
        grouped = torch.cat([action[:, 0:1].repeat(1, 3, 1), action], dim=1)
        grouped = grouped.reshape(action.shape[0], (action.shape[1] + 3) // 4, action.shape[2] * 4)
        action_mod_emb = self.action_mlp2(grouped)
        return action_context_emb, action_mod_emb
