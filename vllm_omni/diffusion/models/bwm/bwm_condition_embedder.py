# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""BWM condition embedder: action-modulated timesteps, no text projection.

BWM fine-tunes the Wan2.2-TI2V-5B DiT with two conditioning changes relative
to the stock model (reference: ``model_fn_wan_video_action`` in the BWM repo):

1. A per-latent-frame action modulation is added to the time embedding
   *before* ``time_proj`` (adaLN injection), so it also flows into the
   final output scale/shift through ``temb``.
2. The text pathway is disabled. The cross-attention context is the action
   encoder's per-frame tokens, which are already in DiT dimension, so the
   text projection must be skipped.

``BWMConditionEmbedder`` subclasses the stock embedder overriding only
``forward``; the pipeline swaps the instance class in place
(``__class__`` assignment) so parameter names, and therefore checkpoint
loading, are unchanged.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTimeTextImageEmbedding


class BWMConditionEmbedder(WanTimeTextImageEmbedding):
    """WanTimeTextImageEmbedding with action adaLN and the text pathway off."""

    # Set per forward by the pipeline: (batch, num_latent_frames, dim).
    action_mod_emb: torch.Tensor | None = None

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        if self.action_mod_emb is not None:
            mod = self.action_mod_emb
            if temb.ndim == 3:
                # Per-token timesteps (expand_timesteps mode): repeat each
                # latent frame's modulation over its spatial tokens.
                num_spatial_tokens = temb.shape[1] // mod.shape[1]
                mod = mod.unsqueeze(2).repeat(1, 1, num_spatial_tokens, 1).flatten(1, 2)
            temb = temb + mod.type_as(temb)

        timestep_proj = self.time_proj(self.act_fn(temb))

        # Text pathway disabled: encoder_hidden_states are pre-embedded action
        # tokens in DiT dimension; do NOT apply the text projection.
        return temb, timestep_proj, encoder_hidden_states, None
