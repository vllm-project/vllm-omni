# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Top-level action head for GR00T-N1.7.

Ported from
``Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py``
(``Gr00tN1d7ActionHead`` + ``get_action_with_features``).  Training-only
paths are dropped; only the inference Euler loop is kept.  RTC (real-time
control) is **not** in the first release.

Submodule names mirror upstream so the GR00T-N1.7 root checkpoint loads via
plain ``load_state_dict`` once the ``action_head.`` prefix is stripped:

    state_encoder, action_encoder, action_decoder, model, vlln,
    vl_self_attention, position_embedding
"""

from __future__ import annotations

import torch
from torch import nn

from vllm_omni.diffusion.models.gr00t.modeling.action_head_modules import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from vllm_omni.diffusion.models.gr00t.modeling.dit import (
    AlternateVLDiT,
    DiT,
    SelfAttentionTransformer,
)
from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config


class Gr00tN1d7ActionHead(nn.Module):
    """Flow-matching action head.

    Builds:
      - ``model``: ``AlternateVLDiT`` (default) or ``DiT`` cross-attending to
        VL embeddings, with action tokens as queries.
      - ``state_encoder`` / ``action_encoder`` / ``action_decoder``:
        per-embodiment MLPs from :mod:`action_head_modules`.
      - ``vlln``: optional LayerNorm over backbone embeddings.
      - ``vl_self_attention``: optional self-attention stack over VL tokens.
      - ``position_embedding``: optional learned positional embedding over the
        state+action sequence.
    """

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.num_timestep_buckets = config.num_timestep_buckets

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln
            else nn.Identity()
        )

        vlsa_cfg = getattr(config, "vl_self_attention_cfg", None)
        if (
            vlsa_cfg
            and vlsa_cfg.get("num_layers", 0) > 0
            and config.use_vl_self_attention
        ):
            self.vl_self_attention = SelfAttentionTransformer(**vlsa_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                config.max_seq_len, self.input_embedding_dim
            )

    def _process_vl(self, vl_embeds: torch.Tensor) -> torch.Tensor:
        return self.vl_self_attention(self.vlln(vl_embeds))

    @torch.no_grad()
    def get_action(
        self,
        *,
        vl_embeds: torch.Tensor,  # [B, S, backbone_embedding_dim]
        vl_attn_mask: torch.Tensor,  # [B, S] bool
        image_mask: torch.Tensor,  # [B, S] bool
        state: torch.Tensor,  # [B, state_history_length, max_state_dim]
        embodiment_id: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        """Run ``num_inference_timesteps`` Euler steps of the flow-matching
        ODE.  Returns ``[B, action_horizon, action_dim]``."""
        vl = self._process_vl(vl_embeds)

        B = vl.shape[0]
        if state.shape[1] != self.config.state_history_length:
            raise ValueError(
                "state history mismatch: got "
                f"{state.shape[1]}, expected {self.config.state_history_length}"
            )
        state_flat = state.reshape(B, 1, -1)
        state_features = self.state_encoder(state_flat, embodiment_id)

        dt = 1.0 / self.num_inference_timesteps
        device, dtype = vl.device, vl.dtype
        x = torch.randn(
            B, self.action_horizon, self.action_dim, device=device, dtype=dtype
        )

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discrete = int(t_cont * self.num_timestep_buckets)
            ts = torch.full((B,), t_discrete, device=device, dtype=torch.long)

            action_features = self.action_encoder(x, ts, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], device=device)
                action_features = action_features + self.position_embedding(
                    pos_ids
                ).unsqueeze(0)

            sa = torch.cat((state_features, action_features), dim=1)
            if self.config.use_alternate_vl_dit:
                mo = self.model(
                    hidden_states=sa,
                    encoder_hidden_states=vl,
                    timestep=ts,
                    image_mask=image_mask,
                    backbone_attention_mask=vl_attn_mask,
                )
            else:
                mo = self.model(
                    hidden_states=sa,
                    encoder_hidden_states=vl,
                    timestep=ts,
                )
            pred = self.action_decoder(mo, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            x = x + dt * pred_velocity

        return x
