# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Conditioned diffusion wrappers for Stable Audio 3.

PORT_FROM: stable_audio_3/models/diffusion.py (253 lines, full file)

Two wrappers, in order from outside in:

  ConditionedDiffusionModelWrapper      ← knows the 6 conditioning slot types
      .conditioner = MultiConditioner   ← runs T5Gemma + NumberConditioner
      .model = DiTWrapper               ← wraps DiffusionTransformer w/ CFG knobs
        .model = DiffusionTransformer   ← the actual DiT (see stable_audio_3_transformer.py)
      .pretransform = AutoencoderPretransform  ← outer wrapper around AudioAutoencoder

The 6 conditioning slot types come from the model_config.json:
  cross_attn_cond_ids[]   — text embeds (cross-attention in DiT blocks)
  global_cond_ids[]        — duration etc. (AdaLN scale/shift)
  input_concat_ids[]       — concat to noisy latents (e.g. inpainting mask)
  local_add_cond_ids[]     — add to latents at each position
  modular_local_cond_ids[] — variable shape per id (used for editing/inpaint)
  prepend_cond_ids[]       — sequence-dim prepend before DiT
"""

from __future__ import annotations

import typing as tp
from typing import Any

import torch
from torch import nn


class ConditionedDiffusionModelWrapper(nn.Module):
    """Top-level diffusion module: routes raw conditioning into DiT.

    PORT_FROM: stable_audio_3/models/diffusion.py:28-198
    """

    def __init__(
        self,
        model: nn.Module,                                          # DiTWrapper
        conditioner: nn.Module,                                    # MultiConditioner
        io_channels: int,
        sample_rate: int,
        min_input_length: int,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        distribution_shift_options: dict | None = None,
        sampling_distribution_shift_options: dict | None = None,
        mask_padding_attention: bool = False,
        use_effective_length_for_schedule: bool = False,
        pretransform: nn.Module | None = None,
        cross_attn_cond_ids: list[str] | None = None,
        global_cond_ids: list[str] | None = None,
        input_concat_ids: list[str] | None = None,
        local_add_cond_ids: list[str] | None = None,
        modular_local_cond_ids: list[str] | None = None,
        prepend_cond_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.min_input_length = min_input_length
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.mask_padding_attention = mask_padding_attention
        self.use_effective_length_for_schedule = use_effective_length_for_schedule

        self.cross_attn_cond_ids = cross_attn_cond_ids or []
        self.global_cond_ids = global_cond_ids or []
        self.input_concat_ids = input_concat_ids or []
        self.local_add_cond_ids = local_add_cond_ids or []
        self.modular_local_cond_ids = modular_local_cond_ids or []
        self.prepend_cond_ids = prepend_cond_ids or []

        # Distribution-shift schedulers
        # PORT_FROM: diffusion.py:64-79 + inference/distribution_shift.py
        self.dist_shift = None
        self.sampling_dist_shift = None
        # TODO(stable-audio-3): init dist_shift from options dicts
        # via _create_dist_shift helper. Default sampling_dist_shift =
        # LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)

    def get_conditioning_inputs(
        self,
        conditioning_tensors: dict[str, Any],
        negative: bool = False,
    ) -> dict[str, Any]:
        """Slot per-id conditioner outputs into the 6 named buckets DiTWrapper expects.

        PORT_FROM: diffusion.py:91-188 — pure data shuffling, no torch ops
        beyond cat/squeeze. Direct port, no vllm-omni-specific changes.
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """PORT_FROM: diffusion.py:190-192. Simply: model(x, t, **routed_cond)."""
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)


class DiTWrapper(nn.Module):
    """Adapter between ConditionedDiffusionModelWrapper and DiffusionTransformer.

    Exposes CFG-related knobs (cfg_scale, cfg_dropout_prob, etc.) at the call
    site so the sampler can vary them per step.

    PORT_FROM: stable_audio_3/models/diffusion.py:195-252
    """

    def __init__(
        self,
        diffusion_objective: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.diffusion_objective = diffusion_objective
        # PORT_FROM: diffusion.py:204
        # Instantiate the actual DiT here. *args/**kwargs come from
        # model_config.json's diffusion.config block.
        from vllm_omni.diffusion.models.stable_audio_3.stable_audio_3_transformer import (
            DiffusionTransformer,
        )

        self.model = DiffusionTransformer(diffusion_objective=diffusion_objective, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        negative_cross_attn_cond: torch.Tensor | None = None,
        negative_cross_attn_mask: torch.Tensor | None = None,
        input_concat_cond: torch.Tensor | None = None,
        local_add_cond: torch.Tensor | None = None,
        negative_input_concat_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        negative_global_cond: torch.Tensor | None = None,
        prepend_cond: torch.Tensor | None = None,
        prepend_cond_mask: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        batch_cfg: bool = True,
        rescale_cfg: bool = False,
        scale_phi: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """PORT_FROM: diffusion.py:206-252. Direct delegation to DiffusionTransformer."""
        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        raise NotImplementedError
