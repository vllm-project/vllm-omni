# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-VL backbone adapter for GR00T-N1.7.

Ported from ``Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py``.  Wraps a
plain ``Qwen3VLForConditionalGeneration`` and:

  1. Truncates ``language_model.layers`` to ``config.select_layer`` so the
     stored checkpoint (which only contains layers 0..select_layer-1) loads
     cleanly via ``load_state_dict``.
  2. Forwards image + text inputs through the truncated model with
     ``output_hidden_states=True`` and returns the last hidden state
     together with the ``image_mask`` and ``backbone_attention_mask`` the
     action head's ``AlternateVLDiT`` needs.

Unlike Isaac, we do NOT call ``from_pretrained`` at construction time.  The
GR00T-N1.7 pipeline reads weights via ``Gr00tN1d7Pipeline.load_weights`` from
the root safetensors, so we instantiate the HF model from a config and the
truncation happens before any weights flow in.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

logger = logging.getLogger(__name__)


class Qwen3VLBackbone(nn.Module):
    """Truncated Qwen3-VL backbone exposing the layer-``select_layer``
    hidden state to the GR00T action head.

    Attribute layout (kept compatible with upstream ``Qwen3Backbone``):
      ``self.model`` : ``Qwen3VLForConditionalGeneration``
      ``self.select_layer`` : ``int``
    """

    def __init__(
        self,
        config: Gr00tN1d7Config,
        *,
        hf_config: Qwen3VLConfig | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.config = config
        self.select_layer = config.select_layer

        hf_config = hf_config if hf_config is not None else self._build_hf_config(config)
        # `from_config` for vision/text via _from_config matches the
        # internvla_a1 pattern; we instantiate the full ConditionalGeneration
        # so we get .visual + .model.language_model + .lm_head ready for
        # checkpoint loading.
        self.model = Qwen3VLForConditionalGeneration(hf_config)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)

        self._truncate_language_model_layers(self.select_layer)

    @staticmethod
    def _build_hf_config(config: Gr00tN1d7Config) -> Qwen3VLConfig:
        """Construct a ``Qwen3VLConfig`` from the overlay sub-configs the
        Gr00tN1d7Config already loaded."""
        text_config = getattr(config, "text_config", None)
        vision_config = getattr(config, "vision_config", None)
        if text_config is None or vision_config is None:
            raise ValueError(
                "Gr00tN1d7Config is missing text_config/vision_config — the "
                "Qwen3-VL backbone overlay did not run.  Pass `hf_config=` "
                "explicitly or set `model_name` to a reachable Cosmos-Reason2 "
                "checkpoint."
            )
        return Qwen3VLConfig(
            text_config=text_config.to_dict() if hasattr(text_config, "to_dict") else text_config,
            vision_config=vision_config.to_dict() if hasattr(vision_config, "to_dict") else vision_config,
        )

    def _truncate_language_model_layers(self, select_layer: int) -> None:
        lm = self.model.model.language_model
        while len(lm.layers) > select_layer:
            lm.layers.pop(-1)
        if len(lm.layers) != select_layer:
            raise ValueError(
                f"Qwen3VL backbone has {len(lm.layers)} layers but "
                f"select_layer={select_layer} requires the model to start with "
                "at least that many layers."
            )

    @property
    def image_token_id(self) -> int:
        return int(self.model.config.image_token_id)

    @torch.no_grad()
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **extra: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the truncated Qwen3-VL on a multimodal batch.

        Returns:
            backbone_features : ``[B, S, hidden_size]`` — hidden state from
                the last (truncated) language-model layer.
            backbone_attention_mask : ``[B, S]`` bool
            image_mask : ``[B, S]`` bool — true where ``input_ids`` matches
                ``config.image_token_id``.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            **extra,
        )
        hidden = outputs.hidden_states[-1]
        image_mask = input_ids == self.image_token_id
        backbone_attention_mask = attention_mask == 1
        return {
            "backbone_features": hidden,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask,
        }
