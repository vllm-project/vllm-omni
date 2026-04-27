# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from Ming repository qwen3_moe_vit.py
# https://github.com/inclusionAI/Ming

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
)
from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


def _adapt_vision_config(vision_config):
    # Adapt Ming's Qwen3VLMoeVisionConfig to be compatible with vLLM's
    # Qwen3Omni_VisionTransformer expectations.
    if not hasattr(vision_config, "image_size") or vision_config.image_size is None:
        if hasattr(vision_config, "num_position_embeddings") and vision_config.num_position_embeddings:
            import math

            num_grid = int(math.sqrt(vision_config.num_position_embeddings))
            vision_config.image_size = num_grid * vision_config.patch_size
        else:
            vision_config.image_size = vision_config.patch_size * 14  # fallback

    if not hasattr(vision_config, "apply_vit_abs_pos_embed"):
        vision_config.apply_vit_abs_pos_embed = True

    return vision_config


class MingVisionEncoder(nn.Module):
    """**Wrapper** around vLLM's Qwen3Omni_VisionTransformer for Ming."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "deepstack_merger_list.": "merger_list.",
            "merger.norm.": "merger.ln_q.",
            "merger.linear_fc1.": "merger.mlp.0.",
            "merger.linear_fc2.": "merger.mlp.2.",
        }
    )

    def __init__(
        self,
        vision_config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        adapted_config = _adapt_vision_config(vision_config)
        norm_eps = 1e-6
        self.encoder = Qwen3Omni_VisionTransformer(
            vision_config=adapted_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )
        self.image_emb_dim = vision_config.out_hidden_size
        self.use_deepstack = (
            hasattr(vision_config, "deepstack_visual_indexes") and vision_config.deepstack_visual_indexes is not None
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.dtype

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """forward method of the vision encoder.

        Args:
            pixel_values: Flattened pixel values.
            grid_thw: [num_images, 3] tensor of (t, h, w) grid sizes.

        Returns:
            If deepstack is enabled, returns concatenated multi-scale features
            along the feature dim: [seq_len, hidden_size * (1 + num_deepstack)].
            Otherwise returns [seq_len, hidden_size].
        """
        return self.encoder(pixel_values, grid_thw=grid_thw)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        import re

        def _remap_merger_list_inner(name: str) -> str:
            name = re.sub(r"(merger_list\.\d+)\.norm\.", r"\1.ln_q.", name)
            name = re.sub(r"(merger_list\.\d+)\.linear_fc1\.", r"\1.mlp.0.", name)
            name = re.sub(r"(merger_list\.\d+)\.linear_fc2\.", r"\1.mlp.2.", name)

            return name

        remapped_weights = self.hf_to_vllm_mapper.apply(weights)
        remapped_weights = ((_remap_merger_list_inner(name), tensor) for name, tensor in remapped_weights)
        loaded_params = self.encoder.load_weights(remapped_weights)

        loaded_params = {f"encoder.{loaded_param}" for loaded_param in loaded_params}

        return loaded_params
