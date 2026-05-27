# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""T5EncoderBlockByT5Mapper — Ming's per-block T5 stack mapping byte5 features
onto the DiT condition space.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.t5_encoder.t5_encoder import T5Block


class T5EncoderBlockByT5Mapper(ModelMixin):
    """Stacks ``num_layers`` T5 encoder blocks on top of byte5 features and
    projects them to ``sdxl_channels`` (= Ming's ``diffusion_c_input_dim``).
    """

    def __init__(self, byte5_config, num_layers: int, sdxl_channels: int | None = None) -> None:
        super().__init__()
        if num_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    T5Block(
                        byte5_config,
                        has_relative_attention_bias=(i == 0),
                        prefix=f"blocks.{i}",
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.blocks = None
        self.layer_norm = RMSNorm(byte5_config.d_model, eps=byte5_config.layer_norm_epsilon)
        if sdxl_channels is not None:
            self.channel_mapper = nn.Linear(byte5_config.d_model, sdxl_channels)
            self.final_layer_norm = RMSNorm(sdxl_channels, eps=byte5_config.layer_norm_epsilon)
        else:
            self.channel_mapper = None
            self.final_layer_norm = None

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Unexpected attention_mask shape {tuple(attention_mask.shape)}")
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        extended_mask = self.get_extended_attention_mask(attention_mask, dtype=self.dtype)

        hidden_states = inputs_embeds
        position_bias = None

        if self.blocks is not None:
            for block in self.blocks:
                hidden_states, position_bias = block(
                    hidden_states,
                    mask=extended_mask,
                    position_bias=position_bias,
                )

        hidden_states = self.layer_norm(hidden_states)
        if self.channel_mapper is not None:
            hidden_states = self.channel_mapper(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Ming's HF-format byte5_mapper checkpoint into the fused
        TP-aware layers.

        Source format (from ``byte5_mapper.pt``):
            blocks.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
            blocks.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
            blocks.{i}.layer.{0,1}.layer_norm.weight
            {layer_norm, channel_mapper, final_layer_norm}.{weight,bias}

        Target format (after ``T5Block`` from t5_encoder.py):
            blocks.{i}.layer.0.SelfAttention.qkv_proj.weight  (fused q+k+v)
            blocks.{i}.layer.1.DenseReluDense.wi.weight       (fused wi_0+wi_1)
            (others identical)
        """
        stacked_params_mapping = [
            (".qkv_proj.", ".q.", "q"),
            (".qkv_proj.", ".k.", "k"),
            (".qkv_proj.", ".v.", "v"),
            (".DenseReluDense.wi.", ".DenseReluDense.wi_0.", 0),
            (".DenseReluDense.wi.", ".DenseReluDense.wi_1.", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            matched = False
            for target_substr, source_substr, shard_id in stacked_params_mapping:
                if source_substr not in name:
                    continue
                target_name = name.replace(source_substr, target_substr, 1)
                if target_name in params_dict:
                    param = params_dict[target_name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(target_name)
                    matched = True
                    break

            if matched:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


__all__ = ["T5EncoderBlockByT5Mapper"]
