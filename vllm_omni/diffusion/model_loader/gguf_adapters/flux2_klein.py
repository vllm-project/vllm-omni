# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.utils import WeightsMapper

from .base import GGUFAdapter, gguf_quant_weights_iterator

FLUX2_TRANSFORMER_KEYS_RENAME_DICT = {
    "single_blocks.": "single_transformer_blocks.",
    # Image and text input projections
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    # Timestep and guidance embeddings
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    # Modulation parameters
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    # Final output layer
    # "final_layer.adaLN_modulation.1": "norm_out.linear",  # Handle separately since we need to swap mod params
    "final_layer.linear": "proj_out",
}

FLUX2_TRANSFORMER_ADA_LAYER_NORM_KEY_MAP = {
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}

FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP = {
    "double_blocks.": "transformer_blocks.",
    # Handle fused QKV projections separately as we need to break into Q, K, V projections
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
    # Additional for fuse qkv
    "img_attn.qkv": "attn.to_qkv",
    "txt_attn.qkv": "attn.add_kv_proj",
}

FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}


class Flux2KleinGGUFAdapter(GGUFAdapter):
    """GGUF adapter for Flux2-Klein models with qkv splitting and adaLN swap."""

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        model_class = od_config.model_class_name or ""
        if model_class.startswith("Flux2"):
            return True
        cfg = od_config.tf_model_config
        if cfg is not None:
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type.startswith("flux"):
                return True
        return False

    gguf_to_hf_mapper = WeightsMapper(
        # double_stream_modulation
        orig_to_new_prefix=FLUX2_TRANSFORMER_KEYS_RENAME_DICT | FLUX2_TRANSFORMER_ADA_LAYER_NORM_KEY_MAP,
        orig_to_new_substr=FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP | FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP,
    )

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        def custom_weights_adapter(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                # Handle the special case for adaLN modulation parameters that require swapping shift and scale
                if name.endswith(".scale"):
                    name = name.replace(".scale", ".weight")
                if name == "norm_out.linear.weight":
                    shift, scale = weight.chunk(2, dim=0)
                    weight = torch.cat([scale, shift], dim=0)
                    yield name, weight
                else:
                    yield name, weight

        weights = gguf_quant_weights_iterator(self.gguf_file)
        weights = self.gguf_to_hf_mapper.apply(weights)
        yield from custom_weights_adapter(weights)
