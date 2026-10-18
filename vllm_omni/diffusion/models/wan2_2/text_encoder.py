# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original Wan UMT5 checkpoint loading shared by causal video pipelines."""

import os

import torch
from transformers import UMT5Config, UMT5EncoderModel


def convert_wan_umt5_encoder_state_dict(source: dict[str, torch.Tensor], *, num_layers: int) -> dict[str, torch.Tensor]:
    """Convert the original Wan T5 encoder names to Transformers UMT5."""

    converted: dict[str, torch.Tensor] = {
        "shared.weight": source["token_embedding.weight"],
        "encoder.embed_tokens.weight": source["token_embedding.weight"],
        "encoder.final_layer_norm.weight": source["norm.weight"],
    }
    consumed = {"token_embedding.weight", "norm.weight"}
    mappings = {
        "norm1.weight": "layer.0.layer_norm.weight",
        "attn.q.weight": "layer.0.SelfAttention.q.weight",
        "attn.k.weight": "layer.0.SelfAttention.k.weight",
        "attn.v.weight": "layer.0.SelfAttention.v.weight",
        "attn.o.weight": "layer.0.SelfAttention.o.weight",
        "pos_embedding.embedding.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
        "norm2.weight": "layer.1.layer_norm.weight",
        "ffn.gate.0.weight": "layer.1.DenseReluDense.wi_0.weight",
        "ffn.fc1.weight": "layer.1.DenseReluDense.wi_1.weight",
        "ffn.fc2.weight": "layer.1.DenseReluDense.wo.weight",
    }
    for index in range(num_layers):
        source_prefix = f"blocks.{index}"
        target_prefix = f"encoder.block.{index}"
        for source_suffix, target_suffix in mappings.items():
            source_name = f"{source_prefix}.{source_suffix}"
            converted[f"{target_prefix}.{target_suffix}"] = source[source_name]
            consumed.add(source_name)
    unexpected = set(source) - consumed
    if unexpected:
        raise KeyError(f"Unexpected keys in Wan UMT5 encoder checkpoint: {sorted(unexpected)[:10]}")
    return converted


def load_wan_umt5_encoder(encoder_pth: str, dtype: torch.dtype) -> UMT5EncoderModel:
    """Load local original-format weights into the existing Transformers encoder."""
    # Match the original Wan UMT5-XXL encoder checkpoint exactly.
    config = UMT5Config(
        d_model=4096,
        d_kv=64,
        d_ff=10240,
        num_layers=24,
        num_decoder_layers=24,
        num_heads=64,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-06,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
        vocab_size=256384,
        tie_word_embeddings=False,
        scalable_attention=True,
    )

    if not os.path.isfile(encoder_pth):
        raise FileNotFoundError(f"UMT5 encoder checkpoint not found at {encoder_pth}")
    source = torch.load(encoder_pth, map_location="cpu", weights_only=True)
    if isinstance(source, dict) and isinstance(source.get("state_dict"), dict):
        source = source["state_dict"]
    converted = convert_wan_umt5_encoder_state_dict(source, num_layers=config.num_layers)
    with torch.device("meta"):
        text_encoder = UMT5EncoderModel(config)
    text_encoder.load_state_dict(converted, strict=True, assign=True)
    return text_encoder.to(dtype=dtype)
