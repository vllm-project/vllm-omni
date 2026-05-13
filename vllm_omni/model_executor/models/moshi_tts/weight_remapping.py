"""Weight remapping from HF Moshi/Hibiki checkpoint to vLLM Llama naming.

HF MoshiForConditionalGeneration weight hierarchy:
  embed_tokens.{0..n_q-1}.weight       → audio_embed_tokens.{0..n_q-1}.weight
  audio_encoder.*                      → (separate Mimi loading)
  decoder.model.embed_tokens.weight    → model.embed_tokens.weight
  decoder.model.layers.{i}.*           → model.layers.{i}.* (with transforms)
  decoder.model.norm.weight            → model.norm.weight
  decoder.lm_head.weight               → lm_head.weight
  depth_decoder.*                      → depth_decoder.* (loaded into custom module)

Key transforms for main decoder layers:
  - .self_attn.{q,k,v,o}_proj.linear.weight → .self_attn.{q,k,v,o}_proj.weight
  - .mlp.fc1.weight [ffn_dim, hidden]       → split into gate_proj + up_proj
  - .mlp.fc2.weight                         → .mlp.down_proj.weight
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def remap_moshi_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    ffn_dim: int = 22528,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Transform HF Moshi weight names to vLLM-compatible names.

    Yields (name, tensor) pairs with remapped names. The fc1 weight is
    split into two separate gate_proj and up_proj tensors.

    Args:
        weights: Iterator of (name, tensor) from the HF checkpoint.
        ffn_dim: Moshi's fc1 output dimension (default 22528).
            intermediate_size = ffn_dim // 2.
    """
    intermediate_size = ffn_dim // 2

    for name, tensor in weights:
        # Skip rotary embedding caches
        if "rotary_emb" in name:
            continue

        # --- Audio encoder (Mimi) --- keep as-is
        if name.startswith("audio_encoder."):
            yield name, tensor
            continue

        # --- Audio embedding tables ---
        if name.startswith("embed_tokens.") and not name.startswith("embed_tokens.weight"):
            # embed_tokens.{i}.weight → audio_embed_tokens.{i}.weight
            new_name = name.replace("embed_tokens.", "audio_embed_tokens.", 1)
            yield new_name, tensor
            continue

        # --- Depth decoder --- keep prefix, loaded into custom module
        if name.startswith("depth_decoder."):
            yield name, tensor
            continue

        # --- Main decoder ---
        new_name = name

        # decoder.model.* → model.*
        if new_name.startswith("decoder.model."):
            new_name = new_name[len("decoder.") :]
        # decoder.lm_head.* → lm_head.*
        elif new_name.startswith("decoder.lm_head."):
            new_name = new_name[len("decoder.") :]

        # Attention: strip .linear. from MoshiLinear wrapper
        # .self_attn.q_proj.linear.weight → .self_attn.q_proj.weight
        new_name = new_name.replace(".q_proj.linear.", ".q_proj.")
        new_name = new_name.replace(".k_proj.linear.", ".k_proj.")
        new_name = new_name.replace(".v_proj.linear.", ".v_proj.")
        new_name = new_name.replace(".o_proj.linear.", ".o_proj.")

        # MLP fc1 → split into gate_proj and up_proj
        if ".mlp.fc1.weight" in new_name:
            gate_name = new_name.replace(".mlp.fc1.weight", ".mlp.gate_proj.weight")
            up_name = new_name.replace(".mlp.fc1.weight", ".mlp.up_proj.weight")
            # fc1 output: [ffn_dim, hidden_size]
            # First half (rows 0:intermediate_size) = gate
            # Second half (rows intermediate_size:ffn_dim) = up
            gate_weight = tensor[:intermediate_size, :]
            up_weight = tensor[intermediate_size:, :]
            yield gate_name, gate_weight
            yield up_name, up_weight
            continue

        # MLP fc2 → down_proj
        new_name = new_name.replace(".mlp.fc2.weight", ".mlp.down_proj.weight")

        yield new_name, tensor


def filter_depth_decoder_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Filter and strip prefix for depth decoder weights.

    Yields weights with 'depth_decoder.' prefix stripped.
    """
    prefix = "depth_decoder."
    for name, tensor in weights:
        if name.startswith(prefix):
            yield name[len(prefix) :], tensor
