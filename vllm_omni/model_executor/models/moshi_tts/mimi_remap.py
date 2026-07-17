"""Remap Kyutai Mimi weight keys to HF MimiModel naming.

Used by both Mimi encoder and decoder.
Mapping derived from tensor-value comparison of Kyutai and HF Mimi weights.
"""

from __future__ import annotations

import re

import torch


def _interleaved_to_half_split(w: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Convert interleaved (GPT-J) layout to half-split (NeoX) layout.

    GPT-J: [r0, i0, r1, i1, ...] per head
    NeoX:  [r0, r1, ..., i0, i1, ...] per head

    For a weight matrix [out_features, in_features], rearranges the out_features
    dimension within each head.
    """
    # w shape: [out_features, in_features] where out_features = num_heads * head_dim
    out_features = w.shape[0]
    num_heads = out_features // head_dim
    # Reshape to [num_heads, head_dim, in_features]
    w = w.view(num_heads, head_dim, -1)
    # Within each head: [r0,i0,r1,i1,...] → [r0,r1,...,i0,i1,...]
    w = w.view(num_heads, head_dim // 2, 2, -1)
    w = w.permute(0, 2, 1, 3).contiguous().view(num_heads, head_dim, -1)
    # Reshape back to [out_features, in_features]
    return w.view(out_features, -1)


def remap_kyutai_mimi_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap Kyutai Mimi weight keys to HF MimiModel naming.

    Key differences:
      - SEANet: model.X.conv.conv → layers.X.conv, convtr.convtr → conv
      - Transformer: .transformer.layers → .layers, norm1→input_layernorm,
        norm2→post_attention_layernorm, linear1→mlp.fc1, linear2→mlp.fc2,
        out_proj→o_proj, layer_scale_1→self_attn_layer_scale,
        layer_scale_2→mlp_layer_scale
      - Quantizer: rvq_first→semantic_residual_vector_quantizer,
        rvq_rest→acoustic_residual_vector_quantizer, .vq.layers→.layers,
        _codebook.embedding_sum→codebook.embed_sum, etc.
      - Attention: fused in_proj_weight [3*d, d] → split q/k/v_proj [d, d]
      - Upsample/downsample: unwrap nested conv wrappers
    """
    remapped: dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        new_key = key

        # --- Fused attention: split in_proj_weight → q_proj, k_proj, v_proj ---
        # Kyutai's Mimi uses interleaved RoPE (GPT-J: [r0,i0,r1,i1,...])
        # but HF's MimiModel uses half-split RoPE (NeoX: [r0,r1,...,i0,i1,...]).
        # Q and K weights must be rearranged; V is unaffected.
        m = re.match(r"(.+)\.self_attn\.in_proj_weight$", new_key)
        if m:
            prefix = m.group(1)
            prefix = prefix.replace("_transformer.transformer.layers.", "_transformer.layers.")
            d = tensor.shape[1]  # hidden_size (= num_heads * head_dim)
            # Mimi always uses 8 attention heads (MimiConfig default).
            # Hardcoded because Mimi weights don't carry a config with num_heads.
            mimi_head_dim = d // 8
            q_w = _interleaved_to_half_split(tensor[:d], mimi_head_dim)
            k_w = _interleaved_to_half_split(tensor[d : 2 * d], mimi_head_dim)
            v_w = tensor[2 * d :]
            remapped[f"{prefix}.self_attn.q_proj.weight"] = q_w
            remapped[f"{prefix}.self_attn.k_proj.weight"] = k_w
            remapped[f"{prefix}.self_attn.v_proj.weight"] = v_w
            continue

        # --- SEANet encoder/decoder: model.X → layers.X ---
        for pfx in ("encoder.", "decoder."):
            if new_key.startswith(pfx + "model."):
                new_key = pfx + "layers." + new_key[len(pfx) + len("model.") :]

        # --- Upsample/downsample (3-level nesting, MUST come before generic 2-level) ---
        new_key = new_key.replace("upsample.convtr.convtr.convtr.", "upsample.conv.")
        new_key = new_key.replace("downsample.conv.conv.conv.", "downsample.conv.")

        # --- SEANet conv wrappers (2-level nesting) ---
        new_key = new_key.replace(".conv.conv.", ".conv.")
        new_key = new_key.replace(".convtr.convtr.", ".conv.")

        # --- Transformer layers ---
        new_key = new_key.replace("_transformer.transformer.layers.", "_transformer.layers.")
        new_key = new_key.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        new_key = new_key.replace(".linear1.", ".mlp.fc1.")
        new_key = new_key.replace(".linear2.", ".mlp.fc2.")
        new_key = new_key.replace(".norm1.", ".input_layernorm.")
        new_key = new_key.replace(".norm2.", ".post_attention_layernorm.")
        new_key = new_key.replace(".layer_scale_1.", ".self_attn_layer_scale.")
        new_key = new_key.replace(".layer_scale_2.", ".mlp_layer_scale.")

        # --- Quantizer ---
        new_key = new_key.replace("quantizer.rvq_first.", "quantizer.semantic_residual_vector_quantizer.")
        new_key = new_key.replace("quantizer.rvq_rest.", "quantizer.acoustic_residual_vector_quantizer.")
        new_key = new_key.replace(".vq.layers.", ".layers.")
        new_key = new_key.replace("._codebook.embedding_sum", ".codebook.embed_sum")
        new_key = new_key.replace("._codebook.cluster_usage", ".codebook.cluster_usage")
        new_key = new_key.replace("._codebook._initialized", ".codebook.initialized")
        new_key = new_key.replace("._codebook.embed", ".codebook.embed")

        remapped[new_key] = tensor

    return remapped
