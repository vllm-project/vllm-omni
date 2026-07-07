# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural test for Qwen3VLModel skeleton (meta-device init + delegates + forward stub).

Usage:
    python _test_dev/_test_qwen3vl_model_init.py
"""
from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    BottleneckPatchEmbed,
    FinalLayer,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
    TimestepEmbedder,
)

text_cfg = Qwen3VLTextConfig(
    vocab_size=1000,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=128,
    tie_word_embeddings=False,
)
vision_cfg = Qwen3VLVisionConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_heads=4,
    in_channels=3,
    patch_size=14,
    temporal_patch_size=1,
    out_hidden_size=128,
)
cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)

with torch.device('meta'):
    model = Qwen3VLModel(cfg)

# structural checks
assert isinstance(model.visual, Qwen3VLVisionModel)
assert isinstance(model.language_model, Qwen3VLTextModel)
assert isinstance(model.x_embedder, BottleneckPatchEmbed)
assert isinstance(model.t_embedder1, TimestepEmbedder)
assert isinstance(model.final_layer2, FinalLayer)
assert model.t_embedder2 is None
assert model.rope_deltas is None

# scalar config checks (upstream-fixed HiDream-O1 constants)
hidden_size = text_cfg.hidden_size
assert model.patch_size == 32
assert model.in_channels == 3
assert model.tms_token_id == 151673
assert model.x_embedder.proj1.in_features == 32 * 32 * 3
assert model.x_embedder.proj1.out_features == hidden_size // 4
assert model.x_embedder.proj2.out_features == hidden_size
assert model.final_layer2.linear.out_features == 32 * 32 * 3

# delegate checks: language_model backed
assert model.get_input_embeddings() is model.language_model.get_input_embeddings()
assert model.get_decoder() is model.language_model
placeholder = torch.nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size, device='meta')
model.set_input_embeddings(placeholder)
assert model.get_input_embeddings() is placeholder
model.set_decoder(model.language_model)  # restore

# state_dict key prefixes
sd_keys = list(model.state_dict().keys())
top_prefixes = {k.split('.', 1)[0] for k in sd_keys}
expected_prefixes = {'visual', 'language_model', 'x_embedder', 't_embedder1', 'final_layer2'}
assert expected_prefixes.issubset(top_prefixes), f'missing prefixes: {expected_prefixes - top_prefixes}'
# t_embedder2 is None so it must not contribute any params
assert not any(k.startswith('t_embedder2') for k in sd_keys), 'unexpected t_embedder2.* in state_dict'

# param-count invariant: model == sum(subs), no orphan params
with torch.device('meta'):
    ref_vision = Qwen3VLVisionModel._from_config(cfg.vision_config)
    ref_text = Qwen3VLTextModel._from_config(cfg.text_config)
    ref_x = BottleneckPatchEmbed(cfg, patch_size=32, in_chans=3, pca_dim=hidden_size // 4, embed_dim=hidden_size, bias=True)
    ref_t = TimestepEmbedder(cfg, hidden_size)
    ref_final = FinalLayer(cfg, hidden_size=hidden_size, patch_size=32, out_channels=3)

n_model = sum(p.numel() for p in model.parameters())
n_ref = sum(sum(p.numel() for p in m.parameters()) for m in [ref_vision, ref_text, ref_x, ref_t, ref_final])
assert n_model == n_ref, f'param count mismatch: model={n_model} sum(subs)={n_ref}'

n_pixel_dit = sum(sum(p.numel() for p in m.parameters()) for m in [ref_x, ref_t, ref_final])

print('pass')
print(f'Qwen3VLModel params: {n_model:,} (backbone={n_ref - n_pixel_dit:,} + pixel-DiT={n_pixel_dit:,})')
print(f'state_dict keys: {len(sd_keys)} (top-level prefixes: {sorted(top_prefixes)})')


# output:
# pass
# Qwen3VLModel params: 2,457,216 (backbone=1,908,992 + pixel-DiT=548,224)
# state_dict keys: 384 (top-level prefixes: ['final_layer2', 'language_model', 't_embedder1', 'visual', 'x_embedder'])
