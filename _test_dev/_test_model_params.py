# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Meta-device structural parity for Qwen3VLTextModel + Qwen3VLVisionModel vs transformers vanilla.

Usage:
    python _test_dev/_test_model_params.py
"""
from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel as Transformers_Qwen3VLTextModel,
    Qwen3VLVisionModel as Transformers_Qwen3VLVisionModel,
)

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
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

with torch.device('meta'):
    text_impl = Qwen3VLTextModel(text_cfg)
    text_transformers = Transformers_Qwen3VLTextModel(text_cfg)
    vision_impl = Qwen3VLVisionModel(vision_cfg)
    vision_transformers = Transformers_Qwen3VLVisionModel(vision_cfg)

n_text_impl = sum(p.numel() for p in text_impl.parameters())
n_text_transformers = sum(p.numel() for p in text_transformers.parameters())
n_vision_impl = sum(p.numel() for p in vision_impl.parameters())
n_vision_transformers = sum(p.numel() for p in vision_transformers.parameters())

keys_text_impl = set(text_impl.state_dict().keys())
keys_text_transformers = set(text_transformers.state_dict().keys())
keys_vision_impl = set(vision_impl.state_dict().keys())
keys_vision_transformers = set(vision_transformers.state_dict().keys())

assert n_text_impl == n_text_transformers, f'text params mismatch: impl={n_text_impl} transformers={n_text_transformers}'
assert n_vision_impl == n_vision_transformers, f'vision params mismatch: impl={n_vision_impl} transformers={n_vision_transformers}'
assert keys_text_impl == keys_text_transformers, f'text state_dict keys mismatch: only_impl={keys_text_impl - keys_text_transformers} only_transformers={keys_text_transformers - keys_text_impl}'
assert keys_vision_impl == keys_vision_transformers, f'vision state_dict keys mismatch: only_impl={keys_vision_impl - keys_vision_transformers} only_transformers={keys_vision_transformers - keys_vision_impl}'

print('pass')
print(f'text_model params: impl={n_text_impl:,} transformers={n_text_transformers:,}')
print(f'vision_model params: impl={n_vision_impl:,} transformers={n_vision_transformers:,}')
print(f'text_model state_dict keys: {len(keys_text_impl)} vision_model state_dict keys: {len(keys_vision_impl)}')


# output:
# pass
# text_model params: impl=423,680 transformers=423,680
# vision_model params: impl=1,485,312 transformers=1,485,312
# text_model state_dict keys: 24 vision_model state_dict keys: 351
