# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical forward parity for Qwen3VLTextModel + Qwen3VLVisionModel vs transformers vanilla.

Usage:
    python _test_dev/_test_forward_parity.py
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

def _extract_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, 'last_hidden_state'):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f'unexpected forward output type: {type(out)}')

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

torch.manual_seed(42)
text_impl = Qwen3VLTextModel(text_cfg).eval()
text_transformers = Transformers_Qwen3VLTextModel(text_cfg).eval()
text_transformers.load_state_dict(text_impl.state_dict())

torch.manual_seed(42)
vision_impl = Qwen3VLVisionModel(vision_cfg).eval()
vision_transformers = Transformers_Qwen3VLVisionModel(vision_cfg).eval()
vision_transformers.load_state_dict(vision_impl.state_dict())

torch.manual_seed(0)
batch, seq_len = 1, 32
input_ids = torch.randint(0, text_cfg.vocab_size, (batch, seq_len))
position_ids = torch.arange(seq_len)[None, :].expand(batch, seq_len)

with torch.no_grad():
    out_text_impl = _extract_tensor(text_impl(input_ids=input_ids, position_ids=position_ids, use_cache=False))
    out_text_transformers = _extract_tensor(text_transformers(input_ids=input_ids, position_ids=position_ids, use_cache=False))

max_abs_text = (out_text_impl - out_text_transformers).abs().max().item()
print(f'text_model forward parity: max|impl - transformers| = {max_abs_text}')

grid_thw = torch.tensor([[1, 4, 4]])
n_patches = int(grid_thw.prod(dim=1).sum().item())
pixel_channels = vision_cfg.in_channels * vision_cfg.temporal_patch_size * vision_cfg.patch_size * vision_cfg.patch_size
pixel_values = torch.randn(n_patches, pixel_channels)

with torch.no_grad():
    out_vision_impl = vision_impl(pixel_values, grid_thw)[0]
    out_vision_transformers = vision_transformers(pixel_values, grid_thw).pooler_output

max_abs_vision = (out_vision_impl - out_vision_transformers).abs().max().item()
print(f'vision_model forward parity: max|impl - transformers| = {max_abs_vision}')

TOL = 1e-5

assert max_abs_text < TOL, f'text forward divergence: max|delta| = {max_abs_text} (tol = {TOL})'
assert max_abs_vision < TOL, f'vision forward divergence: max|delta| = {max_abs_vision} (tol = {TOL})'

print(f'pass (tol = {TOL}, expected ~0.0 on CPU + eager attention + fp32)')


# output:
# text_model forward parity: max|impl - transformers| = 0.0
# vision_model forward parity: max|impl - transformers| = 0.0
# pass (tol = 1e-05, expected ~0.0 on CPU + eager attention + fp32)
