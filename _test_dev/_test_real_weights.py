# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight load + NaN/Inf sanity for Qwen3VLTextModel + Qwen3VLVisionModel.

Not a numerical parity test.

Usage:
    python _test_dev/_test_real_weights.py
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import torch
from safetensors.torch import load_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)

checkpoint_dir = '/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image'

cfg = Qwen3VLConfig.from_pretrained(checkpoint_dir)
print(f'config: text hidden={cfg.text_config.hidden_size} layers={cfg.text_config.num_hidden_layers}, vision hidden={cfg.vision_config.hidden_size} layers={cfg.vision_config.depth}')

shards = sorted(glob.glob(os.path.join(checkpoint_dir, '*.safetensors')))
sd = {}
for shard in shards:
    sd.update(load_file(shard, device='cpu'))
print(f'loaded {len(shards)} shard(s), total keys={len(sd)}')

prefix_hist = defaultdict(int)
for k in sd:
    prefix_hist['.'.join(k.split('.', 2)[:2])] += 1
print('top-level prefixes:')
for p, n in sorted(prefix_hist.items(), key=lambda kv: -kv[1]):
    print(f'  {n:>6}  {p}')

text_prefix = 'model.language_model.'
vision_prefix = 'model.visual.'
text_sd = {k[len(text_prefix):]: v for k, v in sd.items() if k.startswith(text_prefix)}
vision_sd = {k[len(vision_prefix):]: v for k, v in sd.items() if k.startswith(vision_prefix)}

text_impl = Qwen3VLTextModel(cfg.text_config).eval()
res_text = text_impl.load_state_dict(text_sd, strict=False)
print(f'text_impl load: missing={len(res_text.missing_keys)} unexpected={len(res_text.unexpected_keys)}')

vision_impl = Qwen3VLVisionModel(cfg.vision_config).eval()
res_vision = vision_impl.load_state_dict(vision_sd, strict=False)
print(f'vision_impl load: missing={len(res_vision.missing_keys)} unexpected={len(res_vision.unexpected_keys)}')

torch.manual_seed(0)
input_ids = torch.randint(0, cfg.text_config.vocab_size, (1, 32))
position_ids = torch.arange(32)[None, :].expand(1, 32)
with torch.no_grad():
    out_text = text_impl(input_ids=input_ids, position_ids=position_ids, use_cache=False).last_hidden_state
print(f'text forward: shape={tuple(out_text.shape)} mean={out_text.mean().item():+.4f} std={out_text.std().item():.4f} has_nan={torch.isnan(out_text).any().item()} has_inf={torch.isinf(out_text).any().item()}')

grid_thw = torch.tensor([[1, 4, 4]])
n_patches = int(grid_thw.prod(dim=1).sum().item())
pixel_channels = cfg.vision_config.in_channels * cfg.vision_config.temporal_patch_size * cfg.vision_config.patch_size * cfg.vision_config.patch_size
pixel_values = torch.randn(n_patches, pixel_channels)
with torch.no_grad():
    out_vision = vision_impl(pixel_values, grid_thw)[0]
print(f'vision forward: shape={tuple(out_vision.shape)} mean={out_vision.mean().item():+.4f} std={out_vision.std().item():.4f} has_nan={torch.isnan(out_vision).any().item()} has_inf={torch.isinf(out_vision).any().item()}')

assert len(res_text.missing_keys) == 0 and len(res_text.unexpected_keys) == 0, f'text load: missing={res_text.missing_keys[:5]} unexpected={res_text.unexpected_keys[:5]}'
assert len(res_vision.missing_keys) == 0 and len(res_vision.unexpected_keys) == 0, f'vision load: missing={res_vision.missing_keys[:5]} unexpected={res_vision.unexpected_keys[:5]}'
assert not torch.isnan(out_text).any() and not torch.isinf(out_text).any(), 'text forward has NaN or Inf'
assert not torch.isnan(out_vision).any() and not torch.isinf(out_vision).any(), 'vision forward has NaN or Inf'

print('pass')


# output:
# config: text hidden=4096 layers=36, vision hidden=1152 layers=27
# loaded 8 shard(s), total keys=759
# top-level prefixes:
#      398  model.language_model
#      351  model.visual
#        4  model.t_embedder1
#        3  model.x_embedder
#        2  model.final_layer2
#        1  lm_head.weight
# text_impl load: missing=0 unexpected=0
# vision_impl load: missing=0 unexpected=0
# text forward: shape=(1, 32, 4096) mean=+0.0099 std=0.7808 has_nan=False has_inf=False
# vision forward: shape=(4, 4096) mean=-0.0040 std=0.3272 has_nan=False has_inf=False
# pass
