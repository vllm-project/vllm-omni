# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identical parity for 4 Qwen3VLModel public helpers vs inline upstream reference.

Usage:
    python _test_dev/_test_qwen3vl_helpers.py
"""
from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    Qwen3VLModel,
)

# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/qwen3_vl_transformers.py L1067-1255
def upstream_get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask, *, image_token_id, video_token_id, vision_start_token_id, spatial_merge_size):
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, ids_i in enumerate(total_input_ids):
            ids_i = ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(ids_i == vision_start_token_id).squeeze(1)
            vision_tokens = ids_i[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = ids_i.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
    return position_ids, mrope_position_deltas


def upstream_get_placeholder_mask(input_ids, inputs_embeds, image_features, video_features, *, image_token_id, video_token_id):
    special_image_mask = input_ids == image_token_id
    special_video_mask = input_ids == video_token_id
    n_image_tokens = special_image_mask.sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
        raise ValueError(f'Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}')
    n_video_tokens = special_video_mask.sum()
    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
        raise ValueError(f'Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}')
    return special_image_mask, special_video_mask
# --- end upstream ref ---

text_cfg = Qwen3VLTextConfig(vocab_size=200_000, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=256, tie_word_embeddings=False)
vision_cfg = Qwen3VLVisionConfig(hidden_size=64, intermediate_size=128, depth=2, num_heads=4, in_channels=3, patch_size=14, temporal_patch_size=1, out_hidden_size=128, spatial_merge_size=2, deepstack_visual_indexes=(0, 1))
cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)

torch.manual_seed(0)
model = Qwen3VLModel(cfg)

image_token_id = cfg.image_token_id
video_token_id = cfg.video_token_id
vision_start_token_id = cfg.vision_start_token_id
spatial_merge_size = cfg.vision_config.spatial_merge_size
rope_kw = dict(image_token_id=image_token_id, video_token_id=video_token_id, vision_start_token_id=vision_start_token_id, spatial_merge_size=spatial_merge_size)

image_grid_thw = torch.tensor([[1, 4, 4]])
input_ids = torch.tensor([[10, 11, 12, vision_start_token_id, image_token_id, image_token_id, image_token_id, image_token_id, 13, 14]])
pos_impl, delta_impl = model.get_rope_index(input_ids, image_grid_thw, None, None)
pos_upstream, delta_upstream = upstream_get_rope_index(input_ids, image_grid_thw.clone(), None, None, **rope_kw)
pos_equal = torch.equal(pos_impl, pos_upstream)
delta_equal = torch.equal(delta_impl, delta_upstream)
print(f'get_rope_index[1 image]        : pos shape={tuple(pos_impl.shape)} pos_equal={pos_equal} delta_equal={delta_equal}')
assert pos_equal and delta_equal, 'get_rope_index[1 image] not bit-identical to upstream'

input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
pos_impl, delta_impl = model.get_rope_index(input_ids, None, None, attention_mask)
pos_upstream, delta_upstream = upstream_get_rope_index(input_ids, None, None, attention_mask.clone(), **rope_kw)
pos_equal = torch.equal(pos_impl, pos_upstream)
delta_equal = torch.equal(delta_impl, delta_upstream)
print(f'get_rope_index[no vision, mask]: pos shape={tuple(pos_impl.shape)} pos_equal={pos_equal} delta_equal={delta_equal}')
assert pos_equal and delta_equal, 'get_rope_index[no vision, mask] not bit-identical to upstream'

video_grid_thw = torch.tensor([[2, 4, 4]])
input_ids = torch.tensor([[10, vision_start_token_id, video_token_id, video_token_id, video_token_id, video_token_id, video_token_id, video_token_id, video_token_id, video_token_id, 11]])
pos_impl, delta_impl = model.get_rope_index(input_ids, None, video_grid_thw, None)
pos_upstream, delta_upstream = upstream_get_rope_index(input_ids, None, video_grid_thw.clone(), None, **rope_kw)
pos_equal = torch.equal(pos_impl, pos_upstream)
delta_equal = torch.equal(delta_impl, delta_upstream)
print(f'get_rope_index[video split]    : pos shape={tuple(pos_impl.shape)} pos_equal={pos_equal} delta_equal={delta_equal}')
assert pos_equal and delta_equal, 'get_rope_index[video split] not bit-identical to upstream'

n_raw = 1 * 4 * 4
image_grid_thw = torch.tensor([[1, 4, 4]])
pixel_values = torch.randn(n_raw, 3 * 1 * 14 * 14)
with torch.no_grad():
    image_embeds, deepstack = model.get_image_features(pixel_values, image_grid_thw)
n_merged = n_raw // spatial_merge_size ** 2
print(f'get_image_features             : returns {len(image_embeds)} image(s), each shape={tuple(image_embeds[0].shape)}, deepstack layers={len(deepstack)}')
assert image_embeds[0].shape == (n_merged, text_cfg.hidden_size), f'get_image_features shape: {tuple(image_embeds[0].shape)}'
assert len(deepstack) == len(vision_cfg.deepstack_visual_indexes), f'get_image_features deepstack layers: {len(deepstack)}'

torch.manual_seed(1)
pixel_values = torch.randn(n_raw, 3 * 1 * 14 * 14)
with torch.no_grad():
    image_out, image_deepstack = model.get_image_features(pixel_values, image_grid_thw)
    video_out, video_deepstack = model.get_video_features(pixel_values, image_grid_thw)
out_equal = torch.equal(image_out[0], video_out[0])
deepstack_equal = all(torch.equal(a, b) for a, b in zip(image_deepstack, video_deepstack))
print(f'get_video_features             : delegation parity out_equal={out_equal} deepstack_equal={deepstack_equal}')
assert out_equal and deepstack_equal, 'get_video_features delegation not bit-identical to get_image_features'

inputs_embeds = torch.randn(1, 10, text_cfg.hidden_size)
image_features = torch.randn(4, text_cfg.hidden_size)
input_ids = torch.tensor([[10, 11, 12, vision_start_token_id, image_token_id, image_token_id, image_token_id, image_token_id, 13, 14]])
image_mask_impl, video_mask_impl = model.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_features)
image_mask_upstream, video_mask_upstream = upstream_get_placeholder_mask(input_ids, inputs_embeds, image_features, None, image_token_id=image_token_id, video_token_id=video_token_id)
image_mask_equal = torch.equal(image_mask_impl, image_mask_upstream)
video_mask_equal = torch.equal(video_mask_impl, video_mask_upstream)
print(f'get_placeholder_mask[img]      : shape={tuple(image_mask_impl.shape)} image_mask_equal={image_mask_equal} video_mask_equal={video_mask_equal}')
assert image_mask_equal and video_mask_equal, 'get_placeholder_mask not bit-identical to upstream'

try:
    model.get_placeholder_mask(input_ids, inputs_embeds, image_features=torch.randn(3, text_cfg.hidden_size))
    raise AssertionError('expected ValueError for image token count mismatch')
except ValueError as e:
    assert 'tokens' in str(e) and 'features' in str(e)
print('get_placeholder_mask[mismatch] : raises ValueError as expected')

print('pass (all helpers bit-identical to inline upstream reference)')


# output:
# get_rope_index[1 image]        : pos shape=(3, 1, 10) pos_equal=True delta_equal=True
# get_rope_index[no vision, mask]: pos shape=(3, 1, 6) pos_equal=True delta_equal=True
# get_rope_index[video split]    : pos shape=(3, 1, 11) pos_equal=True delta_equal=True
# get_image_features             : returns 1 image(s), each shape=(4, 128), deepstack layers=2
# get_video_features             : delegation parity out_equal=True deepstack_equal=True
# get_placeholder_mask[img]      : shape=(1, 10, 128) image_mask_equal=True video_mask_equal=True
# get_placeholder_mask[mismatch] : raises ValueError as expected
# pass (all helpers bit-identical to inline upstream reference)
