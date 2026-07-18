# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identical parity for pipeline_hidream_o1_image helpers vs inline upstream reference.

Usage:
    python _test_dev/_test_hidream_o1_helpers.py
"""
from __future__ import annotations

import torch

from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    NOISE_SCALE,
    PATCH_SIZE,
    PREDEFINED_RESOLUTIONS,
    T_EPS,
    TIMESTEP_TOKEN_NUM,
    build_t2i_text_sample,
    find_closest_resolution,
    get_rope_index_fix_point,
)

# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/pipeline.py L14-18 @21bcd30471ac
UPSTREAM_PATCH_SIZE = 32
UPSTREAM_TIMESTEP_TOKEN_NUM = 1
UPSTREAM_NOISE_SCALE = 8.0
UPSTREAM_T_EPS = 0.001

# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/utils.py L7-30 @21bcd30471ac
UPSTREAM_PREDEFINED_RESOLUTIONS = [
    (2048, 2048), (2304, 1728), (1728, 2304), (2560, 1440), (1440, 2560),
    (2496, 1664), (1664, 2496), (3104, 1312), (1312, 3104), (2304, 1792), (1792, 2304),
]

def upstream_find_closest_resolution(width, height):
    img_ratio = width / height
    best_res = None
    min_diff = float('inf')
    for w, h in UPSTREAM_PREDEFINED_RESOLUTIONS:
        ratio = w / h
        diff = abs(ratio - img_ratio)
        if diff < min_diff:
            min_diff = diff
            best_res = (w, h)
    return best_res


print(f'constants                : PATCH_SIZE={PATCH_SIZE} TIMESTEP_TOKEN_NUM={TIMESTEP_TOKEN_NUM} NOISE_SCALE={NOISE_SCALE} T_EPS={T_EPS}')
assert PATCH_SIZE == UPSTREAM_PATCH_SIZE, f'PATCH_SIZE {PATCH_SIZE} != {UPSTREAM_PATCH_SIZE}'
assert TIMESTEP_TOKEN_NUM == UPSTREAM_TIMESTEP_TOKEN_NUM, f'TIMESTEP_TOKEN_NUM {TIMESTEP_TOKEN_NUM} != {UPSTREAM_TIMESTEP_TOKEN_NUM}'
assert NOISE_SCALE == UPSTREAM_NOISE_SCALE, f'NOISE_SCALE {NOISE_SCALE} != {UPSTREAM_NOISE_SCALE}'
assert T_EPS == UPSTREAM_T_EPS, f'T_EPS {T_EPS} != {UPSTREAM_T_EPS}'

print(f'PREDEFINED_RESOLUTIONS   : n={len(PREDEFINED_RESOLUTIONS)}')
assert len(PREDEFINED_RESOLUTIONS) == len(UPSTREAM_PREDEFINED_RESOLUTIONS), \
    f'len mismatch: {len(PREDEFINED_RESOLUTIONS)} vs {len(UPSTREAM_PREDEFINED_RESOLUTIONS)}'
for i, (impl, up) in enumerate(zip(PREDEFINED_RESOLUTIONS, UPSTREAM_PREDEFINED_RESOLUTIONS)):
    assert impl == up, f'PREDEFINED_RESOLUTIONS[{i}] {impl} != {up}'

# 11 in-list exact-match + 10 off-bucket cases (aspect boundaries, extremes)
cases = [
    (2048, 2048, (2048, 2048)),      # perfect 1:1 in-list
    (2304, 1728, (2304, 1728)),      # 4:3 in-list
    (1728, 2304, (1728, 2304)),      # 3:4 in-list
    (2560, 1440, (2560, 1440)),      # 16:9 in-list
    (1440, 2560, (1440, 2560)),      # 9:16 in-list
    (2496, 1664, (2496, 1664)),      # 3:2 in-list
    (1664, 2496, (1664, 2496)),      # 2:3 in-list
    (3104, 1312, (3104, 1312)),      # ultra-wide in-list
    (1312, 3104, (1312, 3104)),      # ultra-tall in-list
    (2304, 1792, (2304, 1792)),      # 1.286 in-list
    (1792, 2304, (1792, 2304)),      # 0.778 in-list
    (1024, 1024, (2048, 2048)),      # 1:1 upscaled
    (4096, 4096, (2048, 2048)),      # 1:1 downscaled
    (3000, 1000, (3104, 1312)),      # ~3:1 extreme wide
    (1000, 3000, (1312, 3104)),      # ~1:3 extreme tall
    (2000, 2100, (2048, 2048)),      # near-square off-bucket
    (1920, 1080, (2560, 1440)),      # 16:9 off-bucket
    (1080, 1920, (1440, 2560)),      # 9:16 off-bucket
    (1600, 900,  (2560, 1440)),      # 16:9 off-bucket small
    (960, 540,   (2560, 1440)),      # 16:9 tiny
    (100, 300,   (1312, 3104)),      # tiny 1:3
]
print(f'find_closest_resolution  : n_cases={len(cases)} (11 in-list + 10 off-bucket)')
for i, (w, h, expected) in enumerate(cases):
    impl = find_closest_resolution(w, h)
    up = upstream_find_closest_resolution(w, h)
    assert impl == up, f'case {i} input=({w},{h}) impl={impl} != upstream={up}'
    assert impl == expected, f'case {i} input=({w},{h}) impl={impl} != expected sanity {expected}'

# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/utils.py L67-134 @21bcd30471ac
def upstream_get_rope_index_fix_point(
    spatial_merge_size, image_token_id, video_token_id, vision_start_token_id,
    input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None,
    skip_vision_start_token=None, fix_point=4096,
):
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
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

                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                        fix_point = 0
                else:
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
    else:
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


IMAGE_ID, VIDEO_ID, VS_ID = 200, 201, 202

# Case 1: T2I typical -- 1 image, skip=[1], fix_point=4096 (HiDream layout: VS in vision_tokens block)
# sequence: [1, 2, VS, I×5]; grid=(1,2,3)=6 slots; text_len=(3-0)-1=2; fix_point path
c1_ids = torch.tensor([[1, 2, VS_ID] + [IMAGE_ID] * 5], dtype=torch.long)
c1_grid = torch.tensor([[1, 2, 3]], dtype=torch.long)
c1_impl_p, c1_impl_d = get_rope_index_fix_point(
    spatial_merge_size=1, image_token_id=IMAGE_ID, video_token_id=VIDEO_ID, vision_start_token_id=VS_ID,
    input_ids=c1_ids, image_grid_thw=c1_grid, skip_vision_start_token=[1],
)
c1_up_p, c1_up_d = upstream_get_rope_index_fix_point(
    1, IMAGE_ID, VIDEO_ID, VS_ID, input_ids=c1_ids.clone(), image_grid_thw=c1_grid.clone(),
    skip_vision_start_token=[1],
)
assert torch.equal(c1_impl_p, c1_up_p), f'c1 position_ids diverge:\n impl={c1_impl_p}\n up  ={c1_up_p}'
assert torch.equal(c1_impl_d, c1_up_d), f'c1 mrope_delta diverge: impl={c1_impl_d} up={c1_up_d}'
assert c1_impl_p[0, 0, 2:].tolist() == [4096] * 6, f'c1 t-dim image portion must be all 4096: {c1_impl_p[0, 0, 2:]}'
assert c1_impl_p[1, 0, 2:].tolist() == [4096, 4096, 4096, 4097, 4097, 4097], f'c1 h-dim: {c1_impl_p[1, 0, 2:]}'
assert c1_impl_p[2, 0, 2:].tolist() == [4096, 4097, 4098, 4096, 4097, 4098], f'c1 w-dim: {c1_impl_p[2, 0, 2:]}'
print(f'c1 T2I fix-point         : shape={tuple(c1_impl_p.shape)} delta={c1_impl_d.item()} image_at=4096+')

# Case 2: vanilla behavior -- skip=[0], image positions continue after text (Qwen3-VL native layout)
# sequence: [1, 2, VS, I×6]; grid=(1,2,3)=6; text_len=3 (incl. VS); image offset by 3
c2_ids = torch.tensor([[1, 2, VS_ID] + [IMAGE_ID] * 6], dtype=torch.long)
c2_grid = torch.tensor([[1, 2, 3]], dtype=torch.long)
c2_impl_p, c2_impl_d = get_rope_index_fix_point(
    spatial_merge_size=1, image_token_id=IMAGE_ID, video_token_id=VIDEO_ID, vision_start_token_id=VS_ID,
    input_ids=c2_ids, image_grid_thw=c2_grid, skip_vision_start_token=[0],
)
c2_up_p, c2_up_d = upstream_get_rope_index_fix_point(
    1, IMAGE_ID, VIDEO_ID, VS_ID, input_ids=c2_ids.clone(), image_grid_thw=c2_grid.clone(),
    skip_vision_start_token=[0],
)
assert torch.equal(c2_impl_p, c2_up_p), f'c2 position_ids diverge:\n impl={c2_impl_p}\n up  ={c2_up_p}'
assert torch.equal(c2_impl_d, c2_up_d), f'c2 mrope_delta diverge: impl={c2_impl_d} up={c2_up_d}'
assert c2_impl_p[0, 0, 3:].tolist() == [3] * 6, f'c2 t-dim image portion should be 3 (text_len offset): {c2_impl_p[0, 0, 3:]}'
assert c2_impl_p[2, 0, 3:].tolist() == [3, 4, 5, 3, 4, 5], f'c2 w-dim: {c2_impl_p[2, 0, 3:]}'
print(f'c2 vanilla skip=0        : shape={tuple(c2_impl_p.shape)} delta={c2_impl_d.item()} image_at=text_len+')

# Case 3: text-only, attention_mask=None -- naive arange fallback
c3_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
c3_impl_p, c3_impl_d = get_rope_index_fix_point(
    spatial_merge_size=1, image_token_id=IMAGE_ID, video_token_id=VIDEO_ID, vision_start_token_id=VS_ID,
    input_ids=c3_ids,
)
c3_up_p, c3_up_d = upstream_get_rope_index_fix_point(
    1, IMAGE_ID, VIDEO_ID, VS_ID, input_ids=c3_ids.clone(),
)
assert torch.equal(c3_impl_p, c3_up_p), f'c3 position_ids diverge:\n impl={c3_impl_p}\n up  ={c3_up_p}'
assert torch.equal(c3_impl_d, c3_up_d), f'c3 mrope_delta diverge: impl={c3_impl_d} up={c3_up_d}'
assert c3_impl_p.tolist() == [[[0, 1, 2, 3]]] * 3, f'c3 arange fallback: {c3_impl_p}'
print(f'c3 text-only no mask     : shape={tuple(c3_impl_p.shape)} delta={c3_impl_d.item()}')

# Case 4: text-only with padding mask -- cumsum-based positions, masked slots forced to 1
c4_ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
c4_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
c4_impl_p, c4_impl_d = get_rope_index_fix_point(
    spatial_merge_size=1, image_token_id=IMAGE_ID, video_token_id=VIDEO_ID, vision_start_token_id=VS_ID,
    input_ids=c4_ids, attention_mask=c4_mask,
)
c4_up_p, c4_up_d = upstream_get_rope_index_fix_point(
    1, IMAGE_ID, VIDEO_ID, VS_ID, input_ids=c4_ids.clone(), attention_mask=c4_mask.clone(),
)
assert torch.equal(c4_impl_p, c4_up_p), f'c4 position_ids diverge:\n impl={c4_impl_p}\n up  ={c4_up_p}'
assert torch.equal(c4_impl_d, c4_up_d), f'c4 mrope_delta diverge: impl={c4_impl_d} up={c4_up_d}'
assert c4_impl_p[0, 0].tolist() == [0, 1, 2, 3, 1, 1], f'c4 cumsum masked positions: {c4_impl_p[0, 0]}'
print(f'c4 text-only with pad    : shape={tuple(c4_impl_p.shape)} delta={c4_impl_d.item()}')

# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/pipeline.py L36-77 @21bcd30471ac
def upstream_build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config):
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + boi_token
        + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = upstream_get_rope_index_fix_point(
        1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=None, skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

    vinput_mask = (token_types == 1)
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'token_types': token_types_bin,
        'vinput_mask': vinput_mask,
    }


class _FakeTokenizer:
    """Deterministic char-level tokenizer: token_id = char position + 10 (so ids never collide with 200/201/202)."""
    boi_token = "<BOI>"
    tms_token = "<TMS>"

    def encode(self, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return torch.tensor([[i + 10 for i in range(len(text))]], dtype=torch.int64)


class _FakeProcessor:
    """Deterministic chat template: wraps prompt in <USER>...</USER>."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<USER>{messages[0]['content']}</USER>"


class _FakeConfig:
    image_token_id = 200
    video_token_id = 201
    vision_start_token_id = 202


tok, proc, cfg = _FakeTokenizer(), _FakeProcessor(), _FakeConfig()

# Case 5: T2I cond -- prompt="cat", 64×96 image (H//32=2, W//32=3, image_len=6)
c5_impl = build_t2i_text_sample(prompt="cat", height=64, width=96, tokenizer=tok, processor=proc, model_config=cfg)
c5_up = upstream_build_t2i_text_sample("cat", 64, 96, tok, proc, cfg)
assert c5_impl.keys() == c5_up.keys() == {'input_ids', 'position_ids', 'token_types', 'vinput_mask'}, f'c5 keys diverge: {c5_impl.keys()} vs {c5_up.keys()}'
for k in c5_impl:
    assert torch.equal(c5_impl[k], c5_up[k]), f'c5 {k} diverge:\n impl={c5_impl[k]}\n up  ={c5_up[k]}'
# template = "<USER>cat</USER><BOI><TMS>" = 26 chars -> txt_seq_len = 26; image_len = 6; all_seq_len = 32
assert c5_impl['input_ids'].shape == (1, 26), f"c5 input_ids shape {c5_impl['input_ids'].shape}"
assert c5_impl['position_ids'].shape == (3, 1, 32), f"c5 position_ids shape {c5_impl['position_ids'].shape}"
assert c5_impl['token_types'].shape == (1, 32), f"c5 token_types shape {c5_impl['token_types'].shape}"
assert c5_impl['vinput_mask'].shape == (1, 32) and c5_impl['vinput_mask'].dtype == torch.bool, f"c5 vinput_mask shape/dtype {c5_impl['vinput_mask'].shape}/{c5_impl['vinput_mask'].dtype}"
# token_types: 25 zeros (text incl. boi) + 7 ones (tms + image_len). Note: tms position is also 1 (binary form, upstream sets type 3 then squashes >0).
assert c5_impl['token_types'][0].tolist() == [0] * 25 + [1] * 7, f"c5 token_types: {c5_impl['token_types'][0]}"
# vinput_mask: True only on image_len=6 slots (excludes tms at position 25).
assert c5_impl['vinput_mask'][0].tolist() == [False] * 26 + [True] * 6, f"c5 vinput_mask: {c5_impl['vinput_mask'][0]}"
# Image position ids: fix_point=4096 anchor (verified via 5c logic).
assert c5_impl['position_ids'][0, 0, 26:].tolist() == [4096] * 6, f"c5 t-dim image portion: {c5_impl['position_ids'][0, 0, 26:]}"
print(f"c5 T2I cond              : txt={c5_impl['input_ids'].shape[-1]} all={c5_impl['position_ids'].shape[-1]} vinput_sum={int(c5_impl['vinput_mask'].sum())} token_types_sum={int(c5_impl['token_types'].sum())}")

# Case 6: T2I uncond -- prompt=" " (blank, mirrors generate_image's CFG unconditional branch), same resolution
c6_impl = build_t2i_text_sample(prompt=" ", height=64, width=96, tokenizer=tok, processor=proc, model_config=cfg)
c6_up = upstream_build_t2i_text_sample(" ", 64, 96, tok, proc, cfg)
for k in c6_impl:
    assert torch.equal(c6_impl[k], c6_up[k]), f'c6 {k} diverge:\n impl={c6_impl[k]}\n up  ={c6_up[k]}'
# template = "<USER> </USER><BOI><TMS>" = 24 chars -> txt_seq_len = 24; all_seq_len = 30
assert c6_impl['input_ids'].shape == (1, 24), f"c6 input_ids shape {c6_impl['input_ids'].shape}"
assert c6_impl['position_ids'].shape == (3, 1, 30), f"c6 position_ids shape {c6_impl['position_ids'].shape}"
print(f"c6 T2I uncond            : txt={c6_impl['input_ids'].shape[-1]} all={c6_impl['position_ids'].shape[-1]} vinput_sum={int(c6_impl['vinput_mask'].sum())} token_types_sum={int(c6_impl['token_types'].sum())}")

# Case 7: larger resolution -- 256×256 (H//32=8, W//32=8, image_len=64), same prompt
c7_impl = build_t2i_text_sample(prompt="cat", height=256, width=256, tokenizer=tok, processor=proc, model_config=cfg)
c7_up = upstream_build_t2i_text_sample("cat", 256, 256, tok, proc, cfg)
for k in c7_impl:
    assert torch.equal(c7_impl[k], c7_up[k]), f'c7 {k} diverge:\n impl={c7_impl[k]}\n up  ={c7_up[k]}'
# txt=26, image_len=64, all=90
assert c7_impl['position_ids'].shape == (3, 1, 90), f"c7 position_ids shape {c7_impl['position_ids'].shape}"
assert int(c7_impl['vinput_mask'].sum()) == 64, f"c7 vinput sum {int(c7_impl['vinput_mask'].sum())}"
print(f"c7 T2I 256x256           : txt={c7_impl['input_ids'].shape[-1]} all={c7_impl['position_ids'].shape[-1]} vinput_sum={int(c7_impl['vinput_mask'].sum())} token_types_sum={int(c7_impl['token_types'].sum())}")

# Pipeline sub-step tests: _resolve_generation_params, _validate_static_config,
# _prepare_noise_and_patchify, _forward_once.
from dataclasses import dataclass, field
from types import SimpleNamespace

from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    HiDreamO1ImagePipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

_resolve = HiDreamO1ImagePipeline._resolve_generation_params
_validate_static = HiDreamO1ImagePipeline._validate_static_config
_prep_noise = HiDreamO1ImagePipeline._prepare_noise_and_patchify
_forward_once = HiDreamO1ImagePipeline._forward_once

def _make_req(prompts=None, **sp_kwargs):
    if prompts is None:
        prompts = ["a cat"]
    sp = OmniDiffusionSamplingParams(**sp_kwargs)
    return OmniDiffusionRequest(prompts=prompts, sampling_params=sp, request_id="test")

def _assert_raises(err_type, fn, label):
    try:
        fn()
    except err_type as caught:
        return caught
    raise AssertionError(f'expected {err_type.__name__} for {label}, no raise')

# Case 8: resolve cond-only path -- portrait 1080x1920 snaps to 1440x2560,
# returns (prompt, snapped_h, snapped_w, steps, seed, guidance_scale).
c8_req = _make_req(height=1920, width=1080, guidance_scale=1.0, seed=42)
c8_out = _resolve(None, c8_req)
assert c8_out == ('a cat', 2560, 1440, 50, 42, 1.0), f'c8 unexpected: {c8_out}'
print(f'c8 resolve cond-only     : returned 6-tuple {c8_out}')

# Case 9: resolve generator compatibility.
c9_gen = torch.Generator().manual_seed(7)
c9_out = _resolve(None, _make_req(seed=7, generator=c9_gen, guidance_scale=5.0))
assert c9_out == ('a cat', 2048, 2048, 50, 7, 5.0), f'c9 unexpected: {c9_out}'

c9b_req = _make_req(guidance_scale=1.0)
c9b_req.sampling_params.seed = None
c9b_req.sampling_params.generator = torch.Generator().manual_seed(11)
c9b_out = _resolve(None, c9b_req)
assert c9b_out == ('a cat', 2048, 2048, 50, 11, 1.0), f'c9b unexpected: {c9b_out}'

c9c_err = _assert_raises(
    ValueError,
    lambda: _resolve(
        None,
        _make_req(seed=5, generator=torch.Generator().manual_seed(6)),
    ),
    'seed/generator mismatch',
)
assert 'seed/generator mismatch' in str(c9c_err), f'c9c mismatch message: {c9c_err}'
print(f'c9 generator compat      : explicit seed+generator ok, generator-only seed recovery ok, mismatch rejected')

# Case 10: resolve request-level fail-fast x 5
# (timesteps/latents/num_outputs=0/num_outputs=2/multi-prompt).
# CFG (guidance_scale > 1.0 and do_classifier_free_guidance) is now accepted, see c16.
c10_cases = [
    ('timesteps',     lambda: _resolve(None, _make_req(seed=1, timesteps=torch.tensor([500.0])))),
    ('latents',       lambda: _resolve(None, _make_req(seed=1, latents=torch.zeros(1)))),
    ('num_outputs=0', lambda: _resolve(None, _make_req(seed=1, num_outputs_per_prompt=0))),
    ('num_outputs=2', lambda: _resolve(None, _make_req(seed=1, num_outputs_per_prompt=2))),
    ('multi-prompt',  lambda: _resolve(None, _make_req(prompts=['a', 'b'], seed=1))),
]
for label, fn in c10_cases:
    _assert_raises(NotImplementedError, fn, label)
print(f'c10 resolve fail-fast x 5: all 5 unsupported request-level checks raised NotImplementedError')

# Case 11: resolve prompt type validation -- 3 accepted, 4 rejected
c11_accept = [
    ('str',            'hello',            'hello'),
    ("dict prompt=x",  {'prompt': 'x'},    'x'),
    ("dict prompt=''", {'prompt': ''},     ''),
]
c11_reject = [
    ('empty dict',     {}),
    ('pretokenized',   {'prompt_ids': [1, 2]}),
    ('dict prompt=list', {'prompt': [1, 2]}),
    ('list prompt',    [1, 2]),
]
for label, prompt, expected in c11_accept:
    out = _resolve(None, _make_req(prompts=[prompt], seed=1, guidance_scale=1.0))
    assert out[0] == expected, f'c11 {label}: got {out[0]!r}, expected {expected!r}'
for label, prompt in c11_reject:
    _assert_raises(TypeError, lambda p=prompt: _resolve(None, _make_req(prompts=[p], seed=1, guidance_scale=1.0)), label)
print(f'c11 resolve prompt type  : 3 accepted (str/dict-x/dict-empty-str) + 4 rejected (TypeError)')

# Case 12: resolve boundary -- h/w <= 0 (ValueError), steps <= 0 (ValueError), seed=None+generator=None (RuntimeError)
_assert_raises(ValueError, lambda: _resolve(None, _make_req(height=0, seed=1)), 'h=0')
_assert_raises(ValueError, lambda: _resolve(None, _make_req(width=-1, seed=1)), 'w=-1')
_assert_raises(ValueError, lambda: _resolve(None, _make_req(num_inference_steps=0, seed=1)), 'steps=0')
c12_req = _make_req(seed=1)
c12_req.sampling_params.seed = None
c12_err = _assert_raises(RuntimeError, lambda: _resolve(None, c12_req), 'seed=None')
assert 'request initialization' in str(c12_err), f'c12 seed=None message: {c12_err}'
print(f'c12 resolve boundary     : h<=0/w<=0/steps<=0 (ValueError) + seed=None+generator=None (RuntimeError)')

@dataclass
class _ParallelStub:
    cfg_parallel_size: int = 1

@dataclass
class _ConfigStub:
    parallel_config: _ParallelStub = field(default_factory=_ParallelStub)

def _make_static_stub(dtype=torch.bfloat16, cfg_ps=1):
    stub = SimpleNamespace()
    stub.dtype = dtype
    stub.od_config = _ConfigStub(parallel_config=_ParallelStub(cfg_parallel_size=cfg_ps))
    return stub

# Case 13: init static fail-fast.
_assert_raises(NotImplementedError, lambda: _validate_static(_make_static_stub(dtype=torch.float16)), 'dtype=float16')
_assert_raises(NotImplementedError, lambda: _validate_static(_make_static_stub(cfg_ps=2)), 'cfg_parallel=2')
_validate_static(_make_static_stub())
print(f'c13 static config        : dtype/cfg_parallel raised; bf16+cfg=1 accepted')

# Case 14: noise determinism -- same seed twice on CPU is bit-identical
c14_z1 = _prep_noise(None, 64, 96, seed=42, dtype=torch.float32, device=torch.device('cpu'))
c14_z2 = _prep_noise(None, 64, 96, seed=42, dtype=torch.float32, device=torch.device('cpu'))
assert torch.equal(c14_z1, c14_z2), 'c14 noise not deterministic on CPU'
print(f'c14 noise determinism    : torch.equal on same-seed CPU noise (64x96)')

# Case 15: noise shape + dtype + std -- 64x96 -> 2x3 patches -> image_len=6, patch_dim=3*32*32=3072
c15_z = _prep_noise(None, 64, 96, seed=0, dtype=torch.float32, device=torch.device('cpu'))
assert c15_z.shape == (1, 6, 3072), f'c15 shape {c15_z.shape}'
assert c15_z.dtype == torch.float32, f'c15 dtype {c15_z.dtype}'
c15_std = c15_z.std().item()
assert NOISE_SCALE * 0.8 < c15_std < NOISE_SCALE * 1.2, f'c15 std {c15_std} out of [{NOISE_SCALE*0.8}, {NOISE_SCALE*1.2}]'
print(f'c15 noise shape+dtype+std: shape=(1,6,3072) dtype=fp32 std={c15_std:.3f} in [{NOISE_SCALE*0.8:.1f}, {NOISE_SCALE*1.2:.1f}]')

class _ItemCounter:
    """Monkey-patch Tensor.item to count host-sync calls issued from within a with-block."""
    def __init__(self):
        self.count = 0
        self._orig = torch.Tensor.item
    def __enter__(self):
        counter = self
        original = self._orig
        def counted(tensor):
            counter.count += 1
            return original(tensor)
        torch.Tensor.item = counted
        return self
    def __exit__(self, *_):
        torch.Tensor.item = self._orig

class _FakeModel:
    """Returns x_pred = arange over full sequence so slice indices are verifiable."""
    def __init__(self, patch_dim):
        self.patch_dim = patch_dim
    def __call__(self, **kw):
        seq_len = kw['input_ids'].shape[1] + kw['vinputs'].shape[1]
        x_pred = torch.arange(seq_len * self.patch_dim, dtype=torch.float32).reshape(1, seq_len, self.patch_dim)
        return SimpleNamespace(x_pred=x_pred)

# Case 16: _forward_once FakeModel slice correctness + no host sync.
# Sample: text_len=3, image_len=6, total seq_len=9. vinput_mask True at indices 3..8.
c16_sample = {
    'input_ids':    torch.zeros((1, 3), dtype=torch.long),
    'position_ids': torch.zeros((3, 1, 9), dtype=torch.long),
    'token_types':  torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1]]),
    'vinput_mask':  torch.tensor([[False, False, False, True, True, True, True, True, True]]),
}
c16_z_in = torch.zeros((1, 6, 3072), dtype=torch.float32)
c16_t = torch.tensor(0.5)
c16_pipe = SimpleNamespace(model=_FakeModel(patch_dim=3072), device=torch.device('cpu'), dtype=torch.bfloat16)

with _ItemCounter() as c16_counter:
    c16_out = _forward_once(c16_pipe, c16_sample, c16_z_in, c16_t)
assert c16_out.shape == (1, 6, 3072), f'c16 out shape {c16_out.shape}'
c16_expected = torch.arange(9 * 3072, dtype=torch.float32).reshape(9, 3072)[3:9].unsqueeze(0)
assert torch.equal(c16_out, c16_expected), 'c16 slice mismatch'
assert c16_counter.count == 0, f'c16 _forward_once made {c16_counter.count} host-sync (.item()) calls; expected 0'
print(f'c16 _forward_once slice  : shape=(1,6,3072) + slice correct + {c16_counter.count} host sync (.item() calls)')

# Case 17: resolve CFG accept -- guidance_scale > 1.0 no longer raises; pass through as-is.
# Also verify that setting sp.do_classifier_free_guidance=True alone is accepted (forward() ignores it
# in favor of guidance_scale > 1.0 as the sole CFG criterion, mirroring upstream).
c17_cfg = _resolve(None, _make_req(seed=7, guidance_scale=5.0))
assert c17_cfg == ('a cat', 2048, 2048, 50, 7, 5.0), f'c17_cfg: {c17_cfg}'
c17_do_cfg = _resolve(None, _make_req(seed=7, do_classifier_free_guidance=True))
assert c17_do_cfg == ('a cat', 2048, 2048, 50, 7, 1.0), f'c17_do_cfg: {c17_do_cfg}'
print(f'c17 resolve CFG accept   : guidance=5.0 -> tuple[..., 5.0]; do_cfg=True alone -> tuple[..., 1.0]')

print('pass')


# output:
# constants                : PATCH_SIZE=32 TIMESTEP_TOKEN_NUM=1 NOISE_SCALE=8.0 T_EPS=0.001
# PREDEFINED_RESOLUTIONS   : n=11
# find_closest_resolution  : n_cases=21 (11 in-list + 10 off-bucket)
# c1 T2I fix-point         : shape=(3, 1, 8) delta=4091 image_at=4096+
# c2 vanilla skip=0        : shape=(3, 1, 9) delta=-3 image_at=text_len+
# c3 text-only no mask     : shape=(3, 1, 4) delta=0
# c4 text-only with pad    : shape=(3, 1, 6) delta=-2
# c5 T2I cond              : txt=26 all=32 vinput_sum=6 token_types_sum=7
# c6 T2I uncond            : txt=24 all=30 vinput_sum=6 token_types_sum=7
# c7 T2I 256x256           : txt=26 all=90 vinput_sum=64 token_types_sum=65
# c8 resolve cond-only     : returned 6-tuple ('a cat', 2560, 1440, 50, 42, 1.0)
# [TRACE]* request __post_init__ logs may appear between cases:
#         - unset guidance_scale=0.0 is normalized to effective guidance_scale=1.0
#         - guidance_scale_2 is auto-filled from guidance_scale
#         - guidance_scale=5.0 marks the request as guidance_scale_provided=True
# c9 generator compat      : explicit seed+generator ok, generator-only seed recovery ok, mismatch rejected
# c10 resolve fail-fast x 5: all 5 unsupported request-level checks raised NotImplementedError
# c11 resolve prompt type  : 3 accepted (str/dict-x/dict-empty-str) + 4 rejected (TypeError)
# c12 resolve boundary     : h<=0/w<=0/steps<=0 (ValueError) + seed=None+generator=None (RuntimeError)
# c13 static config        : dtype/cfg_parallel raised; bf16+cfg=1 accepted
# c14 noise determinism    : torch.equal on same-seed CPU noise (64x96)
# c15 noise shape+dtype+std: shape=(1,6,3072) dtype=fp32 std=8.055 in [6.4, 9.6]
# c16 _forward_once slice  : shape=(1,6,3072) + slice correct + 0 host sync (.item() calls)
# c17 resolve CFG accept   : guidance=5.0 -> tuple[..., 5.0]; do_cfg=True alone -> tuple[..., 1.0]
# pass
