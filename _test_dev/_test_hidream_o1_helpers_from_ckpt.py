# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-checkpoint tokenizer/processor/config parity for pipeline_hidream_o1_image helpers.

Usage:
    python _test_dev/_test_hidream_o1_helpers_from_ckpt.py
"""
from __future__ import annotations

import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    PATCH_SIZE,
    TIMESTEP_TOKEN_NUM,
    build_t2i_text_sample,
    get_rope_index_fix_point,
)

checkpoint_dir = '/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image'


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

    position_ids, _ = get_rope_index_fix_point(
        spatial_merge_size=1,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[1],
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


cfg = Qwen3VLConfig.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)

print(f'config class             : {type(cfg).__name__}')
print(f'tokenizer class          : {type(tokenizer).__name__}')
print(f'processor class          : {type(processor).__name__}')
for attr in ('image_token_id', 'video_token_id', 'vision_start_token_id'):
    print(f'cfg.{attr:<22} : {getattr(cfg, attr, "<MISSING>")}')
print(f'boi_token                : {getattr(tokenizer, "boi_token", "<fallback:<|boi_token|>>")!r}')
print(f'tms_token                : {getattr(tokenizer, "tms_token", "<fallback:<|tms_token|>>")!r}')
print(f'vocab_size               : {tokenizer.vocab_size}')

sample_template = processor.apply_chat_template([{"role": "user", "content": "a cat"}], tokenize=False, add_generation_prompt=True)
print(f'chat_template (a cat)    : {sample_template!r}')

prompt = "a dog holds a sign that says HiDream release"
cases = [
    ('r1 real 1024x1024 cond   ', prompt, 1024, 1024),
    ('r2 real 1024x1024 uncond ', ' ',    1024, 1024),
    ('r3 real 2048x2048 cond   ', prompt, 2048, 2048),
]
for label, p, h, w in cases:
    image_len = (h // PATCH_SIZE) * (w // PATCH_SIZE)
    impl = build_t2i_text_sample(prompt=p, height=h, width=w, tokenizer=tokenizer, processor=processor, model_config=cfg)
    up = upstream_build_t2i_text_sample(p, h, w, tokenizer, processor, cfg)
    for k in impl:
        assert torch.equal(impl[k], up[k]), f'{label} {k} bit-identity broke:\n impl={impl[k]}\n up  ={up[k]}'
    txt = impl['input_ids'].shape[-1]
    all_len = impl['position_ids'].shape[-1]
    assert all_len == txt + image_len, f'{label} all_seq_len {all_len} != txt {txt} + image_len {image_len}'
    assert int(impl['vinput_mask'].sum()) == image_len, f'{label} vinput_sum != image_len'
    assert int(impl['token_types'].sum()) == image_len + TIMESTEP_TOKEN_NUM, f'{label} token_types_sum != image_len + tms'
    assert impl['position_ids'][0, 0, txt:].tolist() == [4096] * image_len, f'{label} t-dim image portion not at fix_point=4096'
    print(f'{label}: txt={txt} image_len={image_len} all={all_len} vinput_sum={int(impl["vinput_mask"].sum())} token_types_sum={int(impl["token_types"].sum())}')

print('pass')


# output:
# config class             : Qwen3VLConfig
# tokenizer class          : Qwen2Tokenizer
# processor class          : Qwen3VLProcessor
# cfg.image_token_id         : 151655
# cfg.video_token_id         : 151656
# cfg.vision_start_token_id  : 151652
# boi_token                : '<fallback:<|boi_token|>>'
# tms_token                : '<fallback:<|tms_token|>>'
# vocab_size               : 151643
# chat_template (a cat)    : '<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n'
# r1 real 1024x1024 cond   : txt=20 image_len=1024 all=1044 vinput_sum=1024 token_types_sum=1025
# r2 real 1024x1024 uncond : txt=11 image_len=1024 all=1035 vinput_sum=1024 token_types_sum=1025
# r3 real 2048x2048 cond   : txt=20 image_len=4096 all=4116 vinput_sum=4096 token_types_sum=4097
# pass
