# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identical parity for Qwen3VLModel.forward vanilla path vs inline upstream reference.

Usage:
    python _test_dev/_test_qwen3vl_forward_vanilla.py
"""
from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.utils import is_torchdynamo_compiling

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)

TOL = 1e-5

# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/qwen3_vl_transformers.py L1662-1773
# copy of Qwen3VLModel.forward vanilla path (vinputs is None branch,
# dispatch stripped since we test dispatch separately). Reuses `model`'s
# helpers because they are already bit-identical to upstream in
# _test_qwen3vl_helpers.py; the added coverage here is the forward-level
# control flow (image/video scatter, joint branch, M-RoPE staging).
def upstream_forward(model, *, input_ids=None, attention_mask=None, position_ids=None,
                     past_key_values=None, inputs_embeds=None, pixel_values=None,
                     pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
                     cache_position=None, use_cache=False, **kwargs):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError('You must specify exactly one of input_ids or inputs_embeds')
    if inputs_embeds is None:
        inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask = None
    video_mask = None
    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds
    if position_ids is None:
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask['full_attention']
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or model.rope_deltas is None:
            position_ids, rope_deltas = model.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask_tensor)
            model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (cache_position[0] + model.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    outputs = model.language_model(input_ids=None, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, cache_position=cache_position, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds, use_cache=use_cache, **kwargs)
    return Qwen3VLModelOutputWithPast(last_hidden_state=outputs.last_hidden_state, past_key_values=outputs.past_key_values, rope_deltas=model.rope_deltas)
# --- end upstream ref ---


text_cfg = Qwen3VLTextConfig(vocab_size=200_000, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=256, tie_word_embeddings=False)
vision_cfg = Qwen3VLVisionConfig(hidden_size=64, intermediate_size=128, depth=2, num_heads=4, in_channels=3, patch_size=14, temporal_patch_size=1, out_hidden_size=128, spatial_merge_size=2, deepstack_visual_indexes=(0, 1))
cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)

image_token_id = cfg.image_token_id
video_token_id = cfg.video_token_id
vision_start_token_id = cfg.vision_start_token_id

torch.manual_seed(0)
model = Qwen3VLModel(cfg)
model.eval()


def run_pair(**forward_kwargs):
    """Run model.forward (impl) and upstream_forward with identical inputs,
    resetting model.rope_deltas to None before each call so both start from
    the same state. Returns (out_impl, out_upstream)."""
    model.rope_deltas = None
    with torch.no_grad():
        out_impl = model(**forward_kwargs)
    model.rope_deltas = None
    with torch.no_grad():
        out_upstream = upstream_forward(model, **forward_kwargs)
    return out_impl, out_upstream


input_ids_text = torch.tensor([[10, 11, 12, 13, 14, 15]])
out_impl, out_upstream = run_pair(input_ids=input_ids_text, use_cache=False)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
rope_deltas_equal = torch.equal(out_impl.rope_deltas, out_upstream.rope_deltas)
print(f'forward[text only]           : hidden shape={tuple(out_impl.last_hidden_state.shape)} max_diff={hidden_diff:.2e} (tol={TOL:.0e}) rope_deltas_equal={rope_deltas_equal}')
assert hidden_diff < TOL and rope_deltas_equal, 'forward[text only] parity failed'

n_raw_patches = 4
image_grid_thw = torch.tensor([[1, 2, 2]])
video_grid_thw = torch.tensor([[1, 2, 2]])
pixel_values_image = torch.randn(n_raw_patches, 3 * 1 * 14 * 14)
pixel_values_video = torch.randn(n_raw_patches, 3 * 1 * 14 * 14)

input_ids_image = torch.tensor([[10, 11, vision_start_token_id, image_token_id, 12]])
out_impl, out_upstream = run_pair(input_ids=input_ids_image, pixel_values=pixel_values_image, image_grid_thw=image_grid_thw, use_cache=False)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
rope_deltas_equal = torch.equal(out_impl.rope_deltas, out_upstream.rope_deltas)
print(f'forward[text + image]        : hidden shape={tuple(out_impl.last_hidden_state.shape)} max_diff={hidden_diff:.2e} (tol={TOL:.0e}) rope_deltas_equal={rope_deltas_equal}')
assert hidden_diff < TOL and rope_deltas_equal, 'forward[text + image] parity failed'

input_ids_video = torch.tensor([[10, 11, vision_start_token_id, video_token_id, 12]])
out_impl, out_upstream = run_pair(input_ids=input_ids_video, pixel_values_videos=pixel_values_video, video_grid_thw=video_grid_thw, use_cache=False)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
rope_deltas_equal = torch.equal(out_impl.rope_deltas, out_upstream.rope_deltas)
print(f'forward[text + video]        : hidden shape={tuple(out_impl.last_hidden_state.shape)} max_diff={hidden_diff:.2e} (tol={TOL:.0e}) rope_deltas_equal={rope_deltas_equal}')
assert hidden_diff < TOL and rope_deltas_equal, 'forward[text + video] parity failed'

input_ids_joint = torch.tensor([[10, vision_start_token_id, image_token_id, 11, vision_start_token_id, video_token_id, 12]])
out_impl, out_upstream = run_pair(input_ids=input_ids_joint, pixel_values=pixel_values_image, image_grid_thw=image_grid_thw, pixel_values_videos=pixel_values_video, video_grid_thw=video_grid_thw, use_cache=False)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
rope_deltas_equal = torch.equal(out_impl.rope_deltas, out_upstream.rope_deltas)
print(f'forward[text+image+video]    : hidden shape={tuple(out_impl.last_hidden_state.shape)} max_diff={hidden_diff:.2e} (tol={TOL:.0e}) rope_deltas_equal={rope_deltas_equal}')
assert hidden_diff < TOL and rope_deltas_equal, 'forward[text+image+video] joint scatter parity failed'

attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
out_impl, out_upstream = run_pair(input_ids=input_ids_text, attention_mask=attention_mask, use_cache=False)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
rope_deltas_equal = torch.equal(out_impl.rope_deltas, out_upstream.rope_deltas)
print(f'forward[text + attn_mask]    : hidden shape={tuple(out_impl.last_hidden_state.shape)} max_diff={hidden_diff:.2e} (tol={TOL:.0e}) rope_deltas_equal={rope_deltas_equal}')
assert hidden_diff < TOL and rope_deltas_equal, 'forward[text + attn_mask] parity failed'

print('pass (Qwen3VLModel.forward vanilla path bit-identical to inline upstream reference)')


# output:
# forward[text only]           : hidden shape=(1, 6, 128) max_diff=0.00e+00 (tol=1e-05) rope_deltas_equal=True
# forward[text + image]        : hidden shape=(1, 5, 128) max_diff=0.00e+00 (tol=1e-05) rope_deltas_equal=True
# forward[text + video]        : hidden shape=(1, 5, 128) max_diff=0.00e+00 (tol=1e-05) rope_deltas_equal=True
# forward[text+image+video]    : hidden shape=(1, 7, 128) max_diff=0.00e+00 (tol=1e-05) rope_deltas_equal=True
# forward[text + attn_mask]    : hidden shape=(1, 6, 128) max_diff=0.00e+00 (tol=1e-05) rope_deltas_equal=True
# pass (Qwen3VLModel.forward vanilla path bit-identical to inline upstream reference)
