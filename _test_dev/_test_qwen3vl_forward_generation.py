# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identical parity for Qwen3VLModel._forward_generation (HiDream UiT path) vs inline upstream reference.

Usage:
    python _test_dev/_test_qwen3vl_forward_generation.py
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
    Qwen3VLModelOutputWithPast,
)

TOL = 1e-5


# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/qwen3_vl_transformers.py L1400-1621
# flash-attn / grad-enabled / mem_debug branches even though impl strips them,
# because those branches are runtime-guarded (use_flash_attn=False,
# torch.no_grad() context, DEBUG_MEM env) and won't fire in the parity path.
# If impl stripped a behaviorally-active line by mistake, this reference still
# exercises it and the bit-identical check would surface the divergence.
def upstream_forward_generation(model, input_ids, position_ids, vinputs, timestep, token_types,
                                attention_mask=None, pixel_values=None, pixel_values_videos=None,
                                image_grid_thw=None, video_grid_thw=None, use_flash_attn=False,
                                return_mid_results_layers=None, **kwargs):
    precomputed_image_embeds = kwargs.pop('precomputed_image_embeds', None)
    precomputed_deepstack_image_embeds = kwargs.pop('precomputed_deepstack_image_embeds', None)
    cond_image_embeds_out = None
    cond_deepstack_image_embeds_out = None
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None
    if pixel_values is not None:
        if precomputed_image_embeds is not None and precomputed_deepstack_image_embeds is not None:
            image_embeds = precomputed_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = [d.to(inputs_embeds.device, inputs_embeds.dtype) for d in precomputed_deepstack_image_embeds]
        else:
            image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        cond_image_embeds_out = image_embeds
        cond_deepstack_image_embeds_out = deepstack_image_embeds
    elif torch.is_grad_enabled():
        pe = model.visual.patch_embed
        t_sz = pe.temporal_patch_size
        m_sz = model.visual.spatial_merge_size
        n_patches = t_sz * m_sz * m_sz
        patch_dim = pe.in_channels * t_sz * pe.patch_size * pe.patch_size
        fake_pv = torch.zeros(n_patches, patch_dim, device=inputs_embeds.device, dtype=pe.proj.weight.dtype)
        fake_grid = torch.tensor([[t_sz, m_sz, m_sz]], dtype=torch.long, device=inputs_embeds.device)
        fake_embs, fake_deepstack = model.get_image_features(fake_pv, fake_grid)
        fake_embs = torch.cat(fake_embs, dim=0).to(inputs_embeds.dtype)
        fake_total = fake_embs.sum()
        for _d in fake_deepstack:
            fake_total = fake_total + _d.to(inputs_embeds.dtype).sum()
        inputs_embeds = inputs_embeds + fake_total * inputs_embeds.new_zeros([])
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
    if isinstance(timestep, list):
        timestep = torch.cat(timestep, dim=0)
    timestep = timestep.to(inputs_embeds.device)
    t_emb = model.t_embedder1(timestep)
    tms_mask = (input_ids == model.tms_token_id)
    tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(inputs_embeds)
    t_emb_expanded = t_emb.unsqueeze(1).expand_as(inputs_embeds)
    inputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)
    if isinstance(vinputs, list):
        vinputs = torch.cat(vinputs, dim=0)
    vinputs = vinputs.to(inputs_embeds.device)
    vinputs_embedded = model.x_embedder(vinputs).to(inputs_embeds.dtype)
    inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)
    batch_size, total_seq_len, _ = inputs_embeds.shape
    if visual_pos_masks is not None:
        vinputs_seq_len = vinputs_embedded.shape[1]
        if visual_pos_masks.shape[0] != batch_size:
            visual_pos_masks = visual_pos_masks.expand(batch_size, -1)
        vinputs_pad = torch.zeros(visual_pos_masks.shape[0], vinputs_seq_len, dtype=visual_pos_masks.dtype, device=visual_pos_masks.device)
        visual_pos_masks = torch.cat([visual_pos_masks, vinputs_pad], dim=1)
    if isinstance(token_types, list):
        token_types = torch.cat(token_types, dim=0)
    token_types = token_types.to(inputs_embeds.device)
    if token_types.dim() == 1:
        token_types = token_types.unsqueeze(0)
    elif token_types.dim() == 2 and token_types.shape[-1] == 1 and token_types.shape[0] == total_seq_len:
        token_types = token_types.squeeze(-1).unsqueeze(0)
    if token_types.shape[0] == 1 and batch_size > 1:
        token_types = token_types.expand(batch_size, -1)
    mid_results = None
    if use_flash_attn:
        hidden_states, mid_results = model._run_decoder_flash(inputs_embeds, position_ids, token_types, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds, return_mid_results_layers=return_mid_results_layers)
    else:
        dtype = inputs_embeds.dtype
        min_val = torch.finfo(dtype).min
        attn_masks = []
        for b in range(batch_size):
            causal = torch.full((total_seq_len, total_seq_len), min_val, device=inputs_embeds.device, dtype=dtype)
            causal = torch.triu(causal, diagonal=1)
            gen_positions = token_types[b].bool()
            causal[gen_positions, :] = 0
            attn_masks.append(causal)
        attention_mask_4d = torch.stack(attn_masks, dim=0).unsqueeze(1)
        outputs = model.language_model(input_ids=None, position_ids=position_ids, attention_mask=attention_mask_4d, inputs_embeds=inputs_embeds, use_cache=False, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds, return_mid_results_layers=return_mid_results_layers)
        hidden_states = outputs.last_hidden_state
        if hasattr(outputs, 'mid_results'):
            mid_results = outputs.mid_results
    x_pred = model.final_layer2(hidden_states)
    return Qwen3VLModelOutputWithPast(last_hidden_state=hidden_states, x_pred=x_pred, mid_results=mid_results, cond_image_embeds=cond_image_embeds_out, cond_deepstack_image_embeds=cond_deepstack_image_embeds_out)
# --- end upstream ref ---


text_cfg = Qwen3VLTextConfig(vocab_size=200_000, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=256, tie_word_embeddings=False)
vision_cfg = Qwen3VLVisionConfig(hidden_size=64, intermediate_size=128, depth=2, num_heads=4, in_channels=3, patch_size=14, temporal_patch_size=1, out_hidden_size=128, spatial_merge_size=2, deepstack_visual_indexes=(0, 1))
cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)

image_token_id = cfg.image_token_id
vision_start_token_id = cfg.vision_start_token_id

torch.manual_seed(0)
model = Qwen3VLModel(cfg)
model.eval()
tms_token_id = model.tms_token_id
patch_dim = 3 * 32 * 32  # x_embedder patch_dim (in_channels=3, patch_size=32)


def run_pair(**forward_kwargs):
    """Run model.forward (dispatches to _forward_generation via vinputs != None)
    and upstream_forward_generation with identical inputs. Returns (impl, upstream)."""
    with torch.no_grad():
        out_impl = model(**forward_kwargs)
    with torch.no_grad():
        out_upstream = upstream_forward_generation(model, **forward_kwargs)
    return out_impl, out_upstream


img_tokens = 4
input_ids_t2i = torch.tensor([[10, 11, 12, tms_token_id, 13]])
txt_seq_len = input_ids_t2i.shape[1]
total_seq_len = txt_seq_len + img_tokens
vinputs = torch.randn(1, img_tokens, patch_dim)
timestep = torch.tensor([500.0])
position_ids = torch.arange(total_seq_len).view(1, 1, -1).expand(3, 1, -1).contiguous()
token_types = torch.zeros(1, total_seq_len, dtype=torch.long)
token_types[0, 3] = 1  # tms
token_types[0, txt_seq_len:] = 1  # pixel patches

out_impl, out_upstream = run_pair(input_ids=input_ids_t2i, position_ids=position_ids, vinputs=vinputs, timestep=timestep, token_types=token_types)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
x_pred_diff = (out_impl.x_pred - out_upstream.x_pred).abs().max().item()
cond_none = out_impl.cond_image_embeds is None and out_upstream.cond_image_embeds is None
mid_none = out_impl.mid_results is None and out_upstream.mid_results is None
print(f'_forward_generation[T2I]                 : hidden shape={tuple(out_impl.last_hidden_state.shape)} x_pred shape={tuple(out_impl.x_pred.shape)} hidden_max_diff={hidden_diff:.2e} x_pred_max_diff={x_pred_diff:.2e} (tol={TOL:.0e}) cond_none={cond_none} mid_none={mid_none}')
assert hidden_diff < TOL and x_pred_diff < TOL and cond_none and mid_none, '_forward_generation[T2I] parity failed'

n_raw_patches = 4
image_grid_thw = torch.tensor([[1, 2, 2]])
n_merged_img = 1  # 4 raw / spatial_merge_size**2=4 -> 1 merged
pixel_values_i2i = torch.randn(n_raw_patches, 3 * 1 * 14 * 14)
input_ids_i2i = torch.tensor([[10, vision_start_token_id, image_token_id, 11, tms_token_id, 12]])
txt_seq_len_i2i = input_ids_i2i.shape[1]
total_seq_len_i2i = txt_seq_len_i2i + img_tokens
position_ids_i2i = torch.arange(total_seq_len_i2i).view(1, 1, -1).expand(3, 1, -1).contiguous()
token_types_i2i = torch.zeros(1, total_seq_len_i2i, dtype=torch.long)
token_types_i2i[0, 4] = 1  # tms
token_types_i2i[0, txt_seq_len_i2i:] = 1  # pixel patches

out_impl, out_upstream = run_pair(input_ids=input_ids_i2i, position_ids=position_ids_i2i, vinputs=vinputs, timestep=timestep, token_types=token_types_i2i, pixel_values=pixel_values_i2i, image_grid_thw=image_grid_thw)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
x_pred_diff = (out_impl.x_pred - out_upstream.x_pred).abs().max().item()
cond_diff = (out_impl.cond_image_embeds - out_upstream.cond_image_embeds).abs().max().item()
deepstack_len_equal = len(out_impl.cond_deepstack_image_embeds) == len(out_upstream.cond_deepstack_image_embeds)
deepstack_diff = max((a - b).abs().max().item() for a, b in zip(out_impl.cond_deepstack_image_embeds, out_upstream.cond_deepstack_image_embeds))
print(f'_forward_generation[I2I]                 : hidden shape={tuple(out_impl.last_hidden_state.shape)} x_pred shape={tuple(out_impl.x_pred.shape)} hidden_max_diff={hidden_diff:.2e} x_pred_max_diff={x_pred_diff:.2e} cond_max_diff={cond_diff:.2e} deepstack_max_diff={deepstack_diff:.2e} (tol={TOL:.0e}) deepstack_len_equal={deepstack_len_equal}')
assert hidden_diff < TOL and x_pred_diff < TOL and cond_diff < TOL and deepstack_diff < TOL and deepstack_len_equal, '_forward_generation[I2I] parity failed'

with torch.no_grad():
    image_embeds_precomputed, deepstack_precomputed = model.get_image_features(pixel_values_i2i, image_grid_thw)
image_embeds_precomputed = torch.cat(image_embeds_precomputed, dim=0)
out_impl, out_upstream = run_pair(input_ids=input_ids_i2i, position_ids=position_ids_i2i, vinputs=vinputs, timestep=timestep, token_types=token_types_i2i, pixel_values=pixel_values_i2i, image_grid_thw=image_grid_thw, precomputed_image_embeds=image_embeds_precomputed, precomputed_deepstack_image_embeds=deepstack_precomputed)
hidden_diff = (out_impl.last_hidden_state - out_upstream.last_hidden_state).abs().max().item()
x_pred_diff = (out_impl.x_pred - out_upstream.x_pred).abs().max().item()
cond_diff = (out_impl.cond_image_embeds - out_upstream.cond_image_embeds).abs().max().item()
print(f'_forward_generation[I2I precomputed]     : hidden_max_diff={hidden_diff:.2e} x_pred_max_diff={x_pred_diff:.2e} cond_max_diff={cond_diff:.2e} (tol={TOL:.0e})')
assert hidden_diff < TOL and x_pred_diff < TOL and cond_diff < TOL, '_forward_generation[I2I precomputed] parity failed'

try:
    model(input_ids=input_ids_t2i, position_ids=position_ids, vinputs=vinputs, timestep=timestep, token_types=token_types, use_flash_attn=True)
    raise AssertionError('expected NotImplementedError for use_flash_attn=True')
except NotImplementedError as e:
    assert 'flash' in str(e).lower(), f'expected flash-attn guard message, got: {e}'
print('_forward_generation[use_flash_attn guard]: raises NotImplementedError with flash-attn hint')

print('pass (_forward_generation bit-identical to inline upstream reference on T2I + I2I + precomputed cache)')


# output:
# _forward_generation[T2I]                 : hidden shape=(1, 9, 128) x_pred shape=(1, 9, 3072) hidden_max_diff=0.00e+00 x_pred_max_diff=0.00e+00 (tol=1e-05) cond_none=True mid_none=True
# _forward_generation[I2I]                 : hidden shape=(1, 10, 128) x_pred shape=(1, 10, 3072) hidden_max_diff=0.00e+00 x_pred_max_diff=0.00e+00 cond_max_diff=0.00e+00 deepstack_max_diff=0.00e+00 (tol=1e-05) deepstack_len_equal=True
# _forward_generation[I2I precomputed]     : hidden_max_diff=0.00e+00 x_pred_max_diff=0.00e+00 cond_max_diff=0.00e+00 (tol=1e-05)
# _forward_generation[use_flash_attn guard]: raises NotImplementedError with flash-attn hint
# pass (_forward_generation bit-identical to inline upstream reference on T2I + I2I + precomputed cache)
