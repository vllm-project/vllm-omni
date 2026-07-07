# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identical parity for Qwen3VLForConditionalGeneration + HiDreamO1ImageTransformer alias vs inline upstream reference.

Usage:
    python _test_dev/_test_qwen3vl_for_conditional_generation.py
"""
from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    HiDreamO1ImageTransformer,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
)

TOL = 1e-5


# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/qwen3_vl_transformers.py L1886-1929
def upstream_for_gen_forward(for_gen_model, input_ids=None, attention_mask=None, position_ids=None,
                             past_key_values=None, inputs_embeds=None, labels=None, pixel_values=None,
                             pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
                             cache_position=None, logits_to_keep=0, vinputs=None, timestep=None,
                             token_types=None, use_flash_attn=False, return_mid_results_layers=None, **kwargs):
    outputs = for_gen_model.model(
        input_ids=input_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, position_ids=position_ids,
        attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        cache_position=cache_position, vinputs=vinputs, timestep=timestep, token_types=token_types,
        use_flash_attn=use_flash_attn, return_mid_results_layers=return_mid_results_layers, **kwargs,
    )
    if vinputs is not None:
        return Qwen3VLCausalLMOutputWithPast(
            x_pred=outputs.x_pred,
            mid_results=outputs.mid_results if hasattr(outputs, 'mid_results') else None,
            cond_image_embeds=getattr(outputs, 'cond_image_embeds', None),
            cond_deepstack_image_embeds=getattr(outputs, 'cond_deepstack_image_embeds', None),
        )
    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = for_gen_model.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        loss = for_gen_model.loss_function(logits=logits, labels=labels, vocab_size=for_gen_model.config.text_config.vocab_size)
    return Qwen3VLCausalLMOutputWithPast(
        loss=loss, logits=logits, past_key_values=outputs.past_key_values, rope_deltas=outputs.rope_deltas,
    )
# --- end upstream ref ---


text_cfg = Qwen3VLTextConfig(vocab_size=200_000, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=256, tie_word_embeddings=False)
vision_cfg = Qwen3VLVisionConfig(hidden_size=64, intermediate_size=128, depth=2, num_heads=4, in_channels=3, patch_size=14, temporal_patch_size=1, out_hidden_size=128, spatial_merge_size=2, deepstack_visual_indexes=(0, 1))
cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)

torch.manual_seed(0)
for_gen = Qwen3VLForConditionalGeneration(cfg)
for_gen.eval()

image_token_id = cfg.image_token_id
vision_start_token_id = cfg.vision_start_token_id
tms_token_id = for_gen.model.tms_token_id
patch_dim = 3 * 32 * 32
vocab_size = text_cfg.vocab_size
hidden_size = text_cfg.hidden_size


def run_pair(**forward_kwargs):
    for_gen.model.rope_deltas = None
    with torch.no_grad():
        out_impl = for_gen(**forward_kwargs)
    for_gen.model.rope_deltas = None
    with torch.no_grad():
        out_upstream = upstream_for_gen_forward(for_gen, **forward_kwargs)
    return out_impl, out_upstream


n_for_gen = sum(p.numel() for p in for_gen.parameters())
n_model_only = sum(p.numel() for p in for_gen.model.parameters())
n_lm_head = sum(p.numel() for p in for_gen.lm_head.parameters())
sd_keys = list(for_gen.state_dict().keys())
top_prefixes = {k.split('.', 1)[0] for k in sd_keys}
has_lm_head = 'lm_head.weight' in sd_keys
tied_correct = for_gen._tied_weights_keys == ['lm_head.weight']
alias_ok = HiDreamO1ImageTransformer is Qwen3VLForConditionalGeneration
print(f'structure[params+state_dict+alias] : n_for_gen={n_for_gen:,} (model={n_model_only:,} + lm_head={n_lm_head:,}) top_prefixes={sorted(top_prefixes)} has_lm_head={has_lm_head} tied_correct={tied_correct} alias_ok={alias_ok}')
assert n_for_gen == n_model_only + n_lm_head, 'param count invariant broken'
assert has_lm_head and top_prefixes == {'model', 'lm_head'}, f'unexpected state_dict prefixes: {top_prefixes}'
assert tied_correct, '_tied_weights_keys not [lm_head.weight]'
assert alias_ok, 'HiDreamO1ImageTransformer is not aliased to Qwen3VLForConditionalGeneration'

input_ids_text = torch.tensor([[10, 11, 12, 13, 14, 15]])
out_impl, out_upstream = run_pair(input_ids=input_ids_text, use_cache=False)
logits_diff = (out_impl.logits - out_upstream.logits).abs().max().item()
rope_deltas_equal = torch.equal(out_impl.rope_deltas, out_upstream.rope_deltas)
loss_both_none = out_impl.loss is None and out_upstream.loss is None
print(f'forward[text no labels]            : logits shape={tuple(out_impl.logits.shape)} logits_max_diff={logits_diff:.2e} (tol={TOL:.0e}) rope_deltas_equal={rope_deltas_equal} loss_both_none={loss_both_none}')
assert logits_diff < TOL and rope_deltas_equal and loss_both_none, 'forward[text no labels] parity failed'
assert out_impl.logits.shape == (1, 6, vocab_size), f'expected full-seq logits, got {out_impl.logits.shape}'

labels = torch.tensor([[10, 11, 12, 13, 14, -100]])
out_impl, out_upstream = run_pair(input_ids=input_ids_text, labels=labels, use_cache=False)
logits_diff = (out_impl.logits - out_upstream.logits).abs().max().item()
loss_diff = (out_impl.loss - out_upstream.loss).abs().max().item()
loss_scalar = out_impl.loss.dim() == 0
print(f'forward[text with labels + loss]   : logits shape={tuple(out_impl.logits.shape)} logits_max_diff={logits_diff:.2e} loss_max_diff={loss_diff:.2e} (tol={TOL:.0e}) loss_scalar={loss_scalar}')
assert logits_diff < TOL and loss_diff < TOL and loss_scalar, 'forward[text with labels + loss] parity failed'

out_impl, out_upstream = run_pair(input_ids=input_ids_text, logits_to_keep=2, use_cache=False)
logits_diff = (out_impl.logits - out_upstream.logits).abs().max().item()
kept_shape_ok = out_impl.logits.shape == (1, 2, vocab_size)
print(f'forward[text logits_to_keep=2]     : logits shape={tuple(out_impl.logits.shape)} logits_max_diff={logits_diff:.2e} (tol={TOL:.0e}) kept_shape_ok={kept_shape_ok}')
assert logits_diff < TOL and kept_shape_ok, 'forward[text logits_to_keep=2] parity failed'

n_raw = 4
image_grid_thw = torch.tensor([[1, 2, 2]])
pixel_values = torch.randn(n_raw, 3 * 1 * 14 * 14)
input_ids_img = torch.tensor([[10, 11, vision_start_token_id, image_token_id, 12]])
out_impl, out_upstream = run_pair(input_ids=input_ids_img, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)
logits_diff = (out_impl.logits - out_upstream.logits).abs().max().item()
rope_deltas_equal = torch.equal(out_impl.rope_deltas, out_upstream.rope_deltas)
print(f'forward[text + image]              : logits shape={tuple(out_impl.logits.shape)} logits_max_diff={logits_diff:.2e} (tol={TOL:.0e}) rope_deltas_equal={rope_deltas_equal}')
assert logits_diff < TOL and rope_deltas_equal, 'forward[text + image] parity failed'

img_tokens = 4
input_ids_t2i = torch.tensor([[10, 11, 12, tms_token_id, 13]])
txt_seq_len = input_ids_t2i.shape[1]
total_seq_len = txt_seq_len + img_tokens
vinputs = torch.randn(1, img_tokens, patch_dim)
timestep = torch.tensor([500.0])
position_ids = torch.arange(total_seq_len).view(1, 1, -1).expand(3, 1, -1).contiguous()
token_types = torch.zeros(1, total_seq_len, dtype=torch.long)
token_types[0, 3] = 1
token_types[0, txt_seq_len:] = 1
out_impl, out_upstream = run_pair(input_ids=input_ids_t2i, position_ids=position_ids, vinputs=vinputs, timestep=timestep, token_types=token_types)
x_pred_diff = (out_impl.x_pred - out_upstream.x_pred).abs().max().item()
logits_none = out_impl.logits is None and out_upstream.logits is None
loss_none = out_impl.loss is None and out_upstream.loss is None
print(f'forward[vinputs T2I re-package]    : x_pred shape={tuple(out_impl.x_pred.shape)} x_pred_max_diff={x_pred_diff:.2e} (tol={TOL:.0e}) logits_none={logits_none} loss_none={loss_none}')
assert x_pred_diff < TOL and logits_none and loss_none, 'forward[vinputs T2I re-package] parity failed'

print('pass (Qwen3VLForConditionalGeneration structural + text (no labels / with labels / logits_to_keep / +image) + vinputs re-package all bit-identical to inline upstream reference)')


# output:
# structure[params+state_dict+alias] : n_for_gen=52,593,216 (model=26,993,216 + lm_head=25,600,000) top_prefixes=['lm_head', 'model'] has_lm_head=True tied_correct=True alias_ok=True
# forward[text no labels]            : logits shape=(1, 6, 200000) logits_max_diff=0.00e+00 (tol=1e-05) rope_deltas_equal=True loss_both_none=True
# forward[text with labels + loss]   : logits shape=(1, 6, 200000) logits_max_diff=0.00e+00 loss_max_diff=0.00e+00 (tol=1e-05) loss_scalar=True
# forward[text logits_to_keep=2]     : logits shape=(1, 2, 200000) logits_max_diff=0.00e+00 (tol=1e-05) kept_shape_ok=True
# forward[text + image]              : logits shape=(1, 5, 200000) logits_max_diff=0.00e+00 (tol=1e-05) rope_deltas_equal=True
# forward[vinputs T2I re-package]    : x_pred shape=(1, 9, 3072) x_pred_max_diff=0.00e+00 (tol=1e-05) logits_none=True loss_none=True
# pass (Qwen3VLForConditionalGeneration structural + text (no labels / with labels / logits_to_keep / +image) + vinputs re-package all bit-identical to inline upstream reference)
