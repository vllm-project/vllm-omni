# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural test for Qwen3VLModelOutputWithPast + Qwen3VLCausalLMOutputWithPast.

Usage:
    python _test_dev/_test_dataclass_outputs.py
"""
from __future__ import annotations

import dataclasses

import torch

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
)

expected_model_fields = {
    'last_hidden_state',
    'past_key_values',
    'hidden_states',
    'attentions',
    'rope_deltas',
    'x_pred',
    'mid_results',
    'cond_image_embeds',
    'cond_deepstack_image_embeds',
}
expected_causal_lm_fields = (expected_model_fields - {'last_hidden_state'}) | {'loss', 'logits'}

model_fields = {f.name for f in dataclasses.fields(Qwen3VLModelOutputWithPast)}
causal_lm_fields = {f.name for f in dataclasses.fields(Qwen3VLCausalLMOutputWithPast)}

print(f'Qwen3VLModelOutputWithPast    fields ({len(model_fields)}): {sorted(model_fields)}')
print(f'Qwen3VLCausalLMOutputWithPast fields ({len(causal_lm_fields)}): {sorted(causal_lm_fields)}')

hidden = torch.randn(1, 32, 128)
x_pred = torch.randn(1, 16, 768)
rope_deltas = torch.tensor([0], dtype=torch.long)

out_model = Qwen3VLModelOutputWithPast(last_hidden_state=hidden, x_pred=x_pred, rope_deltas=rope_deltas)
out_causal = Qwen3VLCausalLMOutputWithPast(logits=hidden, x_pred=x_pred, rope_deltas=rope_deltas)

print(f'out_model attr access:  out.last_hidden_state.shape = {tuple(out_model.last_hidden_state.shape)}')
print(f'out_model item access:  out["x_pred"].shape         = {tuple(out_model["x_pred"].shape)}')
print(f'out_model tuple access: out[0].shape                = {tuple(out_model[0].shape)}')
print(f'out_causal: loss={out_causal.loss} logits.shape={tuple(out_causal.logits.shape)} x_pred.shape={tuple(out_causal.x_pred.shape)}')

assert model_fields == expected_model_fields, f'Qwen3VLModelOutputWithPast fields mismatch: got={model_fields} expected={expected_model_fields}'
assert causal_lm_fields == expected_causal_lm_fields, f'Qwen3VLCausalLMOutputWithPast fields mismatch: got={causal_lm_fields} expected={expected_causal_lm_fields}'
assert out_model.last_hidden_state is hidden
assert out_model.x_pred is x_pred
assert out_model.past_key_values is None
assert out_model.mid_results is None
assert out_model['x_pred'] is x_pred
assert out_model[0] is hidden
assert out_causal.loss is None
assert out_causal.logits is hidden

print('pass')


# output:
# Qwen3VLModelOutputWithPast    fields (9): ['attentions', 'cond_deepstack_image_embeds', 'cond_image_embeds', 'hidden_states', 'last_hidden_state', 'mid_results', 'past_key_values', 'rope_deltas', 'x_pred']
# Qwen3VLCausalLMOutputWithPast fields (10): ['attentions', 'cond_deepstack_image_embeds', 'cond_image_embeds', 'hidden_states', 'logits', 'loss', 'mid_results', 'past_key_values', 'rope_deltas', 'x_pred']
# out_model attr access:  out.last_hidden_state.shape = (1, 32, 128)
# out_model item access:  out["x_pred"].shape         = (1, 16, 768)
# out_model tuple access: out[0].shape                = (1, 32, 128)
# out_causal: loss=None logits.shape=(1, 32, 128) x_pred.shape=(1, 16, 768)
# pass
