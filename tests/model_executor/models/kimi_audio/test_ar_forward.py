# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dual-stream execution against the official model's small CPU oracle.

The vLLM decoder forward runs unchanged. Attention uses HF Qwen2 on CPU with
test-owned KV caches; norms deliberately overwrite both residual carriers to
exercise the vLLM fused-kernel contract. This is not native GPU/cache testing.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen2Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

from vllm_omni.model_executor.models.kimi_audio import kimi_audio_ar_stage
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioARStage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = Path(__file__).parent / "fixtures/ar_forward_reference.safetensors"


@pytest.fixture(autouse=True)
def cpu_pp_group(monkeypatch):
    """Single-rank ownership for the numerical comparison, without an engine."""
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(kimi_audio_ar_stage, "get_pp_group", lambda: group)


class CPUAttention(torch.nn.Module):
    """Same HF attention math on both sides of the residual-flow comparison."""

    def __init__(self, config, layer_index, cache):
        super().__init__()
        self.attention = Qwen2Attention(config, layer_index)
        self.rotary = Qwen2RotaryEmbedding(config)
        self.cache = cache
        self.layer_index = layer_index

    def forward(self, *, hidden_states, positions=None, position_ids=None, **kwargs):
        is_vllm = hidden_states.ndim == 2
        states = hidden_states.unsqueeze(0) if is_vllm else hidden_states
        pos = positions.unsqueeze(0) if is_vllm else position_ids
        key_length = self.cache.get_seq_length(self.layer_index) + states.shape[1]
        mask = torch.zeros(1, 1, states.shape[1], key_length, dtype=states.dtype)
        mask.masked_fill_(torch.arange(key_length).view(1, 1, 1, -1) > pos[:, None, :, None], float("-inf"))
        output, _ = self.attention(
            states,
            position_embeddings=self.rotary(states, pos),
            attention_mask=mask,
            past_key_values=self.cache,
        )
        return output[0] if is_vllm else (output, None, None)


class InplaceRMSNorm(Qwen2RMSNorm):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return super().forward(hidden_states)
        residual.add_(hidden_states)
        hidden_states.copy_(super().forward(residual))
        return hidden_states, residual


class CPUDecoder(Qwen2DecoderLayer):
    # Keep the actual vLLM decoder forward; substitute construction only.
    def __init__(self, config, layer_index, cache):
        torch.nn.Module.__init__(self)
        self.self_attn = CPUAttention(config, layer_index, cache)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = InplaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = InplaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def cpu_stage(config):
    stage = KimiAudioARStage.__new__(KimiAudioARStage)
    torch.nn.Module.__init__(stage)
    stage.config = config
    stage.branch_layer = config.kimia_mimo_transformer_from_layer_index
    stage.start_layer, stage.end_layer = 0, config.num_hidden_layers
    stage.mimo_start_layer, stage.mimo_end_layer = 0, config.kimia_mimo_layers
    cache = DynamicCache()
    # HF uses one numbered cache table. Include the separate MIMO layers in
    # that test backend's layer-type table without changing the Kimi config.
    attention_config = copy.deepcopy(config)
    attention_config.num_hidden_layers += config.kimia_mimo_layers
    attention_config.layer_types += ["full_attention"] * config.kimia_mimo_layers
    stage.layers = torch.nn.ModuleList(CPUDecoder(attention_config, i, cache) for i in range(config.num_hidden_layers))
    stage.mimo_layers = torch.nn.ModuleList(
        CPUDecoder(attention_config, config.num_hidden_layers + i, cache) for i in range(config.kimia_mimo_layers)
    )
    stage.norm = InplaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    stage.mimo_norm = InplaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    return stage.eval()


@pytest.fixture
def forward_fixture():
    with safe_open(REFERENCE, framework="pt") as reference:
        config = Qwen2Config(**json.loads(reference.metadata()["config"]))
    config._attn_implementation = "eager"
    tensors = load_file(REFERENCE)
    stage = cpu_stage(config)
    weights = {name.removeprefix("weights."): value for name, value in tensors.items() if name.startswith("weights.")}
    stage.load_state_dict(weights, strict=True)
    return stage, tensors


@torch.inference_mode()
def test_both_branches_match_official_prefill_and_incremental_steps(forward_fixture):
    stage, tensors = forward_fixture
    assert len(stage.layers) == 28 and stage.branch_layer == 21 and len(stage.mimo_layers) == 6
    for case in ("prefill", "step0", "step1"):
        output = stage(
            input_ids=None, positions=tensors[f"{case}.positions"], inputs_embeds=tensors[f"{case}.inputs"].clone()
        )
        text, audio = output.chunk(2, dim=-1)
        torch.testing.assert_close(text, tensors[f"{case}.text"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(audio, tensors[f"{case}.audio"], rtol=1e-5, atol=1e-6)
    # The two branches' cache slots must each have seen the full prefix plus
    # both later steps. These are HF test caches, not native vLLM KV blocks.
    cache = stage.layers[0].self_attn.cache
    for index in range(34):
        assert cache.get_seq_length(index) == tensors["prefill.inputs"].shape[0] + 2


@torch.inference_mode()
def test_text_tail_changes_do_not_leak_into_audio_branch(forward_fixture):
    stage, tensors = forward_fixture
    stage.layers[stage.branch_layer + 1].mlp.down_proj.weight[0].add_(0.2)
    output = stage(None, tensors["prefill.positions"], inputs_embeds=tensors["prefill.inputs"].clone())
    text, audio = output.chunk(2, dim=-1)
    assert not torch.allclose(text, tensors["prefill.text"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(audio, tensors["prefill.audio"], rtol=1e-5, atol=1e-6)
