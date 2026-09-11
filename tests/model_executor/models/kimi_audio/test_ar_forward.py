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

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen2Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

from tests.model_executor.models.kimi_audio.runtime import cpu_pp_group as cpu_pp_group
from vllm_omni.model_executor.models.kimi_audio.kimi_audio import KimiAudioForConditionalGeneration
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioARStage
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = Path(__file__).parent / "fixtures/ar_forward_reference.safetensors"


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
    # This is the runner's selection operation before compute_logits: one row
    # selection must retain BOTH branch states for the same scheduled token.
    rows = torch.tensor([output.shape[0] - 1, 0])
    selected_text, selected_audio = output[rows].chunk(2, dim=-1)
    torch.testing.assert_close(selected_text, text[rows])
    torch.testing.assert_close(selected_audio, audio[rows])
    assert stage.requires_full_prefix_cached_hidden_states is False


def test_forward_requires_prepared_dual_stream_embeddings(forward_fixture):
    stage, tensors = forward_fixture
    with pytest.raises(ValueError, match="requires fused text/audio"):
        stage(torch.zeros_like(tensors["prefill.positions"]), tensors["prefill.positions"])


@pytest.mark.parametrize("partitions", [(14, 14), (22, 6), (20, 4, 4), (21, 1, 3, 3), (23, 5)])
@torch.inference_mode()
def test_pp_cuts_preserve_both_official_branches(forward_fixture, cpu_pp_group, partitions, monkeypatch):
    reference, tensors = forward_fixture
    ranks, start = [], 0
    for rank, count in enumerate(partitions):
        stage = copy.deepcopy(reference)
        stage.start_layer, stage.end_layer = start, start + count
        stage.mimo_start_layer = max(0, start - 22)
        stage.mimo_end_layer = max(0, start + count - 22)
        for index in range(28):
            if not start <= index < start + count:
                stage.layers[index] = PPMissingLayer()
        for index in range(6):
            if not start <= index + 22 < start + count:
                stage.mimo_layers[index] = PPMissingLayer()
        if rank != len(partitions) - 1:
            stage.norm = stage.mimo_norm = PPMissingLayer()
        ranks.append(stage)
        start += count
    cpu_pp_group.world_size = len(ranks)
    for case in ("prefill", "step0", "step1"):
        carriers = None
        for rank, stage in enumerate(ranks):
            cpu_pp_group.rank_in_group = rank
            cpu_pp_group.is_first_rank = rank == 0
            cpu_pp_group.is_last_rank = rank == len(ranks) - 1
            if carriers is not None:
                # Exercise the native allocation/copy contract at each cut.
                incoming = stage.make_empty_intermediate_tensors(
                    len(tensors[f"{case}.positions"]), torch.float32, "cpu"
                )
                assert set(incoming.tensors) == set(carriers.tensors)
                for key, value in carriers.items():
                    incoming[key].copy_(value)
            else:
                incoming = None
            entry = KimiAudioForConditionalGeneration.__new__(KimiAudioForConditionalGeneration)
            torch.nn.Module.__init__(entry)
            entry.model = stage
            runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
            runner.model = entry
            output = entry(
                None,
                tensors[f"{case}.positions"],
                intermediate_tensors=incoming,
                inputs_embeds=tensors[f"{case}.inputs"].clone() if rank == 0 else None,
            )
            if rank < len(ranks) - 1:
                # Unmodified warmup reads a plain tensor; every carrier remains
                # a flat tensor output, suitable for native graph weak refs.
                hidden, _ = runner.extract_multimodal_outputs(output)
                assert all(isinstance(t, torch.Tensor) for t in output)
                assert hidden[torch.tensor([len(hidden) - 1])].shape == (1, stage.config.hidden_size)
                posted = []
                monkeypatch.setattr(stage, "_exchange_pipeline_state", lambda states: posted.append(states))
                packed = stage.make_omni_output(output, model_intermediate_buffer=[{"kimi_audio_generation": {}}])
                carriers, _ = runner.extract_multimodal_outputs(packed)
                assert isinstance(carriers, IntermediateTensors)
                assert posted == [[{}]]
            else:
                carriers = output
        text, audio = carriers.chunk(2, dim=-1)
        torch.testing.assert_close(text, tensors[f"{case}.text"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(audio, tensors[f"{case}.audio"], rtol=1e-5, atol=1e-6)
