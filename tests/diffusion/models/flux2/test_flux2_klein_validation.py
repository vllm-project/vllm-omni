# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Attention,
    Flux2SingleTransformerBlock,
    _get_sequence_parallel_size,
)
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
    Flux2KleinPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline():
    pipeline = object.__new__(Flux2KleinPipeline)
    pipeline.vae_scale_factor = 8
    pipeline.is_distilled = True
    pipeline._guidance_scale = 0.0
    return pipeline


def _check(pipeline, prompt, prompt_embeds=None):
    pipeline.check_inputs(
        prompt=prompt,
        height=512,
        width=512,
        prompt_embeds=prompt_embeds,
        num_inference_steps=4,
        guidance_scale=0.0,
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
    ],
)
def test_rejects_empty_or_whitespace_prompt(prompt):
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _check(pipe, prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        ["valid prompt", ""],
        ["valid prompt", "   "],
        ["   "],
    ],
)
def test_rejects_list_with_empty_or_whitespace_element(prompt):
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _check(pipe, prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "valid prompt",
        ["valid prompt", "another valid prompt"],
    ],
)
def test_accepts_valid_prompts(prompt):
    pipe = _make_pipeline()
    _check(pipe, prompt)


def test_rejects_none_without_embeds():
    pipe = _make_pipeline()
    with pytest.raises(ValueError):
        _check(pipe, None)


def test_accepts_none_with_prompt_embeds():
    pipe = _make_pipeline()
    _check(pipe, None, prompt_embeds=torch.randn(1, 256, 4096))


def test_single_block_forwards_text_length_to_sp_attention():
    class RecordingAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_seq_len = None

        def forward(self, hidden_states, **kwargs):
            self.text_seq_len = kwargs.get("text_seq_len")
            return torch.zeros_like(hidden_states)

    block = object.__new__(Flux2SingleTransformerBlock)
    torch.nn.Module.__init__(block)
    block.norm = torch.nn.Identity()
    block.attn = RecordingAttention()

    hidden_states = torch.randn(1, 3, 4)
    zeros = torch.zeros(1, 1, 4)
    block(
        hidden_states=hidden_states,
        encoder_hidden_states=None,
        temb_mod_params=(zeros, zeros, zeros),
        text_seq_len=2,
    )

    assert block.attn.text_seq_len == 2


def test_double_stream_uses_joint_attention_when_image_only_is_sp_sharded():
    class Projection(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.cat([hidden_states, hidden_states, hidden_states], dim=-1), None

    class PassthroughLinear(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class PassthroughRope(torch.nn.Module):
        def forward(self, hidden_states, cos, sin):
            return hidden_states

    class RecordingAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.metadata = None

        def forward(self, query, key, value, metadata):
            self.metadata = metadata
            return torch.cat([metadata.joint_value, value], dim=1)

    attention = object.__new__(Flux2Attention)
    torch.nn.Module.__init__(attention)
    attention.parallel_config = type("ParallelConfig", (), {"sequence_parallel_size": 8})()
    attention.added_kv_proj_dim = 1
    attention.inner_dim = 1
    attention.query_num_heads = 1
    attention.kv_num_heads = 1
    attention.add_query_num_heads = 1
    attention.add_kv_num_heads = 1
    attention.to_qkv = Projection()
    attention.add_kv_proj = Projection()
    attention.norm_q = torch.nn.Identity()
    attention.norm_k = torch.nn.Identity()
    attention.norm_added_q = torch.nn.Identity()
    attention.norm_added_k = torch.nn.Identity()
    attention.rope = PassthroughRope()
    attention.attn = RecordingAttention()
    attention.to_add_out = PassthroughLinear()
    attention.to_out = torch.nn.ModuleList([PassthroughLinear(), torch.nn.Identity()])

    image = torch.randn(1, 2, 1)
    text = torch.randn(1, 3, 1)
    rotary = (torch.ones(5, 1), torch.zeros(5, 1))
    attention(image, encoder_hidden_states=text, image_rotary_emb=rotary)

    metadata = attention.attn.metadata
    assert metadata is not None
    assert metadata.joint_query.shape[1] == text.shape[1]
    assert metadata.joint_key.shape[1] == text.shape[1]


def test_runtime_sequence_parallel_size_overrides_stale_model_config(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer.model_parallel_is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer.get_sequence_parallel_world_size",
        lambda: 8,
    )
    stale_config = type("ParallelConfig", (), {"sequence_parallel_size": 1})()

    assert _get_sequence_parallel_size(stale_config) == 8


@pytest.mark.parametrize(
    ("configured_size", "expected_size"),
    [
        (None, 1),
        (4, 4),
    ],
)
def test_config_sequence_parallel_size_is_used_before_groups_initialize(monkeypatch, configured_size, expected_size):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer.model_parallel_is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer.get_sequence_parallel_world_size",
        lambda: pytest.fail("runtime SP group must not be queried before initialization"),
    )
    config = DiffusionParallelConfig(
        sequence_parallel_size=configured_size,
        ulysses_degree=expected_size,
    )

    sequence_parallel_size = _get_sequence_parallel_size(config)

    assert sequence_parallel_size == expected_size
    assert isinstance(sequence_parallel_size, int)
