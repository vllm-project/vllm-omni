# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check depth attention against HF Llama and request-local sampling state."""

from collections import defaultdict

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from vllm_omni.model_executor.models.breeze_tts.depth_decoder import BreezeDepthDecoder, sample_logits
from vllm_omni.model_executor.stage_input_processors.breeze_tts import talker2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def depth() -> BreezeDepthDecoder:
    torch.manual_seed(42)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=35,
        max_position_embeddings=16,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 0.001953125,
            "high_freq_factor": 0.0078125,
            "original_max_position_embeddings": 16,
        },
    )
    config.num_codebooks = 4
    config.audio_embed_size = 48
    config._attn_implementation = "eager"
    model = BreezeDepthDecoder(config).eval()
    torch.nn.init.normal_(model.codebooks_head, std=0.1)
    return model


@pytest.mark.parametrize("batch", [1, 2])
@torch.inference_mode()
def test_depth_cached_attention_matches_hf_full_sequence(depth: BreezeDepthDecoder, batch: int) -> None:
    """Wrong cache positions or fused-QKV ordering change the reference hidden states."""
    reference = LlamaModel(depth.config).eval()
    weights = {}
    for name, parameter in depth.named_parameters():
        if name.startswith("layers."):
            if name.endswith("qkv.weight"):
                layer = depth.layers[int(name.split(".")[1])]
                for part, tensor in zip(
                    ("q", "k", "v"), parameter.split([layer.q_size, layer.kv_size, layer.kv_size]), strict=True
                ):
                    weights[name.replace("qkv.weight", f"self_attn.{part}_proj.weight")] = tensor
            elif name.endswith("gate_up.weight"):
                for part, tensor in zip(("gate", "up"), parameter.chunk(2), strict=True):
                    weights[name.replace("gate_up.weight", f"mlp.{part}_proj.weight")] = tensor
            else:
                weights[name.replace(".o_proj.", ".self_attn.o_proj.").replace(".down_proj.", ".mlp.down_proj.")] = (
                    parameter
                )
        elif name == "norm.weight":
            weights[name] = parameter
    result = reference.load_state_dict(weights, strict=False)
    assert result.missing_keys == ["embed_tokens.weight"]
    assert result.unexpected_keys == []
    embeddings = torch.randn(batch, 4, depth.config.hidden_size)
    expected = reference(inputs_embeds=embeddings, use_cache=False).last_hidden_state
    shape = (batch, 2, 4, 8)
    caches = [(torch.empty(shape), torch.empty(shape)) for _ in depth.layers]
    actual = []
    for start, end in [(0, 2), (2, 3), (3, 4)]:
        x = embeddings[:, start:end]
        for layer, cache in zip(depth.layers, caches, strict=True):
            x = layer(x, start, cache, depth.rope_cos[start:end][None, None], depth.rope_sin[start:end][None, None])
        actual.append(depth.norm(x))
    torch.testing.assert_close(torch.cat(actual, 1), expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_frame_generation_does_not_reuse_another_requests_cache(depth: BreezeDepthDecoder) -> None:
    hidden = torch.randn(1, 48)
    first = torch.tensor([7])

    def generate(value: torch.Tensor) -> torch.Tensor:
        return depth.generate_frame(
            value, first, temperature=0.9, top_k=10, top_p=0.8, generator=torch.Generator().manual_seed(99)
        )

    expected = generate(hidden)
    generate(hidden * -3)
    torch.testing.assert_close(generate(hidden), expected, atol=0, rtol=0)


def test_top_k_keeps_tied_candidates_and_nucleus_keeps_crossing_token() -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.tensor([[0.0, 0.0, -100.0]]).expand(128, -1)
    sampled = sample_logits(logits, 1.0, 1, 1.0, generator)
    assert set(sampled.tolist()) == {0, 1}
    sampled = sample_logits(torch.tensor([[10.0, 0.0]]), 1.0, 0, 0.1, generator)
    assert sampled.item() == 0


def test_codec_chunks_preserve_zero_frames_and_flush_each_request_once(mocker) -> None:
    manager = mocker.Mock()
    manager.connector.config = {"extra": {"codec_chunk_frames": 5}}
    manager.code_prompt_token_ids = defaultdict(list)
    manager.request_payload = {}
    first = mocker.Mock(external_req_id="first")
    first.is_finished.return_value = False
    second = mocker.Mock(external_req_id="second")
    second.is_finished.return_value = False
    zeros = {"codes": {"audio": torch.zeros(3, 16, dtype=torch.long)}}
    assert talker2code2wav_async_chunk(manager, zeros, first) is None
    assert talker2code2wav_async_chunk(manager, zeros, second) is None
    codes = torch.arange(64).reshape(4, 16)
    output = talker2code2wav_async_chunk(manager, {"codes": {"audio": codes}}, first)
    expected = torch.cat((torch.zeros(3, 16, dtype=torch.long), codes))
    torch.testing.assert_close(output.codes.audio.reshape(16, -1).T, expected)
    final = talker2code2wav_async_chunk(manager, None, first, is_finished=True)
    assert final.codes.audio.numel() == 0
    assert final.meta.finished.item()
    other = talker2code2wav_async_chunk(manager, None, second, is_finished=True)
    torch.testing.assert_close(other.codes.audio, torch.zeros(48, dtype=torch.long))
