# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check captured cache ownership and padded text masking against eager models."""

import pytest
import torch
from transformers import LlamaConfig, T5Gemma2TextConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.breeze_tts.depth_decoder import BreezeDepthDecoder
from vllm_omni.model_executor.models.breeze_tts.text_encoder_graph import (
    BreezeTextEncoderCompiled,
    BreezeTextEncoderGraph,
)

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_text_graph_masks_padding_and_local_attention_across_replays() -> None:
    torch.manual_seed(42)
    config = T5Gemma2TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        dropout_rate=0.0,
    )
    config._attn_implementation = "sdpa"
    encoder = T5Gemma2TextEncoder(config).to("cuda").eval()
    encoder.embed_tokens.eoi_token_index = 17
    encoder.embed_tokens.eoi_embedding.fill_(0.4)
    projection = torch.nn.Linear(32, 48, bias=False).to("cuda").eval()
    graph = BreezeTextEncoderGraph(encoder, projection, 64)
    for length in (31, 51, 17, 31):
        prompt = torch.randint(0, 64, (1, length), device="cuda")
        prompt[:, 3] = 17
        expected = projection(encoder(input_ids=prompt).last_hidden_state)[0]
        actual = graph.run(prompt)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    batched = BreezeTextEncoderGraph(encoder, projection, 64, batch_size=3)
    prompts = [torch.randint(0, 64, (length,), device="cuda") for length in (31, 51, 17)]
    outputs = batched.run_batch(prompts)
    for prompt, actual in zip(prompts, outputs, strict=True):
        expected = projection(encoder(input_ids=prompt[None]).last_hidden_state)[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    compiled = BreezeTextEncoderCompiled(encoder, projection)
    long_prompts = [torch.randint(0, 64, (length,)) for length in (211, 173, 129)]
    for batch in ([long_prompts[0]], long_prompts, [long_prompts[1]]):
        outputs = compiled.run_batch(batch)
        for prompt, actual in zip(batch, outputs, strict=True):
            expected = projection(encoder(input_ids=prompt[None].to("cuda")).last_hidden_state)[0]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_depth_graph_survives_interleaved_batches_and_workspace_allocations() -> None:
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
    depth = BreezeDepthDecoder(config).to("cuda").eval()
    torch.nn.init.normal_(depth.codebooks_head, std=0.1)
    hidden = torch.randn(2, 48, device="cuda")
    first = torch.tensor([7, 11], device="cuda")
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in (42, 99)]
    expected = torch.cat(
        [
            depth.generate_frame(hidden[i : i + 1], first[i : i + 1], temperature=0, top_k=0, top_p=1, generator=g)
            for i, g in enumerate(generators)
        ]
    )
    for batch in (1, 2, 1, 2):
        # Unowned allocations made before graph capture would be recycled here.
        pressure = [torch.full((batch, 2, 4, 8), float("nan"), device="cuda") for _ in range(32)]
        actual = depth.generate_frames(
            hidden[:batch], first[:batch], temperature=0, top_k=0, top_p=1, generators=generators[:batch]
        )
        torch.testing.assert_close(actual, expected[:batch], atol=0, rtol=0)
        del pressure

    def sample(batch: int) -> torch.Tensor:
        return depth.generate_frames(
            hidden[:batch],
            first[:batch],
            temperature=0.9,
            top_k=10,
            top_p=0.8,
            generators=[torch.Generator(device="cuda").manual_seed(seed) for seed in (42, 99)[:batch]],
        )

    sampled = sample(1)
    first_graph = depth._graphs[(1, 1)]
    sample(2)
    torch.testing.assert_close(sample(1), sampled, atol=0, rtol=0)
    for temperature, top_k, top_p in ((0.0, 0, 1.0), (0.7, 20, 0.95), (1.2, 0, 0.8)):
        depth.generate_frames(
            hidden[:1], first[:1], temperature=temperature, top_k=top_k, top_p=top_p, generators=generators[:1]
        )
        assert depth._graphs[(1, 1)] is first_graph
    torch.testing.assert_close(sample(1), sampled, atol=0, rtol=0)

    graph = None
    for scale in (4.0, 0.5, 4.0):
        expected_cfg = depth._generate_frame(
            hidden, first[:1], depth._allocate_cache(hidden), 0, 0, 1, generators[0], guidance_scale=scale
        )
        actual_cfg = depth.generate_frames(
            hidden, first[:1], temperature=0, top_k=0, top_p=1, generators=generators[:1], guidance_scale=scale
        )
        torch.testing.assert_close(actual_cfg, expected_cfg, atol=0, rtol=0)
        if graph is not None:
            assert depth._graphs[(2, 2)] is graph
        graph = depth._graphs[(2, 2)]
        sample(2)
