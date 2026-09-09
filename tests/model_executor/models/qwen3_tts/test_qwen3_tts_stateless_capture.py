# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise actual stateless decoder capture, not a mocked capture indicator."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("sequence_length", [8, 32])
@pytest.mark.parametrize("input_mode", ["implicit", "positions", "mask"])
@torch.inference_mode()
def test_stateless_decoder_capture_replays_new_inputs(batch_size, sequence_length, input_mode):
    torch.manual_seed(42)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        hidden_size=32,
        latent_dim=16,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        num_hidden_layers=1,
        num_quantizers=2,
        sliding_window=16,
        decoder_dim=32,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
    )
    config._attn_implementation = "sdpa"
    model = Qwen3TTSTokenizerV2DecoderTransformerModel._from_config(config).cuda().eval()
    inputs = torch.randn(batch_size, sequence_length, config.latent_dim, device="cuda")
    positions = {}
    if input_mode == "positions":
        cache_position = torch.arange(sequence_length, device="cuda")
        positions = {"cache_position": cache_position, "position_ids": cache_position.unsqueeze(0)}
    elif input_mode == "mask":
        attention_mask = torch.ones(batch_size, sequence_length, device="cuda", dtype=torch.long)
        attention_mask[:, 0] = 0
        positions = {"attention_mask": attention_mask}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(inputs_embeds=inputs, use_cache=False, **positions)
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()

    # No wrapper or eager fallback: an illegal capture must fail this test.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = model(inputs_embeds=inputs, use_cache=False, **positions).last_hidden_state

    outputs = []
    for _ in range(3):
        new_inputs = torch.randn_like(inputs)
        expected = model(inputs_embeds=new_inputs, use_cache=False, **positions).last_hidden_state
        if input_mode == "implicit":
            # Compare the cached mask against the independent live-mask path,
            # not only against eager execution using the same cached mask.
            reference = model(
                inputs_embeds=new_inputs,
                use_cache=False,
                position_ids=torch.arange(sequence_length, device="cuda").unsqueeze(0),
            ).last_hidden_state
            torch.testing.assert_close(expected, reference, atol=1e-5, rtol=1e-4)
        inputs.copy_(new_inputs)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured_output, expected, atol=1e-5, rtol=1e-4)
        outputs.append(captured_output.clone())
    assert not torch.equal(outputs[0], outputs[1]), "replay must consume the new input"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@torch.inference_mode()
def test_stateless_codec_wrapper_captures_and_replays_waveform(batch_size):
    from tests.model_executor.models.qwen3_tts.test_qwen3_tts_incremental_decode import _make_small_decoder
    from vllm_omni.model_executor.models.qwen3_tts.segmented_graph_wrapper import CUDAGraphDecoderWrapper

    torch.manual_seed(42)
    decoder = _make_small_decoder().cuda().eval()
    wrapper = CUDAGraphDecoderWrapper(decoder, num_quantizers=2, async_chunk=False)
    # Exercise the production capture method, including quantization and
    # waveform convolutions, without accepting warmup's eager fallback.
    wrapper._capture_stateless(batch_size, 8, torch.device("cuda"), torch.long)
    state = wrapper.stateless_states[(batch_size, 8)]
    for _ in range(3):
        codes = torch.randint(0, 32, (batch_size, 2, 8), device="cuda")
        expected = decoder._forward_exact(codes)
        state["input"]["codes"].copy_(codes)
        state["graph"].replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(state["output"], expected, atol=1e-5, rtol=1e-4)
        assert torch.isfinite(state["output"]).all()
