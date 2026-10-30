# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check depth attention against HF Llama and request-local sampling state."""

from collections import defaultdict

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, MimiConfig
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.mimi.modeling_mimi import MimiConv1d

from tests.model_executor.models.breeze_tts_2_helpers import _request_info, _small_talker
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.model_executor.models.breeze_tts_2.depth_decoder import (
    BreezeDepthDecoder,
    sample_graph_logits,
    sample_logits,
)
from vllm_omni.model_executor.models.breeze_tts_2.prompt import CFG_UNCOND_SUFFIX, build_breeze_prompt
from vllm_omni.model_executor.models.breeze_tts_2.reference_encoder import BreezeReferenceConv
from vllm_omni.model_executor.stage_input_processors.breeze_tts_2 import expand_cfg_prompts, talker2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("parameter", ["temperature", "top_p", "repetition_penalty"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_offline_prompt_rejects_nonfinite_sampling(mocker, parameter, value):
    with pytest.raises(ValueError, match="sampling"):
        build_breeze_prompt(mocker.Mock(), "Hello.", **{parameter: value})


@pytest.mark.parametrize("stride", [2, 3])
def test_reference_convolution_batch_preserves_partial_final_frames(stride):
    torch.manual_seed(7)
    config = MimiConfig(use_causal_conv=True, pad_mode="constant")
    first = MimiConv1d(config, 1, 4, kernel_size=2 * stride, stride=stride)
    last = MimiConv1d(config, 4, 4, kernel_size=4, stride=2, pad_mode="replicate", bias=False)
    batched_first = BreezeReferenceConv(first, stride)
    batched_last = BreezeReferenceConv(last, 2 * stride)
    lengths = torch.tensor([37, 62])
    batched_first.sample_lengths = batched_last.sample_lengths = lengths
    waves = [torch.randn(1, 1, int(length)) for length in lengths]
    padded = torch.cat([F.pad(wave, (0, 96 - wave.shape[-1])) for wave in waves])
    actual = batched_last(batched_first(padded))
    for row, waveform in enumerate(waves):
        expected = last(first(waveform))
        torch.testing.assert_close(actual[row : row + 1, :, : expected.shape[-1]], expected)
        assert not actual[row, :, expected.shape[-1] :].count_nonzero()


def test_reference_conditioning_survives_transport_and_cfg_keeps_true_prompt_lengths(mocker) -> None:
    tokenizer = mocker.Mock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text.encode("utf-8"))
    waveform = np.zeros(1921, dtype=np.float32)
    prompt = build_breeze_prompt(
        tokenizer, "Hello.", "Speak happily.", ref_audio=(waveform, 24000), ref_text="Reference.", guidance_scale=4
    )
    companion = expand_cfg_prompts(prompt, None)[0].prompt
    assert len(prompt["prompt_token_ids"]) > len(companion["prompt_token_ids"])
    positive = deserialize_additional_information(serialize_additional_information(prompt["additional_information"]))
    negative = deserialize_additional_information(serialize_additional_information(companion["additional_information"]))
    assert positive["breeze_prompt"]["target_ids"] == list(b"[S0]<ins_bos>Speak happily.<ins_eos>Hello.")
    assert negative["breeze_prompt"]["target_ids"] == list(b"[S0]Hello.")
    assert positive["breeze_prompt"]["reference_ids"] == negative["breeze_prompt"]["reference_ids"]
    assert positive["breeze_prompt"]["reference_frames"] == 2
    assert positive["cfg_group"]["role"] == "cond"
    assert negative["cfg_group"]["role"] == "uncond"
    torch.testing.assert_close(positive["reference_waveform"], torch.from_numpy(waveform), atol=0, rtol=0)
    torch.testing.assert_close(negative["reference_waveform"], positive["reference_waveform"], atol=0, rtol=0)


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


@pytest.fixture
def hf_depth(depth: BreezeDepthDecoder) -> LlamaModel:
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
    return reference


@pytest.mark.parametrize("batch", [1, 2])
@torch.inference_mode()
def test_depth_cached_attention_matches_hf_full_sequence(
    depth: BreezeDepthDecoder, hf_depth: LlamaModel, batch: int
) -> None:
    """Wrong cache positions or fused-QKV ordering change the reference hidden states."""
    embeddings = torch.randn(batch, 4, depth.config.hidden_size)
    expected = hf_depth(inputs_embeds=embeddings, use_cache=False).last_hidden_state
    shape = (batch, 2, 4, 8)
    caches = [(torch.zeros(shape), torch.zeros(shape)) for _ in depth.layers]
    actual = []
    for start, end in [(0, 2), (2, 3), (3, 4)]:
        x = embeddings[:, start:end]
        for layer, cache in zip(depth.layers, caches, strict=True):
            x = layer(
                x,
                depth.positions[start:end],
                cache,
                depth.causal_mask[start:end, :4][None, None],
                depth.rope_cos[start:end][None, None],
                depth.rope_sin[start:end][None, None],
            )
        actual.append(depth.norm(x))
    torch.testing.assert_close(torch.cat(actual, 1), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("guidance_scale", [0.5, 4.0])
@torch.inference_mode()
def test_cfg_depth_matches_hf_full_prefix_at_every_codebook(
    depth: BreezeDepthDecoder, hf_depth: LlamaModel, guidance_scale: float
) -> None:
    hidden = torch.randn(4, depth.config.audio_embed_size)
    first = torch.tensor([7, 11])
    prefix = torch.cat((hidden[:, None], depth.embed_tokens(first.repeat(2)[:, None])), dim=1)
    expected = [first]
    for codebook in range(depth.num_codebooks - 1):
        states = hf_depth(inputs_embeds=depth.inputs_embeds_projector(prefix), use_cache=False).last_hidden_state
        logits = F.linear(states[:, -1].float(), depth.codebooks_head[codebook].T)
        cond, uncond = logits.chunk(2)
        token = (uncond + guidance_scale * (cond - uncond))[:, : depth.vocab_size - 3].argmax(-1)
        expected.append(token)
        embed = depth.embed_tokens((token.repeat(2) + (codebook + 1) * depth.vocab_size)[:, None])
        prefix = torch.cat((prefix, embed), dim=1)
    actual = depth._generate_frame(
        hidden, first, depth._allocate_cache(hidden), 0.0, 0, 1.0, torch.Generator(), guidance_scale=guidance_scale
    )
    torch.testing.assert_close(actual, torch.stack(expected, dim=-1), atol=0, rtol=0)


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


@pytest.mark.parametrize("parameters", [(0.0, 0, 1.0), (0.9, 50, 1.0), (0.7, 100, 0.8), (1.2, 0, 0.95)])
def test_graph_sampler_matches_scalar_filtering_with_identical_noise(parameters) -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(32, 2051, generator=generator)
    logits[:, -3:] = -torch.inf
    logits[:, 10:14] = 4.0
    noise = torch.empty_like(logits).exponential_(generator=generator)
    expected = sample_logits(logits, *parameters, generator, noise)
    actual = sample_graph_logits(logits, torch.tensor(parameters), noise)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("top_p", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("top_k", [0, 2])
def test_graph_sampler_excludes_token_after_exact_nucleus_boundary(top_p, top_k) -> None:
    # Equal probabilities make the CDF boundary exact in float32. Bias the
    # shared noise toward the first token outside that nucleus so retaining
    # one extra candidate deterministically changes the sampled token.
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, -torch.inf, -torch.inf, -torch.inf]])
    noise = torch.ones_like(logits)
    first_excluded = int(top_p * 4)
    noise[:, first_excluded] = 0.001
    parameters = (1.0, top_k, top_p)
    expected = sample_logits(logits, *parameters, torch.Generator().manual_seed(42), noise)
    actual = sample_graph_logits(logits, torch.tensor(parameters), noise)

    assert expected.item() != first_excluded
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_codec_chunks_preserve_zero_frames_and_flush_each_request_once(mocker) -> None:
    manager = mocker.Mock()
    manager.connector.config = {"extra": {"codec_chunk_frames": 5, "initial_codec_chunk_frames": 5}}
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


def test_codec_chunk_ramp_keeps_frame_order_and_flushes_partial_tail(mocker) -> None:
    manager = mocker.Mock()
    manager.connector.config = {"extra": {"codec_chunk_frames": 5, "codec_chunk_ramp": [1, 2, 4, 5]}}
    manager.code_prompt_token_ids = defaultdict(list)
    manager.request_payload = {}
    request = mocker.Mock(external_req_id="ramp")
    request.is_finished.return_value = False
    outputs = []
    for index in range(15):
        result = talker2code2wav_async_chunk(manager, {"codes": {"audio": torch.full((1, 16), index)}}, request)
        if result is not None:
            outputs.append(result.codes.audio.reshape(16, -1).T)
    tail = talker2code2wav_async_chunk(manager, None, request, is_finished=True)
    outputs.append(tail.codes.audio.reshape(16, -1).T)
    assert [len(part) for part in outputs] == [1, 2, 4, 5, 3]
    torch.testing.assert_close(torch.cat(outputs), torch.arange(15)[:, None].expand(-1, 16))
    assert tail.meta.finished.item()


@torch.inference_mode()
def test_talker_history_and_generators_follow_requests_across_reordering_and_new_admission():
    model = _small_talker()
    first, second = _request_info("first"), _request_info("second")
    # Unequal spans must select the last hidden row of each request.
    model.make_omni_output(
        torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        model_intermediate_buffer=[first, second],
        request_token_spans=[(0, 3), (3, 4)],
    )
    reordered = model.make_omni_output(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        model_intermediate_buffer=[second, first],
        request_token_spans=[(0, 1), (1, 2)],
    )
    assert [frame[0, 0].item() for frame in reordered.multimodal_outputs["codes"]["audio"]] == [3, 2]
    assert first["breeze_state"]["history"].tolist() == [[1, 2]]
    assert second["breeze_state"]["history"].tolist() == [[3, 3]]
    generators = model.depth_decoder.generate_frames.call_args.kwargs["generators"]
    assert generators[0] is second["breeze_state"]["generator"]
    assert generators[1] is first["breeze_state"]["generator"]

    newcomer = _request_info("newcomer")
    admitted = model.make_omni_output(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        model_intermediate_buffer=[newcomer, first],
        request_token_spans=[(0, 1), (1, 2)],
    )
    assert [frame[0, 0].item() for frame in admitted.multimodal_outputs["codes"]["audio"]] == [1, 1]
    assert newcomer["breeze_state"]["history"].tolist() == [[1]]
    assert first["breeze_state"]["history"].tolist() == [[1, 2, 1]]
    assert second["breeze_state"]["history"].tolist() == [[3, 3]]


@torch.inference_mode()
def test_talker_eos_does_not_shift_live_state_or_poison_the_next_batch():
    model = _small_talker()
    finished, live = _request_info("finished"), _request_info("live")
    output = model.make_omni_output(
        torch.tensor([[-1.0, -1.0], [0.0, 1.0]]),
        model_intermediate_buffer=[finished, live],
        request_token_spans=[(0, 1), (1, 2)],
    )
    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert finished["breeze_state"]["history"].numel() == 0
    assert live["breeze_state"]["current"].tolist() == [[3, 3, 3]]
    assert model.compute_logits(torch.zeros(2, 2)).argmax(-1).tolist() == [7, 0]

    model.make_omni_output(torch.tensor([[0.0, 1.0]]), model_intermediate_buffer=[live], request_token_spans=[(0, 1)])
    assert live["breeze_state"]["history"].tolist() == [[3, 3]]
    assert model.compute_logits(torch.zeros(1, 2)).argmax(-1).tolist() == [0]


@pytest.mark.parametrize("temperature", [0.0, 0.9])
@torch.inference_mode()
def test_talker_mixed_eos_and_cfg_matches_independent_request_state(temperature):
    """Batch EOS decisions must not mix CFG state or change a request's RNG stream."""
    finished_companion = "finished_cfg" + CFG_UNCOND_SUFFIX
    live_companion = "live_cfg" + CFG_UNCOND_SUFFIX
    order = ["finished_cfg", "live_plain", finished_companion, "finished_plain", live_companion, "live_cfg"]
    hidden_by_id = {
        "finished_cfg": [-1.0, -1.0],
        finished_companion: [-1.0, -1.0],
        "finished_plain": [-1.0, -1.0],
        "live_plain": [0.0, 1.0],
        "live_cfg": [1.0, 0.0],
        live_companion: [1.0, 0.0],
    }

    def make_infos():
        infos = {request_id: _request_info(request_id) for request_id in order}
        for index, (request_id, info) in enumerate(infos.items()):
            info["breeze_sampling"].update(temperature=temperature, top_k=1)
            if "cfg" not in request_id:
                # Interleave distinct sampler groups while each request still
                # matches its independently decoded state and RNG trajectory.
                info["breeze_sampling"].update(temperature=temperature * 0.5, top_p=0.8)
            info["breeze_state"]["generator"].manual_seed(42 + index)
            info["breeze_state"]["history"] = torch.tensor([[3, 3]])
            if "cfg" in request_id:
                info["breeze_prompt"]["guidance_scale"] = 4.0
            if request_id.endswith(CFG_UNCOND_SUFFIX):
                info["breeze_prompt"]["role"] = "uncond"
        return infos

    def sample_depth(hidden, first, *, temperature, generators, **kwargs):
        # Consume request-local RNG state in the downstream phase as well as
        # codebook-0 sampling, without requiring depth CUDA graphs on CPU.
        tail = (
            torch.stack([torch.randint(0, 4, (2,), generator=generator) for generator in generators])
            if temperature > 0
            else first[:, None].expand(-1, 2)
        )
        return torch.cat((first[:, None], tail), dim=1)

    batched, independent = _small_talker(), _small_talker()
    batched.depth_decoder.generate_frames.side_effect = sample_depth
    independent.depth_decoder.generate_frames.side_effect = sample_depth
    batch_infos, independent_infos = make_infos(), make_infos()
    initial_rng = {rid: info["breeze_state"]["generator"].get_state() for rid, info in batch_infos.items()}
    saved_frames = []
    # The second step drops both EOS requests and moves the live CFG companion
    # ahead of its parent, exercising shrinking and reordered batches.
    for step_ids in (order, [live_companion, "live_cfg", "live_plain"]):
        expected_audio, expected_tokens = {}, {}
        for request_id in step_ids:
            if request_id.endswith(CFG_UNCOND_SUFFIX):
                continue
            pair = [request_id]
            if independent_infos[request_id]["breeze_prompt"]["guidance_scale"] != 1.0:
                pair.append(request_id + CFG_UNCOND_SUFFIX)
            result = independent.make_omni_output(
                torch.tensor([hidden_by_id[rid] for rid in pair]),
                model_intermediate_buffer=[independent_infos[rid] for rid in pair],
                request_token_spans=[(i, i + 1) for i in range(len(pair))],
            )
            tokens = independent.compute_logits(torch.zeros(len(pair), 2)).argmax(-1)
            for index, rid in enumerate(pair):
                expected_audio[rid] = result.multimodal_outputs["codes"]["audio"][index].clone()
                expected_tokens[rid] = tokens[index]

        result = batched.make_omni_output(
            torch.tensor([hidden_by_id[rid] for rid in step_ids]),
            model_intermediate_buffer=[batch_infos[rid] for rid in step_ids],
            request_token_spans=[(i, i + 1) for i in range(len(step_ids))],
        )
        actual_tokens = batched.compute_logits(torch.zeros(len(step_ids), 2)).argmax(-1)
        for index, rid in enumerate(step_ids):
            actual = result.multimodal_outputs["codes"]["audio"][index]
            torch.testing.assert_close(actual, expected_audio[rid], atol=0, rtol=0)
            torch.testing.assert_close(actual_tokens[index], expected_tokens[rid], atol=0, rtol=0)
            saved_frames.append((actual, actual.clone()))
        for rid in order:
            actual, expected = batch_infos[rid]["breeze_state"], independent_infos[rid]["breeze_state"]
            for key in ("history", "current"):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            torch.testing.assert_close(actual["generator"].get_state(), expected["generator"].get_state())

    # Unconditional companions never own sampling; an EOS request consumes
    # only codebook-0 sampling, with no downstream depth draws.
    for rid in (finished_companion, live_companion):
        torch.testing.assert_close(batch_infos[rid]["breeze_state"]["generator"].get_state(), initial_rng[rid])
    for rid in ("finished_cfg", "finished_plain"):
        reference_generator = torch.Generator().set_state(initial_rng[rid])
        sampling = batch_infos[rid]["breeze_sampling"]
        sample_logits(
            torch.zeros(1, batched.config.vocab_size),
            sampling["temperature"],
            sampling["top_k"],
            sampling["top_p"],
            reference_generator,
        )
        torch.testing.assert_close(
            batch_infos[rid]["breeze_state"]["generator"].get_state(), reference_generator.get_state()
        )

    for rid in ("finished_cfg", finished_companion, "finished_plain", live_companion):
        assert batch_infos[rid]["breeze_state"]["history"].tolist() == [[3, 3]]
    assert batch_infos["live_cfg"]["breeze_state"]["history"].tolist() == [[3, 3, 1, 2]]
    assert batch_infos["live_plain"]["breeze_state"]["history"].tolist() == [[3, 3, 3, 3]]
    assert batch_infos["live_cfg"]["breeze_state"]["current"] is batch_infos[live_companion]["breeze_state"]["current"]
    for original, snapshot in saved_frames:
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)


@torch.inference_mode()
def test_talker_all_eos_preserves_history_and_skips_depth():
    model = _small_talker()
    infos = [_request_info("first"), _request_info("second")]
    for info in infos:
        info["breeze_state"]["history"] = torch.tensor([[1, 2]])
    output = model.make_omni_output(
        torch.tensor([[-1.0, -1.0], [-2.0, -2.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )

    model.depth_decoder.generate_frames.assert_not_called()
    assert model.compute_logits(torch.zeros(2, 2)).argmax(-1).tolist() == [7, 7]
    assert all(frame.numel() == 0 for frame in output.multimodal_outputs["codes"]["audio"])
    for info in infos:
        assert info["breeze_state"]["history"].tolist() == [[1, 2]]
        assert info["breeze_state"]["current"].tolist() == [[0, 0, 0]]
