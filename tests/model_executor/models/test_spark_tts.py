# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for Spark's prompt, tokenizer and full-utterance transfer contracts."""

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.spark_tts.prompt import clone_prompt, control_prompt
from vllm_omni.model_executor.stage_input_processors.spark_tts import (
    bicodec_token_only,
    llm_to_bicodec_full_payload,
    parse_codes,
)
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_control_prompt_grammar():
    assert control_prompt("  Hello.  ", gender="male", pitch="low", speed="very_high") == (
        "<|task_controllable_tts|><|start_content|>Hello.<|end_content|>"
        "<|start_style_label|><|gender_1|><|pitch_label_1|><|speed_label_4|><|end_style_label|>"
    )


@pytest.mark.parametrize("kwargs", [{"gender": "unknown"}, {"pitch": "unknown"}, {"speed": "unknown"}])
def test_control_prompt_rejects_invalid_style(kwargs):
    with pytest.raises(ValueError, match="Invalid Spark-TTS"):
        control_prompt("Hello.", **kwargs)


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_prompt_builders_reject_empty_target(text):
    with pytest.raises(ValueError, match="text"):
        control_prompt(text)
    with pytest.raises(ValueError, match="text"):
        clone_prompt(text, [7], [])


@pytest.mark.parametrize("ref_text", [None, "", " \t\n"])
def test_clone_without_transcript_uses_only_reference_globals(ref_text):
    assert clone_prompt("  Hello.  ", [0, 4095], [17], ref_text) == (
        "<|task_tts|><|start_content|>Hello.<|end_content|>"
        "<|start_global_token|><|bicodec_global_0|><|bicodec_global_4095|><|end_global_token|>"
    )


def test_clone_with_transcript_preserves_reference_text_and_semantics():
    assert clone_prompt("  Hello.  ", [7], [0, 8191], "Reference. ") == (
        "<|task_tts|><|start_content|>Reference. Hello.<|end_content|>"
        "<|start_global_token|><|bicodec_global_7|><|end_global_token|>"
        "<|start_semantic_token|><|bicodec_semantic_0|><|bicodec_semantic_8191|>"
    )


def test_clone_rejects_missing_reference_codes():
    with pytest.raises(ValueError, match="reference global codes"):
        clone_prompt("Hello.", [], [7])
    with pytest.raises(ValueError, match="reference semantic codes"):
        clone_prompt("Hello.", [7], [], "Reference.")


@pytest.mark.parametrize("close_marker", [[], [165156]])
def test_parse_codes_prefers_prompt_globals_even_without_end_marker(close_marker):
    # Reference semantics must never be replayed as generated speech. Global
    # tokens preceding START_GLOBAL are not part of the reference speaker.
    prompt = [151674, 165150, 151672, 155761, *close_marker, 155778]
    output = [151667, 155762, 163952, 165157, 151645]
    assert parse_codes(output, prompt) == ([1, 8191], [7])


@pytest.mark.parametrize("prompt", [[], [151672], [165150, 42, 165156]])
def test_parse_codes_falls_back_to_generated_globals(prompt):
    assert parse_codes([42, 151664, 151665, 155760, 155761, 163952, 163953], prompt) == (
        [0, 8191],
        [0, 4095],
    )


def test_parse_codes_stops_reference_globals_at_end_marker():
    assert parse_codes([155762], [165150, 151672, 165156, 151673]) == ([1], [7])


def test_malformed_output_still_sends_terminal_payload():
    request = OmniRequest("empty", [0], SamplingParams(max_tokens=1), pooling_params=None)
    request.append_output_token_ids([42])
    payload = llm_to_bicodec_full_payload(None, None, request)
    assert payload.meta.finished.item() is True
    assert payload.meta.req_id == ["empty"]
    assert payload.codes.audio.shape == (0, 1)
    assert payload.codes.ref.shape == (0, 1)


def test_clone_payload_keeps_full_codes_and_external_request_id():
    request = OmniRequest(
        external_req_id="external",
        request_id="internal",
        prompt_token_ids=[165150, 151672, 165156, 155778],
        sampling_params=SamplingParams(max_tokens=3),
        pooling_params=None,
    )
    request.append_output_token_ids([155762, 155763, 151645])
    payload = llm_to_bicodec_full_payload(None, None, request)
    assert payload.meta.req_id == ["external"]
    assert payload.meta.finished.item() is True
    assert payload.codes.audio.tolist() == [[1], [2]]
    assert payload.codes.ref.tolist() == [[7]]


def test_payload_requires_request_id():
    with pytest.raises(ValueError, match="request id"):
        llm_to_bicodec_full_payload(None, None, object())


def test_four_long_clone_requests_use_four_placeholder_tokens(mocker):
    # Even cumulative prompt + output ids must not consume the 4096-token
    # stage-1 budget or split one utterance into multiple decoder calls.
    sources = [
        OmniRequestOutput(
            finished=True,
            outputs=[mocker.Mock(spec=CompletionOutput, cumulative_token_ids=[1] * (1024 + 532))],
        )
        for _ in range(4)
    ]
    sources.append(OmniRequestOutput(finished=False))
    prompts = bicodec_token_only(sources)
    assert [prompt["prompt_token_ids"] for prompt in prompts] == [[0], [0], [0], [0]]


def test_batched_decoder_handles_one_placeholder_per_full_utterance(mocker, model_config):
    from vllm_omni.model_executor.models.spark_tts import bicodec

    codec = torch.nn.Module()
    codec.detokenize = mocker.Mock(side_effect=[torch.full((1, 8), float(i)) for i in range(4)])
    mocker.patch.object(bicodec, "_build_codec", return_value=codec)
    decoder = bicodec.SparkBiCodecDecoder(vllm_config=model_config)
    infos = [{"codes": {"audio": [i] * 1024, "ref": [i] * 32}} for i in range(4)]
    output = decoder.forward(
        input_ids=torch.zeros(4, dtype=torch.long), runtime_additional_information=infos, seq_token_counts=[1] * 4
    )
    assert codec.detokenize.call_count == 4
    for i, (call, audio) in enumerate(zip(codec.detokenize.call_args_list, output.multimodal_outputs["model_outputs"])):
        assert call.args[0].shape == (1, 1024)
        assert call.args[1].shape == (1, 1, 32)
        assert call.args[0].unique().tolist() == [i]
        assert audio.tolist() == [float(i)] * 8


@pytest.fixture
def model_config():
    config = VllmConfig.__new__(VllmConfig)
    config.model_config = OmniModelConfig.__new__(OmniModelConfig)
    config.model_config.model = "unused"
    config.device_config = DeviceConfig(device="cpu")
    return config


@pytest.fixture
def speech_vocab():
    # Published Spark-TTS-0.5B token layout, independent of parser constants.
    vocab = {"<|start_global_token|>": 165150, "<|end_global_token|>": 165156, "<|im_end|>": 151645}
    vocab.update((f"<|bicodec_global_{i}|>", 151665 + i) for i in range(4096))
    vocab.update((f"<|bicodec_semantic_{i}|>", 155761 + i) for i in range(8192))
    return vocab


def test_model_validates_compatible_tokenizer_before_constructing_qwen(mocker, speech_vocab, model_config):
    from vllm_omni.model_executor.models.spark_tts import llm

    tokenizer = mocker.Mock()
    tokenizer.get_vocab.return_value = speech_vocab
    get_tokenizer = mocker.patch.object(llm, "cached_tokenizer_from_config", return_value=tokenizer)
    base_init = mocker.patch.object(llm.Qwen2ForCausalLM, "__init__", return_value=None)
    llm.SparkQwen2ForCausalLM(vllm_config=model_config, prefix="spark")
    get_tokenizer.assert_called_once_with(model_config.model_config)
    tokenizer.get_vocab.assert_called_once_with()
    base_init.assert_called_once_with(vllm_config=model_config, prefix="spark")


@pytest.mark.parametrize(
    "token",
    [
        "<|start_global_token|>",
        "<|end_global_token|>",
        "<|im_end|>",
        "<|bicodec_global_2048|>",
        "<|bicodec_semantic_4096|>",
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_model_rejects_mismatched_tokenizer_before_constructing_qwen(
    mocker, speech_vocab, model_config, token, missing
):
    from vllm_omni.model_executor.models.spark_tts import llm

    if missing:
        del speech_vocab[token]
    else:
        speech_vocab[token] += 1
    tokenizer = mocker.Mock()
    tokenizer.get_vocab.return_value = speech_vocab
    mocker.patch.object(llm, "cached_tokenizer_from_config", return_value=tokenizer)
    base_init = mocker.patch.object(llm.Qwen2ForCausalLM, "__init__", return_value=None)
    with pytest.raises(ValueError, match="Incompatible Spark-TTS tokenizer") as exc:
        llm.SparkQwen2ForCausalLM(vllm_config=model_config)
    assert token in str(exc.value)
    base_init.assert_not_called()


def test_model_rejects_skipping_tokenizer_initialization(mocker, model_config):
    from vllm_omni.model_executor.models.spark_tts import llm

    mocker.patch.object(llm, "cached_tokenizer_from_config", return_value=None)
    base_init = mocker.patch.object(llm.Qwen2ForCausalLM, "__init__", return_value=None)
    with pytest.raises(ValueError, match="requires tokenizer initialization"):
        llm.SparkQwen2ForCausalLM(vllm_config=model_config)
    base_init.assert_not_called()
