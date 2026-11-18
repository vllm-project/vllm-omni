# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.interfaces import TranscriptionLanguageMixin
from vllm_omni.model_executor.stage_input_processors.forced_aligner import (
    ALIGNER_DURATION_MS_KEY,
    ALIGNER_WORDS_KEY,
    _extract_text,
    _extract_waveform,
    asr2aligner,
    code2wav2aligner,
    decode_pooling_output,
    parse_asr_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeCompletion:
    def __init__(self, multimodal_output):
        self.multimodal_output = multimodal_output


class _FakeOutput:
    def __init__(self, multimodal_output, finished=True):
        self.finished = finished
        self.outputs = [_FakeCompletion(multimodal_output)]


def _audio_mm(n=2400, sr=24000):
    return {"audio": torch.zeros(n, dtype=torch.float32), "sr": torch.tensor(sr, dtype=torch.int32)}


def test_extract_waveform_from_audio_key():
    wav, sr = _extract_waveform(_audio_mm(n=100, sr=16000))
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert wav.shape == (100,)
    assert sr == 16000


def test_extract_waveform_falls_back_to_model_outputs_and_default_sr():
    wav, sr = _extract_waveform({"model_outputs": [torch.ones(50)]})
    assert wav.shape == (50,)
    assert sr == 24000  # default when sr missing


def test_extract_text_unwraps_additional_information_list():
    prompt = {"additional_information": {"text": ["Hello world."]}}
    assert _extract_text(prompt) == "Hello world."


def test_code2wav2aligner_builds_text_prompt_and_audio():
    prompt = {"additional_information": {"text": ["U.S.A test"], "language": ["English"]}}
    out = code2wav2aligner([_FakeOutput(_audio_mm())], prompt)

    assert len(out) == 1
    item = out[0]
    # text prompt: tokenization + audio features deferred to the stage preprocessor
    assert "prompt_token_ids" not in item
    # prompt built from official segmentation: "U.S.A" -> "USA"
    assert "USA<timestamp><timestamp>test" in item["prompt"]
    # audio carried as (np.float32, sr) tuple
    audio, sr = item["multi_modal_data"]["audio"]
    assert isinstance(audio, np.ndarray) and sr == 24000
    assert item["additional_information"]["aligner_words"] == [["USA", "test"]]
    assert item["additional_information"]["language"] == ["English"]


def test_code2wav2aligner_skips_unfinished_and_empty():
    # unfinished -> skipped
    assert code2wav2aligner([_FakeOutput(_audio_mm(), finished=False)], {}) == []
    # finished but empty waveform -> skipped
    empty = _FakeOutput({"audio": torch.zeros(0), "sr": torch.tensor(24000)})
    assert code2wav2aligner([empty], {"additional_information": {"text": ["hi"]}}) == []


def test_decode_pooling_output_produces_word_timestamps_ms():
    # 2 words -> 4 <timestamp> markers (id=99). classify_num=5, seg=200ms.
    hf_config = SimpleNamespace(
        timestamp_token_id=99,
        timestamp_segment_time=200.0,
        thinker_config=SimpleNamespace(classify_num=5),
    )
    # prompt: marker positions at indices 2,3,5,6
    prompt_token_ids = [1, 1, 99, 99, 1, 99, 99]
    request = SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        additional_information={ALIGNER_WORDS_KEY: [["hello", "world"]]},
    )
    logits = torch.zeros((7, 5), dtype=torch.float32)
    logits[2, 0] = 1.0  # word0 start -> bin 0
    logits[3, 2] = 1.0  # word0 end   -> bin 2
    logits[5, 2] = 1.0  # word1 start -> bin 2
    logits[6, 4] = 1.0  # word1 end   -> bin 4

    ms = decode_pooling_output(logits, request, hf_config)

    # decoder returns a bare [n_words, 2] int32 tensor (not a dict)
    assert ms.tolist() == [[0, 400], [400, 800]]  # bins x 200ms


def test_decode_pooling_output_handles_hf_config_without_thinker():
    # hf_config without thinker_config falls back to itself for classify_num.
    hf_config = SimpleNamespace(
        timestamp_token_id=99,
        timestamp_segment_time=100.0,
        classify_num=4,
    )
    request = SimpleNamespace(
        prompt_token_ids=[99, 99],
        additional_information={ALIGNER_WORDS_KEY: [["hi"]]},
    )
    logits = torch.zeros((2, 4), dtype=torch.float32)
    logits[0, 1] = 1.0
    logits[1, 3] = 1.0

    ms = decode_pooling_output(logits, request, hf_config)
    assert ms.tolist() == [[100, 300]]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("language English<asr_text>Hello world.", ("Hello world.", "English")),
        ("language Chinese<asr_text>你好。", ("你好。", "Chinese")),
        ("language None<asr_text>", ("", None)),
        ("language Unknown<asr_text>Hello", ("Hello", None)),
        ("language auto<asr_text>Hello", ("Hello", None)),
        ("language <asr_text>Hello", ("Hello", None)),
        ("language English<asr_te", ("language English<asr_te", None)),
        ("Plain transcript", ("Plain transcript", None)),
    ],
)
def test_parse_asr_output(raw, expected, qwen_asr_context):
    assert parse_asr_output(raw, model_config=qwen_asr_context.source_model_config) == expected


def test_parse_asr_output_uses_a_generic_models_postprocessor_without_qwen_cleanup():
    class SpeechModel:
        @classmethod
        def post_process_output(cls, text):
            return text.removeprefix("[transcript]")

    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="another_asr"))
    literal_text = "language English<asr_text>literal<|im_end|>"
    assert parse_asr_output(f"[transcript]{literal_text}", model_config=model_config, model_cls=SpeechModel) == (
        literal_text,
        None,
    )


def test_parse_asr_output_inherits_default_language_without_modifying_text():
    class SpeechModel(TranscriptionLanguageMixin):
        @classmethod
        def post_process_output(cls, text):
            return text

    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="another_asr"))
    text = "  language English<asr_text>literal  "
    assert parse_asr_output(text, model_config=model_config, model_cls=SpeechModel) == (text, None)


@pytest.mark.parametrize("transcript", ["こんにちは", "  <asr_text>literal<|im_end|><|endoftext|>  "])
def test_parse_asr_output_uses_model_language_hook_and_preserves_native_text(mocker, transcript):
    model_cls = SimpleNamespace(
        post_process_output=mocker.Mock(return_value=transcript),
        get_transcription_language=mocker.Mock(return_value="Japanese"),
    )
    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="another_asr"))
    raw_text = "[model-specific language metadata]こんにちは"

    assert parse_asr_output(raw_text, model_config=model_config, model_cls=model_cls) == (transcript, "Japanese")
    model_cls.post_process_output.assert_called_once_with(raw_text)
    model_cls.get_transcription_language.assert_called_once_with(raw_text)


def test_parse_asr_output_resolves_the_source_model_config(mocker):
    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="whisper"))
    model_cls = SimpleNamespace(post_process_output=mocker.Mock(return_value="normalized transcript"))
    resolve = mocker.patch("vllm.model_executor.model_loader.get_model_cls", return_value=model_cls)
    assert parse_asr_output("raw transcript", model_config=model_config) == ("normalized transcript", None)
    resolve.assert_called_once_with(model_config)
    model_cls.post_process_output.assert_called_once_with("raw transcript")


def test_parse_asr_output_explicit_model_class_takes_precedence(mocker):
    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="whisper"))
    resolve = mocker.patch("vllm.model_executor.model_loader.get_model_cls", side_effect=AssertionError("not needed"))
    model_cls = SimpleNamespace(post_process_output=lambda text: "normalized")
    assert parse_asr_output("raw transcript", model_config=model_config, model_cls=model_cls) == ("normalized", None)
    resolve.assert_not_called()


def test_parse_asr_output_rejects_a_non_text_postprocessor_result():
    model_cls = SimpleNamespace(post_process_output=lambda text: None)
    with pytest.raises(TypeError, match="must return a transcript string"):
        parse_asr_output(
            "raw transcript",
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="another_asr")),
            model_cls=model_cls,
        )


def test_qwen_asr_registry_preserves_upstream_inference_and_processor():
    from vllm.model_executor.models.qwen3_asr import (
        Qwen3ASRForConditionalGeneration as UpstreamQwen3ASR,
    )

    from vllm_omni.model_executor.models.qwen3_asr.qwen3_asr import Qwen3ASRForConditionalGeneration
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    model_cls = OmniModelRegistry._try_load_model_cls("Qwen3ASRForConditionalGeneration")
    assert model_cls is Qwen3ASRForConditionalGeneration
    assert issubclass(model_cls, UpstreamQwen3ASR)
    assert model_cls._processor_factory is UpstreamQwen3ASR._processor_factory
    assert model_cls.forward is UpstreamQwen3ASR.forward


@pytest.fixture
def qwen_asr_context(mocker):
    from vllm_omni.model_executor.models.qwen3_asr.qwen3_asr import Qwen3ASRForConditionalGeneration

    mocker.patch("vllm.model_executor.model_loader.get_model_cls", return_value=Qwen3ASRForConditionalGeneration)
    return SimpleNamespace(source_model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_asr")))


def _asr_output(text, *, cumulative_text=None, finished=True):
    return SimpleNamespace(
        finished=finished,
        outputs=[SimpleNamespace(text=text, cumulative_text=cumulative_text)],
    )


def test_asr2aligner_uses_original_audio_and_completed_transcript(qwen_asr_context):
    audio = np.ones(16000, dtype=np.float32)
    prompt = {"multi_modal_data": {"audio": (audio, 16000)}}
    source = _asr_output("world", cumulative_text="language English<asr_text>Hello world")
    out = asr2aligner([source], prompt, _streaming_context=qwen_asr_context)

    assert len(out) == 1
    item = out[0]
    assert item["multi_modal_data"]["audio"][0] is audio
    assert item["multi_modal_data"]["audio"][1] == 16000
    assert "Hello<timestamp><timestamp>world<timestamp><timestamp>" in item["prompt"]
    assert item["additional_information"] == {
        "text": ["Hello world"],
        ALIGNER_WORDS_KEY: [["Hello", "world"]],
        ALIGNER_DURATION_MS_KEY: [1000.0],
        "language": ["English"],
    }


def test_asr2aligner_uses_the_source_model_for_encoder_decoder_audio(mocker):
    audio = np.ones(16000, dtype=np.float32)
    prompt = {
        "encoder_prompt": {"prompt_token_ids": [1], "multi_modal_data": {"audio": (audio, 16000)}},
        "decoder_prompt": {"prompt_token_ids": [2]},
        "additional_information": {"language": ["en"]},
    }
    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="whisper"))
    model_cls = SimpleNamespace(post_process_output=mocker.Mock(return_value="Hello world"))
    mocker.patch("vllm.model_executor.model_loader.get_model_cls", return_value=model_cls)
    result = asr2aligner(
        [_asr_output("[model-specific transcript]")],
        prompt,
        _streaming_context=SimpleNamespace(source_model_config=model_config),
    )
    assert result[0]["multi_modal_data"]["audio"][0] is audio
    assert result[0]["multi_modal_data"]["audio"][1] == 16000
    assert result[0]["additional_information"]["text"] == ["Hello world"]
    assert result[0]["additional_information"][ALIGNER_WORDS_KEY] == [["Hello", "world"]]
    assert result[0]["additional_information"]["language"] == ["en"]
    model_cls.post_process_output.assert_called_once_with("[model-specific transcript]")


def test_asr2aligner_preserves_per_request_audio_and_explicit_language(monkeypatch, qwen_asr_context):
    calls = []

    def segment(text, language):
        calls.append((text, language))
        return [text]

    monkeypatch.setattr("vllm_omni.model_executor.stage_input_processors.forced_aligner.segment_words", segment)
    first = np.zeros(8000, dtype=np.float32)
    second = np.ones(24000, dtype=np.float32)
    prompts = [
        {"multi_modal_data": {"audio": [(first, 16000)]}, "additional_information": {"language": ["ja"]}},
        {"multi_modal_data": {"audio": (second, 24000)}},
    ]
    outputs = [
        _asr_output("language Japanese<asr_text>こんにちは"),
        _asr_output("language Korean<asr_text>안녕하세요"),
    ]
    result = asr2aligner(outputs, prompts, _streaming_context=qwen_asr_context)
    assert calls == [("こんにちは", "ja"), ("안녕하세요", "Korean")]
    assert result[0]["multi_modal_data"]["audio"][0] is first
    assert result[1]["multi_modal_data"]["audio"][0] is second
    assert result[0]["multi_modal_data"]["audio"][1] == 16000
    assert result[1]["multi_modal_data"]["audio"][1] == 24000
    assert [item["additional_information"][ALIGNER_DURATION_MS_KEY] for item in result] == [[500.0], [1000.0]]


@pytest.mark.parametrize("channels_first", [False, True])
@pytest.mark.parametrize("tensor_input", [False, True])
def test_asr2aligner_normalizes_stereo_without_changing_duration(qwen_asr_context, channels_first, tensor_input):
    audio = np.stack([np.zeros(24000, dtype=np.float32), np.ones(24000, dtype=np.float32)])
    if not channels_first:
        audio = audio.T
    if tensor_input:
        audio = torch.from_numpy(audio)
    prompt = {"multi_modal_data": {"audio": (audio, 24000)}}

    result = asr2aligner([_asr_output("hello")], prompt, _streaming_context=qwen_asr_context)

    waveform, sample_rate = result[0]["multi_modal_data"]["audio"]
    assert waveform.shape == (24000,)
    assert waveform.dtype == np.float32
    np.testing.assert_array_equal(waveform, np.full(24000, 0.5, dtype=np.float32))
    assert sample_rate == 24000
    assert result[0]["additional_information"][ALIGNER_DURATION_MS_KEY] == [1000.0]


def test_asr2aligner_silent_transcript_still_completes_alignment(qwen_asr_context):
    prompt = {"multi_modal_data": {"audio": (np.zeros(16000, dtype=np.float32), 16000)}}
    assert asr2aligner([_asr_output("partial", finished=False)], prompt) == []
    result = asr2aligner([_asr_output("language None<asr_text>")], prompt, _streaming_context=qwen_asr_context)
    assert len(result) == 1
    assert result[0]["additional_information"][ALIGNER_WORDS_KEY] == [[]]
    assert "<timestamp>" not in result[0]["prompt"]


def test_asr2aligner_requires_source_model_configuration():
    prompt = {"multi_modal_data": {"audio": (np.zeros(16000, dtype=np.float32), 16000)}}
    with pytest.raises(ValueError, match="source model configuration"):
        asr2aligner([_asr_output("hello")], prompt)


@pytest.mark.parametrize("text", [None, 42])
def test_asr2aligner_rejects_non_text_transcripts(text):
    prompt = {"multi_modal_data": {"audio": (np.zeros(16000, dtype=np.float32), 16000)}}
    with pytest.raises(TypeError, match="ASR output must contain a transcript string"):
        asr2aligner([_asr_output(text)], prompt)


def test_decode_pooling_output_handles_silent_asr_without_timestamp_markers():
    hf_config = SimpleNamespace(timestamp_token_id=99, timestamp_segment_time=100.0, classify_num=4)
    request = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        additional_information={ALIGNER_WORDS_KEY: [[]], ALIGNER_DURATION_MS_KEY: [1000.0]},
    )
    result = decode_pooling_output(torch.zeros((3, 4)), request, hf_config)
    assert result.shape == (0, 2)
    assert result.dtype == torch.int32


@pytest.mark.parametrize(
    "prompt",
    [
        {},
        {"multi_modal_data": {"audio": np.zeros(24000, dtype=np.float32)}},
        {"encoder_prompt": {"multi_modal_data": {"audio": np.zeros(24000, dtype=np.float32)}}},
        {"multi_modal_data": {"audio": (np.zeros(100), 0)}},
    ],
)
def test_asr2aligner_rejects_missing_or_invalid_original_audio(prompt, qwen_asr_context):
    with pytest.raises(ValueError, match="ASR forced alignment requires"):
        asr2aligner([_asr_output("hello")], prompt, _streaming_context=qwen_asr_context)
