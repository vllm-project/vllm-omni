# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Gepard-1.0 speech adapter.

Covers detection, the reject matrix, voice default, empty input, extra_params,
max_new_tokens frame bounds, and build() parity with the offline prompt helper.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.gepard import GepardAdapter
from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "nineninesix/gepard-1.0"


def _adapter(*, revision: str | None = None) -> GepardAdapter:
    engine = SimpleNamespace(model_config=SimpleNamespace(model=_MODEL, revision=revision))
    return GepardAdapter(SpeechServingContext(server=SimpleNamespace(engine_client=engine), engine_client=engine))


def test_gepard_detection_and_registration() -> None:
    assert resolve_adapter("gepard") is GepardAdapter
    assert GepardAdapter.stage_keys == {"gepard"}
    assert GepardAdapter.model_archs == {"GepardTalkerForConditionalGeneration"}
    assert detect_tts_model_type("gepard", None) == "gepard"
    assert detect_tts_model_type("gepard", "GepardTalkerForConditionalGeneration") == "gepard"
    assert detect_tts_model_type(None, "GepardTalkerForConditionalGeneration") == "gepard"


def test_gepard_accepts_omitted_and_default_voice() -> None:
    adapter = _adapter()
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello from Gepard.")) is None
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello from Gepard.", voice="default")) is None
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello from Gepard.", voice="Default")) is None
    assert (
        adapter.validate(
            OpenAICreateSpeechRequest.model_validate({"input": "Hello from Gepard.", "speaker": "default"})
        )
        is None
    )


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"voice": "vivian"}, "voice"),
        ({"speed": 1.2}, "speed"),
        ({"extra_params": {"temperature": 0.3}}, "extra_params"),
        ({"extra_params": {"top_p": 0.9}}, "extra_params"),
        ({"extra_params": {"top_k": 20}}, "extra_params"),
        ({"extra_params": {"foo": 1}}, "extra_params"),
        ({"ref_audio": "https://example.com/ref.wav"}, "ref_audio"),
        ({"ref_text": "transcript"}, "ref_text"),
        ({"ref_audio_2": "https://example.com/ref2.wav"}, "ref_audio_2"),
        ({"speaker_embedding": [0.1, 0.2]}, "speaker_embedding"),
        ({"x_vector_only_mode": True}, "x_vector_only_mode"),
        ({"task_type": "Base"}, "task_type"),
        ({"non_streaming_mode": True}, "non_streaming_mode"),
        ({"instructions": "speak happily"}, "instructions"),
        ({"language": "English"}, "language"),
        ({"ambient_sound": "rain"}, "ambient_sound"),
        ({"duration_seconds": 2.0}, "duration_seconds"),
        ({"initial_codec_chunk_frames": 4}, "initial_codec_chunk_frames"),
        ({"word_timestamps": True}, "word_timestamps"),
        ({"response_format": "opus"}, "opus"),
    ],
)
def test_gepard_rejects_unsupported_fields(kwargs: dict, field: str) -> None:
    adapter = _adapter()
    err = adapter.validate(OpenAICreateSpeechRequest(input="Hello from Gepard.", **kwargs))
    assert err is not None
    assert field in err


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_gepard_rejects_empty_input(text: str) -> None:
    adapter = _adapter()
    err = adapter.validate(OpenAICreateSpeechRequest(input=text))
    assert err is not None
    assert "empty" in err.lower()


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_gepard_schema_rejects_non_positive_max_new_tokens(max_new_tokens: int) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        OpenAICreateSpeechRequest(input="Hello from Gepard.", max_new_tokens=max_new_tokens)


def test_gepard_rejects_max_new_tokens_above_frame_budget() -> None:
    adapter = _adapter()
    err = adapter.validate(OpenAICreateSpeechRequest(input="Hello from Gepard.", max_new_tokens=4097))
    assert err is not None
    assert "max_new_tokens" in err
    assert "frames" in err


def test_gepard_accepts_max_new_tokens_bounds() -> None:
    adapter = _adapter()
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello from Gepard.", max_new_tokens=1)) is None
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello from Gepard.", max_new_tokens=4096)) is None


def test_gepard_apply_sampling_overrides_uses_frames() -> None:
    adapter = _adapter()
    stage_defaults = [SimpleNamespace(max_tokens=1000)]
    overridden = adapter.apply_sampling_overrides(
        stage_defaults,
        OpenAICreateSpeechRequest(input="Hello from Gepard.", max_new_tokens=32),
    )
    assert overridden[0].max_tokens == 32
    assert stage_defaults[0].max_tokens == 1000


def test_gepard_build_matches_offline_prompt_builder(monkeypatch) -> None:
    adapter = _adapter()
    cfg = GepardConfig()
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.gepard.configuration_gepard.GepardConfig.from_checkpoint",
        staticmethod(lambda *_a, **_k: cfg),
    )

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [11, 12, 13]}

    adapter._tokenizer = _Tok()
    adapter._gepard_config = cfg
    text = "Hello from Gepard."
    prepared = asyncio.run(adapter.build(OpenAICreateSpeechRequest(input=text, seed=7), [], False))

    expected = build_gepard_prompt_ids([11, 12, 13], config=cfg)
    assert prepared.prompt["prompt_token_ids"] == expected
    assert prepared.prompt["additional_information"] == {"text": [text]}
    assert prepared.model_type == "gepard"
    assert prepared.output_policy.accumulate_nonstreaming is True


def test_gepard_loads_tokenizer_and_config_at_served_revision(monkeypatch) -> None:
    adapter = _adapter(revision="abc123")
    seen: dict[str, object] = {}

    def fake_from_checkpoint(model, backbone_config=None, revision=None):
        seen["config"] = (model, revision)
        return GepardConfig()

    class _Tok:
        def __init__(self, *args, **kwargs):
            seen["tok"] = (args[0] if args else None, kwargs.get("revision"))

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [11, 12, 13]}

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.gepard.configuration_gepard.GepardConfig.from_checkpoint",
        staticmethod(fake_from_checkpoint),
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _Tok)
    prepared = asyncio.run(adapter.build(OpenAICreateSpeechRequest(input="Hello from Gepard."), [], False))
    assert seen["config"] == (_MODEL, "abc123")
    assert seen["tok"] == (_MODEL, "abc123")
    assert prepared.prompt["prompt_token_ids"] == build_gepard_prompt_ids([11, 12, 13], config=GepardConfig())


def _gepard_mock_output(sample_rate: int = 22050, num_samples: int = 2048) -> OmniRequestOutput:
    class MockCompletionOutput:
        def __init__(self) -> None:
            self.index = 0
            self.text = ""
            self.token_ids: list[int] = []
            self.finish_reason = "stop"
            self.stop_reason = None
            self.logprobs = None

    class MockRequestOutput:
        def __init__(self) -> None:
            self.request_id = "speech-gepard-mock"
            self.outputs = [MockCompletionOutput()]
            self.multimodal_output = {
                "audio": torch.zeros(num_samples),
                "sr": torch.tensor(sample_rate),
            }
            self.finished = True
            self.prompt_token_ids = None
            self.encoder_prompt_token_ids = None
            self.num_cached_tokens = None
            self.prompt_logprobs = None
            self.kv_transfer_params = None

    output = OmniRequestOutput.from_stage_output(MockRequestOutput(), stage_id=0, final_output_type="audio")
    output.metrics = {"stage_metrics": {"0": {"num_tokens_in": 0, "num_tokens_out": 2}}}
    return output


@pytest.fixture
def gepard_server(mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.tts_adapters.base.load_supported_speakers",
        return_value={"default"},
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.tts_adapters.base.load_codec_frame_rate",
        return_value=22050.0,
    )

    async def _gen(*_args, **_kwargs):
        yield _gepard_mock_output()

    mock_engine_client = mocker.MagicMock()
    mock_engine_client.errored = False
    mock_engine_client.model_config = mocker.MagicMock(model=_MODEL, async_chunk=False, revision=None)
    mock_engine_client.default_sampling_params_list = [SimpleNamespace(max_tokens=1000, seed=42, extra_args=None)]
    mock_engine_client.tts_batch_max_items = 32
    mock_engine_client.generate = mocker.MagicMock(side_effect=lambda **_k: _gen())
    mock_engine_client.stage_configs = [
        SimpleNamespace(
            engine_args=SimpleNamespace(model_stage="gepard", model_arch="GepardTalkerForConditionalGeneration"),
            tts_args={},
        )
    ]

    mock_models = mocker.MagicMock()
    mock_models.is_base_model.return_value = True
    server = OmniOpenAIServingSpeech(
        engine_client=mock_engine_client,
        models=mock_models,
        request_logger=mocker.MagicMock(),
    )
    cfg = GepardConfig()
    server._adapter._gepard_config = cfg
    server._adapter._tokenizer = lambda text, add_special_tokens=False: {"input_ids": [11, 12, 13]}
    yield server
    server.shutdown()


def test_gepard_prepare_uses_prompt_builder(gepard_server) -> None:
    request_id, _generator, _ = asyncio.run(
        gepard_server._prepare_speech_generation(OpenAICreateSpeechRequest(input="Hello from Gepard."))
    )
    assert request_id.startswith("speech-")
    prompt = gepard_server.engine_client.generate.call_args.kwargs["prompt"]
    assert prompt["prompt_token_ids"] == build_gepard_prompt_ids([11, 12, 13], config=GepardConfig())


@pytest.mark.parametrize("response_format", ["wav", "pcm", "flac", "mp3"])
def test_gepard_mocked_engine_encodes_response_formats(gepard_server, response_format: str) -> None:
    audio, media_type = asyncio.run(
        gepard_server._generate_audio_bytes(
            OpenAICreateSpeechRequest(input="Hello from Gepard.", response_format=response_format)
        )
    )
    assert isinstance(audio, (bytes, str))
    assert len(audio) > 0
    expected_media_type = {"wav": "audio/wav", "pcm": "audio/pcm", "flac": "audio/flac", "mp3": "audio/mpeg"}
    assert media_type == expected_media_type[response_format]


def test_gepard_streaming_rejects_non_pcm_wav() -> None:
    with pytest.raises(ValidationError, match="requires response_format='pcm' or 'wav'"):
        OpenAICreateSpeechRequest(input="Hello from Gepard.", stream=True, response_format="mp3")
