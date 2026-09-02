# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.omnivoice import pipeline_omnivoice

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeASRModel:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self

    def parameters(self):
        return iter(())


class _FakeASRPipeline:
    def __init__(self, result):
        self.result = result
        self.inputs = []
        self.model = _FakeASRModel()
        self.device = torch.device("cpu")

    def __call__(self, audio_input):
        self.inputs.append(audio_input)
        return self.result


@pytest.fixture
def pipeline() -> pipeline_omnivoice.OmniVoicePipeline:
    model = pipeline_omnivoice.OmniVoicePipeline.__new__(pipeline_omnivoice.OmniVoicePipeline)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model._load_asr_on_startup = False
    model._asr_model_name = pipeline_omnivoice._ASR_MODEL_NAME
    model._asr_device = "cpu"
    model._asr_pipeline = None
    return model


def _install_fake_asr(monkeypatch, result=None):
    asr = _FakeASRPipeline(result={"text": "  hello there  "} if result is None else result)
    load_calls = []

    def load_pipeline(*args, **kwargs):
        load_calls.append((args, kwargs))
        return asr

    monkeypatch.setattr(pipeline_omnivoice, "hf_pipeline", load_pipeline)
    return asr, load_calls


def test_missing_reference_text_loads_asr_and_strips_transcript(pipeline, monkeypatch):
    asr, load_calls = _install_fake_asr(monkeypatch)

    transcript = pipeline._resolve_ref_text((np.zeros((1, 4), dtype=np.float32), 16000), None)

    assert transcript == "hello there"
    assert len(load_calls) == 1
    assert asr.inputs[0]["sampling_rate"] == 16000
    assert asr.inputs[0]["array"].shape == (4,)


def test_second_missing_reference_text_request_reuses_asr(pipeline, monkeypatch):
    asr, load_calls = _install_fake_asr(monkeypatch)
    ref_audio = (np.zeros(4, dtype=np.float32), 16000)

    assert pipeline._resolve_ref_text(ref_audio, None) == "hello there"
    assert pipeline._resolve_ref_text(ref_audio, "   ") == "hello there"

    assert len(load_calls) == 1
    assert len(asr.inputs) == 2


def test_explicit_reference_text_bypasses_asr(pipeline, monkeypatch):
    _install_fake_asr(monkeypatch)
    ref_text = "  caller supplied transcript  "

    assert pipeline._resolve_ref_text((np.zeros(4), 16000), ref_text) == ref_text
    assert pipeline._asr_pipeline is None


def test_text_only_input_bypasses_asr(pipeline, monkeypatch):
    _install_fake_asr(monkeypatch)

    assert pipeline._resolve_ref_text(None, None) is None
    assert pipeline._asr_pipeline is None


@pytest.mark.parametrize("ref_text", [None, "", "   ", "\t\n"])
def test_empty_reference_text_uses_asr(pipeline, monkeypatch, ref_text):
    _install_fake_asr(monkeypatch)

    assert pipeline._resolve_ref_text((np.zeros(4), 16000), ref_text) == "hello there"


def test_empty_asr_result_is_rejected(pipeline, monkeypatch):
    _install_fake_asr(monkeypatch, result={"text": " \t"})

    with pytest.raises(ValueError, match="empty reference transcription"):
        pipeline._resolve_ref_text((np.zeros(4), 16000), None)


def test_malformed_asr_result_is_rejected(pipeline, monkeypatch):
    _install_fake_asr(monkeypatch, result={"chunks": []})

    with pytest.raises(RuntimeError, match="without a 'text' field"):
        pipeline._resolve_ref_text((np.zeros(4), 16000), None)


def test_asr_load_failure_includes_checkpoint_and_device(pipeline, monkeypatch):
    def load_pipeline(*args, **kwargs):
        raise OSError("download failed")

    monkeypatch.setattr(pipeline_omnivoice, "hf_pipeline", load_pipeline)

    with pytest.raises(RuntimeError) as exc_info:
        pipeline._load_asr_pipeline()

    message = str(exc_info.value)
    assert pipeline_omnivoice._ASR_MODEL_NAME in message
    assert "cpu" in message
    assert "download failed" in message


def test_asr_device_placement_failure_includes_checkpoint_and_device(pipeline, monkeypatch):
    asr, _ = _install_fake_asr(monkeypatch)
    pipeline._asr_device = "cuda:1"

    def move_model(device):
        raise RuntimeError(f"cannot move to {device}")

    monkeypatch.setattr(asr.model, "to", move_model)

    with pytest.raises(RuntimeError) as exc_info:
        pipeline._load_asr_pipeline()

    message = str(exc_info.value)
    assert pipeline_omnivoice._ASR_MODEL_NAME in message
    assert "cuda:1" in message
    assert "cannot move" in message


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [
        ({}, (False, pipeline_omnivoice._ASR_MODEL_NAME, None)),
        (
            {"omnivoice_asr": {"load_asr_on_startup": False, "asr_device": None}},
            (False, pipeline_omnivoice._ASR_MODEL_NAME, None),
        ),
        (
            {
                "omnivoice_asr": {
                    "load_asr_on_startup": True,
                    "asr_model_name": "local/whisper",
                    "asr_device": "cuda:1",
                }
            },
            (True, "local/whisper", "cuda:1"),
        ),
    ],
)
def test_asr_config_defaults_and_values(additional_config, expected):
    assert pipeline_omnivoice._parse_asr_config(additional_config) == expected


@pytest.mark.parametrize(
    "additional_config",
    [
        {"omnivoice_asr": {"load_asr_on_startup": "true"}},
        {"omnivoice_asr": {"asr_model_name": "  "}},
        {"omnivoice_asr": {"asr_device": ""}},
        {"omnivoice_asr": []},
    ],
)
def test_asr_config_rejects_invalid_values(additional_config):
    with pytest.raises((TypeError, ValueError)):
        pipeline_omnivoice._parse_asr_config(additional_config)


def test_asr_loader_uses_configured_model_and_device(pipeline, monkeypatch):
    asr, load_calls = _install_fake_asr(monkeypatch)
    pipeline._asr_model_name = "local/whisper"
    pipeline._asr_device = "cpu"

    pipeline._load_asr_pipeline()

    assert asr is pipeline._asr_pipeline
    assert asr.model.to_calls == [torch.device("cpu")]
    assert asr.device == torch.device("cpu")
    assert load_calls == [
        (
            ("automatic-speech-recognition",),
            {"model": "local/whisper", "dtype": torch.float32, "device": "cpu"},
        )
    ]


def test_eager_asr_is_loaded_during_initialization(monkeypatch):
    model = pipeline_omnivoice.OmniVoicePipeline.__new__(pipeline_omnivoice.OmniVoicePipeline)
    torch.nn.Module.__init__(model)
    load_calls = []

    def load_asr(self):
        load_calls.append((self._asr_model_name, self._asr_device))

    monkeypatch.setattr(pipeline_omnivoice.OmniVoicePipeline, "_load_asr_pipeline", load_asr)

    model._initialize_asr(
        {
            "omnivoice_asr": {
                "load_asr_on_startup": True,
                "asr_model_name": "local/whisper",
                "asr_device": "cpu",
            }
        }
    )

    assert model._load_asr_on_startup is True
    assert load_calls == [("local/whisper", "cpu")]


def test_lazy_asr_is_not_loaded_during_initialization(monkeypatch):
    model = pipeline_omnivoice.OmniVoicePipeline.__new__(pipeline_omnivoice.OmniVoicePipeline)
    torch.nn.Module.__init__(model)
    load_calls = []

    def load_asr(self):
        load_calls.append(True)

    monkeypatch.setattr(pipeline_omnivoice.OmniVoicePipeline, "_load_asr_pipeline", load_asr)

    model._initialize_asr({"omnivoice_asr": {"load_asr_on_startup": False}})

    assert model._load_asr_on_startup is False
    assert load_calls == []


def test_lazy_asr_uses_worker_device_on_first_request(pipeline, monkeypatch):
    asr, load_calls = _install_fake_asr(monkeypatch)
    pipeline._asr_device = None

    assert pipeline._resolve_ref_text((np.zeros(4, dtype=np.float32), 16000), None) == "hello there"
    assert asr is pipeline._asr_pipeline
    assert load_calls[0][1]["device"] == torch.device("cpu")


@pytest.mark.parametrize(
    "waveform",
    [
        np.arange(4, dtype=np.float32)[None, :],
        torch.arange(4, dtype=torch.float32)[None, :],
    ],
)
def test_asr_input_accepts_numpy_and_torch_waveforms_without_mutating(waveform, pipeline, monkeypatch):
    asr, _ = _install_fake_asr(monkeypatch)
    original = waveform.clone() if isinstance(waveform, torch.Tensor) else waveform.copy()

    pipeline._transcribe_ref_audio((waveform, 22050))

    asr_input = asr.inputs[0]
    assert asr_input["array"].shape == (4,)
    np.testing.assert_array_equal(asr_input["array"], np.arange(4, dtype=np.float32))
    asr_input["array"][0] = 100
    if isinstance(waveform, torch.Tensor):
        assert torch.equal(waveform, original)
    else:
        np.testing.assert_array_equal(waveform, original)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        (torch.device("cuda:0"), torch.float16),
        ("xpu:0", torch.float16),
    ],
)
def test_loader_receives_checkpoint_device_and_dtype(pipeline, monkeypatch, device, dtype):
    asr, load_calls = _install_fake_asr(monkeypatch)
    pipeline._asr_device = str(device)

    pipeline._load_asr_pipeline()

    assert load_calls[0][1]["model"] == "openai/whisper-large-v3-turbo"
    assert load_calls[0][1]["device"] == str(device)
    assert load_calls[0][1]["dtype"] is dtype
    assert asr.model.to_calls == [torch.device(device)]
    assert asr.device == torch.device(device)


def test_transcript_is_combined_before_target_text(pipeline, monkeypatch):
    _install_fake_asr(monkeypatch)
    ref_audio = (np.zeros(4, dtype=np.float32), 16000)

    ref_text = pipeline._resolve_ref_text(ref_audio, None)

    assert pipeline_omnivoice._combine_text("hello", ref_text) == "hello there hello"


def test_voice_clone_duration_uses_resolved_text_and_reference_token_count(pipeline):
    calls = []

    class DurationEstimator:
        def estimate_duration(self, text, ref_text, ref_duration):
            calls.append((text, ref_text, ref_duration))
            return 17.8

    pipeline.duration_estimator = DurationEstimator()
    ref_audio_tokens = torch.zeros(8, 73, dtype=torch.long)

    assert pipeline._estimate_target_len("target", "transcribed reference", ref_audio_tokens) == 17
    assert calls == [("target", "transcribed reference", 73)]


def test_text_only_duration_keeps_fallback_inputs(pipeline):
    calls = []

    class DurationEstimator:
        def estimate_duration(self, text, ref_text, ref_duration):
            calls.append((text, ref_text, ref_duration))
            return 4

    pipeline.duration_estimator = DurationEstimator()

    assert pipeline._estimate_target_len("target", None, None) == 4
    assert calls == [("target", "Nice to meet you.", 25)]
