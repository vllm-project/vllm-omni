# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from vllm_omni.diffusion.models.omnivoice import audio as audio_utils
from vllm_omni.diffusion.models.omnivoice import pipeline_omnivoice
from vllm_omni.diffusion.models.omnivoice.audio import (
    PreparedReferenceAudio,
    add_reference_punctuation,
    postprocess_generated_audio,
    prepare_reference_audio,
    remove_silence,
)
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SAMPLE_RATE = 24000
_HOP_LENGTH = 960


def _prepare(waveform, sample_rate=_SAMPLE_RATE):
    return prepare_reference_audio(
        waveform,
        sample_rate,
        target_sample_rate=_SAMPLE_RATE,
        hop_length=_HOP_LENGTH,
        trim_long=False,
    )


def test_reference_audio_normalization_and_hop_alignment():
    waveform = np.full(1010, 0.05, dtype=np.float32)

    prepared = _prepare(waveform)

    assert prepared.waveform.shape == (1, _HOP_LENGTH)
    assert prepared.original_rms == pytest.approx(0.05, abs=1e-6)
    assert np.sqrt(np.mean(prepared.waveform**2)) == pytest.approx(0.1, abs=1e-4)


def test_reference_audio_normalization_clips_peaky_quiet_waveform():
    waveform = np.zeros(7056, dtype=np.float32)
    waveform[waveform.size // 2] = 1.0

    prepared = _prepare(waveform)

    assert prepared.original_rms == pytest.approx(0.01190476, abs=1e-8)
    assert np.max(np.abs(prepared.waveform)) == 1.0


def test_reference_audio_at_or_above_target_rms_is_not_rescaled():
    prepared = _prepare(np.full(960, 0.2, dtype=np.float32))

    assert prepared.original_rms == pytest.approx(0.2, abs=1e-6)
    assert np.sqrt(np.mean(prepared.waveform**2)) == pytest.approx(0.2, abs=1e-4)


def test_reference_audio_resamples_and_downmixes():
    waveform = np.full((2, 800), 0.1, dtype=np.float32)

    prepared = _prepare(waveform, sample_rate=16000)

    assert prepared.waveform.shape == (1, 960)
    assert prepared.sample_rate == _SAMPLE_RATE


@pytest.mark.parametrize("waveform", [np.full(960, 0.1, dtype=np.float32), torch.full((960,), 0.1)])
def test_reference_audio_does_not_mutate_input(waveform):
    original = waveform.clone() if isinstance(waveform, torch.Tensor) else waveform.copy()

    _prepare(waveform)

    if isinstance(waveform, torch.Tensor):
        assert torch.equal(waveform, original)
    else:
        np.testing.assert_array_equal(waveform, original)


def test_reference_audio_rejects_empty_after_silence_removal():
    with pytest.raises(ValueError, match="empty after silence removal"):
        _prepare(np.zeros(960, dtype=np.float32))


def test_reference_audio_matches_official_asset_preparation():
    asset_path = Path(__file__).resolve().parents[4] / "tests/assets/qwen3_tts/clone_2.wav"
    waveform, sample_rate = sf.read(asset_path, always_2d=False)

    prepared = prepare_reference_audio(
        waveform,
        sample_rate,
        target_sample_rate=_SAMPLE_RATE,
        hop_length=_HOP_LENGTH,
        trim_long=True,
    )

    assert prepared.waveform.shape == (1, 179520)
    assert prepared.original_rms == pytest.approx(0.07468347, abs=1e-5)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", "hello."),
        ("你好", "你好。"),
        ("hello!", "hello!"),
        ("hello,", "hello,"),
        ("hello]", "hello]"),
        ("", ""),
    ],
)
def test_reference_punctuation(text, expected):
    assert add_reference_punctuation(text) == expected


def test_remove_silence_uses_reference_thresholds():
    waveform = np.concatenate(
        [
            np.zeros(2400, dtype=np.float32),
            np.full(12000, 0.2, dtype=np.float32),
            np.zeros(9600, dtype=np.float32),
            np.full(12000, 0.2, dtype=np.float32),
            np.zeros(7200, dtype=np.float32),
        ]
    )[np.newaxis, :]

    processed = remove_silence(
        waveform,
        _SAMPLE_RATE,
        middle_silence_ms=200,
        leading_silence_ms=100,
        trailing_silence_ms=200,
    )

    assert processed.shape[-1] < waveform.shape[-1]
    assert processed.shape[-1] > 24000


@pytest.mark.parametrize(
    ("gap_ms", "expected_removed_ms"),
    [(199, 0), (200, 0), (500, 100)],
)
def test_remove_silence_preserves_middle_gap_boundary(gap_ms, expected_removed_ms):
    active = np.full(_SAMPLE_RATE, 0.2, dtype=np.float32)
    gap = np.zeros(round(gap_ms * _SAMPLE_RATE / 1000), dtype=np.float32)
    waveform = np.concatenate([active, gap, active])[np.newaxis, :]

    processed = remove_silence(
        waveform,
        _SAMPLE_RATE,
        middle_silence_ms=200,
        leading_silence_ms=0,
        trailing_silence_ms=0,
    )

    removed_ms = round(1000 * (waveform.shape[-1] - processed.shape[-1]) / _SAMPLE_RATE)
    assert removed_ms == expected_removed_ms


@pytest.mark.parametrize(
    ("gap_level", "is_silent"),
    [
        (10 ** (-50 / 20) * 0.99, True),
        (10 ** (-50 / 20), True),
        (10 ** (-50 / 20) * 1.01, False),
    ],
)
def test_remove_silence_uses_reference_threshold(gap_level, is_silent):
    active = np.full(_SAMPLE_RATE, 0.2, dtype=np.float32)
    gap = np.full(round(250 * _SAMPLE_RATE / 1000), gap_level, dtype=np.float32)
    waveform = np.concatenate([active, gap, active])[np.newaxis, :]

    silent_ranges = audio_utils._detect_silence_ranges(
        waveform,
        _SAMPLE_RATE,
        min_silence_ms=200,
        silence_threshold_db=-50,
        seek_step_ms=10,
    )

    assert bool(silent_ranges) is is_silent


@pytest.mark.parametrize("duration_s", [19.9, 20.0, 20.1])
def test_trim_long_audio_only_runs_above_twenty_seconds(duration_s):
    waveform = np.full((1, round(duration_s * _SAMPLE_RATE)), 0.2, dtype=np.float32)

    trimmed = audio_utils.trim_long_audio(waveform, _SAMPLE_RATE)

    if duration_s <= 20.0:
        assert trimmed.shape == waveform.shape
    else:
        assert trimmed.shape[-1] == 15 * _SAMPLE_RATE


def test_generated_audio_scales_with_reference_rms_and_pads():
    audio = np.full((1, 10000), 0.2, dtype=np.float32)

    processed = postprocess_generated_audio(
        audio,
        sample_rate=_SAMPLE_RATE,
        reference_rms=0.05,
    )

    # NumPy keeps the source sample count and uses sample-based padding.
    assert processed.shape[-1] == 14800
    assert processed[0, 5000] == pytest.approx(0.1, abs=1e-5)
    assert processed[0, 0] == 0.0
    assert processed[0, -1] == 0.0


def test_generated_audio_keeps_reference_rms_at_or_above_target():
    processed = postprocess_generated_audio(
        np.full((1, 10000), 0.2, dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        reference_rms=0.2,
    )

    assert processed[0, 5000] == pytest.approx(0.2, abs=3e-5)


def test_generated_audio_uses_output_silence_threshold(monkeypatch):
    calls = []

    def fake_remove_silence(audio, sample_rate, **kwargs):
        calls.append((sample_rate, kwargs))
        return audio

    monkeypatch.setattr(audio_utils, "remove_silence", fake_remove_silence)
    audio_utils.postprocess_generated_audio(
        np.ones((1, 1000), dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        reference_rms=0.05,
    )

    assert calls == [
        (
            _SAMPLE_RATE,
            {
                "middle_silence_ms": 500,
                "leading_silence_ms": 100,
                "trailing_silence_ms": 100,
            },
        )
    ]


def test_generated_zero_audio_remains_finite():
    processed = postprocess_generated_audio(
        np.zeros((1, 1000), dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        reference_rms=0.05,
    )

    assert np.isfinite(processed).all()


class _FakeASR:
    def __init__(self, text: str):
        self.text = text
        self.inputs: list[dict[str, object]] = []

    def __call__(self, audio_input):
        self.inputs.append(audio_input)
        return {"text": self.text}


class _FakeGenerator:
    def __init__(self, num_codebooks: int):
        self.num_codebooks = num_codebooks

    def __call__(self, **kwargs):
        target_len = kwargs["target_lens"][0]
        return torch.zeros((1, self.num_codebooks, target_len), dtype=torch.long)


class _FakeDecoder:
    def __call__(self, tokens):
        return torch.full((1, 1, 10000), 0.2, dtype=torch.float32)


class _FakeDurationEstimator:
    def __init__(self):
        self.calls = []

    def estimate_duration(self, text, ref_text, ref_audio_tokens):
        self.calls.append((text, ref_text, ref_audio_tokens))
        return 4


def _build_fake_pipeline(monkeypatch, prepared_waveform, *, asr_text="reference transcript"):
    model = pipeline_omnivoice.OmniVoicePipeline.__new__(pipeline_omnivoice.OmniVoicePipeline)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.config = SimpleNamespace(num_audio_codebook=2, audio_mask_id=-1)
    model.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sample_rate=_SAMPLE_RATE, hop_length=_HOP_LENGTH))
    model.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[1, 2]))
    model.generator = _FakeGenerator(num_codebooks=2)
    model.decoder = _FakeDecoder()
    model.duration_estimator = _FakeDurationEstimator()
    model._asr_pipeline = _FakeASR(asr_text)
    model.num_step = 1
    model.guidance_scale = 1.0
    model.t_shift = 1.0
    model.layer_penalty_factor = 0.0
    model.position_temperature = 1.0
    model.class_temperature = 1.0
    model.sample_rate = _SAMPLE_RATE
    model._inline_reference_cache = OrderedDict()

    prepare_calls = []
    encoded_calls = []

    def fake_prepare(waveform, sample_rate, **kwargs):
        prepare_calls.append((waveform, sample_rate, kwargs))
        return PreparedReferenceAudio(prepared_waveform, _SAMPLE_RATE, 0.07)

    def fake_encode(self, audio_signal, sample_rate):
        encoded_calls.append((audio_signal.detach().clone(), sample_rate))
        return torch.zeros((2, 3), dtype=torch.long)

    monkeypatch.setattr(pipeline_omnivoice, "prepare_reference_audio", fake_prepare)
    monkeypatch.setattr(pipeline_omnivoice.OmniVoicePipeline, "_encode_ref_audio", fake_encode)
    return model, prepare_calls, encoded_calls


def _request(prompt):
    return SimpleNamespace(
        prompts=[prompt],
        sampling_params=SimpleNamespace(extra_args={}),
    )


def test_pipeline_uses_one_prepared_waveform_for_asr_and_tokenizer(monkeypatch):
    prepared = np.arange(_HOP_LENGTH, dtype=np.float32)[np.newaxis, :]
    model, prepare_calls, encoded_calls = _build_fake_pipeline(monkeypatch, prepared)

    result = model.forward(
        _request(
            {
                "prompt": "hello",
                "multi_modal_data": {"audio": (np.ones(1000, dtype=np.float32), 16000)},
            }
        )
    )

    assert result.error is None
    assert len(prepare_calls) == 1
    assert prepare_calls[0][1] == 16000
    assert prepare_calls[0][2]["trim_long"] is True
    assert prepare_calls[0][2]["target_sample_rate"] == _SAMPLE_RATE
    assert prepare_calls[0][2]["hop_length"] == _HOP_LENGTH
    assert len(model._asr_pipeline.inputs) == 1
    np.testing.assert_array_equal(model._asr_pipeline.inputs[0]["array"], prepared[0])
    assert model._asr_pipeline.inputs[0]["sampling_rate"] == _SAMPLE_RATE
    assert len(encoded_calls) == 1
    torch.testing.assert_close(encoded_calls[0][0], torch.from_numpy(prepared))
    assert encoded_calls[0][1] == _SAMPLE_RATE
    assert model.duration_estimator.calls == [("hello", "reference transcript.", 3)]


def test_pipeline_explicit_reference_text_prepares_audio_without_asr(monkeypatch):
    prepared = np.ones((1, _HOP_LENGTH), dtype=np.float32)
    model, prepare_calls, encoded_calls = _build_fake_pipeline(monkeypatch, prepared)

    result = model.forward(
        _request(
            {
                "prompt": "hello",
                "ref_audio": (np.ones(1000, dtype=np.float32), 16000),
                "ref_text": "caller supplied transcript",
            }
        )
    )

    assert result.error is None
    assert model._asr_pipeline.inputs == []
    assert prepare_calls[0][2]["trim_long"] is False
    assert len(encoded_calls) == 1
    assert model.duration_estimator.calls == [("hello", "caller supplied transcript.", 3)]


def test_pipeline_text_only_skips_reference_audio_processing(monkeypatch):
    model, prepare_calls, encoded_calls = _build_fake_pipeline(
        monkeypatch,
        np.ones((1, _HOP_LENGTH), dtype=np.float32),
    )

    result = model.forward(_request("hello"))

    assert result.error is None
    torch.testing.assert_close(result.output, torch.full((1, 1, 10000), 0.2))
    assert prepare_calls == []
    assert encoded_calls == []
    assert model._asr_pipeline.inputs == []
    assert model.duration_estimator.calls == [("hello", "Nice to meet you.", 25)]


def test_pipeline_text_only_bypasses_audio_postprocessing(monkeypatch):
    model, _, _ = _build_fake_pipeline(
        monkeypatch,
        np.ones((1, _HOP_LENGTH), dtype=np.float32),
    )

    def fail_postprocess(*args, **kwargs):
        raise AssertionError("text-only output must bypass voice-cloning postprocessing")

    monkeypatch.setattr(pipeline_omnivoice, "postprocess_generated_audio", fail_postprocess)

    result = model.forward(_request("hello"))

    assert result.error is None


def test_pipeline_reuses_inline_reference_cache(monkeypatch):
    prepared = np.arange(_HOP_LENGTH, dtype=np.float32)[np.newaxis, :]
    model, prepare_calls, encoded_calls = _build_fake_pipeline(monkeypatch, prepared)
    request = _request(
        {
            "prompt": "hello",
            "ref_audio": (np.ones(1000, dtype=np.float32), 16000),
        }
    )

    first = model.forward(request)
    second = model.forward(request)

    assert first.error is None
    assert second.error is None
    assert len(prepare_calls) == 2
    assert len(model._asr_pipeline.inputs) == 1
    assert len(encoded_calls) == 1


def test_pipeline_keeps_explicit_and_asr_cache_entries_separate(monkeypatch):
    prepared = np.arange(_HOP_LENGTH, dtype=np.float32)[np.newaxis, :]
    model, prepare_calls, encoded_calls = _build_fake_pipeline(monkeypatch, prepared)
    explicit_request = _request(
        {
            "prompt": "hello",
            "ref_audio": (np.ones(1000, dtype=np.float32), 16000),
            "ref_text": "caller supplied transcript",
        }
    )
    missing_request = _request(
        {
            "prompt": "hello",
            "ref_audio": (np.ones(1000, dtype=np.float32), 16000),
        }
    )

    model.forward(explicit_request)
    model.forward(missing_request)
    model.forward(missing_request)

    assert len(prepare_calls) == 3
    assert len(encoded_calls) == 2
    assert len(model._asr_pipeline.inputs) == 1
    assert {key[0] for key in model._inline_reference_cache} == {"explicit", "asr"}


def test_named_cache_separates_explicit_and_asr_preparation(monkeypatch):
    prepared = np.arange(_HOP_LENGTH, dtype=np.float32)[np.newaxis, :]
    model, prepare_calls, encoded_calls = _build_fake_pipeline(monkeypatch, prepared)
    model._speaker_cache = SpeakerEmbeddingCache(max_bytes=1024 * 1024)
    explicit_request = _request(
        {
            "prompt": "hello",
            "ref_audio": (np.ones(1000, dtype=np.float32), 16000),
            "ref_text": "caller supplied transcript",
            "voice_name": "alice",
            "voice_created_at": 1,
        }
    )
    missing_request = _request(
        {
            "prompt": "hello",
            "ref_audio": (np.ones(1000, dtype=np.float32), 16000),
            "voice_name": "alice",
            "voice_created_at": 1,
        }
    )

    model.forward(explicit_request)
    model.forward(missing_request)
    model.forward(missing_request)

    assert model._speaker_cache.stats()["entries"] == 2
    assert len(prepare_calls) == 2
    assert len(encoded_calls) == 2
    assert len(model._asr_pipeline.inputs) == 1


def test_inline_reference_cache_eviction_is_bounded(monkeypatch):
    model, _, _ = _build_fake_pipeline(
        monkeypatch,
        np.ones((1, _HOP_LENGTH), dtype=np.float32),
    )

    for index in range(pipeline_omnivoice._INLINE_CACHE_MAX_ENTRIES + 1):
        model._put_inline_cache(("asr", index), {"index": index})

    assert len(model._inline_reference_cache) == pipeline_omnivoice._INLINE_CACHE_MAX_ENTRIES
    assert ("asr", 0) not in model._inline_reference_cache
    assert ("asr", pipeline_omnivoice._INLINE_CACHE_MAX_ENTRIES) in model._inline_reference_cache


def test_inline_reference_cache_key_preserves_original_rms():
    waveform = np.ones((1, _HOP_LENGTH), dtype=np.float32)
    quiet = PreparedReferenceAudio(waveform, _SAMPLE_RATE, 0.05)
    loud = PreparedReferenceAudio(waveform, _SAMPLE_RATE, 0.08)

    assert pipeline_omnivoice.OmniVoicePipeline._inline_cache_key(
        quiet, "asr"
    ) != pipeline_omnivoice.OmniVoicePipeline._inline_cache_key(loud, "asr")
