# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P2-01 CPU oracles for normalized bytes, conditioning and silence frames."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_processor import Zonos2Processor
from vllm_omni.model_executor.models.zonos2.zonos2_textnorm import Zonos2TextNormalizer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def config():
    counts = {
        "lufs": 12,
        "estimated_snr": 12,
        "max_pause": 12,
        "estimated_bandlimit_hz": 8,
        "leading_silence_s": 8,
        "trailing_silence_s": 8,
    }
    return Zonos2Config(
        quality_features=list(counts), quality_buckets={k: list(map(str, range(n))) for k, n in counts.items()}
    )


@pytest.mark.parametrize("text", ["A", "你好", "café", "😀"])
def test_utf8_bytes_bos_eos_and_independent_padding(config, text):
    prompt = Zonos2Processor(config).build(text, text_normalization=False, quality_buckets={})
    expected = [2, *(v + 192 for v in text.encode("utf-8")), 3]
    assert prompt.frames[:-17, 9].tolist() == expected
    assert torch.all(prompt.frames[:-17, :9] == 1025)
    assert torch.all(prompt.frames[-17:, 9] == 519)
    assert prompt.frames.dtype == torch.int32
    assert prompt.frames.shape == (len(text.encode("utf-8")) + 19, 10)


def test_silence_matches_official_seventeen_rows_without_flushing_delay(config):
    silence = Zonos2Processor(config).silence_frames()
    assert silence.shape == (17, 10)
    assert silence[0].tolist() == [568, *([1025] * 8), 519]
    assert silence[-1].tolist() == [568, 804, 10, 674, 364, 981, 568, 378, 731, 519]
    assert silence[8, :9].tolist() == [568, 804, 10, 674, 364, 981, 568, 378, 90]
    assert silence[16, :9].tolist() == [568, 804, 10, 674, 364, 981, 568, 378, 731]
    for column in range(9):
        assert torch.all(silence[:column, column] == 1025)
        assert torch.all(silence[column:, column] < 1024)


def test_default_quality_and_explicit_bucket_order(config):
    processor = Zonos2Processor(config)
    assert processor.build("A", text_normalization=False).frames[:3, 9].tolist() == [511, 2, 257]
    result = processor.build(
        "A",
        text_normalization=False,
        speaking_rate_bucket=3,
        quality_buckets={"lufs": 1, "trailing_silence_s": 7},
    )
    assert result.frames[:4, 9].tolist() == [451, 457, 515, 2]


@pytest.mark.parametrize("clean,accurate,markers", [(False, True, [519, 517, 518]), (True, False, [519, 516])])
def test_speaker_prefix_and_owned_embedding(config, clean, accurate, markers):
    embedding = torch.arange(2048, dtype=torch.float64)
    result = Zonos2Processor(config).build(
        "A",
        text_normalization=False,
        quality_buckets={},
        speaker_embedding=embedding,
        clean_speaker_background=clean,
        accurate_mode=accurate,
    )
    assert result.frames[: len(markers), 9].tolist() == markers
    assert torch.all(result.frames[: len(markers), :9] == 1025)
    payload = result.to_engine_prompt()
    assert len(payload["prompt_token_ids"]) == len(result.frames)
    info = payload["additional_information"]
    assert info["zonos2_speaker_position"] == 0
    assert info["zonos2_speaker_embedding"].dtype == torch.float32
    embedding.zero_()
    assert info["zonos2_speaker_embedding"][100].item() == 100


@pytest.mark.parametrize(
    "embedding", [torch.zeros(2047), torch.zeros(2048, dtype=torch.int64), torch.full((2048,), float("nan"))]
)
def test_bad_speaker_embedding_is_rejected(config, embedding):
    with pytest.raises((ValueError, TypeError)):
        Zonos2Processor(config).build("A", text_normalization=False, speaker_embedding=embedding)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speaking_rate_bucket": 8},
        {"quality_buckets": {"bad": 0}},
        {"quality_buckets": {"lufs": 12}},
        {"quality_buckets": [0] * 7},
    ],
)
def test_conditioning_ranges_are_not_clamped(config, kwargs):
    with pytest.raises(ValueError):
        Zonos2Processor(config).build("A", text_normalization=False, **kwargs)


@pytest.mark.parametrize("text", ["", " ", "\n"])
def test_empty_text_is_rejected(config, text):
    with pytest.raises(ValueError, match="empty"):
        Zonos2Processor(config).build(text)


def test_normalization_runs_before_byte_tokenization(config, monkeypatch):
    calls = []

    def normalize(text, language):
        calls.append((text, language))
        return "twelve."

    normalizer = Zonos2TextNormalizer()
    monkeypatch.setattr(normalizer, "normalize", normalize)
    result = Zonos2Processor(config, normalizer=normalizer).build("12.", language="en_us", quality_buckets={})
    assert calls == [("12.", "en_us")]
    assert result.normalized_text == "twelve."
    assert result.frames[:-17, 9].tolist() == [2, *(v + 192 for v in b"twelve."), 3]


def test_normalizer_is_lazy_cached_and_preserves_punctuation(monkeypatch, tmp_path):
    normalizer = Zonos2TextNormalizer(str(tmp_path))
    built = []
    seen = []

    def build(lang):
        built.append(lang)

        def normalize(text, punct_post_process):
            seen.append((text, punct_post_process))
            return text.replace("12", "twelve")

        return SimpleNamespace(normalize=normalize)

    monkeypatch.setattr(normalizer, "_build", build)
    assert built == []
    assert normalizer.normalize("12.", "EN-US") == "twelve."
    assert normalizer.normalize("12!", "en_gb") == "twelve!"
    assert built == ["en"]
    assert seen == [("12 .", True), ("12 !", True)]
    with pytest.raises(ValueError, match="Unsupported"):
        normalizer.normalize("12", "unknown")


def test_normalization_failure_does_not_silently_feed_raw_digits(monkeypatch):
    normalizer = Zonos2TextNormalizer()

    def missing(_lang):
        raise ImportError("missing NeMo dependency")

    monkeypatch.setattr(normalizer, "_build", missing)
    with pytest.raises(ImportError, match="NeMo"):
        normalizer.normalize("12", "en_us")


def test_prompt_limit_is_checked_after_utf8_and_silence(config):
    config.max_seqlen = 22
    with pytest.raises(ValueError, match="max_seqlen"):
        Zonos2Processor(config).build("你好", text_normalization=False, quality_buckets={})
