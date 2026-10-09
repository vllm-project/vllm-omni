# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for tests.helpers.speaker_similarity. They use fake embedders and need no model."""

import json

import numpy as np
import pytest

from tests.helpers.speaker_similarity import (
    LABEL_CORRECT,
    LABEL_DISAGREE,
    LABEL_WRONG_BOTH_AGREE,
    SAMPLE_RATE,
    FailureReport,
    classify,
    crop_window,
    min_margins,
    retain_failed_voice_isolation,
    score,
    wrong_both_agree,
)


class _ToneEmbedder:
    """Embeds a clip as one-hot of its dominant tone, so each 'voice' is a pure tone."""

    def __init__(self, name: str, freqs: list[float], remap: dict[int, int] | None = None) -> None:
        self.name = name
        self.freqs = freqs
        self.remap = remap or {}

    def embed(self, wav16: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(wav16))
        bins = np.fft.rfftfreq(len(wav16), 1 / SAMPLE_RATE)
        energy = [spec[np.argmin(np.abs(bins - f))] for f in self.freqs]
        k = int(np.argmax(energy))
        k = self.remap.get(k, k)
        out = np.zeros(len(self.freqs), dtype=np.float32)
        out[k] = 1.0
        return out


def _tone(freq: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


FREQS = [300.0, 600.0, 900.0, 1200.0]
REFS = {f"v{i}": _tone(f) for i, f in enumerate(FREQS)}


@pytest.mark.core_model
@pytest.mark.cpu
def test_classify_labels():
    assert classify("a", {"x": "a", "y": "a"}) == (LABEL_CORRECT, None)
    assert classify("a", {"x": "b", "y": "b"}) == (LABEL_WRONG_BOTH_AGREE, "b")
    assert classify("a", {"x": "b", "y": "c"}) == (LABEL_DISAGREE, None)
    assert classify("a", {"x": "a", "y": "b"}) == (LABEL_DISAGREE, None)


@pytest.mark.core_model
@pytest.mark.cpu
def test_crop_window_short_clip_is_kept_whole():
    wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
    clip, short = crop_window(wav, 2.0)
    assert short and len(clip) == len(wav)
    clip, short = crop_window(np.zeros(3 * SAMPLE_RATE, dtype=np.float32), 2.0)
    assert not short and len(clip) == 2 * SAMPLE_RATE
    clip, short = crop_window(wav, None)
    assert not short and len(clip) == len(wav)


@pytest.mark.core_model
@pytest.mark.cpu
def test_score_flags_swapped_output_only_when_embedders_agree():
    outputs = [("v0", REFS["v0"]), ("v1", REFS["v2"]), ("v2", REFS["v2"]), ("v3", REFS["v3"])]
    both = [_ToneEmbedder("a", FREQS), _ToneEmbedder("b", FREQS)]
    results = score(outputs, REFS, both)
    assert [r.label for r in results] == [LABEL_CORRECT, LABEL_WRONG_BOTH_AGREE, LABEL_CORRECT, LABEL_CORRECT]
    assert results[1].wrong_voice == "v2"
    assert [r.idx for r in wrong_both_agree(results)] == [1]
    assert results[1].margin["a"] == pytest.approx(-1.0)
    assert results[0].margin["a"] == pytest.approx(1.0)

    # One embedder that confuses voice 2 with voice 0 makes the same output a disagreement.
    fooled = [_ToneEmbedder("a", FREQS), _ToneEmbedder("b", FREQS, remap={2: 0})]
    results = score(outputs, REFS, fooled)
    assert results[1].label == LABEL_DISAGREE
    assert not wrong_both_agree(results)


@pytest.mark.core_model
@pytest.mark.cpu
def test_score_first_window_and_short_clip_flag():
    long_then_other = np.concatenate([_tone(300.0, 2.0), _tone(900.0, 4.0)])
    embedders = [_ToneEmbedder("a", FREQS)]
    full = score([("v0", long_then_other)], REFS, embedders, window=None)[0]
    first = score([("v0", long_then_other)], REFS, embedders, window=2.0)[0]
    assert full.argmax["a"] == "v2" and full.label == LABEL_WRONG_BOTH_AGREE
    assert first.argmax["a"] == "v0" and first.label == LABEL_CORRECT
    assert first.scored_s == pytest.approx(2.0) and not first.window_short
    short = score([("v0", _tone(300.0, 1.0))], REFS, embedders, window=2.0)[0]
    assert short.window_short and short.scored_s == pytest.approx(1.0)


@pytest.mark.core_model
@pytest.mark.cpu
def test_min_margins_per_embedder():
    outputs = [("v0", REFS["v0"]), ("v1", REFS["v2"])]
    results = score(outputs, REFS, [_ToneEmbedder("a", FREQS)])
    assert min_margins(results) == {"a": pytest.approx(-1.0)}


@pytest.mark.core_model
@pytest.mark.cpu
def test_retain_failed_voice_isolation_writes_wav_and_json(tmp_path, monkeypatch):
    outputs = [("v0", REFS["v0"]), ("v1", REFS["v2"])]
    emb = [_ToneEmbedder("a", FREQS), _ToneEmbedder("b", FREQS)]
    full = score(outputs, REFS, emb)
    first = score(outputs, REFS, emb, window=2.0)
    audio = [b"RIFF-a", b"RIFF-b"]
    report = FailureReport(
        test="unit",
        voices={"v0": "x.wav", "v1": "y.wav"},
        requests=[{"idx": 0}, {"idx": 1}],
        full=full,
        first=first,
        audio=audio,
        extra={"git_sha": "abc"},
    )
    written = retain_failed_voice_isolation(report, tmp_path)
    names = sorted(p.name for p in written)
    assert names == ["failed_voice_isolation_unit.json", "failed_voice_isolation_unit_1.wav"]
    assert (tmp_path / "failed_voice_isolation_unit_1.wav").read_bytes() == b"RIFF-b"
    data = json.loads((tmp_path / "failed_voice_isolation_unit.json").read_text())
    assert data["full"][1]["label"] == LABEL_WRONG_BOTH_AGREE and data["git_sha"] == "abc"
    monkeypatch.delenv("VLLM_OMNI_FAILED_SPEECH_AUDIO_DIR", raising=False)
    assert retain_failed_voice_isolation(report, None) == []
