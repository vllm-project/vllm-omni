# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/tests/test_codec_triton.py @ d3612b0.
"""Triton codec stream vs the PyTorch CodecStream on the released Haru-4B codec weights.

GPU tests need ``HARU_MODEL_PATH`` (a local ``higurashi-ai/Haru-4B`` snapshot), triton and CUDA.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.haru import codec_triton
from vllm_omni.model_executor.models.haru.configuration_haru import HaruSpec
from vllm_omni.model_executor.models.haru.vibevoice_codec import CodecStream, VibeVoiceCodec

pytestmark = pytest.mark.core_model

FRAME = 3200
MODEL_PATH = os.environ.get("HARU_MODEL_PATH", "")

gpu = pytest.mark.skipif(
    not (codec_triton.is_available() and MODEL_PATH and Path(MODEL_PATH).is_dir()),
    reason="needs triton, a CUDA device and HARU_MODEL_PATH",
)


@pytest.mark.cpu
def test_is_available_reports_missing_requirements(monkeypatch) -> None:
    assert isinstance(codec_triton.is_available(), bool)
    assert codec_triton.__all__ == ["TritonCodecStream", "is_available"]
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        assert codec_triton.is_available() is False
    with monkeypatch.context() as patch:
        patch.setattr(codec_triton, "find_spec", lambda name: None)
        assert codec_triton.is_available() is False
    with pytest.raises(AttributeError):
        codec_triton.missing  # noqa: B018


@pytest.fixture(scope="module")
def codec() -> VibeVoiceCodec:
    from safetensors import safe_open

    spec = HaruSpec.from_file(Path(MODEL_PATH) / "config.json")
    codec = VibeVoiceCodec(spec.codec)
    with safe_open(str(Path(MODEL_PATH) / "model.safetensors"), framework="pt") as weights:
        state = {
            name[len("codec.") :]: weights.get_tensor(name) for name in weights.keys() if name.startswith("codec.")
        }
    codec.load_state_dict(state, strict=True)
    return codec.to(device="cuda", dtype=torch.bfloat16).eval()


def _voice(frames: int, seed: int) -> torch.Tensor:
    """A deterministic speech-like signal: a gliding harmonic tone under a syllable-rate envelope, plus noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(frames * FRAME) / 24_000
    pitch = 120 + 60 * torch.sin(2 * torch.pi * 0.4 * t + seed)
    phase = 2 * torch.pi * torch.cumsum(pitch, 0) / 24_000
    tone = sum(torch.sin(k * phase) / k for k in range(1, 9))
    envelope = torch.sin(2 * torch.pi * 2.5 * t + seed).clamp(min=0) ** 2
    return 0.15 * envelope * tone + 0.01 * torch.randn(t.shape, generator=g)


def _snr_db(reference: torch.Tensor, signal: torch.Tensor) -> float:
    reference, signal = reference.float(), signal.float()
    return float(10 * torch.log10(reference.pow(2).sum() / (reference - signal).pow(2).sum()))


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@gpu
@pytest.mark.parametrize("batch", [1, 4])
def test_triton_stream_matches_the_torch_stream(codec, batch: int) -> None:
    """Rows start at different points of the signal; with several rows, one is reset mid-stream."""
    frames = 100
    audio = torch.stack([_voice(frames, seed) for seed in range(batch)])
    reference, stream = CodecStream(codec, batch), codec_triton.TritonCodecStream(codec, batch)
    collected: list[list[torch.Tensor]] = [[], [], [], []]
    for i in range(frames):
        if batch > 1 and i == 40:
            reference.reset([batch - 1])
            stream.reset([batch - 1])
        frame = audio[:, i * FRAME : (i + 1) * FRAME]
        means = reference.encode(frame)
        for out, result in zip(collected, (means, stream.encode(frame), reference.decode(means), stream.decode(means))):
            out.append(result)
    means, triton_means, wave, triton_wave = (torch.cat(out, dim=1) for out in collected)
    assert triton_means.shape == means.shape == (batch, frames, 64) and triton_means.dtype == means.dtype
    assert triton_wave.shape == wave.shape == (batch, frames * FRAME) and triton_wave.dtype == wave.dtype
    assert _snr_db(means, triton_means) > 30
    assert _snr_db(wave, triton_wave) > 30


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@gpu
def test_row_reset_equals_a_fresh_stream(codec) -> None:
    audio = torch.stack([_voice(12, seed) for seed in range(3)])
    stream, fresh = codec_triton.TritonCodecStream(codec, 3), codec_triton.TritonCodecStream(codec, 3)
    stream.decode(stream.encode(audio[:, : 6 * FRAME]))
    stream.reset([1])
    tail = audio[:, 6 * FRAME :]
    means, fresh_means = stream.encode(tail), fresh.encode(tail)
    assert torch.equal(means[1], fresh_means[1]) and not torch.equal(means[0], fresh_means[0])
    wave, fresh_wave = stream.decode(means), fresh.decode(means)
    assert torch.equal(wave[1], fresh_wave[1]) and not torch.equal(wave[0], fresh_wave[0])
    stream.reset()
    assert torch.equal(stream.encode(tail), fresh_means)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@gpu
def test_several_frames_per_call_equal_single_frames(codec) -> None:
    audio = _voice(5, 0)[None]
    whole, single = codec_triton.TritonCodecStream(codec), codec_triton.TritonCodecStream(codec)
    means = whole.encode(audio)
    frames = [single.encode(audio[:, i * FRAME : (i + 1) * FRAME]) for i in range(5)]
    assert torch.equal(means, torch.cat(frames, 1))
    assert torch.equal(whole.decode(means), torch.cat([single.decode(means[:, i : i + 1]) for i in range(5)], 1))
    with pytest.raises(ValueError):
        whole.encode(audio[:, : FRAME + 1])
    with pytest.raises(ValueError):
        whole.encode(torch.zeros(2, FRAME))


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@gpu
def test_usable_from_another_thread(codec) -> None:
    audio = _voice(2, 0)[None]
    stream = codec_triton.TritonCodecStream(codec)
    stream.warmup()
    expected = stream.encode(audio)
    stream.reset()
    results = []

    def work():
        results.append(stream.encode(audio))
        results.append(codec_triton.TritonCodecStream(codec).encode(audio))  # built in this thread

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert len(results) == 2 and all(torch.equal(result, expected) for result in results)
