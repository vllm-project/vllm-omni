# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``duplex_incremental_fbank``: the incremental Stage-0 streaming fbank is bitwise equal to the full one.

The reference is the remote-code ``StreamingMelProcessorExact._extract_full``
(``processing_minicpmo.py`` of openbmb/MiniCPM-o-4_5), imported from the
transformers dynamic-module cache; without it those tests skip.
"""

from __future__ import annotations

import copy
import importlib
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.duplex.incremental_fbank as incremental_fbank
import vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 as stage0_module
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.incremental_fbank import enable_incremental_fbank
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime, _FbankStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ASSETS = Path(__file__).resolve().parents[3] / "assets" / "minicpmo_4_5"
_SR = 16000
_HOP = 160


@pytest.fixture(scope="module")
def remote() -> ModuleType:
    """The remote ``processing_minicpmo`` module (the newest one in the transformers module cache)."""
    from transformers.utils import HF_MODULES_CACHE

    candidates = [
        path
        for path in Path(HF_MODULES_CACHE, "transformers_modules").glob("*/*/processing_minicpmo.py")
        if "class StreamingMelProcessorExact" in path.read_text()
    ]
    if not candidates:
        pytest.skip("MiniCPM-o 4.5 remote processing code is not in the transformers module cache")
    from transformers.dynamic_module_utils import init_hf_modules

    init_hf_modules()
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return importlib.import_module(".".join(path.relative_to(HF_MODULES_CACHE).with_suffix("").parts))


def _mel(remote: ModuleType, *, incremental: bool) -> object:
    # Stage 0's streaming configuration (_configure_streaming_processor).
    mel = remote.StreamingMelProcessorExact(
        feature_extractor=remote.MiniCPMAAudioProcessor(),
        chunk_ms=1000,
        first_chunk_ms=1035,
        cnn_redundancy_ms=20,
        enable_sliding_window=True,
        slide_trigger_seconds=30.0,
        slide_stride_seconds=10.0,
    )
    if incremental:
        assert enable_incremental_fbank(mel)
    return mel


def _remote_cls(mel: object) -> type:
    return next(cls for cls in type(mel).__mro__ if cls.__name__ == "StreamingMelProcessorExact")


def _original(mel: object) -> torch.Tensor:
    return _remote_cls(mel)._extract_full(mel)


def _speech() -> np.ndarray:
    parts = []
    for name in ("soft_interrupt_16k.wav", "response_required_16k.wav"):
        with wave.open(str(_ASSETS / name)) as wav:
            assert wav.getframerate() == _SR and wav.getsampwidth() == 2 and wav.getnchannels() == 1
            parts.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768)
        parts.append(np.zeros(_SR // 2, dtype=np.float32))
    speech = np.concatenate(parts)
    return np.concatenate([speech, speech, speech[: 10 * _SR]])  # ~72 s: two slides


def _noise(rng: np.random.Generator, seconds: float) -> np.ndarray:
    audio = rng.standard_normal(int(seconds * _SR)) * 0.05
    audio[_SR * 3 : _SR * 7] *= 20  # loud stretch: clipping-level peaks for the dynamic range
    audio[_SR * 12 : _SR * 14] = 0.0  # digital silence (clamped at 1e-10)
    return audio.astype(np.float32)


def _record(mel: object, extract) -> list[torch.Tensor]:
    # Capture every full mel the processor computes inside process().
    seen: list[torch.Tensor] = []

    def wrapper() -> torch.Tensor:
        seen.append(extract(mel))
        return seen[-1]

    mel._extract_full = wrapper
    return seen


def _run_streams(remote: ModuleType, audio: np.ndarray, chunk_sizes) -> dict[str, int]:
    ref, inc = _mel(remote, incremental=False), _mel(remote, incremental=True)
    ref_full = _record(ref, _remote_cls(ref)._extract_full)
    inc_full = _record(inc, type(inc)._extract_full)
    position, calls, slides, short = 0, 0, 0, 0
    while position < len(audio):
        size = ref.get_chunk_size() if chunk_sizes is None else next(chunk_sizes)
        chunk = audio[position : position + size]
        position += size
        dropped = ref.left_samples_dropped
        if len(ref.buffer) + len(chunk) < 400:
            with pytest.raises(ValueError):
                ref.process(chunk)
            with pytest.raises(ValueError):
                inc.process(chunk)
            continue
        expected, expected_info = ref.process(chunk)
        got, got_info = inc.process(chunk)
        assert torch.equal(inc_full[-1], ref_full[-1]), f"call {calls}: full mel differs"
        assert torch.equal(got, expected) and got_info == expected_info
        calls += 1
        slides += ref.left_samples_dropped != dropped
        short += len(ref.buffer) < 5 * _SR
    return {"calls": calls, "slides": slides, "short": short}


@pytest.mark.parametrize("kind", ["below_5s", "above_5s", "slide"])
def test_random_buffers_are_bitwise_equal(remote, kind: str) -> None:
    # 80 cases per kind: a cached buffer, then a longer one (slid by a random number of hops for
    # "slide"); lengths in samples, not hop aligned. "above_5s" also crosses the 5 s switch.
    rng = np.random.default_rng({"below_5s": 1, "above_5s": 2, "slide": 3}[kind])
    inc = _mel(remote, incremental=True)
    for _ in range(80):
        total = int(rng.integers(2_000, 5 * _SR) if kind == "below_5s" else rng.integers(5 * _SR, 640_000))
        audio = (rng.standard_normal(total) * rng.choice([1e-4, 0.01, 0.1, 0.9])).astype(np.float32)
        if rng.random() < 0.3:
            audio[: int(rng.integers(0, total))] = 0.0
        prev_end = int(rng.integers(400, min(total, 480_000) + 1))
        inc.reset()
        inc.buffer, inc.left_samples_dropped = audio[:prev_end], 0
        inc._extract_full()
        dropped = int(rng.integers(1, max(2, (prev_end - 400) // _HOP + 1))) * _HOP if kind == "slide" else 0
        dropped = min(dropped, (prev_end - 400) // _HOP * _HOP)
        start = max(prev_end, 5 * _SR) if kind == "above_5s" else prev_end
        end = int(rng.integers(min(start, total), min(total, dropped + 480_000) + 1))
        inc.buffer, inc.left_samples_dropped = audio[dropped:end], dropped
        got = inc._extract_full()
        expected = _original(inc)
        assert torch.equal(got, expected), (prev_end, dropped, end, (got - expected).abs().max().item())


@pytest.mark.parametrize("source", ["speech", "noise"])
@pytest.mark.parametrize("chunking", ["units", "random"])
def test_streams_are_bitwise_equal(remote, source: str, chunking: str) -> None:
    rng = np.random.default_rng(7)
    audio = _speech() if source == "speech" else _noise(rng, 66.0)
    sizes = None if chunking == "units" else iter(lambda: int(rng.integers(100, 24_000)), None)
    stats = _run_streams(remote, audio, sizes)
    assert stats["slides"] >= 2 and stats["short"] >= 1 and stats["calls"] >= 60, stats


def test_reset_restore_and_copies_stay_exact(remote) -> None:
    rng = np.random.default_rng(3)
    audio = _noise(rng, 60.0)
    ref, inc = _mel(remote, incremental=False), _mel(remote, incremental=True)

    def feed(start: int, stop: int, *pairs) -> None:
        for offset in range(start, stop, _SR):
            for a, b in pairs:
                expected, _ = a.process(audio[offset : offset + _SR])
                got, _ = b.process(audio[offset : offset + _SR])
                assert torch.equal(got, expected)
                assert torch.equal(_original(b), b._extract_full())

    feed(0, 8 * _SR, (ref, inc))
    snapshot_ref, snapshot_inc = ref.get_snapshot(), inc.get_snapshot()
    feed(8 * _SR, 14 * _SR, (ref, inc))
    ref.restore_snapshot(snapshot_ref)
    inc.restore_snapshot(snapshot_inc)
    assert inc._ifb_valid == 0
    feed(20 * _SR, 45 * _SR, (ref, inc))  # another continuation, past the slide
    ref_copy, inc_copy = copy.deepcopy(ref), copy.deepcopy(inc)
    assert type(inc_copy) is type(inc)
    feed(0, 3 * _SR, (ref, inc))
    feed(5 * _SR, 9 * _SR, (ref_copy, inc_copy))
    ref.reset()
    inc.reset()
    assert inc._ifb_valid == 0
    feed(10 * _SR, 18 * _SR, (ref, inc))


def test_unsupported_buffers_take_the_original_path(remote) -> None:
    inc = _mel(remote, incremental=True)
    inc.buffer = np.zeros(399, dtype=np.float32)
    with pytest.raises(ValueError):
        inc._extract_full()
    audio = (np.random.default_rng(5).standard_normal(31 * _SR) * 0.1).astype(np.float32)
    for buffer in (audio, audio[: 6 * _SR].astype(np.float64)):  # truncated at 30 s; float64 cast by Whisper
        inc.buffer = buffer
        assert torch.equal(inc._extract_full(), _original(inc))
        assert inc._ifb_valid == 0


def test_self_check_rejects_a_different_original(remote) -> None:
    class ShiftedMel(remote.StreamingMelProcessorExact):
        def _extract_full(self):
            return super()._extract_full() + 1e-6

    mel = ShiftedMel(feature_extractor=remote.MiniCPMAAudioProcessor(), chunk_ms=1000)
    assert enable_incremental_fbank(mel) is False
    assert type(mel) is ShiftedMel


def test_enable_rejects_other_processors() -> None:
    assert enable_incremental_fbank(SimpleNamespace(buffer=np.zeros(0, dtype=np.float32))) is False


def _session_processor(remote: ModuleType) -> object:
    """The remote MiniCPMOProcessor's streaming surface (its methods, without the tokenizer and image side)."""
    names = (
        "_init_streaming_processor",
        "set_streaming_mode",
        "reset_streaming",
        "get_streaming_chunk_size",
        "process_audio_streaming",
    )
    cls = type("StreamingProcessor", (), {name: getattr(remote.MiniCPMOProcessor, name) for name in names})
    processor = cls()
    processor.audio_processor = remote.MiniCPMAAudioProcessor()
    processor._streaming_mel_processor = None
    return processor


def _runtime(processor: object, *, incremental: bool, stats: _FbankStats | None = None):
    runtime = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.processor = processor
    runtime.stage_model = runtime.thinker = SimpleNamespace()
    runtime._incremental_fbank = incremental
    runtime._fbank_stats = stats
    return runtime


def test_stage0_sessions_stream_identical_features_with_fbank_stats(remote, monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(stage0_module, "logger", logger)
    audio = _speech()[: 40 * _SR]
    off = _runtime(_session_processor(remote), incremental=False)
    on = _runtime(_session_processor(remote), incremental=True, stats=_FbankStats(every=10, incremental=True))
    off_processor = off._configure_streaming_processor(SimpleNamespace(streaming_processor=None))
    on_processor = on._configure_streaming_processor(SimpleNamespace(streaming_processor=None))
    assert type(off_processor._streaming_mel_processor) is remote.StreamingMelProcessorExact
    assert isinstance(on_processor._streaming_mel_processor, remote.StreamingMelProcessorExact)
    assert type(on_processor._streaming_mel_processor) is not remote.StreamingMelProcessorExact
    position, index = 0, 0
    while position < len(audio):
        size = off._streaming_chunk_size(off_processor)
        chunk = audio[position : position + size]
        position += size
        expected = off._process_streaming_audio(chunk, index, processor=off_processor)
        got = on._process_streaming_audio(chunk, index, processor=on_processor)
        assert torch.equal(got["audio_features"], expected["audio_features"])
        index += 1
    assert on._fbank_stats.calls == index
    assert logger.info.call_count == index // 10
    assert "(incremental)" in logger.info.call_args.args[0]


@pytest.mark.parametrize("incremental", [False, True])
def test_configure_streaming_processor_enables_only_when_on(monkeypatch, incremental: bool) -> None:
    enable = MagicMock(return_value=False)
    logger = MagicMock()
    monkeypatch.setattr(incremental_fbank, "enable_incremental_fbank", enable)
    monkeypatch.setattr(stage0_module, "logger", logger)

    class Processor:
        _streaming_mel_processor = None

        def set_streaming_mode(self, **kwargs) -> None:
            self._streaming_mel_processor = SimpleNamespace(**kwargs)

        def reset_streaming(self) -> None:
            pass

    runtime = _runtime(Processor(), incremental=incremental)
    processor = runtime._configure_streaming_processor(SimpleNamespace(streaming_processor=None))
    if incremental:
        enable.assert_called_once_with(processor._streaming_mel_processor)
        logger.warning_once.assert_called_once()  # not supported: stays on the full recompute
    else:
        enable.assert_not_called()


@pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False), ("true", False), (None, False)])
def test_runtime_reads_the_switch_from_the_hf_config(value: object, expected: bool) -> None:
    stage_model = SimpleNamespace(
        config=SimpleNamespace(duplex_incremental_fbank=value, duplex_fbank_stats=True),
        processor=SimpleNamespace(),
    )
    runtime = MiniCPMO45Stage0DuplexRuntime(stage_model, device="cpu")
    assert runtime._incremental_fbank is expected
    assert ("(incremental)" in runtime._fbank_stats._message) is expected
