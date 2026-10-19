# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The CLI's explicit model paths must reach endpoint-reference scoring too."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile

from vllm_omni.engine.duplex import smart_turn, vad

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_model_paths_produce_endpoint_report_without_cache_or_torch_fallback(tmp_path, monkeypatch) -> None:
    from vllm.transformers_utils import repo_utils

    silero_path = tmp_path / "silero.onnx"
    smart_path = tmp_path / "smart-turn.onnx"
    silero_path.write_bytes(b"silero")
    smart_path.write_bytes(b"smart")
    loads: list[Path] = []

    class _SileroBackend:
        def __init__(self, path: Path) -> None:
            loads.append(path)

        def new_state(self) -> object:
            return object()

        def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
            return (0.95 if np.max(np.abs(frame)) > 0.1 else 0.02), state

    class _SmartTurnBackend:
        def __init__(self, path: Path) -> None:
            loads.append(path)

        def features(self, audio: np.ndarray) -> np.ndarray:
            return audio

        def score(self, features: np.ndarray) -> float:
            return 0.9

        def predict(self, audio: np.ndarray) -> float:
            return self.score(self.features(audio))

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Explicit CLI paths must not consult the HF cache or optional silero-vad package")

    monkeypatch.setattr(repo_utils, "try_get_local_file", unexpected_fallback)
    monkeypatch.setattr(vad, "TorchSileroBackend", unexpected_fallback)
    monkeypatch.setattr(vad, "SileroVADBackend", _SileroBackend)
    monkeypatch.setattr(vad, "SILERO_VAD_SHA256", hashlib.sha256(b"silero").hexdigest())
    monkeypatch.setattr(smart_turn, "SmartTurnBackend", _SmartTurnBackend)
    monkeypatch.setattr(smart_turn, "SMART_TURN_SHA256", hashlib.sha256(b"smart").hexdigest())

    wav = tmp_path / "complete.wav"
    soundfile.write(wav, np.full(16000, 0.2, dtype=np.float32), 16000)
    report_path = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[3] / "benchmarks/duplex_vad/bench_vad_latency.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--silero-model",
            str(silero_path),
            "--smart-turn-model",
            str(smart_path),
            "--wav",
            str(wav),
            "--iterations",
            "1",
            "--json",
            str(report_path),
        ],
    )
    runpy.run_path(str(script), run_name="__main__")

    report = json.loads(report_path.read_text())
    assert loads == [silero_path, smart_path]
    endpoints = {row["mode"]: row for row in report["endpoints"]}
    assert set(endpoints) == {"server_vad", "semantic_vad"}
    assert all(row["false_cutoffs"] == 0 for row in endpoints.values())
    assert 500 <= endpoints["server_vad"]["endpoint_delay_ms"] <= 700
    assert 1200 <= endpoints["semantic_vad"]["endpoint_delay_ms"] <= 1400
    assert endpoints["semantic_vad"]["smart_turn_calls"] == 1


@pytest.mark.parametrize("kind", ["server_vad", "semantic_vad"])
@pytest.mark.parametrize("chunk_ms", [100, 127, 10_000])
def test_endpoint_report_counts_commit_before_continuation_in_same_append(monkeypatch, kind, chunk_ms):
    from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, ServerTurnDetector, TurnDetectionConfig

    class _SileroBackend:
        def new_state(self):
            return object()

        def infer(self, frame, state):
            return (0.95 if np.max(np.abs(frame)) > 0.1 else 0.02), state

    provider = vad.SileroVADBackendProvider()
    backend = _SileroBackend()
    monkeypatch.setattr(provider, "get", lambda: backend)
    config = TurnDetectionConfig.from_realtime({"type": kind})
    if kind == "server_vad":
        detector = ServerTurnDetector(config, backend_provider=provider)
    else:
        detector = SemanticTurnDetector(config, backend_provider=provider, turn_scorer=lambda _: 0.9)
    # Two 1 s speech intervals separated by 2 s silence. A 10 s append
    # contains both commits; the first must still count as an early cutoff.
    audio = np.concatenate([np.full(16000, 0.2), np.zeros(32000), np.full(16000, 0.2)]).astype(np.float32)
    script = Path(__file__).resolve().parents[3] / "benchmarks/duplex_vad/bench_vad_latency.py"
    bench = runpy.run_path(str(script))
    row = bench["_endpoint_row"](detector, audio, chunk_ms, provider)
    assert row["false_cutoffs"] == 1
    expected = 500 if kind == "server_vad" else 1200
    assert expected <= row["endpoint_delay_ms"] <= expected + 100
    assert row["smart_turn_calls"] == (2 if kind == "semantic_vad" else 0)
    assert row["append_endpoint_delay_ms"] >= row["endpoint_delay_ms"]
    if chunk_ms == 10_000:
        assert row["append_endpoint_delay_ms"] == 5500


@pytest.mark.parametrize("chunk_ms", [100, 10_000])
def test_chunk_cost_counts_failed_evaluations(monkeypatch, chunk_ms):
    from tests.engine.duplex.test_semantic_vad import _loud_frames_are_speech, _ScriptedTurn

    class _SileroBackend:
        def new_state(self):
            return object()

        def infer(self, frame, state):
            return _loud_frames_are_speech(frame), state

    # bench_chunk_cost replays each clip three times. Both pauses invoke the
    # scorer even when the final invocation produces no probability.
    scorer = _ScriptedTurn(*(value for _ in range(3) for value in (0.1, RuntimeError("scoring failed"))))

    class _SmartBackend:
        predict = staticmethod(scorer)

    silero_provider = vad.SileroVADBackendProvider()
    smart_provider = smart_turn.SmartTurnBackendProvider()
    monkeypatch.setattr(silero_provider, "get", _SileroBackend)
    monkeypatch.setattr(smart_provider, "get", _SmartBackend)
    audio = np.concatenate([np.full(16000, 0.2), np.zeros(32000), np.full(16000, 0.2)]).astype(np.float32)
    script = Path(__file__).resolve().parents[3] / "benchmarks/duplex_vad/bench_vad_latency.py"
    result = runpy.run_path(str(script))["bench_chunk_cost"](
        silero_provider, smart_provider, {"resume": audio}, chunk_ms
    )
    assert len(scorer.windows) == 6
    assert result["semantic_vad"]["smart_turn_chunk"]["n"] == (6 if chunk_ms == 100 else 3)
    assert "smart_turn_chunk" not in result["server_vad"]


@pytest.mark.parametrize("chunk_ms", [100, 127, 20_000])
@pytest.mark.parametrize("last_score", [0.9, RuntimeError("scoring failed")], ids=["complete", "error"])
def test_endpoint_report_counts_every_scorer_call(monkeypatch, chunk_ms, last_score):
    from tests.engine.duplex.test_semantic_vad import _loud_frames_are_speech, _ScriptedTurn
    from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, TurnDetectionConfig

    class _SileroBackend:
        def new_state(self):
            return object()

        def infer(self, frame, state):
            return _loud_frames_are_speech(frame), state

    provider = vad.SileroVADBackendProvider()
    monkeypatch.setattr(provider, "get", _SileroBackend)
    scorer = _ScriptedTurn(0.1, last_score)
    detector = SemanticTurnDetector(
        TurnDetectionConfig.from_realtime({"type": "semantic_vad"}),
        backend_provider=provider,
        turn_scorer=scorer,
    )
    audio = np.concatenate([np.full(16000, 0.2), np.zeros(32000), np.full(16000, 0.2)]).astype(np.float32)
    script = Path(__file__).resolve().parents[3] / "benchmarks/duplex_vad/bench_vad_latency.py"
    row = runpy.run_path(str(script))["_endpoint_row"](detector, audio, chunk_ms, provider)
    assert len(scorer.windows) == 2
    assert row["smart_turn_calls"] == 2
    assert row["false_cutoffs"] == 0
    expected = 4000 if isinstance(last_score, Exception) else 1200
    assert expected <= row["endpoint_delay_ms"] <= expected + 100
