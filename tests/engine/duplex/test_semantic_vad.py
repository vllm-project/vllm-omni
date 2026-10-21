# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``turn_detection: semantic_vad``: Silero pauses, Smart Turn decides the end of turn.

The detector runs here on a scripted Silero scorer (loud frames are speech)
and a scripted Smart Turn, so every endpoint rule is pinned without model
artifacts. The real Smart Turn graph is exercised at the end, and only when
its pinned artifact is already in the Hugging Face cache.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.smart_turn import (
    SMART_TURN_SHA256,
    SMART_TURN_WINDOW_SAMPLES,
    SemanticVADUnavailableError,
    SmartTurnBackendProvider,
    smart_turn_window,
)
from vllm_omni.engine.duplex.turn_detection import (
    SEMANTIC_VAD_CANDIDATE_SILENCE_MS,
    SEMANTIC_VAD_MAX_SILENCE_MS,
    PendingTurnDetectionUpdate,
    SemanticTurnDetector,
    ServerTurnDetector,
    TurnDetectionConfig,
    TurnDetectionResult,
    normalize_turn_detection_session_payload,
    validate_realtime_turn_detection,
)
from vllm_omni.protocol.duplex.errors import realtime_error_type

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RATE = 16_000
SPEECH = 0.2


def _loud_frames_are_speech(frame: np.ndarray) -> float:
    return 0.95 if float(np.abs(frame).max()) >= 0.1 else 0.02


class _ScriptedTurn:
    """Smart Turn stand-in: answers from a list and records what it was shown."""

    def __init__(self, *answers: float | Exception) -> None:
        self._answers = list(answers)
        self.windows: list[np.ndarray] = []

    def __call__(self, audio: np.ndarray) -> float:
        self.windows.append(np.asarray(audio).copy())
        if not self._answers:
            # pytest.fail is a BaseException: the detector's inference guard cannot swallow it.
            pytest.fail("Smart Turn was called more often than the test expects")
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _detector(turn: Callable[[np.ndarray], float], *, eagerness: str = "auto") -> SemanticTurnDetector:
    config = TurnDetectionConfig.from_realtime({"type": "semantic_vad", "eagerness": eagerness})
    return SemanticTurnDetector(config, turn_scorer=turn, frame_scorer=_loud_frames_are_speech)


def _b64(samples: np.ndarray) -> str:
    return base64.b64encode(np.asarray(samples, dtype="<f4").tobytes()).decode()


def _feed(
    detector: SemanticTurnDetector, plan: list[tuple[float, int]], *, chunk_ms: int = 100
) -> list[TurnDetectionResult]:
    """Stream ``(amplitude, duration_ms)`` segments in ``chunk_ms`` chunks."""
    audio = np.concatenate([np.full(ms * RATE // 1000, amp, dtype=np.float32) for amp, ms in plan])
    step = chunk_ms * RATE // 1000
    return [
        segment.result
        for i in range(0, audio.size, step)
        for segment in detector.process_segments(_b64(audio[i : i + step]), fmt="pcm_f32le", sample_rate_hz=RATE)
    ]


def _count(results: list[TurnDetectionResult], name: str) -> int:
    return sum(bool(getattr(result, name)) for result in results)


def _commit_ms(results: list[TurnDetectionResult], chunk_ms: int = 100) -> int:
    """End (ms of input) of the chunk that committed."""
    result = next(result for result in results if result.should_commit)
    assert result.audio_end_ms is not None
    return -(-result.audio_end_ms // chunk_ms) * chunk_ms


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("eagerness", ["low", "medium", "high", "auto"])
def test_each_eagerness_is_accepted_on_both_spellings(eagerness: str) -> None:
    turn_detection = {"type": "semantic_vad", "eagerness": eagerness}
    assert validate_realtime_turn_detection({"turn_detection": turn_detection}) is None
    assert validate_realtime_turn_detection({"audio": {"input": {"turn_detection": turn_detection}}}) is None


@pytest.mark.parametrize(
    ("turn_detection", "message"),
    [
        ({"type": "semantic_vad", "eagerness": "eager"}, "eagerness"),
        ({"type": "semantic_vad", "silence_duration_ms": 500}, "Unknown"),
        ({"type": "semantic_vad", "threshold": 0.5}, "Unknown"),
        ({"type": "semantic_vad", "interrupt_response": False}, "interrupt_response"),
        ({"type": "foo_vad"}, "'server_vad' or 'semantic_vad'"),
    ],
)
def test_invalid_semantic_vad_objects_are_refused(turn_detection: dict, message: str) -> None:
    error = validate_realtime_turn_detection({"turn_detection": turn_detection})
    assert error is not None and message in error


def test_semantic_vad_implies_barge_in_on_speech() -> None:
    payload = {"turn_detection": {"type": "semantic_vad"}, "overlap_policy": "listen_only"}
    assert "conflicts" in (validate_realtime_turn_detection(payload) or "")
    assert (
        validate_realtime_turn_detection(
            {"turn_detection": {"type": "semantic_vad"}, "overlap_policy": "barge_in_on_speech"}
        )
        is None
    )


def test_normalization_fills_the_openai_defaults() -> None:
    payload: dict[str, object] = {"turn_detection": {"type": "semantic_vad"}}
    configured, config = normalize_turn_detection_session_payload(payload)

    assert configured and config is not None
    assert payload["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "auto",
        "create_response": True,
        "interrupt_response": True,
    }
    assert payload["overlap_policy"] == "barge_in_on_speech"
    assert config.type == "semantic_vad" and config.eagerness == "auto"


def test_build_detector_picks_the_detector_by_type() -> None:
    semantic = TurnDetectionConfig.from_realtime({"type": "semantic_vad", "eagerness": "low"}).build_detector()
    server = TurnDetectionConfig.from_realtime({"type": "server_vad"}).build_detector()

    assert isinstance(semantic, SemanticTurnDetector)
    assert semantic.max_silence_ms == SEMANTIC_VAD_MAX_SILENCE_MS["low"] == 8000
    assert semantic.candidate_silence_ms == SEMANTIC_VAD_CANDIDATE_SILENCE_MS["low"] == 1500
    assert isinstance(server, ServerTurnDetector)


def test_the_semantic_vad_error_code_is_a_server_error() -> None:
    assert realtime_error_type("semantic_vad_unavailable") == "server_error"


def test_the_semantic_vad_model_path_is_validated() -> None:
    assert DuplexSessionRuntimeConfig(semantic_vad_model_path="/m/st.onnx").semantic_vad_model_path == "/m/st.onnx"
    with pytest.raises(ValueError, match="semantic_vad_model_path"):
        DuplexSessionRuntimeConfig(semantic_vad_model_path=" ")


# --------------------------------------------------------------------------- #
# Endpoint policy                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_ms", [20, 32, 100, 200])
def test_speech_resuming_before_the_silence_limit_keeps_the_turn_open(chunk_ms: int) -> None:
    # Speech resumes 64 ms before the deadline, inside Silero's 96 ms
    # confirmation interval. A pending onset must not be timed out as silence.
    results = _feed(_detector(_ScriptedTurn(0.1)), [(SPEECH, 288), (0.0, 3936), (SPEECH, 320)], chunk_ms=chunk_ms)
    assert _count(results, "speech_started") == 1
    assert _count(results, "should_commit") == 0


def test_a_short_noise_burst_does_not_extend_the_silence_limit() -> None:
    results = _feed(
        _detector(_ScriptedTurn(0.1)),
        [(SPEECH, 288), (0.0, 3936), (SPEECH, 64), (0.0, 128)],
        chunk_ms=32,
    )
    # The candidate fails on the next silent frame; commit immediately then.
    assert _count(results, "speech_started") == 1
    assert _count(results, "should_commit") == 1
    assert _commit_ms(results, chunk_ms=32) == 4320


@pytest.mark.parametrize("eagerness", ["high", "auto", "low"])
def test_hysteresis_delaying_a_candidate_does_not_extend_the_silence_limit(eagerness: str) -> None:
    turn = _ScriptedTurn(0.1)
    detector = SemanticTurnDetector(
        TurnDetectionConfig(type="semantic_vad", eagerness=eagerness),
        turn_scorer=turn,
        frame_scorer=lambda frame: float(frame[0]),
    )
    # One low frame starts silence. The hysteresis band preserves that timer
    # but cannot close the candidate, even after candidate_silence_ms elapses.
    band_ms = (detector.candidate_silence_ms // 32 + 10) * 32
    results = _feed(detector, [(0.9, 288), (0.1, 32), (0.45, band_ms), (0.1, 9000)], chunk_ms=32)

    assert len(turn.windows) == 1
    deadline_ms = 288 + -(-detector.max_silence_ms // 32) * 32
    assert [r.audio_end_ms for r in results if r.should_commit] == [deadline_ms]


@pytest.mark.parametrize("chunk_ms", [20, 100, 200, 12000])
def test_multiple_turns_and_scoring_windows_are_independent_of_wire_chunk_size(chunk_ms: int) -> None:
    turn = _ScriptedTurn(0.1, 0.1)
    detector = _detector(turn)
    results = _feed(detector, [(SPEECH, 1000), (0.0, 5000), (0.4, 1000), (0.0, 5000)], chunk_ms=chunk_ms)
    assert _count(results, "speech_started") == 2
    assert [result.audio_end_ms for result in results if result.should_commit] == [5024, 11008]
    assert len(turn.windows) == 2
    assert np.max(turn.windows[0]) == pytest.approx(SPEECH), "the first score must not see the next turn"
    assert np.max(turn.windows[1]) == pytest.approx(0.4)
    assert not np.any(turn.windows[1] == SPEECH), "the second score must not include the previous turn"


def test_a_complete_turn_commits_at_the_first_pause() -> None:
    turn = _ScriptedTurn(0.9)
    results = _feed(_detector(turn), [(SPEECH, 1000), (0.0, 1600)])

    assert _count(results, "speech_started") == 1
    assert _count(results, "speech_stopped") == 1
    assert _count(results, "should_commit") == 1
    # ``auto`` makes 1200 ms of silence a candidate: 38 Silero frames after the
    # frame-aligned end of speech (1024 ms), then the 100 ms chunk grid.
    assert _commit_ms(results) == 2300
    assert len(turn.windows) == 1
    committed = next(result for result in results if result.should_commit)
    assert committed.turn_complete_probability == pytest.approx(0.9)
    assert committed.backend == "silero+smart_turn"
    assert committed.as_payload()["turn_complete_probability"] == pytest.approx(0.9)


def test_default_waits_through_a_short_clause_pause_before_scoring() -> None:
    # Even a confidently complete model answer must not split a 900 ms
    # hesitation: the default only calls it after the 1200 ms candidate pause.
    turn = _ScriptedTurn(0.99)
    results = _feed(_detector(turn), [(SPEECH, 800), (0.0, 900), (SPEECH, 800), (0.0, 1600)])
    assert len(turn.windows) == 1
    assert _count(results, "speech_started") == _count(results, "should_commit") == 1
    assert 3700 <= _commit_ms(results) <= 3800


def test_a_thinking_pause_stays_inside_the_turn() -> None:
    turn = _ScriptedTurn(0.1, 0.8)
    detector = _detector(turn)
    results = _feed(detector, [(SPEECH, 1000), (0.0, 1600), (SPEECH, 800), (0.0, 1600)])

    assert _count(results, "speech_started") == 1, "resumed speech must not start a second turn"
    assert _count(results, "speech_stopped") == 1
    assert _count(results, "should_commit") == 1
    # The second pause starts at 3400 ms; 1200 ms on the frame grid, then the chunk grid.
    assert 4600 <= _commit_ms(results) <= 4700
    # Re-scored with the whole, longer turn the second time.
    assert len(turn.windows) == 2
    assert turn.windows[1].size > turn.windows[0].size
    # The pause chunks stay speech so a turn session buffers them.
    pause = results[10:26]
    assert all(result.is_speech and result.speech_active for result in pause)
    assert not any(result.speech_stopped for result in pause)


@pytest.mark.parametrize(("eagerness", "max_ms"), [("high", 2000), ("auto", 4000), ("low", 8000)])
def test_an_incomplete_pause_commits_at_the_eagerness_limit(eagerness: str, max_ms: int) -> None:
    turn = _ScriptedTurn(0.05)
    results = _feed(_detector(turn, eagerness=eagerness), [(SPEECH, 1000), (0.0, max_ms + 500)])

    assert _count(results, "should_commit") == 1
    # Silence starts on the 32 ms frame grid (1024 ms); the commit lands on the next chunk.
    assert 1000 + max_ms <= _commit_ms(results) <= 1000 + max_ms + 100
    assert len(turn.windows) == 1, "Smart Turn runs once per pause, not per chunk"


def test_silence_between_turns_is_not_speech() -> None:
    results = _feed(_detector(_ScriptedTurn()), [(0.0, 2000)])
    assert not any(result.is_speech or result.speech_active or result.speech_started for result in results)


@pytest.mark.parametrize("chunk_ms", [20, 100, 12000])
@pytest.mark.parametrize(
    ("answer", "decision", "reason"),
    [
        (0.9, "smart_turn_complete", "semantic_complete"),
        (0.1, "smart_turn_incomplete", "silence_timeout"),
        (RuntimeError("inference failed"), "smart_turn_error", "silence_timeout"),
    ],
)
def test_commit_diagnostics_distinguish_incomplete_from_failed_inference(chunk_ms, answer, decision, reason) -> None:
    detector = _detector(_ScriptedTurn(answer), eagerness="high")
    results = _feed(detector, [(SPEECH, 600), (0.0, 2600)], chunk_ms=chunk_ms)
    commits = [result for result in results if result.should_commit]
    assert len(commits) == 1
    assert commits[0].semantic_decision == decision
    assert commits[0].commit_reason == reason
    # Keep diagnostics at decisions/commits, not on every silent audio frame.
    diagnostics = [result for result in results if result.semantic_decision is not None]
    assert len(diagnostics) == (1 if reason == "semantic_complete" or chunk_ms == 12000 else 2)
    assert all(result.semantic_decision == decision for result in diagnostics)
    idle = _feed(detector, [(0.0, 500)])
    assert all(result.semantic_decision is None and result.commit_reason is None for result in idle)


@pytest.mark.parametrize("reset", [False, True])
def test_new_pause_does_not_inherit_previous_inference_error(reset: bool) -> None:
    detector = _detector(_ScriptedTurn(RuntimeError("inference failed"), 0.1), eagerness="high")
    first = _feed(detector, [(SPEECH, 600), (0.0, 600)])
    assert any(result.semantic_decision == "smart_turn_error" for result in first)
    if reset:
        detector.reset()
    resumed = _feed(detector, [(SPEECH, 600)])
    assert all(result.semantic_decision is None for result in resumed)
    final = _feed(detector, [(0.0, 2600)])
    commit = next(result for result in final if result.should_commit)
    assert commit.semantic_decision == "smart_turn_incomplete"
    assert commit.commit_reason == "silence_timeout"


def test_failed_score_does_not_report_an_earlier_pauses_probability_in_one_append() -> None:
    detector = _detector(_ScriptedTurn(0.1, RuntimeError("inference failed")), eagerness="high")
    results = _feed(detector, [(SPEECH, 600), (0.0, 600), (SPEECH, 600), (0.0, 2600)], chunk_ms=12000)
    commit = next(result for result in results if result.should_commit)
    assert commit.semantic_decision == "smart_turn_error"
    assert commit.commit_reason == "silence_timeout"
    assert commit.turn_complete_probability is None


def test_stopped_and_active_are_never_reported_together() -> None:
    turn = _ScriptedTurn(0.1, 0.1, 0.9)
    results = _feed(
        _detector(turn), [(SPEECH, 500), (0.0, 1600), (SPEECH, 500), (0.0, 1600), (SPEECH, 300), (0.0, 1600)]
    )
    assert not any(result.speech_stopped and result.speech_active for result in results)
    assert _count(results, "speech_started") == 1
    assert _count(results, "should_commit") == 1


@pytest.mark.parametrize("chunk_ms", [20, 100, 400])
def test_decisions_do_not_depend_on_chunking(chunk_ms: int) -> None:
    turn = _ScriptedTurn(0.1, 0.9)
    results = _feed(_detector(turn), [(SPEECH, 1200), (0.0, 1600), (SPEECH, 800), (0.0, 1600)], chunk_ms=chunk_ms)

    assert _count(results, "speech_started") == 1
    assert _count(results, "should_commit") == 1
    assert len(turn.windows) == 2
    # Commit lands within one frame and one chunk of 1200 ms after the second pause began.
    assert 3600 + 1200 <= _commit_ms(results, chunk_ms) <= 3600 + 1200 + 32 + chunk_ms


def test_the_detector_resets_itself_after_a_commit() -> None:
    turn = _ScriptedTurn(0.9, 0.9)
    detector = _detector(turn)
    first = _feed(detector, [(SPEECH, 600), (0.0, 1600)])
    second = _feed(detector, [(SPEECH, 600), (0.0, 1600)])

    assert _count(first, "should_commit") == _count(second, "should_commit") == 1
    assert _count(second, "speech_started") == 1
    started = next(result for result in second if result.speech_started)
    assert started.audio_start_ms is not None and started.audio_start_ms >= 2200 - 300 - 32


def test_reset_drops_a_pending_pause() -> None:
    turn = _ScriptedTurn(0.1)
    detector = _detector(turn, eagerness="high")
    _feed(detector, [(SPEECH, 600), (0.0, 1000)])
    detector.reset()
    results = _feed(detector, [(0.0, 3000)])
    assert not any(result.should_commit or result.speech_stopped for result in results)


def test_a_failed_inference_falls_back_to_the_silence_limit() -> None:
    turn = _ScriptedTurn(RuntimeError("boom"))
    results = _feed(_detector(turn, eagerness="high"), [(SPEECH, 600), (0.0, 2600)])

    assert _count(results, "should_commit") == 1
    assert 600 + 2000 <= _commit_ms(results) <= 600 + 2000 + 100
    assert all(result.turn_complete_probability is None for result in results)


def test_sessions_sharing_one_scorer_keep_their_own_state() -> None:
    shared = _ScriptedTurn(0.1, 0.9)
    a, b = _detector(shared), _detector(shared)
    paused = _feed(a, [(SPEECH, 600), (0.0, 1600)])
    complete = _feed(b, [(SPEECH, 600), (0.0, 1600)])

    assert _count(paused, "should_commit") == 0 and paused[-1].speech_active
    assert _count(complete, "should_commit") == 1


def test_smart_turn_sees_the_turn_with_its_prefix_padding_only() -> None:
    turn = _ScriptedTurn(0.9)
    _feed(_detector(turn), [(0.0, 3000), (SPEECH, 1000), (0.0, 1600)])
    window = turn.windows[0]
    # From the start of speech minus 300 ms of padding (2700 ms) to the chunk
    # that ended the 1200 ms pause, not the 3 s of silence before it.
    assert 2400 * RATE // 1000 <= window.size <= 2700 * RATE // 1000


def test_the_window_does_not_drift_into_earlier_turns() -> None:
    """Resets drop Silero's partial frame; the window must stay on Silero's clock anyway."""
    turns = 40
    turn = _ScriptedTurn(*([0.9] * turns))
    detector = _detector(turn)
    for _ in range(turns):
        # 100 ms chunks are not a multiple of Silero's 512-sample frame.
        assert _count(_feed(detector, [(SPEECH, 1000), (0.0, 1600)]), "should_commit") == 1
    # The first turn's prefix padding is clipped at the start of the stream;
    # every later turn sees the same window, however many resets came before.
    sizes = [window.size for window in turn.windows[1:]]
    assert max(sizes) - min(sizes) <= 512, sizes
    # Speech plus padding plus the candidate pause, never a previous turn.
    assert max(sizes) <= 2600 * RATE // 1000


def test_a_non_string_eagerness_is_refused() -> None:
    error = validate_realtime_turn_detection({"turn_detection": {"type": "semantic_vad", "eagerness": ["high"]}})
    assert error is not None and "eagerness" in error


def test_a_long_turn_is_cut_to_the_last_eight_seconds() -> None:
    turn = _ScriptedTurn(0.9)
    _feed(_detector(turn), [(SPEECH, 10_000), (0.0, 1600)])
    assert turn.windows[0].size == SMART_TURN_WINDOW_SAMPLES


def test_the_window_pads_at_the_front() -> None:
    audio = np.ones(RATE, dtype=np.float32)
    window = smart_turn_window(audio)
    assert window.shape == (SMART_TURN_WINDOW_SAMPLES,)
    assert not window[: SMART_TURN_WINDOW_SAMPLES - RATE].any()
    assert window[-RATE:].all()
    long = np.arange(SMART_TURN_WINDOW_SAMPLES + 10, dtype=np.float32)
    assert smart_turn_window(long)[0] == 10


def test_a_session_update_swaps_and_resets_the_old_detector() -> None:
    old = _detector(_ScriptedTurn(0.1))
    _feed(old, [(SPEECH, 600), (0.0, 1600)])
    payload: dict[str, object] = {"turn_detection": {"type": "server_vad"}}
    pending = PendingTurnDetectionUpdate.prepare(payload)

    assert pending is not None
    config, detector = pending.commit(old)
    assert isinstance(detector, ServerTurnDetector) and config is not None and config.type == "server_vad"
    assert not old.vad.speech_active
    assert _feed(old, [(0.0, 5000)])[-1].speech_active is False


# --------------------------------------------------------------------------- #
# Backend provider                                                            #
# --------------------------------------------------------------------------- #


def test_a_configured_model_path_that_does_not_exist_is_an_error(tmp_path) -> None:
    with pytest.raises(SemanticVADUnavailableError, match="does not exist"):
        SmartTurnBackendProvider(model_path=str(tmp_path / "missing.onnx")).get()


def test_a_configured_model_with_the_wrong_checksum_is_refused(tmp_path) -> None:
    path = tmp_path / "smart-turn.onnx"
    path.write_bytes(b"not the model")
    with pytest.raises(SemanticVADUnavailableError, match="checksum"):
        SmartTurnBackendProvider(model_path=str(path)).get()


def test_the_backend_is_loaded_once_and_shared(monkeypatch, tmp_path) -> None:
    import vllm_omni.engine.duplex.smart_turn as smart_turn

    path = tmp_path / "smart-turn.onnx"
    path.write_bytes(b"x")
    loads: list[str] = []
    gate = threading.Barrier(8)

    class _Backend:
        def __init__(self, model_path) -> None:
            loads.append(str(model_path))

    monkeypatch.setattr(smart_turn, "SmartTurnBackend", _Backend)
    monkeypatch.setattr(smart_turn, "SMART_TURN_SHA256", hashlib.sha256(b"x").hexdigest())
    provider = SmartTurnBackendProvider(model_path=str(path))

    def get():
        gate.wait()
        return provider.get()

    with ThreadPoolExecutor(8) as pool:
        backends = list(pool.map(lambda _: get(), range(8)))
    assert len(loads) == 1
    assert all(backend is backends[0] for backend in backends)


def test_a_missing_backend_fails_the_first_chunk_explicitly(tmp_path) -> None:
    config = TurnDetectionConfig.from_realtime({"type": "semantic_vad"})
    detector = SemanticTurnDetector(
        config,
        smart_turn_provider=SmartTurnBackendProvider(model_path=str(tmp_path / "missing.onnx")),
        frame_scorer=_loud_frames_are_speech,
    )
    with pytest.raises(SemanticVADUnavailableError):
        detector.process_segments(_b64(np.zeros(1600, dtype=np.float32)), fmt="pcm_f32le", sample_rate_hz=RATE)


def test_first_audio_chunk_loads_backend_before_any_candidate_pause(monkeypatch, tmp_path) -> None:
    import vllm_omni.engine.duplex.smart_turn as smart_turn

    path = tmp_path / "smart-turn.onnx"
    path.write_bytes(b"x")
    calls: list[str] = []

    class _Backend:
        def __init__(self, model_path) -> None:
            calls.append("load")

        def predict(self, audio: np.ndarray) -> float:
            calls.append("score")
            return 0.9

    monkeypatch.setattr(smart_turn, "SmartTurnBackend", _Backend)
    monkeypatch.setattr(smart_turn, "SMART_TURN_SHA256", hashlib.sha256(b"x").hexdigest())
    provider = SmartTurnBackendProvider(model_path=str(path))
    config = TurnDetectionConfig.from_realtime({"type": "semantic_vad", "eagerness": "high"})
    for _ in range(2):
        detector = SemanticTurnDetector(config, smart_turn_provider=provider, frame_scorer=_loud_frames_are_speech)
        before = calls.count("score")
        _feed(detector, [(0.0, 100)])
        assert calls.count("load") == 1
        assert calls.count("score") == before
        _feed(detector, [(SPEECH, 600), (0.0, 600)])
        assert calls.count("score") == before + 1


# --------------------------------------------------------------------------- #
# The real Smart Turn v3.2 graph (only when the pinned artifact is cached)     #
# --------------------------------------------------------------------------- #


def _cached_smart_turn():
    provider = SmartTurnBackendProvider()
    try:
        return provider.get()
    except SemanticVADUnavailableError as exc:
        pytest.skip(f"Smart Turn artifact not available: {exc}")


def test_the_pinned_artifact_holds_the_smart_turn_contract() -> None:
    backend = _cached_smart_turn()
    assert hashlib.sha256(backend.model_path.read_bytes()).hexdigest() == SMART_TURN_SHA256
    features = backend.features(np.zeros(RATE, dtype=np.float32))
    assert features.shape == (1, 80, 800) and features.dtype == np.float32
    probability = backend.predict(np.random.default_rng(0).standard_normal(RATE).astype(np.float32) * 0.1)
    assert 0.0 <= probability <= 1.0
