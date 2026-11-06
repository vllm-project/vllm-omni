# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cross-session turn detection (:class:`TurnDetectionBatcher`) against each session alone.

Every test runs twice. ``[fake]`` scores with a deterministic recurrent stand-in
for Silero whose probabilities depend on the stream's history, so a state routed
to the wrong session, or advanced once too often, changes the scores; it always
runs. ``[onnx]`` scores with the pinned Silero ONNX artifact (and, for
``semantic_vad``, the pinned Smart Turn artifact) from the local Hugging Face
cache, and is skipped when they are not cached -- the provider never downloads.

The unbatched reference is the pre-batching path: one session alone, the append
re-encoded to base64 and converted, then ``process_segments``, with each frame
scored by a single-frame model call that does not go through ``infer_batch``.
The batched path is production's: ``decode_audio_append`` decodes the wire append
once and hands the PCM to the batcher.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import random
import threading
import time
import wave
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.smart_turn import SemanticVADUnavailableError, SmartTurnBackendProvider
from vllm_omni.engine.duplex.turn_detection import (
    SemanticTurnDetector,
    ServerTurnDetector,
    TurnDetectionConfig,
    TurnDetectionSegment,
)
from vllm_omni.engine.duplex.vad import (
    FrameScoreRequest,
    ServerVADUnavailableError,
    SileroVADBackend,
    SileroVADBackendProvider,
    _SileroVADState,
)
from vllm_omni.engine.duplex.vad_batcher import TurnDetectionBatcher
from vllm_omni.protocol.realtime.audio import convert_input_audio_with_rate
from vllm_omni.protocol.realtime.audio_input import decode_audio_append
from vllm_omni.protocol.realtime.session import RealtimeInputDefaults

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RATE = 16_000
FRAME = 512
ASSETS = Path(__file__).parents[2] / "assets" / "semantic_vad"
SPEECH_FILES = ("complete.wav", "short_pause.wav", "thinking_pause.wav", "abandoned.wav")
#: Append sizes; none is a whole number of 512-sample frames, so frames straddle appends.
CHUNK_MS = (20, 40, 80, 100, 200, 330)
#: Probabilities and recurrent states may differ in the last bits between a batched and a single-row model run.
ATOL = 1e-6


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #


class _RecurrentFakeSilero:
    """A Silero stand-in: probability from the frame's energy and two leaky states carried across frames."""

    frame_samples = FRAME

    def new_state(self) -> np.ndarray:
        return np.zeros(2)

    def infer(self, frame: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        energy = min(1.0, 30.0 * float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))))
        fast, slow = state
        next_state = np.array([0.5 * fast + 0.5 * energy, 0.95 * slow + 0.05 * energy])
        return min(1.0, 0.8 * next_state[0] + 0.4 * next_state[1]), next_state

    def infer_batch(self, frames, states):
        scored = [self.infer(frame, state) for frame, state in zip(frames, states)]
        return [p for p, _ in scored], [s for _, s in scored]


class _PerFrameOnnx:
    """Silero's single-frame ONNX run as it was before batching: no ``infer_batch``, one row per call."""

    frame_samples = FRAME

    def __init__(self, backend: SileroVADBackend) -> None:
        self._backend = backend

    def new_state(self) -> _SileroVADState:
        return self._backend.new_state()

    def infer(self, frame: np.ndarray, state: _SileroVADState) -> tuple[float, _SileroVADState]:
        model_input = np.concatenate((state.context, np.asarray(frame, dtype=np.float32).reshape(1, -1)), axis=1)
        probability, model_state = self._backend._session.run(
            None, {"input": model_input, "state": state.model_state, "sr": np.asarray(RATE, dtype=np.int64)}
        )[:2]
        return float(np.asarray(probability).reshape(-1)[0]), _SileroVADState(
            model_state=np.ascontiguousarray(model_state, dtype=np.float32),
            context=np.ascontiguousarray(model_input[:, -64:]),
        )


class _Unbatched:
    """``backend`` without ``infer_batch``, so every frame is scored alone."""

    def __init__(self, backend) -> None:
        self.frame_samples = FRAME
        self.new_state = backend.new_state
        self.infer = backend.infer


class _Instrumented:
    """Wraps a batch-capable backend: counts model runs, records batch sizes, can slow or fail a run."""

    def __init__(self, backend, *, delay_s: float = 0.0, fail_on_run: int | None = None) -> None:
        self.frame_samples = FRAME
        self._backend = backend
        self._delay_s = delay_s
        self._fail_on_run = fail_on_run
        self.runs = 0
        self.batch_sizes: list[int] = []
        self.new_state = backend.new_state
        self.infer = backend.infer

    def infer_batch(self, frames, states):
        self.runs += 1
        self.batch_sizes.append(len(frames))
        if self._delay_s:
            time.sleep(self._delay_s)
        if self.runs == self._fail_on_run:
            raise RuntimeError("injected model failure")
        return self._backend.infer_batch(frames, states)


def _provider(backend) -> SimpleNamespace:
    return SimpleNamespace(get=lambda: backend)


def _fake_turn_scorer(window: np.ndarray) -> float:
    """Longer turns look complete, so both Smart Turn decisions occur."""
    return 0.9 if window.size > 2.5 * RATE else 0.2


@dataclasses.dataclass
class _Variant:
    name: str
    #: The shared, batch-capable backend production uses.
    backend: object
    #: A single-frame backend scoring like ``backend`` did before batching.
    unbatched: object
    #: Smart Turn for ``semantic_vad`` detectors: a provider (real) or a scorer (fake).
    smart_turn: SmartTurnBackendProvider | Callable[[np.ndarray], float]


_ONNX: dict[str, object] = {}


@pytest.fixture(params=["fake", "onnx"])
def variant(request) -> _Variant:
    if request.param == "fake":
        fake = _RecurrentFakeSilero()
        return _Variant("fake", fake, _Unbatched(fake), _fake_turn_scorer)
    if "silero" not in _ONNX:
        try:
            _ONNX["silero"] = SileroVADBackendProvider().get()
        except ServerVADUnavailableError as exc:
            _ONNX["silero"] = exc
        provider = SmartTurnBackendProvider()
        try:
            provider.get()
            _ONNX["smart_turn"] = provider
        except SemanticVADUnavailableError as exc:
            _ONNX["smart_turn"] = exc
    silero, smart_turn = _ONNX["silero"], _ONNX["smart_turn"]
    if not isinstance(silero, SileroVADBackend):
        pytest.skip(f"no locally cached Silero ONNX artifact at the pinned revision ({silero!r}); [fake] still runs")
    if not isinstance(smart_turn, SmartTurnBackendProvider):
        pytest.skip(f"no locally cached Smart Turn artifact at the pinned revision ({smart_turn!r}); [fake] still runs")
    return _Variant("onnx", silero, _PerFrameOnnx(silero), smart_turn)


# --------------------------------------------------------------------------- #
# Streams and drivers                                                         #
# --------------------------------------------------------------------------- #


def _speech(name: str) -> np.ndarray:
    with wave.open(str(ASSETS / name)) as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype="<i2")


@dataclasses.dataclass
class _Session:
    """One session's wire appends (base64 PCM16) and how it is detected."""

    appends: list[str]
    semantic: bool = False
    #: Append indices before which the detector is reset (as a barge-in or a buffer clear does).
    reset_before: frozenset[int] = frozenset()
    start_delay_s: float = 0.0


def _sessions(count: int, *, semantic_every: int = 3, seed: int = 0) -> list[_Session]:
    """Distinct histories: speech files in a different order and number, different pauses and append sizes."""
    rng = random.Random(seed)
    sessions = []
    for i in range(count):
        parts = [_silence(0.2 + 0.1 * (i % 4))]
        for k in range(1 + i % 3):
            parts += [_speech(SPEECH_FILES[(i + k) % len(SPEECH_FILES)]), _silence(rng.uniform(0.4, 2.0))]
        parts.append(_silence(1.5))
        pcm = np.concatenate(parts).tobytes()
        appends, offset = [], 0
        while offset < len(pcm):
            size = rng.choice(CHUNK_MS) * RATE // 1000 * 2
            appends.append(base64.b64encode(pcm[offset : offset + size]).decode("ascii"))
            offset += size
        sessions.append(
            _Session(
                appends,
                semantic=semantic_every > 0 and i % semantic_every == 0,
                reset_before=frozenset({len(appends) // 2}) if i % 4 == 1 else frozenset(),
                start_delay_s=0.02 * i,
            )
        )
    return sessions


@dataclasses.dataclass
class _Trace:
    """What one session's detector did: every frame's raw score, the state and segments after each append."""

    scores: list[float] = dataclasses.field(default_factory=list)
    states: list[object] = dataclasses.field(default_factory=list)
    segments: list[list[TurnDetectionSegment] | str] = dataclasses.field(default_factory=list)


def _traced(detector, trace: _Trace):
    """Record the raw score sent back for each frame the detector asks for, however it is driven."""
    segment_steps = detector.segment_steps

    def steps(audio, **kwargs):
        inner = segment_steps(audio, **kwargs)
        try:
            request = next(inner)
            while True:
                try:
                    value = yield request
                except Exception as exc:
                    request = inner.throw(exc)
                    continue
                if isinstance(request, FrameScoreRequest):
                    trace.scores.append(value)
                request = inner.send(value)
        except StopIteration as stop:
            return stop.value

    detector.segment_steps = steps
    return detector


def _detector(session: _Session, backend, variant: _Variant, trace: _Trace, *, turn_scorer=None):
    if not session.semantic:
        detector = ServerTurnDetector(TurnDetectionConfig(silence_duration_ms=300), backend_provider=_provider(backend))
    elif turn_scorer is not None or callable(variant.smart_turn):
        detector = SemanticTurnDetector(
            TurnDetectionConfig(type="semantic_vad"),
            backend_provider=_provider(backend),
            turn_scorer=turn_scorer or variant.smart_turn,
        )
    else:
        detector = SemanticTurnDetector(
            TurnDetectionConfig(type="semantic_vad"),
            backend_provider=_provider(backend),
            smart_turn_provider=variant.smart_turn,
        )
    return _traced(detector, trace)


def _state(detector) -> object:
    state = detector.vad._detector_state
    return state.copy() if isinstance(state, np.ndarray) else state


def _reference(session: _Session, variant: _Variant, **detector_kwargs) -> _Trace:
    """The session alone, through the pre-batching path, every frame scored by its own model call."""
    trace = _Trace()
    detector = _detector(session, variant.unbatched, variant, trace, **detector_kwargs)
    for k, append in enumerate(session.appends):
        if k in session.reset_before:
            detector.reset()
            trace.segments.append("reset")
        audio, fmt, rate = convert_input_audio_with_rate(append, "pcm16", sample_rate_hz=RATE)
        trace.segments.append(detector.process_segments(audio, fmt=fmt, sample_rate_hz=rate))
        trace.states.append(_state(detector))
    return trace


async def _run_batched(
    batcher: TurnDetectionBatcher, session: _Session, detector, trace: _Trace, *, jitter_seed: int = 0
) -> None:
    """The session through production's path: decode the wire append once, detect through the batcher."""
    rng = random.Random(jitter_seed)
    await asyncio.sleep(session.start_delay_s)
    for k, append in enumerate(session.appends):
        if k in session.reset_before:
            detector.reset()
            trace.segments.append("reset")
        decoded = decode_audio_append({"audio": append}, defaults=RealtimeInputDefaults())
        trace.segments.append(
            await batcher.process_segments(
                detector, decoded.audio, fmt=decoded.format, sample_rate_hz=decoded.sample_rate_hz
            )
        )
        trace.states.append(_state(detector))
        await asyncio.sleep(rng.random() * 0.004)


def _assert_close(got: object, want: object, where: str) -> None:
    """Equal, except that floats and arrays may differ by :data:`ATOL`."""
    if isinstance(want, float) and isinstance(got, float):
        assert got == pytest.approx(want, abs=ATOL, rel=0), where
    elif isinstance(want, np.ndarray):
        np.testing.assert_allclose(got, want, atol=ATOL, rtol=0, err_msg=where)
    elif isinstance(want, _SileroVADState):
        assert isinstance(got, _SileroVADState), where
        _assert_close(got.model_state, want.model_state, f"{where}.model_state")
        _assert_close(got.context, want.context, f"{where}.context")
    elif dataclasses.is_dataclass(want):
        assert type(got) is type(want), where
        for field in dataclasses.fields(want):
            _assert_close(getattr(got, field.name), getattr(want, field.name), f"{where}.{field.name}")
    elif isinstance(want, list):
        assert isinstance(got, list), where
        assert len(got) == len(want), f"{where}: {len(got)} vs {len(want)} items"
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_close(g, w, f"{where}[{i}]")
    else:
        assert got == want, where


def _assert_same_trace(got: _Trace, want: _Trace, where: str) -> None:
    _assert_close(got.scores, want.scores, f"{where} frame scores")
    _assert_close(got.states, want.states, f"{where} state after each append")
    _assert_close(got.segments, want.segments, f"{where} segments")


def _batcher() -> tuple[TurnDetectionBatcher, ThreadPoolExecutor]:
    executor = ThreadPoolExecutor(4)
    return TurnDetectionBatcher(executor), executor


# --------------------------------------------------------------------------- #
# Parity                                                                      #
# --------------------------------------------------------------------------- #


async def test_batched_turn_detection_matches_each_session_alone(variant: _Variant) -> None:
    """Concurrent server_vad and semantic_vad sessions with distinct histories, joining and leaving
    (so the batch size changes), appends that split frames, and resets mid-stream, score, keep state
    and cut segments exactly as each session alone through the pre-batching path."""
    sessions = _sessions(9)
    references = [_reference(session, variant) for session in sessions]

    backend = _Instrumented(variant.backend)
    batcher, executor = _batcher()
    traces = [_Trace() for _ in sessions]
    await asyncio.gather(
        *(
            _run_batched(batcher, session, _detector(session, backend, variant, traces[i]), traces[i], jitter_seed=i)
            for i, session in enumerate(sessions)
        )
    )
    batcher.close()
    executor.shutdown()

    for i, (trace, reference) in enumerate(zip(traces, references)):
        _assert_same_trace(trace, reference, f"session {i} ({'semantic' if sessions[i].semantic else 'server'}_vad)")
    # The comparison covered what it is meant to cover.
    assert len(set(backend.batch_sizes)) >= 3 and max(backend.batch_sizes) >= 4, backend.batch_sizes
    results = [s.result for r in references for append in r.segments if isinstance(append, list) for s in append]
    assert sum(r.speech_started for r in results) >= len(sessions)
    assert any(r.commit_reason == "semantic_complete" for r in results)
    assert any(r.should_commit and r.backend == "silero" for r in results)
    assert sum(len(reference.scores) for reference in references) == sum(backend.batch_sizes)


# --------------------------------------------------------------------------- #
# Head-of-line blocking                                                       #
# --------------------------------------------------------------------------- #


async def test_a_short_append_is_not_held_behind_a_long_one(variant: _Variant) -> None:
    """A one-frame append finishes within a couple of iterations, whether it arrives with a 60-frame
    append or while that append is running; the long one keeps going."""
    backend = _Instrumented(variant.backend, delay_s=0.005)
    speech = _speech("complete.wav").astype(np.float32) / 32768.0
    long_audio = speech[: 60 * FRAME].tobytes()
    short_audio = speech[:FRAME].tobytes()
    session = _Session([])
    batcher, executor = _batcher()

    def detector():
        return _detector(session, backend, variant, _Trace())

    async def append(audio) -> int:
        await batcher.process_segments(detector(), audio, fmt="pcm_f32le", sample_rate_hz=RATE)
        return backend.runs

    # Submitted together.
    long_task = asyncio.ensure_future(append(long_audio))
    short_done = await append(short_audio)
    long_done = await long_task
    assert short_done <= 3 < 60 <= long_done

    # Submitted while the long one runs.
    start = backend.runs
    long_task = asyncio.ensure_future(append(long_audio))
    while backend.runs < start + 10:
        await asyncio.sleep(0.001)
    submitted = backend.runs
    short_done = await append(short_audio)
    long_done = await long_task
    batcher.close()
    executor.shutdown()
    assert short_done - submitted <= 3
    assert long_done - start >= 60


async def test_a_slow_smart_turn_does_not_stall_server_vad_sessions(variant: _Variant) -> None:
    """While one semantic_vad session waits on Smart Turn, server_vad sessions keep detecting; the
    semantic session then commits with Smart Turn's answer."""
    entered, release = threading.Event(), threading.Event()

    def slow_turn_scorer(window: np.ndarray) -> float:
        entered.set()
        release.wait(10)
        return 0.9

    backend = _Instrumented(variant.backend)
    batcher, executor = _batcher()
    pcm = np.concatenate([_speech("complete.wav"), _silence(2.0)]).tobytes()
    semantic, semantic_trace = _Session([base64.b64encode(pcm).decode("ascii")], semantic=True), _Trace()
    semantic_detector = _detector(semantic, backend, variant, semantic_trace, turn_scorer=slow_turn_scorer)
    semantic_task = asyncio.ensure_future(_run_batched(batcher, semantic, semantic_detector, semantic_trace))
    while not entered.is_set():
        await asyncio.sleep(0.002)

    servers = _sessions(3, semantic_every=0, seed=1)
    references = [_reference(session, variant) for session in servers]
    traces = [_Trace() for _ in servers]
    await asyncio.wait_for(
        asyncio.gather(
            *(
                _run_batched(batcher, s, _detector(s, backend, variant, traces[i]), traces[i], jitter_seed=i)
                for i, s in enumerate(servers)
            )
        ),
        timeout=30,
    )
    assert not semantic_task.done()
    for i, (trace, reference) in enumerate(zip(traces, references)):
        _assert_same_trace(trace, reference, f"server_vad session {i}")

    release.set()
    await asyncio.wait_for(semantic_task, timeout=10)
    batcher.close()
    executor.shutdown()
    segments = semantic_trace.segments[0]
    assert isinstance(segments, list)
    committed = [s.result for s in segments if s.result.should_commit]
    assert [(r.commit_reason, r.turn_complete_probability) for r in committed] == [("semantic_complete", 0.9)]


# --------------------------------------------------------------------------- #
# Cancellation and failures                                                   #
# --------------------------------------------------------------------------- #


async def test_cancelling_a_session_leaves_the_others_and_a_replacement_unchanged(variant: _Variant) -> None:
    """A caller cancelled while its append is being scored gets CancelledError; other sessions and a
    replacement session replaying the same stream match the reference."""
    sessions = _sessions(5, seed=2)
    references = [_reference(session, variant) for session in sessions]
    backend = _Instrumented(variant.backend, delay_s=0.002)
    batcher, executor = _batcher()
    traces = [_Trace() for _ in sessions]
    others = asyncio.gather(
        *(
            _run_batched(batcher, s, _detector(s, backend, variant, traces[i]), traces[i], jitter_seed=i)
            for i, s in enumerate(sessions)
            if i
        )
    )

    # Session 0 stops in the middle of an append of at least 6 frames.
    cancelled = sessions[0]
    detector = _detector(cancelled, backend, variant, _Trace())
    at = next(k for k, a in enumerate(cancelled.appends) if k >= 3 and len(base64.b64decode(a)) >= 6 * FRAME * 2)
    for append in cancelled.appends[:at]:
        decoded = decode_audio_append({"audio": append}, defaults=RealtimeInputDefaults())
        await batcher.process_segments(detector, decoded.audio, fmt=decoded.format, sample_rate_hz=RATE)
    decoded = decode_audio_append({"audio": cancelled.appends[at]}, defaults=RealtimeInputDefaults())
    in_flight = asyncio.ensure_future(
        batcher.process_segments(detector, decoded.audio, fmt=decoded.format, sample_rate_hz=RATE)
    )
    submitted = backend.runs
    while backend.runs < submitted + 2:
        await asyncio.sleep(0.001)
    assert not in_flight.done()
    in_flight.cancel()
    with pytest.raises(asyncio.CancelledError):
        await in_flight

    await asyncio.wait_for(others, timeout=60)
    for i in range(1, len(sessions)):
        _assert_same_trace(traces[i], references[i], f"session {i}")

    replacement = _Trace()
    await _run_batched(batcher, sessions[0], _detector(sessions[0], backend, variant, replacement), replacement)
    batcher.close()
    executor.shutdown()
    _assert_same_trace(replacement, references[0], "replacement session")


async def test_a_failure_stays_with_its_session(variant: _Variant) -> None:
    """Undecodable audio, an unavailable Silero backend, a Smart Turn model that will not load and a
    Smart Turn inference error each affect only their own session."""

    def broken_turn_scorer(window: np.ndarray) -> float:
        raise RuntimeError("injected Smart Turn failure")

    def unavailable() -> object:
        raise ServerVADUnavailableError("injected: no Silero backend")

    def no_smart_turn() -> object:
        raise SemanticVADUnavailableError("injected: no Smart Turn artifact")

    backend = _Instrumented(variant.backend)
    batcher, executor = _batcher()
    good = _sessions(4, semantic_every=0, seed=3)
    references = [_reference(session, variant) for session in good]
    traces = [_Trace() for _ in good]
    speech = _speech("complete.wav").astype(np.float32) / 32768.0
    audio = np.concatenate([speech, np.zeros(RATE * 2, dtype=np.float32)]).tobytes()

    async def fails_with(detector, data: bytes) -> BaseException:
        try:
            await batcher.process_segments(detector, data, fmt="pcm_f32le", sample_rate_hz=RATE)
        except Exception as exc:
            return exc
        raise AssertionError("expected the append to fail")

    semantic = TurnDetectionConfig(type="semantic_vad", eagerness="high")
    bad_audio = ServerTurnDetector(TurnDetectionConfig(), backend_provider=_provider(backend))
    no_silero = ServerTurnDetector(TurnDetectionConfig(), backend_provider=SimpleNamespace(get=unavailable))
    no_turn_model = SemanticTurnDetector(
        semantic, backend_provider=_provider(backend), smart_turn_provider=SimpleNamespace(get=no_smart_turn)
    )
    turn_error = SemanticTurnDetector(semantic, backend_provider=_provider(backend), turn_scorer=broken_turn_scorer)
    turn_error_inline = SemanticTurnDetector(
        semantic, backend_provider=_provider(variant.unbatched), turn_scorer=broken_turn_scorer
    )

    *_, errors, turn_error_segments = await asyncio.wait_for(
        asyncio.gather(
            *(
                _run_batched(batcher, s, _detector(s, backend, variant, traces[i]), traces[i], jitter_seed=i)
                for i, s in enumerate(good)
            ),
            asyncio.gather(
                fails_with(bad_audio, b"\x00" * (4 * FRAME + 2)),
                fails_with(no_silero, audio),
                fails_with(no_turn_model, audio),
            ),
            batcher.process_segments(turn_error, audio, fmt="pcm_f32le", sample_rate_hz=RATE),
        ),
        timeout=60,
    )
    batcher.close()
    executor.shutdown()

    assert [type(exc) for exc in errors] == [ValueError, ServerVADUnavailableError, SemanticVADUnavailableError]
    # A Smart Turn inference error is not fatal: the pause counts as incomplete, as when scored in line.
    inline = turn_error_inline.process_segments(
        base64.b64encode(audio).decode("ascii"), fmt="pcm_f32le", sample_rate_hz=RATE
    )
    _assert_close(turn_error_segments, inline, "Smart Turn error segments")
    assert any(s.result.semantic_decision == "smart_turn_error" for s in inline)
    for i, (trace, reference) in enumerate(zip(traces, references)):
        _assert_same_trace(trace, reference, f"session {i}")


async def test_a_failed_model_run_does_not_stop_the_batcher(variant: _Variant) -> None:
    """A shared model run that raises fails the appends it was scoring -- none is left waiting -- and
    later appends are detected normally."""
    backend = _Instrumented(variant.backend, fail_on_run=3)
    batcher, executor = _batcher()
    speech = _speech("complete.wav").astype(np.float32) / 32768.0
    session = _Session([])

    outcomes = await asyncio.wait_for(
        asyncio.gather(
            *(
                batcher.process_segments(
                    _detector(session, backend, variant, _Trace()),
                    speech[: (10 + i) * FRAME].tobytes(),
                    fmt="pcm_f32le",
                    sample_rate_hz=RATE,
                )
                for i in range(4)
            ),
            return_exceptions=True,
        ),
        timeout=30,
    )
    assert [str(o) for o in outcomes] == ["injected model failure"] * 4

    after = _sessions(3, semantic_every=0, seed=4)
    references = [_reference(s, variant) for s in after]
    traces = [_Trace() for _ in after]
    await asyncio.wait_for(
        asyncio.gather(
            *(
                _run_batched(batcher, s, _detector(s, backend, variant, traces[i]), traces[i])
                for i, s in enumerate(after)
            )
        ),
        timeout=60,
    )
    batcher.close()
    executor.shutdown()
    for i, (trace, reference) in enumerate(zip(traces, references)):
        _assert_same_trace(trace, reference, f"session {i} after the failure")


async def test_a_cancelled_smart_turn_job_stays_with_its_session(variant: _Variant) -> None:
    """Shutting the executor down with ``cancel_futures=True`` (as the session manager does) cancels a
    queued Smart Turn job: only its session degrades to ``smart_turn_error``; the session whose job was
    already running and server_vad sessions are unaffected."""
    entered, release = threading.Event(), threading.Event()
    queued = threading.Event()
    cancelled_job: list[Future] = []

    class _Recording(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            parked = entered.is_set()  # read before submitting: this very job may set it
            future = super().submit(fn, *args, **kwargs)
            if parked and not cancelled_job:
                # the running session is parked in its scorer: this is the other session's job
                cancelled_job.append(future)
                queued.set()
            return future

    def slow_turn_scorer(window: np.ndarray) -> float:
        entered.set()
        release.wait(10)
        return 0.9

    backend = _Instrumented(variant.backend)
    executor = _Recording(1)  # one worker: the second Smart Turn job queues behind the first
    batcher = TurnDetectionBatcher(executor)
    pcm = np.concatenate([_speech("complete.wav"), _silence(2.0)]).tobytes()

    running, running_trace = _Session([base64.b64encode(pcm).decode("ascii")], semantic=True), _Trace()
    running_task = asyncio.ensure_future(
        _run_batched(
            batcher,
            running,
            _detector(running, backend, variant, running_trace, turn_scorer=slow_turn_scorer),
            running_trace,
        )
    )
    while not entered.is_set():
        await asyncio.sleep(0.002)
    cancelled, cancelled_trace = _Session([base64.b64encode(pcm).decode("ascii")], semantic=True), _Trace()
    cancelled_task = asyncio.ensure_future(
        _run_batched(
            batcher,
            cancelled,
            _detector(cancelled, backend, variant, cancelled_trace, turn_scorer=lambda w: 0.9),
            cancelled_trace,
        )
    )
    while not queued.is_set():
        await asyncio.sleep(0.002)

    servers = _sessions(3, semantic_every=0, seed=5)
    references = [_reference(session, variant) for session in servers]
    traces = [_Trace() for _ in servers]
    server_tasks = asyncio.gather(
        *(
            _run_batched(batcher, s, _detector(s, backend, variant, traces[i]), traces[i], jitter_seed=i)
            for i, s in enumerate(servers)
        )
    )
    executor.shutdown(wait=False, cancel_futures=True)
    assert cancelled_job[0].cancelled()
    release.set()
    await asyncio.wait_for(asyncio.gather(running_task, cancelled_task, server_tasks), timeout=30)
    batcher.close()

    for i, (trace, reference) in enumerate(zip(traces, references)):
        _assert_same_trace(trace, reference, f"server_vad session {i}")
    running_segments, cancelled_segments = running_trace.segments[0], cancelled_trace.segments[0]
    assert isinstance(running_segments, list) and isinstance(cancelled_segments, list)
    running_results = [s.result for s in running_segments if s.result.should_commit]
    assert [(r.commit_reason, r.turn_complete_probability) for r in running_results] == [("semantic_complete", 0.9)]
    assert any(s.result.semantic_decision == "smart_turn_error" for s in cancelled_segments)


async def test_close_finishes_submitted_appends_and_refuses_new_ones(variant: _Variant) -> None:
    backend = _Instrumented(variant.backend, delay_s=0.002)
    batcher, executor = _batcher()
    session = _Session([])
    speech = _speech("complete.wav").astype(np.float32) / 32768.0
    pending = asyncio.ensure_future(
        batcher.process_segments(
            _detector(session, backend, variant, _Trace()),
            speech[: 20 * FRAME].tobytes(),
            fmt="pcm_f32le",
            sample_rate_hz=RATE,
        )
    )
    while not backend.runs:
        await asyncio.sleep(0.001)
    batcher.close()
    segments = await asyncio.wait_for(pending, timeout=10)
    assert backend.runs == 20 and len(segments) >= 1
    with pytest.raises(RuntimeError, match="closed"):
        await batcher.process_segments(
            _detector(session, backend, variant, _Trace()),
            speech[:FRAME].tobytes(),
            fmt="pcm_f32le",
            sample_rate_hz=RATE,
        )
    executor.shutdown()
