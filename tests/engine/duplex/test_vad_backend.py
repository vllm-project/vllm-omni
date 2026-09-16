# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The engine-side Silero VAD: backend selection and endpoint rules.

The endpoint rules are the part worth pinning hardest. They were reimplemented
when turn detection moved into the engine, and drifted from upstream's
``ThresholdEndpointPolicy`` in two ways that only show up mid-utterance: a loud
frame used to cancel an in-progress silence timer, and ``audio_end_ms`` used to
exclude the trailing silence OpenAI says it should include. The parity cases
below hold upstream's own outputs, so neither can come back unnoticed.
"""

from __future__ import annotations

import base64
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.turn_detection import validate_realtime_turn_detection
from vllm_omni.engine.duplex.vad import (
    SILERO_VAD_SHA256,
    ServerVADUnavailableError,
    SileroStreamingVAD,
    SileroVADBackend,
    SileroVADBackendProvider,
    SileroVADConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAME = 512
SAMPLE_RATE_HZ = 16_000


def _drive(config: SileroVADConfig, probabilities: list[float]) -> list[tuple[str, int | None]]:
    """Feed one probability per frame and collect the endpoint events."""
    scores = iter(probabilities)
    vad = SileroStreamingVAD(config, frame_scorer=lambda _frame: next(scores))
    events: list[tuple[str, int | None]] = []
    for _ in probabilities:
        result = vad.process(np.zeros(FRAME, dtype=np.float32))
        if result.speech_started:
            events.append(("start", result.speech_start_ms))
        if result.speech_stopped:
            events.append(("stop", result.speech_end_ms))
    return events


# --------------------------------------------------------------------------- #
# Parity with the serving-side policy this was ported from                    #
# --------------------------------------------------------------------------- #

#: Probability sequences with the endpoint events upstream's
#: ``ThresholdEndpointPolicy`` produced for them, captured from
#: ``entrypoints/duplex/server_vad.py`` at `e2d2617f` before that module was
#: removed. Golden rather than computed because the oracle no longer ships: the
#: serving layer is transport only, so the VAD it used to carry is gone. Any
#: drift in the engine's rules still breaks these.
_PARITY_CASES = {
    "one utterance": (
        [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        {"threshold": 0.5, "prefix_padding_ms": 0, "silence_duration_ms": 100, "min_speech_duration_ms": 32},
        [("start", 32), ("stop", 256)],
    ),
    "prefix padding reaches back": (
        [0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0],
        {"threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 64, "min_speech_duration_ms": 32},
        [("start", 0), ("stop", 160)],
    ),
    "loud frames inside the silence timer": (
        [0.9, 0.9, 0.1, 0.45, 0.1, 0.1, 0.1],
        {"threshold": 0.5, "prefix_padding_ms": 0, "silence_duration_ms": 96, "min_speech_duration_ms": 32},
        [("start", 0), ("stop", 160)],
    ),
    "min speech duration rejects a blip": (
        [0.9, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0],
        {"threshold": 0.5, "prefix_padding_ms": 0, "silence_duration_ms": 64, "min_speech_duration_ms": 96},
        [("start", 64), ("stop", 224)],
    ),
    "negative threshold floor at a low threshold": (
        [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
        {"threshold": 0.16, "prefix_padding_ms": 0, "silence_duration_ms": 64, "min_speech_duration_ms": 32},
        [("start", 0)],
    ),
}


@pytest.mark.parametrize(("probabilities", "options", "expected"), _PARITY_CASES.values(), ids=list(_PARITY_CASES))
def test_endpoint_decisions_match_the_serving_side_policy(
    probabilities: list[float],
    options: dict[str, object],
    expected: list[tuple[str, int | None]],
) -> None:
    """The engine detector reproduces the policy server VAD shipped with upstream."""
    assert _drive(SileroVADConfig(**options), probabilities) == expected


# --------------------------------------------------------------------------- #
# The two rules that had drifted                                              #
# --------------------------------------------------------------------------- #


def test_a_loud_frame_does_not_restart_the_silence_timer() -> None:
    """Silero v6.2 hysteresis: once silence is running, a loud frame only delays it.

    The frame at 0.45 sits above the negative threshold (0.35) but below the
    activation threshold. It must not cancel the pending endpoint, or a speaker
    who trails off unevenly never gets their turn committed.
    """
    config = SileroVADConfig(threshold=0.5, prefix_padding_ms=0, silence_duration_ms=96, min_speech_duration_ms=32)
    assert _drive(config, [0.9, 0.9, 0.1, 0.45, 0.1, 0.1]) == [("start", 0), ("stop", 160)]


def test_audio_end_ms_includes_the_trailing_silence() -> None:
    """OpenAI defines audio_end_ms as the end of the audio sent to the model.

    That includes the silence spent deciding the turn was over, so it is the
    frame boundary the detector stopped on, not where speech last was.
    """
    config = SileroVADConfig(threshold=0.5, prefix_padding_ms=0, silence_duration_ms=100, min_speech_duration_ms=32)
    events = _drive(config, [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stop_ms = next(ms for kind, ms in events if kind == "stop")
    # 8 frames consumed when the timer expires: 8 * 512 / 16 kHz = 256 ms.
    assert stop_ms == 256


def test_reset_keeps_the_session_clock_and_clamps_the_prefix() -> None:
    """A barge-in resets the detector but not the session's timeline.

    Timestamps have to stay comparable with the rest of the session, and prefix
    padding must not reach back into audio the reset discarded.
    """
    config = SileroVADConfig(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=64, min_speech_duration_ms=32)
    scores = iter([0.0] * 10 + [0.9, 0.9, 0.9])
    vad = SileroStreamingVAD(config, frame_scorer=lambda _frame: next(scores))
    for _ in range(10):
        vad.process(np.zeros(FRAME, dtype=np.float32))

    stream_start_ms = round(10 * FRAME * 1000 / SAMPLE_RATE_HZ)
    vad.reset()
    result = vad.process(np.zeros(FRAME * 3, dtype=np.float32))

    assert result.speech_started
    # Without the clock the start would land at 0; without the clamp, 300 ms of
    # prefix would reach back before the reset.
    assert result.speech_start_ms == stream_start_ms


# --------------------------------------------------------------------------- #
# Input handling                                                              #
# --------------------------------------------------------------------------- #


def _pcm16_b64(samples: np.ndarray) -> str:
    return base64.b64encode((samples * 32767).astype("<i2").tobytes()).decode("ascii")


def test_pcm16_input_is_accepted_alongside_float32() -> None:
    config = SileroVADConfig(threshold=0.5)
    tone = np.zeros(FRAME * 2, dtype=np.float32)
    seen: list[int] = []
    vad = SileroStreamingVAD(config, frame_scorer=lambda frame: seen.append(frame.size) or 0.0)
    vad.process_base64(_pcm16_b64(tone), fmt="pcm16", sample_rate_hz=SAMPLE_RATE_HZ)
    assert seen == [FRAME, FRAME]


def test_an_odd_pcm16_byte_count_is_rejected() -> None:
    vad = SileroStreamingVAD(SileroVADConfig(), frame_scorer=lambda _frame: 0.0)
    with pytest.raises(ValueError, match="incomplete pcm16"):
        vad.process_base64(base64.b64encode(b"\x00\x01\x02").decode(), fmt="pcm16", sample_rate_hz=SAMPLE_RATE_HZ)


def test_resampling_keeps_the_frame_grid_from_drifting() -> None:
    """The 24 kHz remainder carries across chunks, so frame boundaries cannot drift.

    This is a weaker claim than upstream's ``StreamingAudioResampler``: the
    interpolation is per chunk, so values near a chunk edge differ slightly.
    What must hold is the frame *count* and alignment, which is what the
    endpoint timestamps are derived from.
    """
    source = np.sin(np.linspace(0, 40 * np.pi, 24_000, dtype=np.float32))

    def frames_for(chunk_sizes: list[int]) -> int:
        frames = 0

        def count(_frame: np.ndarray) -> float:
            nonlocal frames
            frames += 1
            return 0.0

        vad = SileroStreamingVAD(SileroVADConfig(), frame_scorer=count)
        offset = 0
        for size in chunk_sizes:
            vad.process_base64(_pcm16_b64(source[offset : offset + size]), fmt="pcm16", sample_rate_hz=24_000)
            offset += size
        assert offset == source.size
        return frames

    whole = frames_for([source.size])
    split = frames_for([7, 13, 511, 512, 513, source.size - 1_556])
    assert whole == split == 16_000 // FRAME


def test_a_sample_rate_change_mid_stream_is_rejected() -> None:
    vad = SileroStreamingVAD(SileroVADConfig(), frame_scorer=lambda _frame: 0.0)
    chunk = _pcm16_b64(np.zeros(1_024, dtype=np.float32))
    vad.process_base64(chunk, fmt="pcm16", sample_rate_hz=24_000)
    with pytest.raises(ValueError, match="sample rate cannot change"):
        vad.process_base64(chunk, fmt="pcm16", sample_rate_hz=48_000)


# --------------------------------------------------------------------------- #
# Backend selection                                                           #
# --------------------------------------------------------------------------- #


def test_a_configured_model_path_that_does_not_exist_is_an_error() -> None:
    """An operator who names an artifact gets told it is missing, not a fallback."""
    provider = SileroVADBackendProvider(model_path="/nonexistent/silero_vad.onnx")
    with pytest.raises(ServerVADUnavailableError, match="does not exist"):
        provider.get()


def test_a_configured_model_with_the_wrong_checksum_is_refused(tmp_path) -> None:
    """A different Silero revision changes endpointing, so the pin is enforced."""
    impostor = tmp_path / "silero_vad.onnx"
    impostor.write_bytes(b"not the pinned model")
    provider = SileroVADBackendProvider(model_path=str(impostor))
    with pytest.raises(ServerVADUnavailableError, match="checksum mismatch"):
        provider.get()
    assert hashlib.sha256(impostor.read_bytes()).hexdigest() != SILERO_VAD_SHA256


def test_the_onnx_backend_is_shared_by_every_session() -> None:
    """One ONNX session for the whole engine: its state is passed in, not held."""
    provider = SileroVADBackendProvider()
    try:
        backend = provider.get()
    except ServerVADUnavailableError as exc:
        pytest.skip(f"no Silero backend available here: {exc}")
    if not isinstance(backend, SileroVADBackend):
        pytest.skip("no local ONNX artifact; the torch fallback is per-session by design")

    assert provider.get() is backend
    # State is the caller's, so two sessions on one backend cannot interfere.
    first, second = backend.new_state(), backend.new_state()
    assert first is not second
    probability, next_state = backend.infer(np.zeros(FRAME, dtype=np.float32), first)
    assert 0.0 <= probability <= 1.0
    assert next_state is not first


def test_the_onnx_backend_holds_the_upstream_silero_v62_contract(monkeypatch, tmp_path) -> None:
    """Pin the ONNX call shape without needing the real artifact or runtime.

    The context carry is the part worth pinning: Silero v6.2 is fed 64 samples
    of the previous frame ahead of the current 512, and that tail travels in the
    caller's state rather than inside the session -- which is exactly what lets
    one session serve every duplex session at once.
    """
    sessions: list[FakeInferenceSession] = []

    class FakeSessionOptions:
        inter_op_num_threads = 0
        intra_op_num_threads = 0

    class FakeInferenceSession:
        def __init__(self, path: str, *, providers: list[str], sess_options: object) -> None:
            self.providers = providers
            self.sess_options = sess_options
            self.calls: list[dict[str, np.ndarray]] = []
            self.run_barrier: threading.Barrier | None = None
            sessions.append(self)

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=name) for name in ("input", "state", "sr")]

        def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.calls.append({name: np.array(value, copy=True) for name, value in inputs.items()})
            if self.run_barrier is not None:
                self.run_barrier.wait(timeout=5)
            return [
                np.asarray([[0.75]], dtype=np.float32),
                np.full((2, 1, 128), len(self.calls), dtype=np.float32),
            ]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(SessionOptions=FakeSessionOptions, InferenceSession=FakeInferenceSession),
    )

    model_path = tmp_path / "silero_vad.onnx"
    model_path.write_bytes(b"fake-model")
    backend = SileroVADBackend(model_path)
    session = sessions[0]

    # Single-threaded on purpose: many sessions share this one ORT session, so
    # it must not fan out per call.
    assert session.providers == ["CPUExecutionProvider"]
    assert session.sess_options.inter_op_num_threads == 1
    assert session.sess_options.intra_op_num_threads == 1
    # Warm-up already ran: 64 context + 512 frame.
    assert session.calls[0]["input"].shape == (1, 576)
    assert session.calls[0]["state"].shape == (2, 1, 128)
    assert session.calls[0]["sr"].item() == SAMPLE_RATE_HZ

    state = backend.new_state()
    first_frame = np.arange(backend.frame_samples, dtype=np.float32)
    probability, state = backend.infer(first_frame, state)
    first_call = session.calls[1]

    assert probability == pytest.approx(0.75)
    np.testing.assert_array_equal(
        first_call["input"][:, : backend.context_samples],
        np.zeros((1, backend.context_samples), dtype=np.float32),
    )
    np.testing.assert_array_equal(first_call["input"][:, backend.context_samples :], first_frame[None, :])

    second_frame = -first_frame
    backend.infer(second_frame, state)
    second_call = session.calls[2]
    # The previous frame's tail becomes this frame's context.
    np.testing.assert_array_equal(
        second_call["input"][:, : backend.context_samples],
        first_frame[None, -backend.context_samples :],
    )
    np.testing.assert_array_equal(second_call["state"], np.full((2, 1, 128), 2, dtype=np.float32))

    # Two sessions may enter the shared ORT session at the same time.
    session.run_barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(backend.infer, first_frame, backend.new_state()),
            executor.submit(backend.infer, second_frame, backend.new_state()),
        ]
        assert [future.result()[0] for future in futures] == pytest.approx([0.75, 0.75])


# --------------------------------------------------------------------------- #
# Session payload validation                                                  #
# --------------------------------------------------------------------------- #


def test_an_unknown_server_vad_field_is_refused() -> None:
    """A misspelled knob that silently does nothing is worse than a refusal.

    The endpointing would still look configured while running on defaults.
    """
    error = validate_realtime_turn_detection({"turn_detection": {"type": "server_vad", "semantic_eagerness": "high"}})
    assert error is not None
    assert "semantic_eagerness" in error


def test_the_documented_server_vad_fields_are_all_accepted() -> None:
    assert (
        validate_realtime_turn_detection(
            {
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.6,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 400,
                    "create_response": True,
                    "interrupt_response": True,
                    "min_speech_duration_ms": 120,
                },
                "overlap_policy": "barge_in_on_speech",
            }
        )
        is None
    )
