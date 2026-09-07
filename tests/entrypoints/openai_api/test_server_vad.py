# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import io
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.entrypoints.duplex.protocol import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.entrypoints.duplex.server_vad import (
    ServerVADConfig,
    ServerVADFrame,
    ServerVADPipeline,
    SileroVADBackend,
    SileroVADBackendProvider,
    SpeechEndpointDecision,
    ThresholdEndpointPolicy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class SequenceDetector:
    frame_samples = 160

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def new_state(self) -> int:
        return 0

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
        index = int(state)
        probability = self.probabilities[index] if index < len(self.probabilities) else 0.0
        return probability, index + 1


def test_server_vad_config_defaults_and_validation():
    config = ServerVADConfig.from_value({"type": "server_vad"})

    assert config.type == "server_vad"
    assert config.threshold == 0.5
    assert config.prefix_padding_ms == 300
    assert config.silence_duration_ms == 500
    assert config.create_response is True
    assert config.interrupt_response is True
    assert config.min_speech_duration_ms is None
    assert config.as_dict()["type"] == config.type

    interrupting = ServerVADConfig.from_value(
        {
            "type": "server_vad",
            "interrupt_response": True,
            "min_speech_duration_ms": 96,
        }
    )
    assert interrupting.interrupt_response is True
    assert interrupting.min_speech_duration_ms == 96
    with pytest.raises(ValueError, match="Unknown"):
        ServerVADConfig.from_value({"type": "server_vad", "semantic_eagerness": "high"})


@pytest.mark.asyncio
async def test_server_vad_pipeline_handles_arbitrary_chunk_boundaries():
    detector = SequenceDetector([0.0, 0.9, 0.8, 0.0, 0.0])
    pipeline = ServerVADPipeline(
        detector,
        ServerVADConfig(prefix_padding_ms=10, silence_duration_ms=20),
    )
    samples = np.zeros(detector.frame_samples * 5, dtype=np.float32)

    batches = [
        await pipeline.push(samples[:73]),
        await pipeline.push(samples[73:431]),
        await pipeline.push(samples[431:]),
    ]
    frames = [frame for batch in batches for frame in batch.frames]

    assert len(frames) == 5
    assert [index for index, frame in enumerate(frames) if frame.decision.speech_started] == [1]
    assert [index for index, frame in enumerate(frames) if frame.decision.speech_stopped] == [4]

    pipeline.reset()
    reset_batch = await pipeline.push(samples)
    assert [index for index, frame in enumerate(reset_batch.frames) if frame.decision.speech_started] == [1]
    assert [index for index, frame in enumerate(reset_batch.frames) if frame.decision.speech_stopped] == [4]


@pytest.mark.asyncio
async def test_server_vad_pipeline_pcm16_resampling_is_split_invariant():
    sample_index = np.arange(2_400, dtype=np.int32)
    source = ((sample_index * 7919) % 65_536 - 32_768).astype("<i2")
    # Supply ample silence after the signal so the streaming resampler can
    # emit a complete set of VAD frames without depending on its FIR width.
    right_context = np.zeros(240, dtype="<i2")
    continuous_source = np.concatenate((source, right_context))

    async def process(chunk_sizes: list[int], *, use_bytes: bool) -> tuple[np.ndarray, list[SpeechEndpointDecision]]:
        pipeline = ServerVADPipeline(
            SequenceDetector([0.0] * 10),
            ServerVADConfig(),
        )
        frames: list[ServerVADFrame] = []
        offset = 0
        for chunk_size in chunk_sizes:
            chunk = continuous_source[offset : offset + chunk_size]
            payload = chunk.tobytes() if use_bytes else chunk
            batch = await pipeline.push_pcm16(payload, source_sample_rate_hz=24_000)
            frames.extend(batch.frames)
            offset += chunk_size
        assert offset == continuous_source.size
        return np.concatenate([frame.samples for frame in frames]), [frame.decision for frame in frames]

    whole_samples, whole_decisions = await process([continuous_source.size], use_bytes=False)
    split_samples, split_decisions = await process(
        [7, 13, 511, 512, 513, continuous_source.size - 1_556],
        use_bytes=True,
    )

    np.testing.assert_array_equal(split_samples, whole_samples)
    assert whole_samples.size == 1_600
    assert split_decisions == whole_decisions


@pytest.mark.asyncio
async def test_server_vad_pipeline_reset_clears_resampler_and_source_rate_lock():
    pipeline = ServerVADPipeline(
        SequenceDetector([0.0]),
        ServerVADConfig(),
    )

    batch = await pipeline.push_pcm16(
        np.asarray([30_000, -30_000], dtype="<i2"),
        source_sample_rate_hz=24_000,
    )
    assert not batch.frames

    pipeline.reset()

    source = np.zeros(160, dtype="<i2")
    source[:3] = [-32768, 0, 32767]
    batch = await pipeline.push_pcm16(source.tobytes(), source_sample_rate_hz=16_000)

    assert len(batch.frames) == 1
    np.testing.assert_array_equal(
        batch.frames[0].samples[:3],
        np.asarray([-1.0, 0.0, 32767 / 32768], dtype=np.float32),
    )


def test_threshold_endpoint_policy_uses_silero_v62_exit_hysteresis():
    policy = ThresholdEndpointPolicy(
        ServerVADConfig(threshold=0.5, silence_duration_ms=20),
        sample_rate_hz=16_000,
    )
    frame_samples = 160

    started = policy.update(0.8, frame_start_sample=0, frame_samples=frame_samples)
    assert started.speech_started is True

    # Values below the activation threshold but above threshold - 0.15 must
    # not begin a silence candidate or close an active speech segment.
    for frame_index in range(1, 5):
        decision = policy.update(
            0.4,
            frame_start_sample=frame_index * frame_samples,
            frame_samples=frame_samples,
        )
        assert decision.speech_stopped is False

    first_silence = policy.update(
        0.2,
        frame_start_sample=5 * frame_samples,
        frame_samples=frame_samples,
    )
    stopped = policy.update(
        0.2,
        frame_start_sample=6 * frame_samples,
        frame_samples=frame_samples,
    )

    assert first_silence.speech_stopped is False
    assert stopped.speech_stopped is True


def test_threshold_endpoint_policy_honors_minimum_speech_duration():
    policy = ThresholdEndpointPolicy(
        ServerVADConfig(prefix_padding_ms=0, min_speech_duration_ms=25),
        sample_rate_hz=16_000,
    )
    frame_samples = 160

    assert not policy.update(0.9, frame_start_sample=0, frame_samples=frame_samples).speech_started
    assert not policy.update(0.9, frame_start_sample=160, frame_samples=frame_samples).speech_started
    started = policy.update(0.9, frame_start_sample=320, frame_samples=frame_samples)

    assert started.speech_started is True
    assert started.audio_start_ms == 0


def test_threshold_endpoint_policy_clamps_silero_exit_threshold():
    policy = ThresholdEndpointPolicy(
        ServerVADConfig(threshold=0.1, silence_duration_ms=20),
        sample_rate_hz=16_000,
    )
    frame_samples = 160

    started = policy.update(0.9, frame_start_sample=0, frame_samples=frame_samples)
    assert started.speech_started is True

    first_silence = policy.update(
        0.0,
        frame_start_sample=frame_samples,
        frame_samples=frame_samples,
    )
    stopped = policy.update(
        0.0,
        frame_start_sample=2 * frame_samples,
        frame_samples=frame_samples,
    )

    assert first_silence.speech_stopped is False
    assert stopped.speech_stopped is True
    # The committed audio ends at 30 ms: 10 ms of speech followed by the
    # 20 ms of silence required to detect the endpoint.
    assert stopped.audio_end_ms == 30
    assert stopped.endpoint_delay_ms == 20


def test_silero_backend_matches_upstream_v62_streaming_contract(monkeypatch, tmp_path):
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

    sessions: list[FakeInferenceSession] = []
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=FakeSessionOptions,
            InferenceSession=FakeInferenceSession,
        ),
    )

    model_path = tmp_path / "silero_vad.onnx"
    model_path.write_bytes(b"fake-model")
    backend = SileroVADBackend(model_path)
    session = sessions[0]

    assert session.providers == ["CPUExecutionProvider"]
    assert session.sess_options.inter_op_num_threads == 1
    assert session.sess_options.intra_op_num_threads == 1
    assert session.calls[0]["input"].shape == (1, 576)
    assert session.calls[0]["state"].shape == (2, 1, 128)
    np.testing.assert_array_equal(session.calls[0]["state"], np.zeros((2, 1, 128), dtype=np.float32))
    assert session.calls[0]["sr"].shape == ()
    assert session.calls[0]["sr"].item() == 16_000

    state = backend.new_state()
    first_frame = np.arange(backend.frame_samples, dtype=np.float32)
    probability, state = backend.infer(first_frame, state)
    first_call = session.calls[1]

    assert probability == pytest.approx(0.75)
    np.testing.assert_array_equal(
        first_call["input"][:, : backend.context_samples],
        np.zeros((1, backend.context_samples), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        first_call["input"][:, backend.context_samples :],
        first_frame[None, :],
    )
    np.testing.assert_array_equal(first_call["state"], np.zeros((2, 1, 128), dtype=np.float32))

    second_frame = -first_frame
    backend.infer(second_frame, state)
    second_call = session.calls[2]
    np.testing.assert_array_equal(
        second_call["input"][:, : backend.context_samples],
        first_frame[None, -backend.context_samples :],
    )
    np.testing.assert_array_equal(
        second_call["input"][:, backend.context_samples :],
        second_frame[None, :],
    )
    np.testing.assert_array_equal(
        second_call["state"],
        np.full((2, 1, 128), 2, dtype=np.float32),
    )

    # Independent pipeline states may enter the shared ORT session concurrently.
    session.run_barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(backend.infer, first_frame, backend.new_state()),
            executor.submit(backend.infer, second_frame, backend.new_state()),
        ]
        assert [future.result()[0] for future in futures] == pytest.approx([0.75, 0.75])


def test_duplex_session_owns_server_vad_prefix_and_utterance_audio():
    session = DuplexSession("server-vad-session", DuplexSessionConfig())
    frame = np.zeros(160, dtype=np.float32)

    session.reserve_input_bytes(frame.nbytes * 3, limit=frame.nbytes * 4)
    released = session.append_server_vad_frame(
        frame,
        speech_started=False,
        speech_stopped=False,
        prefix_samples=160,
    )
    assert released == 0
    released = session.append_server_vad_frame(
        frame,
        speech_started=False,
        speech_stopped=False,
        prefix_samples=160,
    )
    assert released == frame.nbytes
    session.release_input_bytes(released)
    session.append_server_vad_frame(
        frame,
        speech_started=True,
        speech_stopped=False,
        prefix_samples=160,
    )

    assert session.server_vad_utterance_bytes == frame.nbytes * 2
    assert session.stage_server_vad_audio_for_commit() is True
    committed = session.commit_user_input()
    assert committed is not None
    assert committed.message["role"] == "user"
    audio_url = committed.message["content"][0]["audio_url"]["url"]
    assert audio_url.startswith("data:audio/wav;base64,")
    wav_bytes = base64.b64decode(audio_url.partition(",")[2])
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == frame.size * 2


def test_silero_provider_rejects_missing_or_invalid_local_artifact(tmp_path):
    missing = SileroVADBackendProvider(model_path=str(tmp_path / "missing.onnx"))
    with pytest.raises(RuntimeError, match="does not exist"):
        missing.get()

    invalid_path = tmp_path / "silero_vad.onnx"
    invalid_path.write_bytes(b"not-the-pinned-model")
    invalid = SileroVADBackendProvider(model_path=str(invalid_path))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        invalid.get()
