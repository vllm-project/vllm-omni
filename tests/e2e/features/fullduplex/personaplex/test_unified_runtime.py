# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from tests.e2e.online_serving import personaplex_realtime_duplex as e2e_driver
from vllm_omni.config.stage_config import (
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexInputMode,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.personaplex.data_plane import (
    PersonaPlexDataPlaneContext,
    PersonaPlexDataPlaneSession,
)
from vllm_omni.experimental.fullduplex.personaplex.input import (
    PersonaPlexPcmAppendBuffer,
)
from vllm_omni.experimental.fullduplex.personaplex.runtime_extension import (
    PersonaPlexDuplexRuntimeExtension,
)
from vllm_omni.experimental.fullduplex.personaplex.serving_adapter import (
    PersonaPlexServingRuntimeAdapter,
)
from vllm_omni.model_executor.models.personaplex.pipeline import (
    PERSONAPLEX_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pcm_payload(samples: np.ndarray, *, sample_rate_hz: int = 24000) -> dict[str, object]:
    samples = np.ascontiguousarray(samples, dtype="<f4")
    return {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": sample_rate_hz,
        "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
    }


def _audio_client(
    frame_count: int = 10,
    *,
    voiced_frames: int = 0,
) -> SimpleNamespace:
    frames = np.full((frame_count, e2e_driver.FRAME_SAMPLES), 7, dtype="<i2")
    frames[:voiced_frames] = 1000
    raw = frames.tobytes()
    event = {
        "type": "response.audio.delta",
        "response_id": "response-1",
        "sample_rate_hz": e2e_driver.SAMPLE_RATE_HZ,
        "metadata": {
            "vllm_omni": {
                "runtime_impl": "scheduler_data_plane",
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
            }
        },
    }
    events = SimpleNamespace(
        events=[event],
        response_audio={"response-1": [raw]},
        audio_bytes=lambda: raw,
    )
    return SimpleNamespace(events=events)


def test_realtime_audio_frame_stats_separate_voiced_and_silent_frames() -> None:
    silent = np.zeros(e2e_driver.FRAME_SAMPLES, dtype="<i2")
    voiced = np.full(e2e_driver.FRAME_SAMPLES, 1000, dtype="<i2")

    stats = e2e_driver._audio_frame_stats(
        silent.tobytes() + voiced.tobytes(),
        input_frames=3,
        voiced_frame_rms_threshold=1e-4,
    )

    assert stats == {
        "output_frames": 2,
        "voiced_frames": 1,
        "silent_frames": 1,
        "frame_deficit": 1,
        "frame_coverage_ratio": pytest.approx(2 / 3),
    }


def test_realtime_audio_frame_stats_reject_partial_codec_frame() -> None:
    partial_frame = np.zeros(e2e_driver.FRAME_SAMPLES - 1, dtype="<i2")

    with pytest.raises(AssertionError, match="whole 80 ms codec frames"):
        e2e_driver._audio_frame_stats(
            partial_frame.tobytes(),
            input_frames=1,
            voiced_frame_rms_threshold=1e-4,
        )


@pytest.mark.parametrize("frame_count", [10, 40])
def test_e2e_driver_rejects_inaudible_sessions(frame_count: int) -> None:
    args = SimpleNamespace(
        max_frame_deficit=4,
        voiced_frame_rms_threshold=1e-3,
        min_voiced_frames=5,
    )

    with pytest.raises(AssertionError, match="audible speech"):
        e2e_driver._session_result(
            _audio_client(frame_count),
            input_frames=frame_count,
            args=args,
            minimum_chunks=1,
        )


def test_e2e_driver_uses_absolute_audible_floor() -> None:
    args = SimpleNamespace(
        max_frame_deficit=4,
        voiced_frame_rms_threshold=1e-3,
        min_voiced_frames=5,
    )

    e2e_driver._session_result(
        _audio_client(200, voiced_frames=5),
        input_frames=200,
        args=args,
        minimum_chunks=1,
    )


def test_e2e_driver_defaults_replacement_to_full_input() -> None:
    args = e2e_driver.parse_args(
        [
            "--model",
            "/model",
            "--input-wav",
            "/input.wav",
        ]
    )

    assert args.replacement_frames == 0
    assert args.min_voiced_frames == 5


def test_e2e_driver_records_and_validates_input_sha256(tmp_path: Path) -> None:
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"fixture-wav")

    identity = e2e_driver._input_identity(
        input_wav,
        expected_sha256="bade4b3c163edde390ff391207d34a887257d8a9cc3b621cc8c618b6e6761304",
    )

    assert identity == {
        "path": str(input_wav.resolve()),
        "sha256": "bade4b3c163edde390ff391207d34a887257d8a9cc3b621cc8c618b6e6761304",
    }


def test_e2e_driver_rejects_unexpected_input_sha256(tmp_path: Path) -> None:
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"fixture-wav")

    with pytest.raises(ValueError, match="input WAV SHA-256 mismatch"):
        e2e_driver._input_identity(input_wav, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        (
            "duplex_runtime_extension",
            "vllm_omni.experimental.fullduplex.personaplex.runtime_extension.PersonaPlexDuplexRuntimeExtension",
        ),
        (
            "duplex_serving_adapter",
            "vllm_omni.experimental.fullduplex.personaplex.serving_adapter.PersonaPlexServingRuntimeAdapter",
        ),
    ],
)
def test_personaplex_pipeline_registers_unified_duplex_components(
    attribute: str,
    expected: str,
) -> None:
    assert PERSONAPLEX_PIPELINE.duplex_control_enabled is True
    assert getattr(PERSONAPLEX_PIPELINE, attribute) == expected


@pytest.mark.parametrize(
    ("engine_arg", "expected"),
    [
        ("skip_tokenizer_init", True),
        ("enable_prefix_caching", False),
    ],
)
def test_personaplex_stage_engine_args(
    engine_arg: str,
    expected: bool,
) -> None:
    deploy_path = Path(__file__).parents[5] / "vllm_omni" / "deploy" / "personaplex.yaml"
    stages = merge_pipeline_deploy(
        PERSONAPLEX_PIPELINE,
        load_deploy_config(deploy_path),
    )

    assert [stage.yaml_engine_args.get(engine_arg) for stage in stages] == [
        expected,
        expected,
    ]


def test_personaplex_duplex_capacity_is_propagated_to_all_model_stages() -> None:
    deploy_path = Path(__file__).parents[5] / "vllm_omni" / "deploy" / "personaplex.yaml"
    deploy = load_deploy_config(deploy_path)
    stages = merge_pipeline_deploy(PERSONAPLEX_PIPELINE, deploy)

    assert deploy.duplex_session.max_sessions == 2
    assert [stage.yaml_engine_args.get("duplex_max_sessions") for stage in stages] == [2, 2]
    assert "personaplex_codec_max_sessions" not in deploy.connectors["connector_of_shared_memory"]["extra"]


def test_personaplex_capabilities_are_honest() -> None:
    single = PersonaPlexServingRuntimeAdapter(lambda *_: None).capabilities(max_sessions=1)
    multi = PersonaPlexServingRuntimeAdapter(lambda *_: None).capabilities(max_sessions=2)

    assert multi.implementation_level == "model_native_duplex"
    assert multi.input_modes == ["append_audio_chunk"]
    assert multi.chunk_period_ms == 80
    assert single.supports_multi_session is False
    assert multi.supports_multi_session is True
    assert multi.supports_multi_session_same_replica is True
    assert multi.supports_barge_in is False


def test_pcm_buffer_emits_one_80ms_frame_transactionally() -> None:
    buffer = PersonaPlexPcmAppendBuffer()
    reservation = buffer.prepare_append(
        _pcm_payload(np.arange(1920, dtype=np.float32)),
        operation_id="op-1",
        chunk_period_ms=80,
        allow_emit=True,
    )

    assert reservation is not None
    assert reservation.byte_count == 1920 * 4
    assert buffer.pending_byte_count == 0
    reservation.rollback()
    assert buffer.pending_byte_count == 1920 * 4


def test_pcm_buffer_rejects_changed_sample_rate() -> None:
    buffer = PersonaPlexPcmAppendBuffer()
    assert (
        buffer.prepare_append(
            _pcm_payload(np.zeros(100, np.float32)),
            operation_id="op-1",
            chunk_period_ms=80,
            allow_emit=False,
        )
        is None
    )

    with pytest.raises(ValueError, match="sample_rate"):
        buffer.prepare_append(
            _pcm_payload(np.zeros(100, np.float32), sample_rate_hz=16000),
            operation_id="op-2",
            chunk_period_ms=80,
            allow_emit=False,
        )


def test_pcm_buffer_rejects_non_finite_samples() -> None:
    buffer = PersonaPlexPcmAppendBuffer()

    with pytest.raises(ValueError, match="finite"):
        buffer.prepare_append(
            _pcm_payload(np.array([0.0, np.nan], np.float32)),
            operation_id="op-1",
            chunk_period_ms=80,
            allow_emit=False,
        )


def test_runtime_extension_builds_one_frame_scheduler_append() -> None:
    extension = PersonaPlexDuplexRuntimeExtension()
    defaults = (
        SamplingParams(temperature=0.8, top_k=10, max_tokens=10),
        SamplingParams(max_tokens=1024),
    )

    configured = extension.configure_sampling_params(runtime_config={}, defaults=defaults)
    assert configured[0].temperature == 0.0
    assert configured[0].top_k == 1
    assert configured[0].max_tokens == 1
    assert configured[1] is defaults[1]

    plan = extension.plan_append(
        request_id="req",
        fence=DuplexFence("session", incarnation=2, epoch=3),
        session_config={"instructions": "Be concise."},
        runtime_config={"personaplex_prefill_slots": 4},
        seq=1,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=_pcm_payload(np.zeros(1920, np.float32)),
        final=False,
        sampling_params=configured[0],
    )
    assert len(plan.prompt["prompt_token_ids"]) == 5
    duplex = plan.prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["session_id"] == "session"
    assert duplex["incarnation"] == 2
    assert duplex["epoch"] == 3
    assert duplex["seq"] == 1
    assert duplex["data_plane"] is True


def test_projector_emits_only_cumulative_audio_and_text_suffixes() -> None:
    encoded_sizes: list[int] = []

    def encode_audio(audio, _sample_rate, _response_format, _speed):
        size = int(np.asarray(audio, dtype=np.float32).size)
        encoded_sizes.append(size)
        return f"audio-{size}"

    projector = PersonaPlexDataPlaneSession(encode_audio)
    projector.begin_request("req")
    context = PersonaPlexDataPlaneContext(response_format="wav")

    first = SimpleNamespace(
        request_id="req",
        outputs=[SimpleNamespace(text="he", multimodal_output={})],
        multimodal_output={"model_outputs": np.arange(4, dtype=np.float32), "sr": 24000},
        finished=False,
    )
    second = SimpleNamespace(
        request_id="req",
        outputs=[SimpleNamespace(text="hello", multimodal_output={})],
        multimodal_output={"model_outputs": np.arange(6, dtype=np.float32), "sr": 24000},
        finished=False,
    )

    projected = list(
        projector.project(
            {"data_plane_outputs": [first, second]},
            context=context,
        )
    )

    assert encoded_sizes == [4, 2]
    assert [item["text"] for item in projected] == ["he", "llo"]
    assert [item["audio_data"] for item in projected] == ["audio-4", "audio-2"]
    assert all(item["sample_rate_hz"] == 24000 for item in projected)
    assert all(item["end_of_turn"] is False for item in projected)
