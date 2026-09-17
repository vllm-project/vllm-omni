# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""PersonaPlex on the unified full-duplex framework: the plugin contract and the E2E driver helpers."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from tests.e2e.online_serving import personaplex_realtime_duplex as e2e_driver
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexDataPlaneContext,
    DuplexRuntimeConfigError,
    load_duplex_plugin,
    validate_duplex_plugin_sampling,
)
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.model_executor.models.personaplex.duplex.config import DEFAULT_PERSONA, FRAME_SIZE, SAMPLE_RATE
from vllm_omni.model_executor.models.personaplex.duplex.data_plane import PersonaPlexDataPlaneSession
from vllm_omni.model_executor.models.personaplex.duplex.input import PersonaPlexPcmAppendBuffer
from vllm_omni.model_executor.models.personaplex.duplex.plugin import (
    PRIVATE_RUNTIME_CONFIG_KEYS,
    PersonaPlexDuplexPlugin,
    PersonaPlexSessionState,
)
from vllm_omni.model_executor.models.personaplex.pipeline import PERSONAPLEX_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_PATH = Path(__file__).parents[5] / "vllm_omni" / "deploy" / "personaplex.yaml"


def _pcm_payload(samples: np.ndarray, *, sample_rate_hz: int = SAMPLE_RATE) -> dict[str, object]:
    samples = np.ascontiguousarray(samples, dtype="<f4")
    return {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": sample_rate_hz,
        "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
    }


def _encode_audio(audio, _sample_rate, _response_format, _speed):
    if audio is None:
        return None
    size = int(np.asarray(audio, dtype=np.float32).size)
    return f"audio-{size}" if size else None


def _plugin() -> PersonaPlexDuplexPlugin:
    return PersonaPlexDuplexPlugin(_encode_audio)


# --------------------------------------------------------------------------- #
# Pipeline binding                                                            #
# --------------------------------------------------------------------------- #


def test_pipeline_declares_the_duplex_plugin_and_no_pre_framework_wiring() -> None:
    assert PERSONAPLEX_PIPELINE.duplex_plugin == (
        "vllm_omni.model_executor.models.personaplex.duplex.plugin.PersonaPlexDuplexPlugin"
    )
    assert PERSONAPLEX_PIPELINE.duplex_runtime_extension is None
    assert PERSONAPLEX_PIPELINE.duplex_serving_adapter is None
    assert PERSONAPLEX_PIPELINE.duplex_control_enabled is False

    plugin = load_duplex_plugin(PERSONAPLEX_PIPELINE.duplex_plugin, _encode_audio)

    assert isinstance(plugin, PersonaPlexDuplexPlugin)
    assert plugin.plugin_id == "personaplex"
    assert isinstance(plugin.data_plane, PersonaPlexDataPlaneSession)
    validate_duplex_plugin_sampling(plugin, sampling_defaults=(SamplingParams(), SamplingParams()))


@pytest.mark.parametrize(
    ("engine_arg", "expected"),
    [("skip_tokenizer_init", True), ("enable_prefix_caching", False)],
)
def test_personaplex_stage_engine_args(engine_arg: str, expected: bool) -> None:
    stages = merge_pipeline_deploy(PERSONAPLEX_PIPELINE, load_deploy_config(DEPLOY_PATH))

    assert [stage.yaml_engine_args.get(engine_arg) for stage in stages] == [expected, expected]


def test_personaplex_deploy_is_duplex_and_propagates_capacity_to_all_model_stages() -> None:
    deploy = load_deploy_config(DEPLOY_PATH)
    stages = merge_pipeline_deploy(PERSONAPLEX_PIPELINE, deploy)

    assert deploy.session_mode == "duplex"
    assert deploy.duplex_session.max_sessions == 2
    assert [stage.yaml_engine_args.get("duplex_max_sessions") for stage in stages] == [2, 2]
    assert "personaplex_codec_max_sessions" not in deploy.connectors["connector_of_shared_memory"]["extra"]


# --------------------------------------------------------------------------- #
# Session policy                                                              #
# --------------------------------------------------------------------------- #


def test_capabilities_are_honest() -> None:
    single = _plugin().capabilities(max_sessions=1)
    multi = _plugin().capabilities(max_sessions=2)

    assert multi.as_dict()["implementation_level"] == "model_native_duplex"
    assert multi.as_dict()["input_modes"] == ["append_audio_chunk"]
    assert multi.chunk_period_ms == 80
    assert single.supports_multi_session is False
    assert multi.supports_multi_session is True
    assert multi.supports_multi_session_same_replica is True
    assert multi.supports_barge_in is False
    assert multi.supports_client_commit is False
    assert multi.supports_external_turn_signal is False
    assert multi.supports_session_resume is False
    assert multi.supports_audio_truncate is False
    assert multi.supports_chat_completions is False
    assert multi.session_admission_mode == "engine_managed"


def test_session_state_is_the_framework_default_with_the_personaplex_buffer() -> None:
    state = _plugin().create_session_state()

    assert isinstance(state, PersonaPlexSessionState)
    assert isinstance(state, DefaultDuplexModelSessionState)
    assert isinstance(state.audio_buffer, PersonaPlexPcmAppendBuffer)
    state.retain_committed_audio({"audio": ""}, operation_id="op", reserved_bytes=8)
    assert state.clear_committed_audio() == 8
    assert state.committed_audio_payload is None


def test_private_runtime_keys_are_rejected_in_extra_body() -> None:
    plugin = _plugin()

    assert PRIVATE_RUNTIME_CONFIG_KEYS == {
        "personaplex_prefill_slots",
        "personaplex_model_path",
        "personaplex_voice_prompt",
        "personaplex_persona",
    }
    plugin.validate_client_extra_body({"auto_response": True})
    plugin.validate_client_extra_body(None)
    with pytest.raises(DuplexRuntimeConfigError, match="personaplex_persona") as excinfo:
        plugin.validate_client_extra_body({"personaplex_persona": "x", "auto_response": True})
    assert excinfo.value.code == "invalid_duplex_runtime_config"


@pytest.mark.asyncio
async def test_prepare_runtime_config_resolves_voice_and_prefill_slots_once(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_prefill_slots(model_path: str, voice: str, persona: str) -> int:
        calls.append((model_path, voice, persona))
        return 42

    monkeypatch.setattr(stage0, "personaplex_prefill_slots", fake_prefill_slots)
    plugin = _plugin()
    model_config = SimpleNamespace(model="/models/personaplex")

    runtime_config = await plugin.prepare_runtime_config(
        DuplexSessionConfig(voice="NATM1.pt", instructions="Be brief."),
        model_config=model_config,
    )
    again = await plugin.prepare_runtime_config(
        DuplexSessionConfig(voice="NATM1.pt", instructions="Be brief."),
        model_config=model_config,
    )
    defaults = await plugin.prepare_runtime_config(DuplexSessionConfig(), model_config=model_config)

    assert runtime_config == {
        "personaplex_model_path": "/models/personaplex",
        "personaplex_voice_prompt": "NATM1.pt",
        "personaplex_persona": "Be brief.",
        "personaplex_prefill_slots": 42,
    }
    assert again == runtime_config
    assert defaults["personaplex_voice_prompt"] == "NATF2.pt"
    assert defaults["personaplex_persona"] == DEFAULT_PERSONA
    # The voice bundle and tokenizer are read once per (model, voice, persona).
    assert calls == [
        ("/models/personaplex", "NATM1.pt", "Be brief."),
        ("/models/personaplex", "NATF2.pt", DEFAULT_PERSONA),
    ]


@pytest.mark.asyncio
async def test_prepare_runtime_config_reports_prefill_failures_as_runtime_config_errors(monkeypatch) -> None:
    def failing_prefill_slots(model_path: str, voice: str, persona: str) -> int:
        raise FileNotFoundError("voices.tgz missing")

    monkeypatch.setattr(stage0, "personaplex_prefill_slots", failing_prefill_slots)

    with pytest.raises(DuplexRuntimeConfigError, match="voices.tgz missing") as excinfo:
        await _plugin().prepare_runtime_config(DuplexSessionConfig(), model_config=SimpleNamespace(model="/m"))
    assert excinfo.value.code == "prefill_unavailable"

    with pytest.raises(DuplexRuntimeConfigError, match="model path") as excinfo:
        await _plugin().prepare_runtime_config(DuplexSessionConfig(), model_config=None)
    assert excinfo.value.code == "model_path_unavailable"


@pytest.mark.parametrize("voice", ["../NATF2.pt", "voices/NATF2.pt", "NATF2.wav", "/abs/NATF2.pt"])
@pytest.mark.asyncio
async def test_prepare_runtime_config_rejects_voices_that_are_not_bundled_basenames(voice: str) -> None:
    with pytest.raises(DuplexRuntimeConfigError, match="bundled .pt basename") as excinfo:
        await _plugin().prepare_runtime_config(
            DuplexSessionConfig(voice=voice), model_config=SimpleNamespace(model="/m")
        )
    assert excinfo.value.code == "invalid_voice"


def test_runtime_config_update_rejects_changed_persona() -> None:
    plugin = _plugin()
    short_persona = "You are a helpful assistant."
    current = {"personaplex_persona": short_persona, "personaplex_voice_prompt": "NATF2.pt"}

    with pytest.raises(DuplexRuntimeConfigError, match="persona") as excinfo:
        plugin.runtime_config_for_update(DuplexSessionConfig(instructions="You are a pirate."), current)
    assert excinfo.value.code == "persona_update_unsupported"

    unchanged = plugin.runtime_config_for_update(DuplexSessionConfig(instructions=short_persona), current)
    assert unchanged["personaplex_persona"] == short_persona
    assert unchanged is not current


def test_runtime_config_update_rejects_changed_voice() -> None:
    plugin = _plugin()
    current = {"personaplex_persona": DEFAULT_PERSONA, "personaplex_voice_prompt": "NATF2.pt"}

    with pytest.raises(DuplexRuntimeConfigError, match="voice") as excinfo:
        plugin.runtime_config_for_update(DuplexSessionConfig(voice="NATM1.pt"), current)
    assert excinfo.value.code == "voice_update_unsupported"

    unchanged = plugin.runtime_config_for_update(DuplexSessionConfig(voice="NATF2.pt"), current)
    assert unchanged["personaplex_voice_prompt"] == "NATF2.pt"


def test_data_plane_context_is_the_framework_default() -> None:
    context = _plugin().data_plane_context(
        epoch=1,
        turn_id=2,
        active_response_turn_id=None,
        active_response_id="resp",
        auto_responds=True,
        response_format="pcm16",
        speed=None,
        modalities=("audio",),
    )

    assert isinstance(context, DuplexDataPlaneContext)
    assert context.response_format == "pcm16" and context.epoch == 1


# --------------------------------------------------------------------------- #
# Engine policy                                                               #
# --------------------------------------------------------------------------- #


def test_sampling_params_are_greedy_one_token_on_stage0_only() -> None:
    defaults = (SamplingParams(temperature=0.8, top_k=10, max_tokens=10), SamplingParams(max_tokens=1024))

    configured = _plugin().configure_sampling_params(runtime_config={}, defaults=defaults)

    assert configured[0].temperature == 0.0
    assert configured[0].top_k == 1
    assert configured[0].max_tokens == 1
    assert configured[1] is defaults[1]
    assert defaults[0].max_tokens == 10
    assert _plugin().configure_sampling_params(runtime_config={}, defaults=()) == ()


def test_plan_append_reserves_one_slot_plus_prefill_on_the_first_seq() -> None:
    plugin = _plugin()
    fence = DuplexFence("session", epoch=3, turn_id=1)
    common = {
        "request_id": "req",
        "fence": fence,
        "session_config": {"instructions": "Be concise."},
        "runtime_config": {"personaplex_prefill_slots": 4},
        "turn_seq": 1,
        "payload": _pcm_payload(np.zeros(FRAME_SIZE, np.float32)),
        "final": False,
        "sampling_params": SamplingParams(),
    }

    first = plugin.plan_append(seq=1, **common)
    second = plugin.plan_append(seq=2, **common)

    assert len(first.prompt["prompt_token_ids"]) == 5
    assert len(second.prompt["prompt_token_ids"]) == 1
    duplex = first.prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["session_id"] == "session"
    assert duplex["epoch"] == 3
    assert duplex["turn_id"] == 1
    assert duplex["seq"] == 1
    assert duplex["data_plane"] is True
    assert duplex["fence"] == fence
    assert duplex["scheduler_token_budget"] == 5
    assert duplex["runtime_config"] == {"personaplex_prefill_slots": 4}
    assert "incarnation" not in duplex
    assert first.prompt["model_intermediate_buffer"]["global_request_id"] == ["session"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_pcm_payload(np.zeros(FRAME_SIZE - 1, np.float32)), "exactly 1920 samples"),
        (_pcm_payload(np.zeros(FRAME_SIZE, np.float32), sample_rate_hz=16000), "sample_rate_hz must be 24000"),
        ({"format": "pcm16", "sample_rate_hz": SAMPLE_RATE, "audio": ""}, "format must be pcm_f32le"),
        ("frame", "must be a mapping"),
    ],
)
def test_plan_append_rejects_anything_but_one_24k_frame(payload: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _plugin().plan_append(
            request_id="req",
            fence=DuplexFence("session"),
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload=payload,
            final=False,
            sampling_params=None,
        )


def test_plan_append_rejects_a_malformed_prefill_slot_count() -> None:
    with pytest.raises(ValueError, match="personaplex_prefill_slots"):
        _plugin().plan_append(
            request_id="req",
            fence=DuplexFence("session"),
            session_config={},
            runtime_config={"personaplex_prefill_slots": "many"},
            seq=1,
            turn_seq=1,
            payload=_pcm_payload(np.zeros(FRAME_SIZE, np.float32)),
            final=False,
            sampling_params=None,
        )


def test_decide_output_never_decides() -> None:
    assert (
        _plugin().decide_output(
            stage_id=0,
            final_stage_id=1,
            segment_finished=True,
            segment_token_ids=(1,),
            segment_output_metadata={},
            output=SimpleNamespace(outputs=[]),
        )
        is None
    )


def test_silence_unit_is_one_24k_frame_that_plan_append_accepts() -> None:
    plugin = _plugin()

    unit = plugin.silence_unit_payload()

    assert unit["format"] == "pcm_f32le"
    assert unit["sample_rate_hz"] == SAMPLE_RATE
    assert len(base64.b64decode(unit["audio"])) == FRAME_SIZE * 4
    plan = plugin.plan_append(
        request_id="req",
        fence=DuplexFence("session"),
        session_config={},
        runtime_config={},
        seq=7,
        turn_seq=7,
        payload=unit,
        final=False,
        sampling_params=None,
    )
    assert len(plan.prompt["prompt_token_ids"]) == 1


# --------------------------------------------------------------------------- #
# E2E driver helpers (CPU-checked parts of the GPU driver)                    #
# --------------------------------------------------------------------------- #


def _audio_client(frame_count: int = 10, *, voiced_frames: int = 0) -> SimpleNamespace:
    frames = np.full((frame_count, e2e_driver.FRAME_SAMPLES), 7, dtype="<i2")
    frames[:voiced_frames] = 1000
    raw = frames.tobytes()
    event = {
        "type": "response.output_audio.delta",
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
    events = SimpleNamespace(events=[event], response_audio={"response-1": [raw]}, audio_bytes=lambda: raw)
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
        e2e_driver._audio_frame_stats(partial_frame.tobytes(), input_frames=1, voiced_frame_rms_threshold=1e-4)


@pytest.mark.parametrize("frame_count", [10, 40])
def test_e2e_driver_rejects_inaudible_sessions(frame_count: int) -> None:
    args = SimpleNamespace(max_frame_deficit=4, voiced_frame_rms_threshold=1e-3, min_voiced_frames=5)

    with pytest.raises(AssertionError, match="audible speech"):
        e2e_driver._session_result(_audio_client(frame_count), input_frames=frame_count, args=args, minimum_chunks=1)


def test_e2e_driver_uses_absolute_audible_floor() -> None:
    args = SimpleNamespace(max_frame_deficit=4, voiced_frame_rms_threshold=1e-3, min_voiced_frames=5)

    e2e_driver._session_result(_audio_client(200, voiced_frames=5), input_frames=200, args=args, minimum_chunks=1)


def test_e2e_driver_defaults_replacement_to_full_input() -> None:
    args = e2e_driver.parse_args(["--model", "/model", "--input-wav", "/input.wav"])

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


def test_e2e_driver_lets_the_server_allocate_the_session_id() -> None:
    url = e2e_driver._realtime_url("ws://127.0.0.1:8099/v1/realtime?duplex=1", "nvidia/personaplex-7b-v1")

    assert "session_id=" not in url
    assert "duplex=1" in url and "autostart=0" in url
