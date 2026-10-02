# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for typed deployment-only runtime configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import vllm_omni
from tests.helpers.stage_config import get_deploy_duplex_max_sessions
from vllm_omni.config.stage_config import (
    DuplexPacingConfig,
    DuplexSessionRuntimeConfig,
    load_deploy_config,
    resolve_deploy_yaml,
)

_DEPLOY_DIR = Path(vllm_omni.__file__).parent / "deploy"
_OMP_CAPS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE"}

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duplex_session_runtime_defaults_are_typed_and_immutable(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text("duplex_session: {}\nstages: []\n", encoding="utf-8")

    deploy = load_deploy_config(deploy_path)

    assert isinstance(deploy.duplex_session, DuplexSessionRuntimeConfig)
    assert deploy.duplex_session.idle_ttl_s == 300.0
    assert deploy.duplex_session.disconnect_grace_s == 30.0
    assert deploy.duplex_session.reaper_interval_s == 5.0
    assert deploy.duplex_session.resume_replay_ttl_s == 60.0
    assert deploy.duplex_session.resume_replay_max_bytes_per_session == 8 * 1024 * 1024
    assert deploy.duplex_session.max_pending_input_bytes_per_session == 16 * 1024 * 1024
    assert deploy.duplex_session.max_pending_turns_per_session == 4
    assert deploy.duplex_session.max_sessions == 1
    assert deploy.duplex_session.completed_append_cache_size == 256
    assert deploy.duplex_session.server_vad_model_path is None
    with pytest.raises(FrozenInstanceError):
        deploy.duplex_session.idle_ttl_s = 1.0  # type: ignore[misc]


def test_duplex_session_runtime_accepts_disabled_idle_expiry(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text("duplex_session:\n  idle_ttl_s: null\nstages: []\n", encoding="utf-8")

    deploy = load_deploy_config(deploy_path)

    assert deploy.duplex_session.idle_ttl_s is None


def test_duplex_session_runtime_accepts_local_server_vad_model(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text(
        "duplex_session:\n  server_vad_model_path: /models/silero_vad.onnx\nstages: []\n",
        encoding="utf-8",
    )

    deploy = load_deploy_config(deploy_path)

    assert deploy.duplex_session.server_vad_model_path == "/models/silero_vad.onnx"


def test_duplex_session_runtime_rejects_empty_server_vad_model_path(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text(
        "duplex_session:\n  server_vad_model_path: ''\nstages: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="server_vad_model_path"):
        load_deploy_config(deploy_path)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("idle_ttl_s", 0),
        ("idle_ttl_s", -1),
        ("disconnect_grace_s", 0),
        ("reaper_interval_s", -1),
        ("resume_replay_ttl_s", 0),
        ("resume_replay_max_bytes_per_session", -1),
        ("max_pending_input_bytes_per_session", 0),
        ("max_pending_turns_per_session", -1),
        ("max_sessions", 0),
        ("completed_append_cache_size", 0),
    ],
)
def test_duplex_session_runtime_rejects_non_positive_values(tmp_path, name: str, value: int) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text(f"duplex_session:\n  {name}: {value}\nstages: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"duplex_session\.{name} must be positive"):
        load_deploy_config(deploy_path)


@pytest.mark.parametrize(
    ("deploy_yaml", "expected"),
    [
        ("minicpmo_4_5.yaml", 16),
        ("minicpmo_4_5_duplex_h200.yaml", 16),
        ("minicpmo_4_5_8x4090.yaml", 1),
        ("minicpmo_4_5_3gpu_stage1_replicas.yaml", 16),
        ("minicpmo_4_5_dxsched.yaml", 16),
        ("minicpmo_4_5_dxsched_on.yaml", 16),
        ("minicpmo_4_5_h100.yaml", 20),
    ],
)
def test_deploy_duplex_max_sessions_tracks_the_deploy_config(deploy_yaml: str, expected: int) -> None:
    # Guards the admission probe. A capacity edit, a config that declares none,
    # or an overlay inheriting one from its base must surface here rather than
    # as another nightly duplex admission timeout.
    assert get_deploy_duplex_max_sessions(deploy_yaml) == expected


def _stage(raw: dict, stage_id: int) -> dict:
    return next(stage for stage in raw["stages"] if stage["stage_id"] == stage_id)


def test_dxsched_profile_keeps_the_inherited_switches_and_adds_the_omp_caps() -> None:
    # duplex_session and stage env / hf_overrides are replaced, not merged, by
    # resolve_deploy_yaml: the profile has to restate what it inherits.
    raw = resolve_deploy_yaml(_DEPLOY_DIR / "minicpmo_4_5_dxsched.yaml")
    stage0, stage1, stage2 = (_stage(raw, stage_id) for stage_id in (0, 1, 2))
    assert stage0["hf_overrides"]["duplex_audio_encoder_pinned_h2d"] is True
    assert stage0["hf_overrides"]["duplex_fbank_stats"] is False
    assert stage0["hf_overrides"]["duplex_incremental_fbank"] is False
    assert stage2["async_scheduling"] is False
    assert stage2["env"]["VLLM_OMNI_STAGE_IDLE_WAIT_S"] == "0.05"
    for stage in (stage0, stage1, stage2):
        assert {key: stage["env"][key] for key in _OMP_CAPS} == _OMP_CAPS
    # connectors are deep-merged: the base connector definition survives.
    connector = raw["connectors"]["connector_of_shared_memory"]
    assert connector["name"] == "SharedMemoryConnector"
    assert connector["extra"]["hift_graph_codec_chunk_frames"] == [25, 75]
    assert "connector_get_sleep_s" in connector["extra"]

    deploy = load_deploy_config(_DEPLOY_DIR / "minicpmo_4_5_dxsched.yaml")
    assert deploy.duplex_session.pacing == DuplexPacingConfig()
    assert deploy.duplex_session.barge_cut_on_model_yield is False
    # Code2Wav bucket switches are listed off.
    assert connector["extra"]["code2wav_bucket_stats"] is False
    assert connector["extra"]["cfm_cross_turn_buckets"] is False
    assert connector["extra"]["cfm_row_offset_merge"] is False


def test_dxsched_profile_values_match_the_code_defaults() -> None:
    raw = resolve_deploy_yaml(_DEPLOY_DIR / "minicpmo_4_5_dxsched.yaml")["duplex_session"]
    runtime_defaults = DuplexSessionRuntimeConfig()
    barge_names = {name for name in raw if name.startswith("barge_")}
    assert barge_names == {f.name for f in fields(DuplexSessionRuntimeConfig) if f.name.startswith("barge_")}
    for name in barge_names:
        assert raw[name] == getattr(runtime_defaults, name), name
    pacing_defaults = DuplexPacingConfig()
    assert set(raw["pacing"]) == {f.name for f in fields(DuplexPacingConfig)}
    for name, value in raw["pacing"].items():
        assert value == getattr(pacing_defaults, name), name


def test_duplex_session_pacing_parses_from_a_mapping(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text(
        "duplex_session:\n  pacing:\n    enabled: true\n    onset_lead_max_s: 0.9\n    fire_grid_ms: 250\nstages: []\n",
        encoding="utf-8",
    )
    pacing = load_deploy_config(deploy_path).duplex_session.pacing
    assert isinstance(pacing, DuplexPacingConfig)
    assert pacing.enabled is True and pacing.onset_lead_max_s == 0.9 and pacing.fire_grid_ms == 250
    with pytest.raises(FrozenInstanceError):
        pacing.enabled = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("pacing:\n    onset_lead_max_s: 1.0", "onset_lead_max_s must be < 1.0"),
        ("pacing:\n    onset_lead_max_s: -0.1", "onset_lead_max_s must be a non-negative number"),
        ("pacing:\n    critical_stall_s: -1", "critical_stall_s must be a non-negative number"),
        ("pacing:\n    fire_grid_ms: 300", "fire_grid_ms must be 0 or a positive divisor of 1000"),
        ("pacing:\n    fire_grid_ms: -250", "fire_grid_ms must be 0 or a positive divisor of 1000"),
        ("pacing:\n    enabled: 1", "pacing.enabled must be a boolean"),
        ("pacing:\n    client_prebuffer_s: 0.5", "client_prebuffer_s"),
        ("pacing: 3", "pacing must be a mapping"),
        ("barge_cut_on_model_yield: 1", "barge_cut_on_model_yield must be a boolean"),
        ("barge_arm_min_silence_s: -0.5", "barge_arm_min_silence_s must be a non-negative number"),
        ("barge_arm_min_speech_s: -0.1", "barge_arm_min_speech_s must be a non-negative number"),
    ],
)
def test_duplex_session_rejects_invalid_dxsched_values(tmp_path, body: str, match: str) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text(f"duplex_session:\n  {body}\nstages: []\n", encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=match):
        load_deploy_config(deploy_path)


def test_dxsched_profile_passes_codec_deadline_to_stage2_with_the_code_defaults() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.codec_deadline import CodecDeadlineConfig

    deploy = load_deploy_config(_DEPLOY_DIR / "minicpmo_4_5_dxsched.yaml")
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)
    raw = stage2.engine_extras["additional_config"]["codec_deadline"]
    assert CodecDeadlineConfig.from_additional_config({"codec_deadline": raw}) == CodecDeadlineConfig()
    assert set(raw).issubset({f.name for f in fields(CodecDeadlineConfig)})
    assert stage2.env["VLLM_OMNI_STAGE_IDLE_WAIT_S"] == "0.05"


def test_dxsched_on_profile_turns_on_the_measured_switches_only() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.codec_deadline import CodecDeadlineConfig

    raw = resolve_deploy_yaml(_DEPLOY_DIR / "minicpmo_4_5_dxsched_on.yaml")
    base = resolve_deploy_yaml(_DEPLOY_DIR / "minicpmo_4_5_dxsched.yaml")
    stage0, stage1, stage2 = (_stage(raw, stage_id) for stage_id in (0, 1, 2))
    assert stage0["hf_overrides"] == {
        **_stage(base, 0)["hf_overrides"],
        "duplex_incremental_fbank": True,
        "duplex_audio_encoder_cuda_graph_batch_sizes_from_sessions": True,
        "duplex_audio_encoder_resident_slots": 20,
        "vision_fused_layers": True,
        "vision_cuda_graph": True,
    }
    # Stage env and the S2 scheduling mode are inherited untouched.
    for stage_id, stage in ((0, stage0), (1, stage1), (2, stage2)):
        assert stage["env"] == _stage(base, stage_id)["env"]
    assert stage2["async_scheduling"] is False
    extra = raw["connectors"]["connector_of_shared_memory"]["extra"]
    base_extra = base["connectors"]["connector_of_shared_memory"]["extra"]
    assert extra == {
        **base_extra,
        "cfm_row_offset_merge": True,
        "cfm_drop_upstream_att_buffer": True,
        "cfm_precapture_empty_cache": True,
        "token2wav_s3tokenizer_device": "cpu",
    }
    assert extra["cfm_cross_turn_buckets"] is False

    deploy = load_deploy_config(_DEPLOY_DIR / "minicpmo_4_5_dxsched_on.yaml")
    session = deploy.duplex_session
    assert session.barge_cut_on_model_yield is True
    assert session.pacing == DuplexPacingConfig(
        enabled=True, onset_lead_max_s=0.9, onset_skip_response_wait=True, fire_grid_ms=250, idle_grid=True
    )
    codec = next(stage for stage in deploy.stages if stage.stage_id == 2).engine_extras["additional_config"]
    assert CodecDeadlineConfig.from_additional_config(codec) == CodecDeadlineConfig(enabled=True)


def test_h100_profile_has_all_multimodal_and_scheduling_optimizations() -> None:
    raw = resolve_deploy_yaml(_DEPLOY_DIR / "minicpmo_4_5_h100.yaml")
    stage0 = _stage(raw, 0)
    assert stage0["hf_overrides"]["vision_cuda_graph"] is True
    assert stage0["hf_overrides"]["vision_fused_layers"] is True
    assert stage0["hf_overrides"]["duplex_audio_encoder_cuda_graph_batch_sizes_from_sessions"] is True
    assert stage0["hf_overrides"]["duplex_audio_encoder_resident_slots"] == 20
    assert stage0["hf_overrides"]["duplex_incremental_fbank"] is True
    assert raw["duplex_session"]["max_sessions"] == 20
    assert raw["duplex_session"]["barge_cut_on_model_yield"] is True
    assert raw["duplex_session"]["pacing"]["enabled"] is True
    stage2 = _stage(raw, 2)
    assert stage2["engine_extras"]["additional_config"]["codec_deadline"]["enabled"] is True
