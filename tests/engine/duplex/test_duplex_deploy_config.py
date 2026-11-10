# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for typed deployment-only runtime configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tests.helpers.stage_config import get_deploy_config_path, get_deploy_duplex_max_sessions
from vllm_omni.config.stage_config import (
    DuplexSessionRuntimeConfig,
    load_deploy_config,
)

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
    assert deploy.duplex_session.replica_policy == "legacy"
    assert deploy.duplex_session.max_sessions_per_replica is None
    assert deploy.duplex_session.replica_groups is None
    assert deploy.duplex_session.worker_max_sessions == 1
    assert deploy.duplex_session.completed_append_cache_size == 256
    assert deploy.duplex_session.server_vad_model_path is None
    with pytest.raises(FrozenInstanceError):
        deploy.duplex_session.idle_ttl_s = 1.0  # type: ignore[misc]


def test_duplex_session_runtime_accepts_disabled_idle_expiry(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text("duplex_session:\n  idle_ttl_s: null\nstages: []\n", encoding="utf-8")

    deploy = load_deploy_config(deploy_path)

    assert deploy.duplex_session.idle_ttl_s is None


def test_duplex_session_runtime_loads_replica_placement_fields(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text(
        "session_mode: duplex\n"
        "duplex_session:\n"
        "  max_sessions: 9\n"
        "  replica_policy: least_sessions\n"
        "  max_sessions_per_replica: 3\n"
        "  replica_groups:\n"
        "    - {0: 0, 1: 0}\n"
        "    - {0: 1, 1: 0}\n"
        "stages: []\n",
        encoding="utf-8",
    )

    config = load_deploy_config(deploy_path).duplex_session

    assert config.replica_policy == "least_sessions"
    assert config.max_sessions == 9
    assert config.max_sessions_per_replica == 3
    assert config.worker_max_sessions == 3
    assert config.replica_groups == [{0: 0, 1: 0}, {0: 1, 1: 0}]


@pytest.mark.parametrize("policy", ["round_robin", "", None, True])
def test_duplex_replica_policy_rejects_unknown_values(policy) -> None:
    with pytest.raises(ValueError, match="duplex_session.replica_policy"):
        DuplexSessionRuntimeConfig(replica_policy=policy)


def test_duplex_least_sessions_requires_replica_capacity() -> None:
    with pytest.raises(ValueError, match="max_sessions_per_replica is required"):
        DuplexSessionRuntimeConfig(replica_policy="least_sessions")


@pytest.mark.parametrize("capacity", [0, -1, True, False, 1.5, "2"])
def test_duplex_replica_capacity_rejects_non_positive_or_non_integer_values(capacity) -> None:
    with pytest.raises(ValueError, match="max_sessions_per_replica must be an integer >= 1"):
        DuplexSessionRuntimeConfig(replica_policy="least_sessions", max_sessions_per_replica=capacity)


@pytest.mark.parametrize(
    "groups",
    [
        [],
        {0: 0},
        [{0: 0}, []],
        [{}],
        [{"0": 0}],
        [{True: 0}],
        [{-1: 0}],
        [{0: "0"}],
        [{0: True}],
        [{0: -1}],
    ],
)
def test_duplex_replica_groups_reject_malformed_mappings(groups) -> None:
    with pytest.raises(ValueError, match="duplex_session.replica_groups"):
        DuplexSessionRuntimeConfig(
            replica_policy="least_sessions",
            max_sessions_per_replica=1,
            replica_groups=groups,
        )


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
    ("deploy_yaml", "platform", "expected"),
    [
        ("minicpmo_4_5.yaml", "cuda", 32),
        ("minicpmo_4_5.yaml", "npu", 16),
        ("minicpmo_4_5_duplex_h200.yaml", "cuda", 32),
        ("minicpmo_4_5_8x4090.yaml", "cuda", 1),
        ("minicpmo_4_5_3gpu_stage1_replicas.yaml", "cuda", 32),
        ("minicpmo_4_5_stage2_replicas.yaml", "npu", 4),
    ],
)
def test_deploy_duplex_max_sessions_tracks_the_deploy_config(deploy_yaml: str, platform: str, expected: int) -> None:
    # Guards the admission probe. A capacity edit, a config that declares none,
    # or an overlay inheriting one from its base must surface here rather than
    # as another nightly duplex admission timeout.
    assert get_deploy_duplex_max_sessions(deploy_yaml, platform) == expected


@pytest.mark.parametrize(
    "patch",
    [
        {"history_asr_url": "file:///tmp/model", "history_asr_model": "asr"},
        {"history_asr_url": "http://localhost/transcribe"},
        {"history_calibration_timeout_ms": 0},
        {"history_calibration_timeout_ms": float("nan")},
        {"history_audio_max_bytes_per_session": True},
        {"history_asr_max_concurrency": 0},
        {"history_asr_max_concurrency": True},
        {"history_asr_max_concurrency": 1.5},
        {"history_ms_per_token": 0},
        {"history_ms_per_token": True},
        {"history_ms_per_token": "260"},
        {"history_ms_per_token": float("nan")},
        {"history_ms_per_token": float("inf")},
        {"history_ms_per_token": 260, "history_asr_url": "http://localhost/transcribe", "history_asr_model": "asr"},
    ],
)
def test_history_calibration_requires_bounded_server_configuration(patch):
    with pytest.raises(ValueError):
        DuplexSessionRuntimeConfig(**patch)


@pytest.mark.parametrize("platform,capacity", [("cuda", 32), ("npu", 16)])
def test_platform_session_admission_matches_worker(monkeypatch, platform: str, capacity: int) -> None:
    from vllm_omni.config.stage_config import _apply_platform_overrides
    from vllm_omni.engine.omni_engine_base import OmniEngineBase, _load_engine_deploy_config
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    path = get_deploy_config_path("minicpmo_4_5.yaml")
    admission = _load_engine_deploy_config(path).duplex_session
    worker = _apply_platform_overrides(load_deploy_config(path)).duplex_session
    assert admission == worker
    assert admission.max_sessions == capacity
    engine = object.__new__(OmniEngineBase)
    engine.deploy_config = None
    engine._deploy_config_source = None
    engine._set_pipeline_runtime_config(None, path)
    assert engine.duplex_session_config == admission


@pytest.mark.parametrize("override", ["[]", "{unknown: 4}", "{max_sessions: 0}"])
def test_invalid_platform_session_override(tmp_path, override: str) -> None:
    from vllm_omni.config.stage_config import resolve_platform_duplex_session

    path = tmp_path / "deploy.yaml"
    path.write_text(f"platforms:\n  npu:\n    duplex_session: {override}\nstages: []\n")
    with pytest.raises(ValueError):
        resolve_platform_duplex_session(load_deploy_config(path), "npu")


def test_platform_session_override_preserves_base_and_unset_fields(tmp_path) -> None:
    from vllm_omni.config.stage_config import resolve_platform_duplex_session

    path = tmp_path / "deploy.yaml"
    path.write_text(
        "duplex_session: {max_sessions: 32, idle_ttl_s: 600}\n"
        "platforms: {npu: {duplex_session: {max_sessions: 16}}}\nstages: []\n"
    )
    base = load_deploy_config(path)
    resolved = resolve_platform_duplex_session(base, "npu")
    assert resolved.max_sessions == 16
    assert resolved.idle_ttl_s == 600
    assert base.duplex_session.max_sessions == 32
    assert resolve_platform_duplex_session(base, "cuda") == base.duplex_session
