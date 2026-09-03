# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for typed deployment-only runtime configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tests.helpers.stage_config import get_deploy_duplex_max_sessions
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
    assert deploy.duplex_session.completed_append_cache_size == 256
    with pytest.raises(FrozenInstanceError):
        deploy.duplex_session.idle_ttl_s = 1.0  # type: ignore[misc]


def test_duplex_session_runtime_accepts_disabled_idle_expiry(tmp_path) -> None:
    deploy_path = tmp_path / "duplex.yaml"
    deploy_path.write_text("duplex_session:\n  idle_ttl_s: null\nstages: []\n", encoding="utf-8")

    deploy = load_deploy_config(deploy_path)

    assert deploy.duplex_session.idle_ttl_s is None


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
        ("minicpmo_4_5.yaml", 4),
        ("minicpmo_4_5_8x4090.yaml", 1),
        ("minicpmo_4_5_3gpu_stage1_replicas.yaml", 4),
    ],
)
def test_deploy_duplex_max_sessions_tracks_the_deploy_config(deploy_yaml: str, expected: int) -> None:
    # Guards the admission probe. A capacity edit, a config that declares none,
    # or an overlay inheriting one from its base must surface here rather than
    # as another nightly duplex admission timeout.
    assert get_deploy_duplex_max_sessions(deploy_yaml) == expected
