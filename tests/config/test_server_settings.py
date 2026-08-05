# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ServerSettings env-var parameterization."""

import pytest

from vllm_omni.config.server_settings import ServerSettings

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ENV_VARS = [
    "VLLM_OMNI_SERVER_STORAGE__PATH",
    "VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY",
    "VLLM_OMNI_SERVER_STORAGE__FILE_TTL",
    "VLLM_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL",
    "VLLM_OMNI_STORAGE_PATH",
    "VLLM_OMNI_STORAGE_MAX_CONCURRENCY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_no_env_set():
    storage = ServerSettings().storage

    assert storage.path == "/tmp/storage"
    assert storage.file_concurrency == 4
    assert storage.file_ttl is None
    assert storage.ttl_sweep_interval is None


def test_nested_env_vars_populate_storage(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", "/var/storage")
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY", "8")
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__FILE_TTL", "120")
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL", "30")

    storage = ServerSettings().storage

    assert storage.path == "/var/storage"
    assert storage.file_concurrency == 8
    assert storage.file_ttl == 120
    assert storage.ttl_sweep_interval == 30


def test_ttl_sweep_interval_defaults_when_ttl_set(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__FILE_TTL", "60")

    storage = ServerSettings().storage

    assert storage.file_ttl == 60
    assert storage.ttl_sweep_interval == 300


def test_deprecated_path_alias_populates_and_warns(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_STORAGE_PATH", "/legacy/path")

    with pytest.warns(DeprecationWarning, match="VLLM_OMNI_STORAGE_PATH"):
        storage = ServerSettings().storage

    assert storage.path == "/legacy/path"


def test_deprecated_concurrency_alias_populates_and_warns(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_STORAGE_MAX_CONCURRENCY", "16")

    with pytest.warns(DeprecationWarning, match="VLLM_OMNI_STORAGE_MAX_CONCURRENCY"):
        storage = ServerSettings().storage

    assert storage.file_concurrency == 16


def test_new_env_var_wins_over_deprecated_alias(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", "/new/path")
    monkeypatch.setenv("VLLM_OMNI_STORAGE_PATH", "/legacy/path")

    with pytest.warns(DeprecationWarning, match="VLLM_OMNI_STORAGE_PATH"):
        storage = ServerSettings().storage

    assert storage.path == "/new/path"
