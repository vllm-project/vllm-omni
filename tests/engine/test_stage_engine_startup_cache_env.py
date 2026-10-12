# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for per-replica compile-cache isolation at spawn time.

Replicas of the same stage compile the same model concurrently, and vLLM
0.27's AOT compile (TritonBundler) races on the shared VLLM_CACHE_ROOT —
a replica then dies with 'Cubin file saved by TritonBundler not found'
(build 2954, Omni Perf Multi-Replica). Replica 0 must keep the default
shared cache; additional replicas must spawn with an isolated cache dir,
and the parent env must be restored afterwards.
"""

import os

import pytest

from vllm_omni.engine.stage_engine_startup import scoped_spawn_device_env

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_replica_zero_keeps_shared_cache_root(monkeypatch):
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    with scoped_spawn_device_env(None, None, stage_id=1, replica_id=0):
        assert "VLLM_CACHE_ROOT" not in os.environ
    assert "VLLM_CACHE_ROOT" not in os.environ


def test_additional_replica_gets_isolated_cache_root(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/tmp/base-cache")
    with scoped_spawn_device_env(None, None, stage_id=2, replica_id=1):
        assert os.environ["VLLM_CACHE_ROOT"] == os.path.join("/tmp/base-cache", "omni_stage2_replica1")
    assert os.environ["VLLM_CACHE_ROOT"] == "/tmp/base-cache"


def test_isolated_cache_root_defaults_without_env(monkeypatch):
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    with scoped_spawn_device_env(None, None, stage_id=0, replica_id=3):
        assert os.environ["VLLM_CACHE_ROOT"].endswith("omni_stage0_replica3")
        assert os.environ["VLLM_CACHE_ROOT"].startswith(os.path.expanduser("~/.cache/vllm"))
    assert "VLLM_CACHE_ROOT" not in os.environ


def test_missing_ids_leave_env_untouched(monkeypatch):
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    with scoped_spawn_device_env(None, None):
        assert "VLLM_CACHE_ROOT" not in os.environ
