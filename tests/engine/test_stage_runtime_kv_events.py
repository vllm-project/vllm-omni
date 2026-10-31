# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for per-replica KV-event publisher diversification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.config.kv_events import KVEventsConfig

from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _kv_events_config(**overrides) -> KVEventsConfig:
    fields = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": "tcp://*:5556",
        "replay_endpoint": "tcp://*:5559",
        "topic": "kv@test:8000@model",
    }
    fields.update(overrides)
    return KVEventsConfig(**fields)


def _replica_metadata(replica_id: int, stage_id: int = 0) -> StageMetadata:
    return SimpleNamespace(replica_id=replica_id, stage_id=stage_id)


def test_replica_zero_leaves_kv_events_config_unchanged():
    stage_vllm_config = SimpleNamespace(kv_events_config=_kv_events_config(), kv_transfer_config=None)

    result = StageRuntime._prepare_replica_vllm_config(
        stage_vllm_config, replica_cfg=None, replica_metadata=_replica_metadata(0), native_kv=False
    )

    assert result.kv_events_config.endpoint == "tcp://*:5556"
    assert result.kv_events_config.replay_endpoint == "tcp://*:5559"
    assert result.kv_events_config.topic == "kv@test:8000@model-r0"


def test_distinct_replicas_get_distinct_endpoints_and_topics():
    stage_vllm_config = SimpleNamespace(kv_events_config=_kv_events_config(), kv_transfer_config=None)

    replica0 = StageRuntime._prepare_replica_vllm_config(
        stage_vllm_config, replica_cfg=None, replica_metadata=_replica_metadata(0), native_kv=False
    )
    replica1 = StageRuntime._prepare_replica_vllm_config(
        stage_vllm_config, replica_cfg=None, replica_metadata=_replica_metadata(1), native_kv=False
    )

    assert replica0.kv_events_config.endpoint != replica1.kv_events_config.endpoint
    assert replica0.kv_events_config.replay_endpoint != replica1.kv_events_config.replay_endpoint
    assert replica0.kv_events_config.topic != replica1.kv_events_config.topic
    # The source config is untouched: each call deep-copies before mutating.
    assert stage_vllm_config.kv_events_config.topic == "kv@test:8000@model"


def test_disabled_kv_events_config_is_not_mutated():
    stage_vllm_config = SimpleNamespace(
        kv_events_config=_kv_events_config(enable_kv_cache_events=False), kv_transfer_config=None
    )

    result = StageRuntime._prepare_replica_vllm_config(
        stage_vllm_config, replica_cfg=None, replica_metadata=_replica_metadata(1), native_kv=False
    )

    assert result is stage_vllm_config


def test_no_kv_events_config_and_no_native_kv_is_a_noop():
    stage_vllm_config = SimpleNamespace(kv_events_config=None, kv_transfer_config=None)

    result = StageRuntime._prepare_replica_vllm_config(
        stage_vllm_config, replica_cfg=None, replica_metadata=_replica_metadata(2), native_kv=False
    )

    assert result is stage_vllm_config


def test_none_stage_vllm_config_returns_none():
    assert (
        StageRuntime._prepare_replica_vllm_config(
            None, replica_cfg=None, replica_metadata=_replica_metadata(0), native_kv=False
        )
        is None
    )
