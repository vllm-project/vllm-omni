# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PD pair detection and validation on structured (typed) stage configs.

``tests/entrypoints/test_pd_disaggregation.py`` is module-skipped, so these
checks live in their own CPU module.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.config.stage_config import PDRole
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _typed_stage(stage_id: int, pd_role, input_sources=(), kv_role: str | None = None, tp: int = 1):
    """Shape of ``BaseVllmOmniStageConfig``: no ``engine_args`` / ``engine_input_source``."""
    kv = None
    if kv_role is not None:
        kv = SimpleNamespace(kv_connector="MooncakeConnector", kv_role=kv_role)
    return SimpleNamespace(
        stage_id=stage_id,
        pd_role=pd_role,
        input_sources=list(input_sources),
        connector_config=SimpleNamespace(kv_transfer_config=kv),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


def _pd(stage_configs):
    pd = PDDisaggregationMixin()
    pd.stage_configs = stage_configs
    pd._pd_separation_pair = PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stage_configs)
    return pd


def test_detects_pair_from_pd_role_on_typed_configs():
    stages = [
        _typed_stage(0, PDRole.PREFILL),
        _typed_stage(1, PDRole.DECODE, input_sources=(0,)),
    ]

    assert PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages) == (0, 1)


def test_detects_pair_from_pd_role_string_values():
    stages = [
        _typed_stage(0, "prefill"),
        _typed_stage(1, "decode", input_sources=(0,)),
    ]

    assert PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages) == (0, 1)


def test_no_pair_without_roles():
    stages = [_typed_stage(0, None), _typed_stage(1, None, input_sources=(0,))]

    assert PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages) is None


def test_decode_must_consume_the_prefill_stage():
    stages = [
        _typed_stage(0, PDRole.PREFILL),
        _typed_stage(1, None),
        _typed_stage(2, PDRole.DECODE, input_sources=(1,)),
    ]

    assert PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages) is None


def test_legacy_boolean_markers_still_work():
    stages = [
        SimpleNamespace(stage_id=0, is_prefill_only=True, engine_input_source=[]),
        SimpleNamespace(stage_id=1, is_decode_only=True, engine_input_source=[0]),
    ]

    assert PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages) == (0, 1)


def test_validation_reads_typed_connector_and_parallel_configs():
    pd = _pd(
        [
            _typed_stage(0, PDRole.PREFILL, kv_role="kv_producer"),
            _typed_stage(1, PDRole.DECODE, input_sources=(0,), kv_role="kv_consumer"),
        ]
    )

    pd._validate_pd_separation_config()


def test_validation_rejects_typed_tp_mismatch():
    pd = _pd(
        [
            _typed_stage(0, PDRole.PREFILL, kv_role="kv_producer", tp=1),
            _typed_stage(1, PDRole.DECODE, input_sources=(0,), kv_role="kv_consumer", tp=2),
        ]
    )

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        pd._validate_pd_separation_config()


def test_validation_rejects_typed_stage_without_kv_transfer_config():
    pd = _pd(
        [
            _typed_stage(0, PDRole.PREFILL),
            _typed_stage(1, PDRole.DECODE, input_sources=(0,), kv_role="kv_consumer"),
        ]
    )

    with pytest.raises(ValueError, match="no 'kv_transfer_config'"):
        pd._validate_pd_separation_config()


def _mooncake_prefill(extra: dict | None = None):
    kv = {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer", "kv_connector_extra_config": extra or {}}
    return SimpleNamespace(
        stage_id=0,
        pd_role=PDRole.PREFILL,
        input_sources=[],
        connector_config=SimpleNamespace(kv_transfer_config=kv),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        # Same rule as the stage runtime: bootstrap_port (default 8998) + replica id.
        ({}, "http://127.0.0.1:8998"),
        ({"bootstrap_port": 9100, "mooncake_protocol": "tcp"}, "http://127.0.0.1:9100"),
        # The legacy key, which the stage runtime never reads, is ignored.
        ({"mooncake_bootstrap_port": 25201}, "http://127.0.0.1:8998"),
    ],
)
def test_connector_info_uses_the_stage_runtime_bootstrap_port(extra, expected):
    pd = _pd([_mooncake_prefill(extra), _typed_stage(1, PDRole.DECODE, input_sources=(0,))])

    assert pd._get_pd_connector_info() == {"prefill_bootstrap_addr": expected}


def test_connector_info_has_no_bootstrap_address_for_mooncake_store():
    prefill = _mooncake_prefill()
    prefill.connector_config.kv_transfer_config["kv_connector"] = "MooncakeStoreConnector"
    pd = _pd([prefill, _typed_stage(1, PDRole.DECODE, input_sources=(0,))])

    assert pd._get_pd_connector_info() == {}


def test_engine_reads_the_prefill_replica_bootstrap_address_and_engine_id():
    """The orchestrator's decode params use the per-replica values the stage
    runtime publishes on the prefill client's KV transfer config."""
    from vllm.config import KVTransferConfig

    from vllm_omni.engine.omni_engine_base import OmniEngineBase

    engine = OmniEngineBase.__new__(OmniEngineBase)
    engine.stage_configs = [
        _typed_stage(0, PDRole.PREFILL, kv_role="kv_producer"),
        _typed_stage(1, PDRole.DECODE, input_sources=(0,), kv_role="kv_consumer"),
    ]
    prefill_kv = KVTransferConfig(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        engine_id="thinker-s0-r0",
        kv_connector_extra_config={"bootstrap_port": 9100, "bootstrap_addr": "http://127.0.0.1:9100"},
    )
    engine.stage_clients = [
        SimpleNamespace(vllm_config=SimpleNamespace(kv_transfer_config=prefill_kv)),
        SimpleNamespace(vllm_config=None),
    ]

    assert engine._detect_pd_config() == {
        "pd_pair": (0, 1),
        "bootstrap_addr": "http://127.0.0.1:9100",
        "prefill_engine_id": "thinker-s0-r0",
    }


class _PastThePDGuardError(Exception):
    """Raised by the first engine init step that runs after the PD guard."""


def _init_engine_up_to_runtime_config(monkeypatch, stages) -> None:
    from vllm_omni.engine import omni_engine_base
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    def _past_the_guard(*_args):
        raise _PastThePDGuardError

    engine = object.__new__(AsyncOmniEngine)
    monkeypatch.setattr(omni_engine_base.StageConfigFactory, "get_pipeline_config", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_resolve_stage_configs", lambda *a, **k: (None, stages))
    monkeypatch.setattr(engine, "_set_pipeline_runtime_config", _past_the_guard)
    engine.__init__("test-model")


def test_engine_refuses_a_pd_pair_before_any_stage_launches(monkeypatch):
    """Until the prefill->decode KV handoff lands, a detected PD pair would fail
    every request at the handoff, so the engine stops right after config resolution."""
    stages = [
        _typed_stage(0, PDRole.PREFILL, kv_role="kv_producer"),
        _typed_stage(1, PDRole.DECODE, input_sources=(0,), kv_role="kv_consumer"),
    ]

    with pytest.raises(NotImplementedError, match=r"prefill stage 0, decode stage 1.*KV handoff"):
        _init_engine_up_to_runtime_config(monkeypatch, stages)


def test_engine_init_continues_without_a_pd_pair(monkeypatch):
    stages = [_typed_stage(0, None), _typed_stage(1, None, input_sources=(0,))]

    with pytest.raises(_PastThePDGuardError):
        _init_engine_up_to_runtime_config(monkeypatch, stages)
