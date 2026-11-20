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
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin, reject_pd_until_runtime_ready

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


@pytest.mark.parametrize("bad_role", ["prefll", "Prefill", "kv_producer", 1])
def test_unknown_pd_role_is_rejected_with_stage_id_and_allowed_values(bad_role):
    """A misspelled role must not turn the stage into an ordinary one: both
    detection and the startup guard reject it, naming the declared stage id."""
    stages = [
        _typed_stage(4, bad_role),
        _typed_stage(5, PDRole.DECODE, input_sources=(4,)),
    ]
    expected = rf"Stage 4 has unknown pd_role {bad_role!r}; allowed values are 'prefill', 'decode'"

    with pytest.raises(ValueError, match=expected):
        PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages)
    with pytest.raises(ValueError, match=expected):
        reject_pd_until_runtime_ready(stages)


@pytest.mark.parametrize("bad_role", ["prefll", ""], ids=["typo", "empty-string"])
def test_guard_rejects_an_unknown_pd_role_when_no_other_stage_declares_one(bad_role):
    """The guard validates roles itself: with no other role declared it never
    reaches pair detection, and the stage must not launch as an ordinary one."""
    stages = [_typed_stage(4, bad_role), _typed_stage(5, None, input_sources=(4,))]

    with pytest.raises(ValueError, match=rf"Stage 4 has unknown pd_role {bad_role!r}; allowed values are"):
        reject_pd_until_runtime_ready(stages)


def test_guard_passes_when_no_stage_declares_a_pd_role():
    stages = [
        _typed_stage(0, None),
        SimpleNamespace(stage_id=1, is_prefill_only=False, is_decode_only=False, engine_input_source=[0]),
    ]

    assert reject_pd_until_runtime_ready(stages) is None


def test_guard_names_declared_stage_ids_not_positions():
    stages = [
        _typed_stage(3, PDRole.PREFILL),
        _typed_stage(7, PDRole.DECODE, input_sources=(3,)),
    ]

    assert PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages) == (0, 1)
    with pytest.raises(NotImplementedError, match=r"prefill stage 3, decode stage 7.*KV handoff.*single host only"):
        reject_pd_until_runtime_ready(stages)


def _lone_prefill_stages():
    return [_typed_stage(2, PDRole.PREFILL), _typed_stage(3, None, input_sources=(2,))]


def _lone_decode_stages():
    return [_typed_stage(2, None), _typed_stage(3, PDRole.DECODE, input_sources=(2,))]


_HALF_DECLARED_CASES = [
    pytest.param(
        _lone_prefill_stages,
        r"Stage\(s\) \[2\] declare the prefill role, but no stage declares the decode role",
        id="lone-prefill",
    ),
    pytest.param(
        _lone_decode_stages,
        r"Stage\(s\) \[3\] declare the decode role, but no stage declares the prefill role",
        id="lone-decode",
    ),
]


@pytest.mark.parametrize(("build_stages", "expected"), _HALF_DECLARED_CASES)
def test_guard_refuses_a_half_declared_pair(build_stages, expected):
    stages = build_stages()

    assert PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages) is None
    with pytest.raises(NotImplementedError, match=expected + r".*KV handoff"):
        reject_pd_until_runtime_ready(stages)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        pytest.param(
            "is_prefill_only",
            r"Stage\(s\) \[2\] declare the prefill role, but no stage declares the decode role",
            id="is_prefill_only",
        ),
        pytest.param(
            "is_decode_only",
            r"Stage\(s\) \[2\] declare the decode role, but no stage declares the prefill role",
            id="is_decode_only",
        ),
    ],
)
def test_guard_refuses_a_lone_legacy_marker(marker, expected):
    stages = [
        SimpleNamespace(stage_id=1, engine_input_source=[]),
        SimpleNamespace(stage_id=2, engine_input_source=[1], **{marker: True}),
    ]

    with pytest.raises(NotImplementedError, match=expected + r".*is_prefill_only / is_decode_only"):
        reject_pd_until_runtime_ready(stages)


def test_guard_refuses_prefill_and_decode_that_are_not_connected():
    stages = [
        _typed_stage(3, PDRole.PREFILL),
        _typed_stage(5, None),
        _typed_stage(7, PDRole.DECODE, input_sources=(5,)),
    ]

    with pytest.raises(
        NotImplementedError,
        match=r"Stage\(s\) \[3\] declare the prefill role and stage\(s\) \[7\] the decode role, "
        r"but no decode stage lists a prefill stage in its input sources.*ordinary stages.*single host only",
    ):
        reject_pd_until_runtime_ready(stages)


def test_multiple_pd_pairs_stay_a_value_error_that_names_declared_stage_ids():
    stages = [
        _typed_stage(10, PDRole.PREFILL),
        _typed_stage(11, PDRole.DECODE, input_sources=(10,)),
        _typed_stage(12, PDRole.PREFILL),
        _typed_stage(13, PDRole.DECODE, input_sources=(12,)),
    ]
    expected = r"Multiple PD pairs detected \(prefill stage, decode stage\): \[\(10, 11\), \(12, 13\)\]"

    with pytest.raises(ValueError, match=expected):
        PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages)
    with pytest.raises(ValueError, match=expected):
        reject_pd_until_runtime_ready(stages)


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


@pytest.mark.parametrize(("build_stages", "expected"), _HALF_DECLARED_CASES)
def test_engine_refuses_a_half_declared_pair_before_any_stage_launches(monkeypatch, build_stages, expected):
    with pytest.raises(NotImplementedError, match=expected):
        _init_engine_up_to_runtime_config(monkeypatch, build_stages())


def test_engine_rejects_an_unknown_pd_role_before_any_stage_launches(monkeypatch):
    stages = [_typed_stage(0, "prefll"), _typed_stage(1, PDRole.DECODE, input_sources=(0,))]

    with pytest.raises(ValueError, match=r"Stage 0 has unknown pd_role 'prefll'"):
        _init_engine_up_to_runtime_config(monkeypatch, stages)
