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
