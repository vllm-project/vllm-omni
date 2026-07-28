# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

import pytest

from vllm_omni.diffusion.attention.parallel.cost_selector import (
    CalibrationPoint,
    EmpiricalCostModel,
    Interconnect,
    SPCostSelector,
    SPStrategy,
    SPWorkload,
    StrategyCapabilities,
)


def _model(*, ring_ms: float = 3.0) -> EmpiricalCostModel:
    points = []
    values = {
        SPStrategy.ULYSSES: (2.0, 8.0),
        SPStrategy.ALLGATHER_KV: (1.0, 9.0),
        SPStrategy.RING: (ring_ms, 6.0),
    }
    for strategy, (short_ms, long_ms) in values.items():
        points.extend(
            [
                CalibrationPoint(strategy, Interconnect.NVLINK, 2, 0.125, 2048, short_ms),
                CalibrationPoint(strategy, Interconnect.NVLINK, 2, 0.125, 8192, long_ms),
            ]
        )
    return EmpiricalCostModel(points)


def _workload(**overrides) -> SPWorkload:
    values = {
        "seq_len": 2048,
        "sp_degree": 2,
        "num_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 128,
        "interconnect": "nvlink",
    }
    values.update(overrides)
    return SPWorkload(**values)


def test_selects_lowest_cost_feasible_strategy():
    decision = SPCostSelector(_model()).select(_workload())
    assert decision.strategy == SPStrategy.ALLGATHER_KV
    assert decision.predicted_ms == 1.0
    assert not decision.rejected


def test_ring_divisibility_is_a_hard_constraint():
    decision = SPCostSelector(_model(ring_ms=0.1)).select(_workload(seq_len=2049))
    assert decision.strategy == SPStrategy.ALLGATHER_KV
    assert "divisible" in decision.rejected[SPStrategy.RING]


def test_ring_checks_post_sharding_length_with_auto_pad():
    model = EmpiricalCostModel(
        [
            CalibrationPoint(strategy, Interconnect.NVLINK, 4, 0.125, 14040, float(i + 1))
            for i, strategy in enumerate(SPStrategy)
        ]
    )
    decision = SPCostSelector(model).select(
        _workload(
            seq_len=14040,
            sp_degree=4,
            sequence_auto_pad=True,
        )
    )
    assert SPStrategy.RING in decision.rejected
    assert "local sequence" in decision.rejected[SPStrategy.RING]
    assert "3510" in decision.rejected[SPStrategy.RING]


def test_causal_rejects_allgather_kv():
    decision = SPCostSelector(_model()).select(_workload(causal=True))
    assert SPStrategy.ALLGATHER_KV in decision.rejected
    assert decision.strategy == SPStrategy.ULYSSES


def test_attention_mask_rejects_ring():
    decision = SPCostSelector(_model(ring_ms=0.1)).select(_workload(has_attention_mask=True))
    assert SPStrategy.RING in decision.rejected


def test_strict_ulysses_requires_divisible_kv_heads():
    points = [
        CalibrationPoint(strategy, Interconnect.NVLINK, 4, 0.125, 2048, float(i + 1))
        for i, strategy in enumerate(SPStrategy)
    ]
    decision = SPCostSelector(EmpiricalCostModel(points)).select(
        _workload(sp_degree=4, num_kv_heads=2)
    )
    assert SPStrategy.ULYSSES in decision.rejected


def test_missing_backend_is_rejected():
    caps = StrategyCapabilities(allgather_kv=False)
    decision = SPCostSelector(_model(), caps).select(_workload())
    assert decision.strategy == SPStrategy.ULYSSES
    assert "not available" in decision.rejected[SPStrategy.ALLGATHER_KV]


def test_log_sequence_interpolation():
    # 2048 -> 1 ms and 8192 -> 4 ms: midpoint in log(seq) is 4096 -> 2 ms.
    points = [
        CalibrationPoint(SPStrategy.ULYSSES, Interconnect.NVLINK, 2, 1.0, 2048, 1.0),
        CalibrationPoint(SPStrategy.ULYSSES, Interconnect.NVLINK, 2, 1.0, 8192, 4.0),
    ]
    model = EmpiricalCostModel(points)
    assert model.predict_ms(
        SPStrategy.ULYSSES,
        _workload(seq_len=4096, num_kv_heads=32),
    ) == pytest.approx(2.0)


def test_loads_p1d_jsonl_schema(tmp_path):
    path = tmp_path / "profile.jsonl"
    path.write_text(
        json.dumps(
            {
                "strategy": "ring",
                "sp": 2,
                "f": 0.125,
                "seq": 2048,
                "interconnect": "nvlink",
                "p50_ms": 1.25,
                "status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = EmpiricalCostModel.from_jsonl(path)
    assert model.predict_ms(SPStrategy.RING, _workload()) == 1.25
