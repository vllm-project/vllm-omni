# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.attention.parallel.cost_selector import (
    CalibrationPoint,
    EmpiricalCostModel,
    Interconnect,
    SPCostSelector,
    SPStrategy,
    SPWorkload,
    StrategyCapabilities,
    resolve_auto_sp_strategy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _point(
    strategy: SPStrategy,
    *,
    sp_degree: int = 2,
    kv_ratio: float = 0.125,
    seq_len: int = 2048,
    batch_size: int = 1,
    head_dim: int = 128,
    dtype_bytes: int = 2,
    latency_ms: float = 1.0,
) -> CalibrationPoint:
    return CalibrationPoint(
        strategy=strategy,
        interconnect=Interconnect.NVLINK,
        sp_degree=sp_degree,
        kv_ratio=kv_ratio,
        seq_len=seq_len,
        batch_size=batch_size,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        latency_ms=latency_ms,
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
                _point(strategy, seq_len=2048, latency_ms=short_ms),
                _point(strategy, seq_len=8192, latency_ms=long_ms),
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
        "batch_size": 1,
        "dtype_bytes": 2,
    }
    values.update(overrides)
    return SPWorkload(**values)


def test_selects_lowest_cost_feasible_strategy():
    decision = SPCostSelector(_model()).select(_workload())
    assert decision.strategy == SPStrategy.ALLGATHER_KV
    assert decision.predicted_ms == 1.0
    assert not decision.rejected


def test_non_divisible_sequence_rejects_equal_shard_strategies():
    decision = SPCostSelector(_model(ring_ms=0.1)).select(_workload(seq_len=2049))
    assert decision.strategy == SPStrategy.ULYSSES
    assert "divisible" in decision.rejected[SPStrategy.ALLGATHER_KV]
    assert "divisible" in decision.rejected[SPStrategy.RING]


def test_causal_rejects_allgather_kv():
    decision = SPCostSelector(_model()).select(_workload(causal=True))
    assert SPStrategy.ALLGATHER_KV in decision.rejected
    assert decision.strategy == SPStrategy.ULYSSES


def test_attention_mask_rejects_ring():
    decision = SPCostSelector(_model(ring_ms=0.1)).select(_workload(has_attention_mask=True))
    assert SPStrategy.RING in decision.rejected


def test_strict_ulysses_requires_divisible_kv_heads():
    points = [_point(strategy, sp_degree=4, latency_ms=float(i + 1)) for i, strategy in enumerate(SPStrategy)]
    decision = SPCostSelector(EmpiricalCostModel(points)).select(_workload(sp_degree=4, num_kv_heads=2))
    assert SPStrategy.ULYSSES in decision.rejected


def test_missing_backend_is_rejected():
    caps = StrategyCapabilities(allgather_kv=False)
    decision = SPCostSelector(_model(), caps).select(_workload())
    assert decision.strategy == SPStrategy.ULYSSES
    assert "not available" in decision.rejected[SPStrategy.ALLGATHER_KV]


def test_log_sequence_interpolation():
    # 2048 -> 1 ms and 8192 -> 4 ms: midpoint in log(seq) is 4096 -> 2 ms.
    points = [
        _point(SPStrategy.ULYSSES, kv_ratio=1.0, seq_len=2048, latency_ms=1.0),
        _point(SPStrategy.ULYSSES, kv_ratio=1.0, seq_len=8192, latency_ms=4.0),
    ]
    model = EmpiricalCostModel(points)
    assert model.predict_ms(
        SPStrategy.ULYSSES,
        _workload(seq_len=4096, num_kv_heads=32),
    ) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("workload_override", "expected_dimension"),
    [
        ({"batch_size": 2}, "batch_size=2"),
        ({"head_dim": 64}, "head_dim=64"),
        ({"dtype_bytes": 4}, "dtype_bytes=4"),
    ],
)
def test_calibration_shape_mismatch_is_rejected(workload_override, expected_dimension):
    with pytest.raises(RuntimeError, match=expected_dimension):
        SPCostSelector(_model()).select(_workload(**workload_override))


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
                "batch": 1,
                "dim": 128,
                "dtype": "bf16",
                "p50_ms": 1.25,
                "status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = EmpiricalCostModel.from_jsonl(path)
    assert model.predict_ms(SPStrategy.RING, _workload()) == 1.25


def test_calibration_jsonl_requires_full_shape_identity(tmp_path):
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid calibration row"):
        EmpiricalCostModel.from_jsonl(path)


def test_auto_strategy_updates_degrees_before_group_initialization(tmp_path):
    profile = tmp_path / "profile.jsonl"
    rows = [
        {
            "strategy": strategy,
            "sp": 2,
            "f": 0.125,
            "seq": 2048,
            "interconnect": "nvlink",
            "batch": 1,
            "dim": 128,
            "dtype": "bf16",
            "p50_ms": latency,
        }
        for strategy, latency in (
            ("ulysses", 2.0),
            ("allgather_kv", 1.0),
            ("ring", 0.5),
        )
    ]
    profile.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = SimpleNamespace(
        sp_strategy="auto",
        sp_selector_profile=str(profile),
        sp_selector_workload={
            "seq_len": 2048,
            "sp_degree": 2,
            "num_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
            "interconnect": "nvlink",
            "batch_size": 1,
            "dtype_bytes": 2,
        },
        sp_selector_allow_ring=False,
        ulysses_mode="strict",
        sequence_parallel_size=2,
        ulysses_degree=1,
        ring_degree=1,
        allgather_degree=1,
    )

    decision = resolve_auto_sp_strategy(config)

    assert decision is not None
    assert decision.strategy == SPStrategy.ALLGATHER_KV
    assert config.ulysses_degree == 1
    assert config.ring_degree == 1
    assert config.allgather_degree == 2
    assert config.sequence_parallel_size == 2


def test_auto_strategy_requires_explicit_workload_identity():
    config = SimpleNamespace(
        sp_strategy="auto",
        sp_selector_profile="unused.jsonl",
        sp_selector_workload={"seq_len": 2048},
        sequence_parallel_size=2,
        ulysses_mode="strict",
    )

    with pytest.raises(ValueError, match=r"missing: .*batch_size.*dtype_bytes"):
        resolve_auto_sp_strategy(config)
