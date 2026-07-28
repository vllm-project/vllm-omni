# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate the SP cost selector against measured strategy latencies.

Example:
    python benchmarks/diffusion/sp_cost_selector_oracle.py \
        --calibration p1d_results.jsonl \
        --evaluation p1d_results.jsonl

Use separate calibration/evaluation files for a real holdout result.  The
``--leave-one-seq-out`` mode is a stricter interpolation stress test when only
one measured grid is available.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vllm_omni.diffusion.attention.parallel.cost_selector import (
    CalibrationPoint,
    EmpiricalCostModel,
    Interconnect,
    PhysicsInformedCostModel,
    SPCostSelector,
    SPStrategy,
    SPWorkload,
)


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and json.loads(line).get("status", "ok") == "ok"]


def _point(row: dict) -> CalibrationPoint:
    return CalibrationPoint(
        strategy=SPStrategy(row["strategy"]),
        interconnect=Interconnect(row["interconnect"]),
        sp_degree=int(row["sp_degree"] if "sp_degree" in row else row["sp"]),
        kv_ratio=float(row["kv_ratio"] if "kv_ratio" in row else row["f"]),
        seq_len=int(row["seq_len"] if "seq_len" in row else row["seq"]),
        latency_ms=float(row["latency_ms"] if "latency_ms" in row else row["p50_ms"]),
    )


def _cell_key(row: dict) -> tuple:
    return (
        row["interconnect"],
        int(row["sp_degree"] if "sp_degree" in row else row["sp"]),
        float(row["kv_ratio"] if "kv_ratio" in row else row["f"]),
        int(row["seq_len"] if "seq_len" in row else row["seq"]),
    )


def _workload(row: dict) -> SPWorkload:
    num_heads = int(row.get("hq", 32))
    kv_ratio = float(row["kv_ratio"] if "kv_ratio" in row else row["f"])
    return SPWorkload(
        seq_len=int(row["seq_len"] if "seq_len" in row else row["seq"]),
        sp_degree=int(row["sp_degree"] if "sp_degree" in row else row["sp"]),
        num_heads=num_heads,
        num_kv_heads=int(row.get("hkv", round(num_heads * kv_ratio))),
        head_dim=int(row.get("dim", row.get("head_dim", 128))),
        interconnect=row["interconnect"],
        batch_size=int(row.get("batch", row.get("batch_size", 1))),
    )


def evaluate(
    calibration_rows: list[dict],
    evaluation_rows: list[dict],
    *,
    leave_one_seq_out: bool,
    physics_informed: bool = False,
    pcie_sp2_ring_kv_tokens: float | None = None,
) -> dict:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in evaluation_rows:
        cells[_cell_key(row)].append(row)

    decisions = []
    for key, rows in sorted(cells.items()):
        workload = _workload(rows[0])
        if leave_one_seq_out:
            training = [_point(row) for row in calibration_rows if _cell_key(row)[3] != workload.seq_len]
        else:
            training = [_point(row) for row in calibration_rows]
        model_cls = PhysicsInformedCostModel if physics_informed else EmpiricalCostModel
        selector = SPCostSelector(
            model_cls(training),
            pcie_sp2_ring_kv_tokens=pcie_sp2_ring_kv_tokens,
        )
        decision = selector.select(workload)
        measured = {SPStrategy(row["strategy"]): _point(row).latency_ms for row in rows}
        feasible_measured = {strategy: ms for strategy, ms in measured.items() if strategy in decision.costs_ms}
        oracle = min(feasible_measured, key=feasible_measured.__getitem__)
        chosen_ms = feasible_measured[decision.strategy]
        oracle_ms = feasible_measured[oracle]
        decisions.append(
            {
                "interconnect": key[0],
                "sp": key[1],
                "f": key[2],
                "seq": key[3],
                "selected": decision.strategy,
                "oracle": oracle,
                "hit": decision.strategy == oracle,
                "regret": chosen_ms / oracle_ms - 1,
                "selected_ms": chosen_ms,
                "oracle_ms": oracle_ms,
                "predicted_ms": decision.predicted_ms,
            }
        )

    regrets = [row["regret"] for row in decisions]
    return {
        "cells": len(decisions),
        "winner_hits": sum(row["hit"] for row in decisions),
        "winner_hit_rate": sum(row["hit"] for row in decisions) / len(decisions),
        "mean_regret": sum(regrets) / len(regrets),
        "max_regret": max(regrets),
        "regret_over_5pct": sum(regret > 0.05 for regret in regrets),
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--leave-one-seq-out", action="store_true")
    parser.add_argument("--physics-informed", action="store_true")
    parser.add_argument("--pcie-sp2-ring-kv-tokens", type=float)
    args = parser.parse_args()
    result = evaluate(
        _read_rows(args.calibration),
        _read_rows(args.evaluation),
        leave_one_seq_out=args.leave_one_seq_out,
        physics_informed=args.physics_informed,
        pcie_sp2_ring_kv_tokens=args.pcie_sp2_ring_kv_tokens,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
