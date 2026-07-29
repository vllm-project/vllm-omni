# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost-based selection for diffusion sequence-parallel attention.

The selector is deliberately independent from process-group initialization and
attention kernels.  A deployment planner can therefore resolve a strategy
before workers start, while tests and benchmarks can compare the decision with
an oracle without importing CUDA or torch.distributed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class SPStrategy(StrEnum):
    ULYSSES = "ulysses"
    ALLGATHER_KV = "allgather_kv"
    RING = "ring"


class Interconnect(StrEnum):
    NVLINK = "nvlink"
    PCIE = "pcie"


@dataclass(frozen=True, slots=True)
class SPWorkload:
    """Shape and semantic constraints known before an SP deployment starts.

    ``seq_len`` is the global (pre-sharding) attention sequence length.
    """

    seq_len: int
    sp_degree: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    interconnect: Interconnect | str
    batch_size: int = 1
    dtype_bytes: int = 2
    causal: bool = False
    has_attention_mask: bool = False
    ulysses_mode: str = "strict"

    def __post_init__(self) -> None:
        positive = {
            "seq_len": self.seq_len,
            "sp_degree": self.sp_degree,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "batch_size": self.batch_size,
            "dtype_bytes": self.dtype_bytes,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"SP workload dimensions must be positive: {', '.join(invalid)}")
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads must be divisible by num_kv_heads, got {self.num_heads} and {self.num_kv_heads}"
            )
        if self.ulysses_mode not in {"strict", "advanced_uaa"}:
            raise ValueError(f"ulysses_mode must be 'strict' or 'advanced_uaa', got {self.ulysses_mode!r}")
        object.__setattr__(self, "interconnect", Interconnect(self.interconnect))

    @property
    def kv_ratio(self) -> float:
        return self.num_kv_heads / self.num_heads


@dataclass(frozen=True, slots=True)
class StrategyCapabilities:
    """Capabilities of the concrete implementations in the current build."""

    ulysses: bool = True
    allgather_kv: bool = True
    ring: bool = True
    allgather_kv_causal: bool = False
    ring_attention_mask: bool = False

    def enabled(self, strategy: SPStrategy) -> bool:
        return {
            SPStrategy.ULYSSES: self.ulysses,
            SPStrategy.ALLGATHER_KV: self.allgather_kv,
            SPStrategy.RING: self.ring,
        }[strategy]


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    strategy: SPStrategy
    interconnect: Interconnect
    sp_degree: int
    kv_ratio: float
    seq_len: int
    latency_ms: float

    def __post_init__(self) -> None:
        if self.sp_degree <= 1:
            raise ValueError("calibration sp_degree must be > 1")
        if not 0 < self.kv_ratio <= 1:
            raise ValueError("calibration kv_ratio must be in (0, 1]")
        if self.seq_len <= 0 or self.latency_ms <= 0:
            raise ValueError("calibration seq_len and latency_ms must be positive")


class StrategyCostModel(Protocol):
    def predict_ms(self, strategy: SPStrategy, workload: SPWorkload) -> float: ...


class EmpiricalCostModel:
    """Log-sequence interpolation over an offline calibration table.

    No profiling occurs in the serving hot path.  For a new sequence length,
    latency is interpolated in log(seq)-log(time) space.  If the model's GQA
    ratio was not calibrated exactly, predictions from the two nearest ratios
    are linearly interpolated.
    """

    def __init__(self, points: Iterable[CalibrationPoint]) -> None:
        self._points = tuple(points)
        if not self._points:
            raise ValueError("at least one calibration point is required")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EmpiricalCostModel:
        """Load either selector calibration rows or the #5092 P1-D JSONL."""
        points: list[CalibrationPoint] = []
        with Path(path).open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status", "ok") != "ok":
                    continue
                try:
                    strategy = SPStrategy(row["strategy"])
                    points.append(
                        CalibrationPoint(
                            strategy=strategy,
                            interconnect=Interconnect(row["interconnect"]),
                            sp_degree=int(row["sp_degree"] if "sp_degree" in row else row["sp"]),
                            kv_ratio=float(row["kv_ratio"] if "kv_ratio" in row else row["f"]),
                            seq_len=int(row["seq_len"] if "seq_len" in row else row["seq"]),
                            latency_ms=float(row["latency_ms"] if "latency_ms" in row else row["p50_ms"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid calibration row {line_number} in {path}: {exc}") from exc
        return cls(points)

    def predict_ms(self, strategy: SPStrategy, workload: SPWorkload) -> float:
        matching = [
            p
            for p in self._points
            if p.strategy == strategy and p.interconnect == workload.interconnect and p.sp_degree == workload.sp_degree
        ]
        if not matching:
            raise LookupError(
                "no calibration for "
                f"strategy={strategy}, interconnect={workload.interconnect}, "
                f"sp_degree={workload.sp_degree}"
            )

        ratios = sorted({p.kv_ratio for p in matching})
        lower = max((ratio for ratio in ratios if ratio <= workload.kv_ratio), default=ratios[0])
        upper = min((ratio for ratio in ratios if ratio >= workload.kv_ratio), default=ratios[-1])
        lower_ms = self._predict_at_ratio(matching, lower, workload.seq_len)
        if math.isclose(lower, upper):
            return lower_ms
        upper_ms = self._predict_at_ratio(matching, upper, workload.seq_len)
        weight = (workload.kv_ratio - lower) / (upper - lower)
        return lower_ms + weight * (upper_ms - lower_ms)

    @staticmethod
    def _predict_at_ratio(points: list[CalibrationPoint], ratio: float, seq_len: int) -> float:
        curve = sorted((p for p in points if math.isclose(p.kv_ratio, ratio)), key=lambda p: p.seq_len)
        exact = next((p for p in curve if p.seq_len == seq_len), None)
        if exact is not None:
            return exact.latency_ms
        if len(curve) < 2:
            # A single point can still provide a conservative linear-in-sequence
            # extrapolation. Production profiles should contain at least two.
            return curve[0].latency_ms * seq_len / curve[0].seq_len

        pair = None
        for left, right in zip(curve, curve[1:]):
            if left.seq_len <= seq_len <= right.seq_len:
                pair = left, right
                break
        if pair is None:
            pair = (curve[0], curve[1]) if seq_len < curve[0].seq_len else (curve[-2], curve[-1])
        left, right = pair
        weight = (math.log(seq_len) - math.log(left.seq_len)) / (math.log(right.seq_len) - math.log(left.seq_len))
        log_ms = math.log(left.latency_ms) + weight * (math.log(right.latency_ms) - math.log(left.latency_ms))
        return math.exp(log_ms)


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    strategy: SPStrategy
    predicted_ms: float
    costs_ms: dict[SPStrategy, float]
    rejected: dict[SPStrategy, str]


class SPCostSelector:
    """Filter illegal strategies, then select the lowest predicted latency."""

    def __init__(
        self,
        cost_model: StrategyCostModel,
        capabilities: StrategyCapabilities | None = None,
    ) -> None:
        self._cost_model = cost_model
        self._capabilities = capabilities or StrategyCapabilities()

    def select(self, workload: SPWorkload) -> StrategyDecision:
        costs: dict[SPStrategy, float] = {}
        rejected: dict[SPStrategy, str] = {}
        for strategy in SPStrategy:
            reason = self._infeasible_reason(strategy, workload)
            if reason is not None:
                rejected[strategy] = reason
                continue
            try:
                costs[strategy] = self._cost_model.predict_ms(strategy, workload)
            except LookupError as exc:
                rejected[strategy] = str(exc)

        if not costs:
            details = "; ".join(f"{strategy}: {reason}" for strategy, reason in rejected.items())
            raise RuntimeError(f"no feasible calibrated SP strategy: {details}")
        selected = min(costs, key=costs.__getitem__)
        return StrategyDecision(
            strategy=selected,
            predicted_ms=costs[selected],
            costs_ms=costs,
            rejected=rejected,
        )

    def _infeasible_reason(self, strategy: SPStrategy, workload: SPWorkload) -> str | None:
        caps = self._capabilities
        if not caps.enabled(strategy):
            return "implementation is not available in this build"
        if strategy == SPStrategy.ULYSSES:
            if workload.ulysses_mode == "strict" and (
                workload.num_heads % workload.sp_degree or workload.num_kv_heads % workload.sp_degree
            ):
                return "strict Ulysses requires Q and KV heads divisible by SP degree"
            return None
        if strategy == SPStrategy.ALLGATHER_KV:
            if workload.causal and not caps.allgather_kv_causal:
                return "AllGather-KV does not support causal attention"
            return None
        if workload.seq_len % workload.sp_degree:
            return "Ring requires global sequence length divisible by SP degree"
        if workload.has_attention_mask and not caps.ring_attention_mask:
            return "Ring does not support attention masks"
        return None
