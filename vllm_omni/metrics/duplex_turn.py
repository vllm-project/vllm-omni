# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-assistant-turn log tables for native duplex.

A duplex session keeps one long-lived stage-0 request, so chat generate()
finalize never runs for that id. Each turn owns its own aggregator keyed by
response_id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import cast

from vllm.logger import init_logger

from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats, StageStats

logger = init_logger(__name__)

_DUPLEX_RESOURCE_PREFIX = "duplex-s."


def is_duplex_resource_request_id(request_id: str | None) -> bool:
    """Return whether request_id is a native duplex stage resource id."""
    if not request_id or not request_id.startswith(_DUPLEX_RESOURCE_PREFIX):
        return False
    parts = request_id.split(".")
    return len(parts) == 8 and parts[2] == "i" and parts[4] == "e" and parts[6] == "r"


def _normalize_finished_reason(reason: str) -> str:
    return "stop" if reason == "stop" else "abort"


@dataclass
class DuplexTurnMetrics:
    """Metrics owner for one duplex assistant turn (one response_id)."""

    request_id: str
    response_id: str
    turn_id: int | None = None
    arrival_ts: float = 0.0
    aggregator: OrchestratorAggregator | None = None
    finalized: bool = False
    finished_reason: str | None = None


def _copy_stage_request_stats(
    metrics: StageRequestStats,
    *,
    stage_id: int,
    request_id: str,
    final_output_type: str | None,
) -> StageRequestStats:
    stage_stats = metrics.stage_stats
    if isinstance(stage_stats, StageStats):
        stage_stats = replace(stage_stats)
    else:
        stage_stats = StageStats()
    return replace(
        metrics,
        stage_id=stage_id,
        request_id=request_id,
        final_output_type=final_output_type or metrics.final_output_type,
        stage_stats=stage_stats,
        inter_output_latencies_ms=list(metrics.inter_output_latencies_ms or []),
        vllm_itls_ms=list(metrics.vllm_itls_ms or []),
        pipeline_timings=dict(metrics.pipeline_timings) if metrics.pipeline_timings else None,
        diffusion_metrics=dict(metrics.diffusion_metrics) if metrics.diffusion_metrics else {},
    )


def accumulate_turn_stage_metrics(
    turn: DuplexTurnMetrics,
    stage_id: int,
    metrics: StageRequestStats,
    *,
    final_output_type: str | None = None,
) -> None:
    """Append one stage snapshot; segments collapse at finalize."""
    aggregator = turn.aggregator
    if aggregator is None or turn.finalized:
        return
    stats = _copy_stage_request_stats(
        metrics,
        stage_id=stage_id,
        request_id=turn.response_id,
        final_output_type=final_output_type,
    )
    aggregator.on_stage_metrics(stage_id, turn.response_id, stats, final_output_type)
    now = time.time()
    if 0 <= stage_id < aggregator.num_stages:
        first_ts = cast(list[float | None], aggregator.stage_first_ts)
        last_ts = cast(list[float | None], aggregator.stage_last_ts)
        if first_ts[stage_id] is None:
            first_ts[stage_id] = now
        last_ts[stage_id] = max(last_ts[stage_id] or 0.0, now)


def _collapse_stage_events(aggregator: OrchestratorAggregator, request_id: str) -> None:
    events = aggregator.stage_events.get(request_id) or []
    if len(events) <= 1:
        return
    merged_rows: dict[int, dict] = {}
    first_evt: dict[int, StageRequestStats] = {}
    for evt in events:
        sid = int(evt.stage_id) if evt.stage_id is not None else -1
        first_evt.setdefault(sid, evt)
        merged_rows[sid] = OrchestratorAggregator._merge_stage_metric_event(merged_rows.get(sid), evt)
    collapsed: list[StageRequestStats] = []
    field_aliases = {"audio_frames": "audio_generated_frames"}
    for sid in sorted(merged_rows):
        base = first_evt[sid]
        for key, value in merged_rows[sid].items():
            attr = field_aliases.get(key, key)
            if attr != "stage_id" and hasattr(base, attr):
                setattr(base, attr, value)
        collapsed.append(base)
    aggregator.stage_events[request_id] = collapsed


def finalize_duplex_turn_metrics(turn: DuplexTurnMetrics, *, reason: str) -> bool:
    """Finalize one turn once. Returns True when a log summary was attempted."""
    if turn.finalized:
        return False
    turn.finalized = True
    turn.finished_reason = _normalize_finished_reason(reason)
    aggregator = turn.aggregator
    if aggregator is None:
        return False
    try:
        e2e_key = turn.response_id
        _collapse_stage_events(aggregator, e2e_key)
        if e2e_key not in aggregator.e2e_done:
            final_stage = 0
            if isinstance(aggregator.final_stage_id_for_e2e, int):
                final_stage = aggregator.final_stage_id_for_e2e
            elif aggregator.num_stages > 0:
                final_stage = aggregator.num_stages - 1
            aggregator.on_finalize_request(final_stage, e2e_key, turn.arrival_ts or aggregator.wall_start_ts)
        if aggregator.log_stats:
            logger.info(
                "[OmniTiming] req=%s response=%s turn=%s reason=%s",
                turn.request_id,
                turn.response_id,
                turn.turn_id,
                turn.finished_reason,
            )
            aggregator.build_and_log_summary()
    except Exception:
        logger.exception(
            "Failed to log duplex turn summary req=%s response=%s",
            turn.request_id,
            turn.response_id,
        )
    return True


def make_turn_aggregator(*, num_stages: int, log_stats: bool, wall_start_ts: float) -> OrchestratorAggregator:
    return OrchestratorAggregator(
        max(1, int(num_stages)),
        log_stats,
        wall_start_ts,
        max(0, int(num_stages) - 1),
    )


__all__ = [
    "DuplexTurnMetrics",
    "accumulate_turn_stage_metrics",
    "finalize_duplex_turn_metrics",
    "is_duplex_resource_request_id",
    "make_turn_aggregator",
]
