# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-assistant-turn log tables for native duplex.

A duplex session keeps one long-lived stage-0 request, so chat generate()
finalize never runs for that id. Each turn owns its own aggregator keyed by
response_id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger

from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats, StageStats

if TYPE_CHECKING:
    from vllm_omni.entrypoints.client_request_state import ClientRequestState

logger = init_logger(__name__)

_DUPLEX_RESOURCE_PREFIX = "duplex-s."
_KNOWN_FINISHED_REASONS = frozenset({"stop", "barge_in", "cancel", "close", "abort", "error"})
_CANCEL_SOURCE_TO_FINISHED = {
    "barge_in": "barge_in",
    "session_close": "close",
    "disconnect": "close",
    "disconnect_grace_expired": "close",
    "timeout": "cancel",
    "new_response": "cancel",
    "output_audio_buffer_clear": "cancel",
    "input.cancel": "cancel",
    "response.cancel": "cancel",
    "client_cancelled": "cancel",
}


def is_duplex_resource_request_id(request_id: str | None) -> bool:
    """Return whether request_id is a native duplex stage resource id."""
    if not request_id or not request_id.startswith(_DUPLEX_RESOURCE_PREFIX):
        return False
    parts = request_id.split(".")
    return len(parts) == 8 and parts[2] == "i" and parts[4] == "e" and parts[6] == "r"


def _normalize_finished_reason(reason: str) -> str:
    if reason in _KNOWN_FINISHED_REASONS:
        return reason
    return "abort"


def finished_reason_for_cancel(reason: str) -> str:
    """Map a native cancel source onto a log-table finished_reason."""
    if reason in _KNOWN_FINISHED_REASONS:
        return reason
    return _CANCEL_SOURCE_TO_FINISHED.get(reason, "cancel")


@dataclass(frozen=True)
class PendingTurnStageMetric:
    """Stage snapshot that arrived before the assistant response began."""

    stage_id: int
    metrics: StageRequestStats
    final_output_type: str | None
    stage_submit_ts: float | None


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
    tx_baseline: dict[tuple[int, int, str], tuple[float, int]] = field(default_factory=dict)


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
    stage_submit_ts: float | None = None,
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
    first_value = stage_submit_ts if stage_submit_ts is not None else now
    if 0 <= stage_id < aggregator.num_stages:
        first_ts = cast(list[float | None], aggregator.stage_first_ts)
        last_ts = cast(list[float | None], aggregator.stage_last_ts)
        if first_ts[stage_id] is None:
            first_ts[stage_id] = first_value
        last_ts[stage_id] = max(last_ts[stage_id] or 0.0, now)


def queue_turn_stage_metrics(
    req_state: ClientRequestState,
    stage_id: int,
    metrics: StageRequestStats,
    *,
    final_output_type: str | None = None,
    stage_submit_ts: float | None = None,
) -> None:
    """Accumulate into the open turn, or buffer until ``begin_response``."""
    if metrics is None:
        return
    turn = req_state.duplex_turn
    if turn is None or turn.finalized:
        copied = _copy_stage_request_stats(
            metrics,
            stage_id=stage_id,
            request_id="",
            final_output_type=final_output_type,
        )
        req_state.duplex_turn_pending.append(
            PendingTurnStageMetric(
                stage_id=stage_id,
                metrics=copied,
                final_output_type=final_output_type,
                stage_submit_ts=stage_submit_ts,
            )
        )
        if req_state.duplex_turn_arrival_ts is None:
            req_state.duplex_turn_arrival_ts = float(stage_submit_ts) if stage_submit_ts is not None else time.time()
        return
    accumulate_turn_stage_metrics(
        turn,
        stage_id,
        metrics,
        final_output_type=final_output_type,
        stage_submit_ts=stage_submit_ts,
    )


def flush_pending_turn_metrics(req_state: ClientRequestState) -> None:
    """Replay snapshots that arrived before the turn aggregator existed."""
    pending = req_state.duplex_turn_pending
    req_state.duplex_turn_pending = []
    turn = req_state.duplex_turn
    if turn is None:
        return
    for item in pending:
        accumulate_turn_stage_metrics(
            turn,
            item.stage_id,
            item.metrics,
            final_output_type=item.final_output_type,
            stage_submit_ts=item.stage_submit_ts,
        )


def snapshot_transfer_tx(
    aggregator: OrchestratorAggregator | None,
) -> dict[tuple[int, int, str], tuple[float, int]]:
    if aggregator is None:
        return {}
    return {key: (float(evt.tx_time_ms), int(evt.size_bytes)) for key, evt in aggregator.transfer_events.items()}


def copy_session_transfer_tx(
    turn: DuplexTurnMetrics,
    session_aggregator: OrchestratorAggregator | None,
) -> None:
    """Copy connector TX that landed on the long-lived session aggregator."""
    turn_agg = turn.aggregator
    if turn_agg is None or session_aggregator is None:
        return
    baseline = turn.tx_baseline
    for key, evt in session_aggregator.transfer_events.items():
        from_stage, to_stage, rid = key
        if rid != turn.request_id:
            continue
        prev_tx, prev_size = baseline.get(key, (0.0, 0))
        d_tx = float(evt.tx_time_ms) - prev_tx
        d_size = int(evt.size_bytes) - prev_size
        if d_tx <= 0 and d_size <= 0:
            continue
        dest = turn_agg._get_or_create_transfer_event(from_stage, to_stage, turn.response_id)
        dest.tx_time_ms += max(d_tx, 0.0)
        dest.used_shm = dest.used_shm or bool(evt.used_shm)
        if dest.size_bytes == 0 and d_size > 0:
            dest.size_bytes = d_size


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
        unmapped: list[str] = []
        for key, value in merged_rows[sid].items():
            if key.startswith("_"):
                continue
            attr = field_aliases.get(key, key)
            if attr == "stage_id":
                continue
            if hasattr(base, attr):
                setattr(base, attr, value)
            else:
                unmapped.append(key)
        if unmapped:
            logger.debug(
                "duplex turn collapse dropped unmapped keys request_id=%s stage=%s keys=%s",
                request_id,
                sid,
                unmapped,
            )
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
            aggregator.timing_identity = {
                "req": turn.request_id,
                "response": turn.response_id,
                "turn": turn.turn_id,
                "reason": turn.finished_reason,
            }
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
    "PendingTurnStageMetric",
    "accumulate_turn_stage_metrics",
    "copy_session_transfer_tx",
    "finalize_duplex_turn_metrics",
    "finished_reason_for_cancel",
    "flush_pending_turn_metrics",
    "is_duplex_resource_request_id",
    "make_turn_aggregator",
    "queue_turn_stage_metrics",
    "snapshot_transfer_tx",
]
