from __future__ import annotations

import time
from dataclasses import dataclass
from pprint import pformat
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def log_transfer_tx(
    from_stage: int,
    to_stage: int,
    request_id: Any,
    size_bytes: int,
    tx_time_ms: float,
    used_shm: bool,
) -> None:
    logger.info(
        pformat(
            {
                "type": "transfer_stats",
                "from_stage": from_stage,
                "to_stage": to_stage,
                "request_id": request_id,
                "size_bytes": int(size_bytes),
                "tx_time_ms": float(tx_time_ms),
                "tx_mbps": (float(size_bytes) * 8.0) / (max(tx_time_ms, 1e-6) * 1000.0),
                "used_shm": bool(used_shm),
            },
            sort_dicts=False,
        )
    )


def log_transfer_rx(
    from_stage: int,
    to_stage: int,
    request_id: Any,
    rx_bytes: int,
    rx_decode_time_ms: float,
    in_flight_time_ms: float,
) -> None:
    logger.info(
        pformat(
            {
                "type": "transfer_rx_stats",
                "from_stage": from_stage,
                "to_stage": to_stage,
                "request_id": request_id,
                "rx_bytes": int(rx_bytes),
                "rx_decode_time_ms": float(rx_decode_time_ms),
                "in_flight_time_ms": float(in_flight_time_ms),
                "rx_time_per_kb_ms": (
                    (float(rx_decode_time_ms) / max(float(rx_bytes) / 1024.0, 1e-6)) if rx_bytes > 0 else 0.0
                ),
            },
            sort_dicts=False,
        )
    )


def log_transfer_total(
    from_stage: int,
    to_stage: int,
    request_id: Any,
    size_bytes: int,
    tx_time_ms: float,
    in_flight_time_ms: float,
    rx_decode_time_ms: float,
    total_time_ms: float,
) -> None:
    logger.info(
        pformat(
            {
                "type": "transfer_total_stats",
                "from_stage": from_stage,
                "to_stage": to_stage,
                "request_id": request_id,
                "size_bytes": int(size_bytes),
                "tx_time_ms": float(tx_time_ms),
                "in_flight_time_ms": float(in_flight_time_ms),
                "rx_decode_time_ms": float(rx_decode_time_ms),
                "total_time_ms": float(total_time_ms),
                "total_time_per_kb_ms": (
                    float(total_time_ms) / max(float(size_bytes) / 1024.0, 1e-6) if size_bytes > 0 else 0.0
                ),
            },
            sort_dicts=False,
        )
    )


def log_stage_request_stats(
    stage_id: int,
    request_id: Any,
    batch_size: int,
    num_tokens_out: int,
    stage_gen_time_ms: float,
    tokens_per_s: float,
    rx_transfer_bytes: int,
    rx_decode_time_ms: float,
    rx_mbps: float,
) -> None:
    logger.info(
        pformat(
            {
                "type": "Request_stage_stats",
                "stage_id": stage_id,
                "request_id": request_id,
                "batch_size": int(batch_size),
                "num_tokens_out": int(num_tokens_out),
                "stage_gen_time_ms": float(stage_gen_time_ms),
                "tokens_per_s": float(tokens_per_s),
                "rx_transfer_bytes": int(rx_transfer_bytes),
                "rx_decode_time_ms": float(rx_decode_time_ms),
                "rx_mbps": float(rx_mbps),
            },
            sort_dicts=False,
        )
    )


def compute_and_log_stage_request_stats(
    stage_id: int,
    request_id: Any,
    batch_size: int,
    num_engine_outputs: int,
    stage_gen_time_ms: float,
    rx_transfer_bytes: int,
    rx_decode_time_ms: float,
) -> None:
    """Compute per-request metrics and log them in one call."""
    tokens_per_s = (num_engine_outputs * 1000.0 / stage_gen_time_ms) if stage_gen_time_ms > 0 else 0.0
    rx_mbps = (
        (float(rx_transfer_bytes) * 8.0) / (max(float(rx_decode_time_ms), 1e-6) * 1000.0)
        if rx_transfer_bytes > 0
        else 0.0
    )
    log_stage_request_stats(
        stage_id,
        request_id,
        int(batch_size),
        int(num_engine_outputs),
        float(stage_gen_time_ms),
        float(tokens_per_s),
        int(rx_transfer_bytes),
        float(rx_decode_time_ms),
        float(rx_mbps),
    )


# ----------------- Aggregation helpers for orchestrator -----------------


def record_stage_metrics(
    per_request: dict[str, dict[str, Any]],
    stage_req_counts: list[int],
    stage_total_time_ms: list[float],
    stage_total_tokens: list[int],
    stage_id: int,
    req_id: Any,
    metrics: dict[str, Any],
) -> None:
    try:
        stage_req_counts[stage_id] += 1
        stage_total_tokens[stage_id] += int(metrics.get("num_tokens_out", 0))
        rid_key = str(req_id)
        pr = per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
        pr_stages = pr["stages"]  # type: ignore[index]
        stage_data: dict[str, Any] = {
            "stage_gen_time_ms": float(metrics.get("stage_gen_time_ms", 0.0)),
            "num_tokens_out": int(metrics.get("num_tokens_out", 0)),
        }
        # Only record num_tokens_in for stage 0 (initial prompt)
        if stage_id == 0:
            stage_data["num_tokens_in"] = int(metrics.get("num_tokens_in", 0))
        pr_stages[stage_id] = stage_data
    except Exception:
        pass


def aggregate_rx_and_maybe_total(
    transfer_edge_req: dict[tuple[int, int, str], dict[str, float]],
    transfer_agg: dict[tuple[int, int], dict[str, float]],
    per_request: dict[str, dict[str, Any]],
    stage_id: int,
    req_id: Any,
    rx_bytes: float,
    rx_ms: float,
    in_flight_ms: float,
) -> tuple[int, float, float] | None:
    try:
        # Update RX aggregates for (stage_id-1 -> stage_id)
        if stage_id > 0:
            key = (stage_id - 1, stage_id)
            agg = transfer_agg.get(key)
            if agg is None:
                agg = {
                    "sum_bytes": 0.0,
                    "sum_ms": 0.0,
                    "count": 0.0,
                    "sum_rx_bytes": 0.0,
                    "sum_rx_ms": 0.0,
                    "rx_count": 0.0,
                    "sum_total_ms": 0.0,
                    "total_count": 0.0,
                }
                transfer_agg[key] = agg
            agg["sum_rx_bytes"] += float(rx_bytes)
            agg["sum_rx_ms"] += float(rx_ms)
            agg["rx_count"] += 1.0

            # Try combine with sender-side timing if present
            rid_key = str(req_id)
            s = transfer_edge_req.get((stage_id - 1, stage_id, rid_key))
            if s is None:
                return None
            tx_ms = float(s.get("tx_ms", 0.0))
            size_b = float(s.get("size_bytes", rx_bytes))
            total_ms = tx_ms + float(in_flight_ms) + float(rx_ms)
            agg["sum_total_ms"] += total_ms
            agg["total_count"] += 1.0
            # accumulate per-request transfer totals
            try:
                pr = per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
                pr["transfers_ms"] = float(pr.get("transfers_ms", 0.0)) + total_ms  # type: ignore[index]
                pr["transfers_bytes"] = int(pr.get("transfers_bytes", 0)) + int(rx_bytes)  # type: ignore[index]
            except Exception:
                pass
            return int(size_b), float(tx_ms), float(total_ms)
        return None
    except Exception:
        return None


def record_sender_transfer_agg(
    transfer_agg: dict[tuple[int, int], dict[str, float]],
    transfer_edge_req: dict[tuple[int, int, str], dict[str, float]],
    from_stage: int,
    to_stage: int,
    req_id: Any,
    size_bytes: int,
    tx_ms: float,
) -> None:
    try:
        key = (from_stage, to_stage)
        agg = transfer_agg.get(key)
        if agg is None:
            agg = {
                "sum_bytes": 0.0,
                "sum_ms": 0.0,
                "count": 0.0,
                "sum_rx_bytes": 0.0,
                "sum_rx_ms": 0.0,
                "rx_count": 0.0,
                "sum_total_ms": 0.0,
                "total_count": 0.0,
            }
            transfer_agg[key] = agg
        agg["sum_bytes"] += float(size_bytes)
        agg["sum_ms"] += float(tx_ms)
        agg["count"] += 1.0
        # Store sender-side timing for per-request combination
        rid_key = str(req_id)
        transfer_edge_req[(from_stage, to_stage, rid_key)] = {
            "tx_ms": float(tx_ms),
            "size_bytes": float(size_bytes),
        }
    except Exception:
        pass


def count_tokens_from_outputs(engine_outputs: list[Any]) -> int:
    total = 0
    for _ro in engine_outputs:
        try:
            outs = getattr(_ro, "outputs", None)
            if outs and len(outs) > 0:
                tokens = getattr(outs[0], "token_ids", None)
                if tokens is not None:
                    total += len(tokens)
        except Exception:
            pass
    return total


def build_stage_summary(
    stage_req_counts: list[int],
    stage_total_tokens: list[int],
    stage_total_time_ms: list[float],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for sid in range(len(stage_req_counts)):
        reqs = stage_req_counts[sid]
        tokens = stage_total_tokens[sid]
        total_ms = float(stage_total_time_ms[sid])
        avg_req = (total_ms / reqs) if reqs > 0 else 0.0
        avg_tok = (tokens * 1000.0 / total_ms) if total_ms > 0 else 0.0
        summary.append(
            {
                "stage_id": sid,
                "requests": int(reqs),
                "tokens": int(tokens),
                "total_time_ms": total_ms,
                "avg_time_per_request_ms": avg_req,
                "avg_tokens_per_s": avg_tok,
            }
        )
    return summary


def build_transfer_summary(
    transfer_agg: dict[tuple[int, int], dict[str, float]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for (src, dst), agg in transfer_agg.items():
        sum_bytes = float(agg.get("sum_bytes", 0.0))
        sum_ms = float(agg.get("sum_ms", 0.0))
        samples = int(agg.get("count", 0.0))
        tx_mbps = (sum_bytes * 8.0) / (max(sum_ms, 1e-6) * 1000.0) if sum_bytes > 0 else 0.0
        sum_rx_bytes = float(agg.get("sum_rx_bytes", 0.0))
        sum_rx_ms = float(agg.get("sum_rx_ms", 0.0))
        samples_rx = int(agg.get("rx_count", 0.0))
        rx_mbps = (sum_rx_bytes * 8.0) / (max(sum_rx_ms, 1e-6) * 1000.0) if sum_rx_bytes > 0 else 0.0
        sum_total_ms = float(agg.get("sum_total_ms", 0.0))
        samples_total = int(agg.get("total_count", 0.0))
        total_mbps = (sum_bytes * 8.0) / (max(sum_total_ms, 1e-6) * 1000.0) if sum_bytes > 0 else 0.0
        summary.append(
            {
                "from_stage": src,
                "to_stage": dst,
                "samples": samples,
                "total_bytes": int(sum_bytes),
                "total_time_ms": sum_ms,
                "tx_mbps": tx_mbps,
                "rx_samples": samples_rx,
                "rx_total_bytes": int(sum_rx_bytes),
                "rx_total_time_ms": sum_rx_ms,
                "rx_mbps": rx_mbps,
                "total_samples": samples_total,
                "total_transfer_time_ms": sum_total_ms,
                "total_mbps": total_mbps,
            }
        )
    return summary


@dataclass
class StageStats:
    total_token: int
    total_gen_time: float


@dataclass
class StageRequestMetrics:
    num_tokens_in: int
    num_tokens_out: int
    stage_gen_time_ms: float
    batch_id: int
    batch_size: int
    rx_decode_time_ms: float
    rx_transfer_bytes: int
    rx_in_flight_time_ms: float

    stage_stats: StageStats


class OrchestratorMetrics:
    def __init__(
        self,
        num_stages: int,
        enable_stats: bool,
        wall_start_ts: float,
    ) -> None:
        self.num_stages = int(num_stages)
        self.enable_stats = bool(enable_stats)
        self.stage_total_time_ms: list[float] = [0.0 for _ in range(self.num_stages)]
        self.stage_total_tokens: list[int] = [0 for _ in range(self.num_stages)]
        self.stage_req_counts: list[int] = [0 for _ in range(self.num_stages)]
        self.transfer_agg: dict[tuple[int, int], dict[str, float]] = {}
        self.transfer_edge_req: dict[tuple[int, int, str], dict[str, float]] = {}
        self.e2e_total_ms: float = 0.0
        self.e2e_total_tokens: int = 0
        self.e2e_count: int = 0
        self.e2e_done: set[str] = set()
        self.per_request: dict[str, dict[str, Any]] = {}
        self.sum_per_request_transfer_ms: float = 0.0
        self.wall_start_ts: float = float(wall_start_ts)
        self.last_finish_ts: float = float(wall_start_ts)
        self.stage_seen_batches: dict[int, set] = {sid: set() for sid in range(self.num_stages)}
        self.stage_first_ts: list[float | None] = [None for _ in range(self.num_stages)]
        self.stage_last_ts: list[float | None] = [None for _ in range(self.num_stages)]

    def on_stage_metrics(self, stage_id: int, req_id: Any, metrics: dict[str, Any]) -> None:
        record_stage_metrics(
            self.per_request,
            self.stage_req_counts,
            self.stage_total_time_ms,
            self.stage_total_tokens,
            stage_id,
            req_id,
            metrics,
        )
        if self.enable_stats:
            compute_and_log_stage_request_stats(
                stage_id=stage_id,
                request_id=req_id,
                batch_size=metrics.get("batch_size"),
                num_engine_outputs=metrics.get("num_tokens_out"),
                stage_gen_time_ms=metrics.get("stage_gen_time_ms"),
                rx_decode_time_ms=metrics.get("rx_decode_time_ms"),
                rx_transfer_bytes=metrics.get("rx_transfer_bytes"),
            )
            if stage_stats := metrics.get("stage_stats", None):
                total_token = int(stage_stats.get("total_token"))
                total_gen_time = float(stage_stats.get("total_gen_time"))
                _avg_tokens_per_s = (total_token * 1000.0 / total_gen_time) if total_gen_time > 0 else 0.0
                logger.info(
                    pformat(
                        {
                            "type": "Stage_running_avg",
                            "stage_id": stage_id,
                            "total_tokens": total_token,
                            "total_gen_time_ms": total_gen_time,
                            "avg_tokens_per_s": _avg_tokens_per_s,
                        },
                        sort_dicts=False,
                    )
                )
        try:
            batch_id_raw = metrics.get("batch_id", None)
            if batch_id_raw is not None:
                batch_id = int(batch_id_raw)
                if batch_id not in self.stage_seen_batches[stage_id]:
                    self.stage_total_time_ms[stage_id] += float(metrics.get("stage_gen_time_ms", 0.0))
                    self.stage_seen_batches[stage_id].add(batch_id)
        except Exception:
            pass
        rx_b = float(metrics.get("rx_transfer_bytes", 0.0))
        rx_ms = float(metrics.get("rx_decode_time_ms", 0.0))
        in_flight_ms = float(metrics.get("rx_in_flight_time_ms", 0.0))
        combined = aggregate_rx_and_maybe_total(
            self.transfer_edge_req,
            self.transfer_agg,
            self.per_request,
            stage_id,
            req_id,
            rx_b,
            rx_ms,
            in_flight_ms,
        )
        if self.enable_stats and stage_id > 0:
            log_transfer_rx(
                stage_id - 1,
                stage_id,
                req_id,
                int(rx_b),
                float(rx_ms),
                float(in_flight_ms),
            )
            if combined is not None:
                size_b_c, tx_ms_c, total_ms_c = combined
                log_transfer_total(
                    stage_id - 1,
                    stage_id,
                    req_id,
                    int(size_b_c),
                    float(tx_ms_c),
                    float(in_flight_ms),
                    float(rx_ms),
                    float(total_ms_c),
                )

    def on_forward(
        self,
        from_stage: int,
        to_stage: int,
        req_id: Any,
        size_bytes: int,
        tx_ms: float,
        used_shm: bool,
    ) -> None:
        # Mark first input time for the destination stage if not set
        if self.stage_first_ts[to_stage] is None:
            self.stage_first_ts[to_stage] = time.time()
        if self.enable_stats:
            log_transfer_tx(
                from_stage,
                to_stage,
                req_id,
                int(size_bytes),
                float(tx_ms),
                bool(used_shm),
            )
        record_sender_transfer_agg(
            self.transfer_agg,
            self.transfer_edge_req,
            from_stage,
            to_stage,
            req_id,
            int(size_bytes),
            float(tx_ms),
        )

    def on_finalize_request(
        self,
        stage_id: int,
        req_id: Any,
        req_start_ts: float,
    ) -> None:
        rid_key = str(req_id)
        _t0 = float(req_start_ts)
        _t1 = time.time()
        # Update last output time for this stage
        prev_last = self.stage_last_ts[stage_id]
        self.stage_last_ts[stage_id] = _t1 if prev_last is None else max(prev_last, _t1)
        self.last_finish_ts = max(self.last_finish_ts, _t1)
        e2e_ms = (_t1 - _t0) * 1000.0

        # Sum tokens from all stages for this request
        # Include input tokens from stage 0 + output tokens from all stages
        pr = self.per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
        total_tokens = 0
        stages_info = pr.get("stages", {})
        for sid, stage_data in stages_info.items():
            # Add input tokens only from stage 0 (initial prompt)
            if sid == 0:
                total_tokens += int(stage_data.get("num_tokens_in", 0))
            total_tokens += int(stage_data.get("num_tokens_out", 0))

        self.e2e_total_ms += e2e_ms
        self.e2e_total_tokens += total_tokens
        self.e2e_count += 1
        self.e2e_done.add(rid_key)
        per_req_record = {
            "type": "request_level_metrics",
            "request_id": rid_key,
            "e2e_time_ms": e2e_ms,
            "e2e_tpt": (e2e_ms / total_tokens) if total_tokens > 0 else 0.0,
            "e2e_total_tokens": total_tokens,
            "transfers_total_time_ms": float(pr.get("transfers_ms", 0.0)),
            "transfers_total_bytes": int(pr.get("transfers_bytes", 0)),
            "stages": stages_info,
        }
        self.sum_per_request_transfer_ms += float(pr.get("transfers_ms", 0.0))
        logger.info(pformat(per_req_record, sort_dicts=False))

    def build_and_log_summary(self, final_stage_id_to_prompt: dict[str, int]) -> dict[str, Any]:
        # Compute stage summary using wall time between first input and last output per stage
        stage_summary: list[dict[str, Any]] = []
        for sid in range(self.num_stages):
            first_ts = self.stage_first_ts[sid]
            last_ts = self.stage_last_ts[sid]
            total_ms = (
                (max(0.0, (last_ts - first_ts)) * 1000.0) if (first_ts is not None and last_ts is not None) else 0.0
            )
            reqs = self.stage_req_counts[sid]
            tokens = self.stage_total_tokens[sid]
            avg_req = (total_ms / reqs) if reqs > 0 else 0.0
            avg_tok = (tokens * 1000.0 / total_ms) if total_ms > 0 else 0.0
            stage_summary.append(
                {
                    "stage_id": sid,
                    "requests": int(reqs),
                    "tokens": int(tokens),
                    "total_time_ms": float(total_ms),
                    "avg_time_per_request_ms": float(avg_req),
                    "avg_tokens_per_s": float(avg_tok),
                }
            )
        transfer_summary = build_transfer_summary(self.transfer_agg)
        e2e_avg_req = (self.e2e_total_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        e2e_avg_tok = (self.e2e_total_tokens * 1000.0 / self.e2e_total_ms) if self.e2e_total_ms > 0 else 0.0
        wall_time_ms = max(0.0, (self.last_finish_ts - self.wall_start_ts) * 1000.0)
        summary: dict[str, Any] = {
            "e2e_requests": int(self.e2e_count),
            "e2e_total_time_ms": float(wall_time_ms),
            "e2e_sum_time_ms": float(self.e2e_total_ms),
            "e2e_total_tokens": int(self.e2e_total_tokens),
            "e2e_avg_time_per_request_ms": e2e_avg_req,
            "e2e_avg_tokens_per_s": e2e_avg_tok,
            "wall_time_ms": wall_time_ms,
            "final_stage_id": final_stage_id_to_prompt,
            "stages": stage_summary,
            "transfers": transfer_summary,
        }
        return summary
