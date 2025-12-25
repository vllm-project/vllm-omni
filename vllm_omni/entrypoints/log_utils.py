from __future__ import annotations

import logging
import os
import time
from typing import Any

from vllm_omni.entrypoints.stage_utils import append_jsonl as _append_jsonl

# =========================
# File/log helpers
# =========================


def remove_old_logs(log_file: str | None, num_stages: int) -> None:
    try:
        if not log_file:
            return
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
        except Exception:
            pass
        # Per-stage logs and stats
        for sid in range(num_stages):
            try:
                p = f"{log_file}.stage{sid}.log"
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
            try:
                p = f"{log_file}.stage{sid}.stats.jsonl"
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        # Orchestrator stats files
        try:
            p = f"{log_file}.orchestrator.stats.jsonl"
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        try:
            p = f"{log_file}.overall.stats.jsonl"
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    except Exception:
        pass


def configure_orchestrator_logger(logger: logging.Logger, log_file: str | None) -> None:
    try:
        if not log_file:
            return
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        if not has_file_handler:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s [PID:%(process)d] %(levelname)s: %(message)s"))
            logger.addHandler(fh)
            logger.setLevel(logging.DEBUG)
    except Exception:
        pass


def init_stats_paths(enable_stats: bool, log_file: str | None) -> tuple[str | None, str | None]:
    stats_file: str | None = None
    overall_file: str | None = None
    try:
        if enable_stats and log_file:
            stats_file = f"{log_file}.orchestrator.stats.jsonl"
            overall_file = f"{log_file}.overall.stats.jsonl"
    except Exception:
        stats_file = None
        overall_file = None
    return stats_file, overall_file


def _safe_append_jsonl(path: str | None, record: dict[str, Any]) -> None:
    if not path:
        return
    try:
        _append_jsonl(path, record)  # type: ignore[arg-type]
    except Exception:
        pass


# =========================
# Transfer logging
# =========================


def log_transfer_tx(
    stats_file: str | None,
    from_stage: int,
    to_stage: int,
    request_id: Any,
    size_bytes: int,
    tx_time_ms: float,
    used_shm: bool,
) -> None:
    _safe_append_jsonl(
        stats_file,
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
    )


def log_transfer_rx(
    stats_file: str | None,
    from_stage: int,
    to_stage: int,
    request_id: Any,
    rx_bytes: int,
    rx_decode_time_ms: float,
    in_flight_time_ms: float,
) -> None:
    _safe_append_jsonl(
        stats_file,
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
    )


def log_transfer_total(
    stats_file: str | None,
    from_stage: int,
    to_stage: int,
    request_id: Any,
    size_bytes: int,
    tx_time_ms: float,
    in_flight_time_ms: float,
    rx_decode_time_ms: float,
    total_time_ms: float,
) -> None:
    _safe_append_jsonl(
        stats_file,
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
    )


# =========================
# Orchestrator logging
# =========================


def log_orchestrator_e2e(
    stats_file: str | None,
    request_id: Any,
    final_stage_id: int,
    e2e_time_ms: float,
    num_tokens_out: int,
    e2e_time_per_token_ms: float,
) -> None:
    _safe_append_jsonl(
        stats_file,
        {
            "type": "orchestrator_request_e2e",
            "request_id": request_id,
            "final_stage_id": final_stage_id,
            "e2e_time_ms": float(e2e_time_ms),
            "num_tokens_out": int(num_tokens_out),
            "e2e_time_per_token_ms": float(e2e_time_per_token_ms),
        },
    )


def log_orchestrator_summary(stats_file: str | None, summary: dict[str, Any]) -> None:
    _safe_append_jsonl(stats_file, {"type": "orchestrator_summary", **summary})


def log_overall_summary(overall_stats_file: str | None, summary: dict[str, Any]) -> None:
    _safe_append_jsonl(overall_stats_file, {"type": "overall_summary", **summary})


def log_overall_record(overall_stats_file: str | None, record: dict[str, Any]) -> None:
    _safe_append_jsonl(overall_stats_file, record)


# =========================
# Stage logging
# =========================


def log_stage_request_stats(
    stats_file: str | None,
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
    _safe_append_jsonl(
        stats_file,
        {
            "type": "stage_request_stats",
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
    )


def log_stage_running_avg(
    stats_file: str | None,
    stage_id: int,
    total_tokens: int,
    total_gen_time_ms: float,
    avg_tokens_per_s: float,
) -> None:
    _safe_append_jsonl(
        stats_file,
        {
            "type": "stage_running_avg",
            "stage_id": stage_id,
            "total_tokens": int(total_tokens),
            "total_gen_time_ms": float(total_gen_time_ms),
            "avg_tokens_per_s": float(avg_tokens_per_s),
        },
    )


def log_stage_batch_stats(
    stats_file: str | None,
    stage_id: int,
    batch_size: int,
    batch_gen_time_ms: float,
    request_ids: list[Any],
) -> None:
    _safe_append_jsonl(
        stats_file,
        {
            "type": "stage_batch_stats",
            "stage_id": stage_id,
            "batch_size": int(batch_size),
            "batch_gen_time_ms": float(batch_gen_time_ms),
            "request_ids": list(request_ids),
        },
    )


def compute_and_log_stage_request_stats(
    stats_file: str | None,
    stage_id: int,
    request_id: Any,
    batch_size: int,
    engine_outputs: list[Any],
    stage_gen_time_ms: float,
    rx_transfer_bytes: int,
    rx_decode_time_ms: float,
) -> None:
    """Compute per-request metrics and log them in one call."""
    num_tokens = count_tokens_from_outputs(engine_outputs)
    tokens_per_s = (num_tokens * 1000.0 / stage_gen_time_ms) if stage_gen_time_ms > 0 else 0.0
    rx_mbps = (
        (float(rx_transfer_bytes) * 8.0) / (max(float(rx_decode_time_ms), 1e-6) * 1000.0)
        if rx_transfer_bytes > 0
        else 0.0
    )
    log_stage_request_stats(
        stats_file,
        stage_id,
        request_id,
        int(batch_size),
        int(num_tokens),
        float(stage_gen_time_ms),
        float(tokens_per_s),
        int(rx_transfer_bytes),
        float(rx_decode_time_ms),
        float(rx_mbps),
    )


# =========================
# Aggregation helpers
# =========================


def record_stage_metrics(
    per_request: dict[str, dict[str, Any]],
    stage_req_counts: list[int],
    stage_total_time_ms: list[float],
    stage_total_tokens: list[int],
    stage_id: int,
    req_id: Any,
    metrics: dict[str, Any],
) -> None:
    """Store per-request stage stats and aggregate counts/tokens."""
    try:
        stage_req_counts[stage_id] += 1
        stage_total_tokens[stage_id] += int(metrics.get("num_tokens_out", 0))
        rid_key = str(req_id)
        pr = per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
        pr_stages = pr["stages"]  # type: ignore[index]
        pr_stages[stage_id] = {
            "stage_gen_time_ms": float(metrics.get("stage_gen_time_ms", 0.0)),
            "num_tokens_out": int(metrics.get("num_tokens_out", 0)),
        }
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
    """Record RX aggregates and (if TX exists) compute total transfer ms for an edge."""
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
    """Record TX aggregate; store per-request TX timing for later combination with RX."""
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
                "tx_total_time_ms": float(sum_ms),
                "tx_mbps": float(tx_mbps),
                "rx_samples": samples_rx,
                "rx_total_bytes": int(sum_rx_bytes),
                "rx_total_time_ms": float(sum_rx_ms),
                "rx_mbps": float(rx_mbps),
                "total_samples": samples_total,
                "total_transfer_time_ms": float(sum_total_ms),
                "total_mbps": float(total_mbps),
            }
        )
    return summary


# =========================
# Orchestrator metrics class
# =========================


class OrchestratorMetrics:
    """
    Collects per-stage, per-request, transfer, and end-to-end metrics.

    Key improvement vs your current output:
      - Adds E2E breakdown: service_ms (sum of stage compute),
        transfer_ms (sum of edge transfers), queue_ms (everything else)
      - Adds wall_tokens_per_s vs service_tokens_per_s
    """

    def __init__(
        self,
        num_stages: int,
        enable_stats: bool,
        stats_file: str | None,
        overall_stats_file: str | None,
        wall_start_ts: float,
    ) -> None:
        self.num_stages = int(num_stages)
        self.enable_stats = bool(enable_stats)
        self.stats_file = stats_file
        self.overall_stats_file = overall_stats_file

        # Aggregates
        self.stage_total_time_ms: list[float] = [0.0 for _ in range(self.num_stages)]  # sum of unique batches
        self.stage_total_tokens: list[int] = [0 for _ in range(self.num_stages)]
        self.stage_req_counts: list[int] = [0 for _ in range(self.num_stages)]

        # Transfers
        self.transfer_agg: dict[tuple[int, int], dict[str, float]] = {}
        self.transfer_edge_req: dict[tuple[int, int, str], dict[str, float]] = {}

        # E2E
        self.e2e_total_ms: float = 0.0  # sum of per-request e2e
        self.e2e_total_tokens: int = 0
        self.e2e_count: int = 0
        self.e2e_done: set[str] = set()

        # Per-request store
        self.per_request: dict[str, dict[str, Any]] = {}

        # For wall time
        self.wall_start_ts: float = float(wall_start_ts)
        self.last_finish_ts: float = float(wall_start_ts)

        # For stage window (busy interval)
        self.stage_first_ts: list[float | None] = [None for _ in range(self.num_stages)]
        self.stage_last_ts: list[float | None] = [None for _ in range(self.num_stages)]

        # Avoid counting stage_total_time_ms multiple times for same batch_id
        self.stage_seen_batches: dict[int, set] = {sid: set() for sid in range(self.num_stages)}

        # New: breakdown accumulators
        self.sum_service_ms: float = 0.0
        self.sum_queue_ms: float = 0.0
        self.sum_transfer_ms: float = 0.0

    def _compute_request_breakdown_ms(self, rid_key: str, e2e_ms: float) -> dict[str, float]:
        pr = self.per_request.get(rid_key, {})
        stages = pr.get("stages", {}) if isinstance(pr, dict) else {}

        service_ms = 0.0
        try:
            for _sid, sm in stages.items():
                if isinstance(sm, dict):
                    service_ms += float(sm.get("stage_gen_time_ms", 0.0))
        except Exception:
            pass

        transfer_ms = 0.0
        try:
            transfer_ms = float(pr.get("transfers_ms", 0.0)) if isinstance(pr, dict) else 0.0
        except Exception:
            transfer_ms = 0.0

        queue_ms = max(0.0, float(e2e_ms) - service_ms - transfer_ms)
        return {
            "service_ms": float(service_ms),
            "transfer_ms": float(transfer_ms),
            "queue_ms": float(queue_ms),
        }

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

        # count stage_total_time_ms only once per unique batch_id
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

        if self.enable_stats and self.stats_file and stage_id > 0:
            log_transfer_rx(
                self.stats_file,
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
                    self.stats_file,
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
        # Mark first input time for destination stage if not set
        if self.stage_first_ts[to_stage] is None:
            self.stage_first_ts[to_stage] = time.time()

        if self.enable_stats and self.stats_file:
            log_transfer_tx(
                self.stats_file,
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

    def on_finalize_request(self, stage_id: int, req_id: Any, engine_outputs: list[Any], req_start_ts: float) -> None:
        rid_key = str(req_id)
        _t0 = float(req_start_ts)
        _t1 = time.time()

        # Update stage last output time
        prev_last = self.stage_last_ts[stage_id]
        self.stage_last_ts[stage_id] = _t1 if prev_last is None else max(prev_last, _t1)

        self.last_finish_ts = max(self.last_finish_ts, _t1)

        e2e_ms = (_t1 - _t0) * 1000.0
        num_tokens = count_tokens_from_outputs(engine_outputs)

        self.e2e_total_ms += e2e_ms
        self.e2e_total_tokens += int(num_tokens)
        self.e2e_count += 1
        self.e2e_done.add(rid_key)

        pr = self.per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})

        breakdown = self._compute_request_breakdown_ms(rid_key, e2e_ms)
        self.sum_service_ms += breakdown["service_ms"]
        self.sum_transfer_ms += breakdown["transfer_ms"]
        self.sum_queue_ms += breakdown["queue_ms"]

        per_req_record = {
            "type": "overall_request",
            "request_id": rid_key,
            "final_stage_id": int(stage_id),
            "e2e_time_ms": float(e2e_ms),
            "num_tokens_out": int(num_tokens),
            "service_time_ms": float(breakdown["service_ms"]),
            "transfer_time_ms": float(breakdown["transfer_ms"]),
            "queue_time_ms": float(breakdown["queue_ms"]),
            "transfers_total_time_ms": float(pr.get("transfers_ms", 0.0)),
            "transfers_total_bytes": int(pr.get("transfers_bytes", 0)),
            "stages": pr.get("stages", {}),
        }

        if self.enable_stats and self.overall_stats_file:
            log_overall_record(self.overall_stats_file, per_req_record)

        if self.enable_stats and self.stats_file:
            e2e_tpt = (e2e_ms / num_tokens) if num_tokens > 0 else 0.0
            log_orchestrator_e2e(self.stats_file, req_id, stage_id, e2e_ms, int(num_tokens), e2e_tpt)

    def build_and_log_summary(self, final_stage_id_to_prompt: dict[str, int]) -> dict[str, Any]:
        # Stage summary: report both wall-busy (first->last) and service sum (batch times)
        stage_summary: list[dict[str, Any]] = []
        for sid in range(self.num_stages):
            first_ts = self.stage_first_ts[sid]
            last_ts = self.stage_last_ts[sid]
            wall_busy_ms = (
                (max(0.0, (last_ts - first_ts)) * 1000.0) if (first_ts is not None and last_ts is not None) else 0.0
            )

            reqs = self.stage_req_counts[sid]
            tokens = self.stage_total_tokens[sid]
            service_ms = float(self.stage_total_time_ms[sid])

            stage_summary.append(
                {
                    "stage_id": sid,
                    "requests": int(reqs),
                    "tokens": int(tokens),
                    "wall_busy_time_ms": float(wall_busy_ms),
                    "service_time_ms": float(service_ms),
                    "avg_service_time_per_request_ms": float(service_ms / reqs) if reqs > 0 else 0.0,
                    "avg_tokens_per_s_service": float(tokens * 1000.0 / service_ms) if service_ms > 0 else 0.0,
                    "avg_tokens_per_s_wall": float(tokens * 1000.0 / wall_busy_ms) if wall_busy_ms > 0 else 0.0,
                }
            )

        transfer_summary = build_transfer_summary(self.transfer_agg)

        e2e_avg_req = (self.e2e_total_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        e2e_avg_tok = (self.e2e_total_tokens * 1000.0 / self.e2e_total_ms) if self.e2e_total_ms > 0 else 0.0

        wall_time_ms = max(0.0, (self.last_finish_ts - self.wall_start_ts) * 1000.0)

        # New: breakdown averages
        avg_service_ms = (self.sum_service_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        avg_transfer_ms = (self.sum_transfer_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        avg_queue_ms = (self.sum_queue_ms / self.e2e_count) if self.e2e_count > 0 else 0.0

        # New: throughput definitions
        wall_toks_per_s = (self.e2e_total_tokens * 1000.0 / wall_time_ms) if wall_time_ms > 0 else 0.0
        service_toks_per_s = (self.e2e_total_tokens * 1000.0 / self.sum_service_ms) if self.sum_service_ms > 0 else 0.0

        summary: dict[str, Any] = {
            "e2e_requests": int(self.e2e_count),
            "e2e_total_time_ms": float(wall_time_ms),  # total wall time (what you waited)
            "e2e_sum_time_ms": float(self.e2e_total_ms),  # sum of per-request e2e (includes queueing)
            "e2e_total_tokens": int(self.e2e_total_tokens),
            "e2e_avg_time_per_request_ms": float(e2e_avg_req),
            "e2e_avg_tokens_per_s": float(e2e_avg_tok),
            "wall_time_ms": float(wall_time_ms),
            "final_stage_id": final_stage_id_to_prompt,
            "breakdown_avg_ms": {
                "service_ms": float(avg_service_ms),
                "transfer_ms": float(avg_transfer_ms),
                "queue_ms": float(avg_queue_ms),
            },
            "throughput": {
                "wall_tokens_per_s": float(wall_toks_per_s),
                "service_tokens_per_s": float(service_toks_per_s),
            },
            "stages": stage_summary,
            "transfers": transfer_summary,
        }

        if self.enable_stats and self.stats_file:
            log_orchestrator_summary(self.stats_file, summary)
        if self.enable_stats and self.overall_stats_file:
            log_overall_summary(self.overall_stats_file, summary)

        return summary
