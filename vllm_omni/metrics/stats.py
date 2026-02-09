from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.metrics.utils import _build_field_defs, _build_row, _format_table

logger = init_logger(__name__)


@dataclass
class StageStats:
    total_token: int = 0
    total_gen_time_ms: float = 0.0

    @property
    def avg_tokens_per_s(self) -> float:
        return (self.total_token * 1000.0 / self.total_gen_time_ms) if self.total_gen_time_ms > 0 else 0.0


@dataclass
class StageRequestStats:
    batch_id: int
    batch_size: int
    num_tokens_in: int
    num_tokens_out: int
    stage_gen_time_ms: float
    rx_transfer_bytes: int
    rx_decode_time_ms: float
    rx_in_flight_time_ms: float
    stage_stats: StageStats
    stage_id: int | None = None
    final_output_type: str | None = None
    request_id: str | None = None
    postprocess_time_ms: float = 0.0
    diffusion_metrics: dict[str, int] = None
    audio_generated_frames: int = 0

    @property
    def rx_mbps(self) -> float:
        return (
            (float(self.rx_transfer_bytes) * 8.0) / (max(float(self.rx_decode_time_ms), 1e-6) * 1000.0)
            if self.rx_transfer_bytes > 0
            else 0.0
        )

    @property
    def tokens_per_s(self) -> float:
        return (self.num_tokens_out * 1000.0 / self.stage_gen_time_ms) if (self.stage_gen_time_ms > 0) else 0.0


@dataclass
class TransferEdgeStats:
    from_stage: int
    to_stage: int
    request_id: str
    size_bytes: int
    tx_time_ms: float
    used_shm: bool = False
    rx_decode_time_ms: float = 0.0
    in_flight_time_ms: float = 0.0

    @property
    def total_time_ms(self) -> float:
        return float(self.tx_time_ms) + float(self.rx_decode_time_ms) + float(self.in_flight_time_ms)


@dataclass
class RequestE2EStats:
    request_id: str
    e2e_total_ms: float
    e2e_total_tokens: int
    transfers_total_time_ms: float
    transfers_total_bytes: int

    @property
    def e2e_tpt(self) -> float:
        return (self.e2e_total_ms / self.e2e_total_tokens) if self.e2e_total_tokens > 0 else 0.0


# === Field Configuration ===
# Fields requiring unit conversion:  original_field_name -> (display_name, transform_fn)
FIELD_TRANSFORMS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "rx_transfer_bytes": ("rx_transfer_kbytes", lambda v: v / 1024.0),
    "size_bytes": ("size_kbytes", lambda v: v / 1024.0),
    "transfers_total_bytes": ("transfers_total_kbytes", lambda v: v / 1024.0),
}

# Fields to exclude from table display for each event type
STAGE_EXCLUDE = {
    "stage_stats",
    "stage_id",
    "request_id",
    "rx_transfer_bytes",
    "rx_decode_time_ms",
    "rx_in_flight_time_ms",
    "final_output_type",
}
TRANSFER_EXCLUDE = {"from_stage", "to_stage", "request_id", "used_shm"}
E2E_EXCLUDE = {"request_id"}

# Decide the order of overall summary fields, or None for auto
OVERALL_FIELDS: list[str] | None = None
STAGE_FIELDS = _build_field_defs(StageRequestStats, STAGE_EXCLUDE, FIELD_TRANSFORMS)
TRANSFER_FIELDS = _build_field_defs(TransferEdgeStats, TRANSFER_EXCLUDE, FIELD_TRANSFORMS)
E2E_FIELDS = _build_field_defs(RequestE2EStats, E2E_EXCLUDE, FIELD_TRANSFORMS)


class OrchestratorAggregator:
    def __init__(
        self,
        num_stages: int,
        log_stats: bool,
        wall_start_ts: float,
        final_stage_id_for_e2e: dict[str, int] | int,
    ) -> None:
        self.num_stages = int(num_stages)
        self.log_stats = bool(log_stats)
        self.final_stage_id_for_e2e = final_stage_id_for_e2e
        self.init_run_state(wall_start_ts)
        self.stage_events: dict[str, list[StageRequestStats]] = {}
        self.transfer_events: dict[
            tuple[int, int, str], TransferEdgeStats
        ] = {}  # Key: (from_stage, to_stage, request_id)
        self.e2e_events: list[RequestE2EStats] = []

    def init_run_state(self, wall_start_ts: float) -> None:
        # Per-run aggregates and timing state
        self.stage_total_tokens = [0 for _ in range(self.num_stages)]
        self.e2e_total_ms = 0.0
        self.e2e_total_tokens = 0
        self.e2e_count = 0
        self.e2e_done = set()
        self.wall_start_ts = float(wall_start_ts)
        self.last_finish_ts = float(wall_start_ts)
        self.stage_first_ts = [None for _ in range(self.num_stages)]
        self.stage_last_ts = [None for _ in range(self.num_stages)]
        self.accumulated_gen_time_ms: defaultdict[str, defaultdict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )  # {request_id: {stage_id:accumulated_gen_time_ms}}
        self.diffusion_metrics: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )  # {request_id: {diffusion_metrics_key: accumulated_metrics_data}}

    def _get_or_create_transfer_event(
        self,
        from_stage: int,
        to_stage: int,
        request_id: str,
    ) -> TransferEdgeStats:
        key = (from_stage, to_stage, request_id)
        evt = self.transfer_events.get(key)
        if evt is None:
            evt = TransferEdgeStats(
                from_stage=from_stage,
                to_stage=to_stage,
                request_id=request_id,
                size_bytes=0,
                tx_time_ms=0.0,
                used_shm=False,
                rx_decode_time_ms=0.0,
                in_flight_time_ms=0.0,
            )
            self.transfer_events[key] = evt
        return evt

    def record_transfer_tx(
        self,
        from_stage: int,
        to_stage: int,
        request_id: Any,
        size_bytes: int,
        tx_time_ms: float,
        used_shm: bool,
    ) -> TransferEdgeStats | None:
        try:
            evt = self._get_or_create_transfer_event(
                int(from_stage),
                int(to_stage),
                str(request_id),
            )
            # Accumulate tx metrics
            evt.size_bytes += int(size_bytes)
            evt.tx_time_ms += float(tx_time_ms)
            evt.used_shm = evt.used_shm or bool(used_shm)
            return evt
        except Exception:
            return None

    def record_transfer_rx(
        self,
        stats: StageRequestStats,
    ) -> TransferEdgeStats | None:
        try:
            if stats.stage_id is None or stats.stage_id <= 0:
                return None
            from_stage = int(stats.stage_id) - 1
            to_stage = int(stats.stage_id)
            rid_key = str(stats.request_id)
            evt = self._get_or_create_transfer_event(from_stage, to_stage, rid_key)
            # Accumulate rx metrics
            if evt.size_bytes == 0:
                # size_bytes has been recorded in tx phase
                evt.size_bytes = int(stats.rx_transfer_bytes)
            evt.rx_decode_time_ms += float(stats.rx_decode_time_ms)
            evt.in_flight_time_ms += float(stats.rx_in_flight_time_ms)
            return evt
        except Exception:
            return None

    def record_audio_generated_frames(
        self,
        output_to_yield: Any,
        finished: bool,
        stage_id: int,
        request_id: str,
    ) -> None:
        if (
            output_to_yield.final_output_type == "audio"
            and finished
            and (multimodal_output := output_to_yield.multimodal_output.get("audio")) is not None
        ):
            nframes = int(multimodal_output[-1].shape[0])
            stage_events_for_req = self.stage_events.get(request_id, [])
            if stage_events_for_req:
                for stage_event in stage_events_for_req:
                    if stage_event.stage_id == stage_id:
                        stage_event.audio_generated_frames += nframes
                        break
            else:
                logger.warning(
                    "Failed to record audio generated frames for request %s at stage %s: no stage event found",
                    request_id,
                    stage_id,
                )

    def _as_stage_request_stats(
        self,
        stage_id: int,
        req_id: str,
        metrics: StageRequestStats,
        final_output_type: str | None = None,
    ) -> StageRequestStats:
        "Convert dict to StageRequestStats if needed."
        stats = metrics
        stats.stage_id = stage_id
        stats.request_id = req_id
        stats.final_output_type = final_output_type
        stats.diffusion_metrics = (
            {k: int(v) for k, v in self.diffusion_metrics.pop(req_id, {}).items()}
            if req_id in self.diffusion_metrics
            else None
        )
        return stats

    def on_stage_metrics(
        self,
        stage_id: int,
        req_id: Any,
        metrics: StageRequestStats,
        final_output_type: str | None = None,
    ) -> None:
        stats = self._as_stage_request_stats(stage_id, req_id, metrics, final_output_type)
        self.stage_total_tokens[stats.stage_id] += int(stats.num_tokens_out)
        if stats.stage_id == 0:
            self.stage_total_tokens[stats.stage_id] += int(stats.num_tokens_in)
        self.stage_events.setdefault(str(stats.request_id), []).append(stats)

        self.record_transfer_rx(stats)

    def record_stage_postprocess_time(self, stage_id: int, req_id: Any, postproc_time_ms: float) -> None:
        if req_id in self.stage_events:
            for stats in self.stage_events[req_id]:
                if stats.stage_id == stage_id:
                    stats.postprocess_time_ms = float(postproc_time_ms)
                    break
        else:
            logger.warning(
                "Failed to record postprocess time for request %s at stage %s: no stage event found",
                req_id,
                stage_id,
            )

    @contextmanager
    def stage_postprocess_timer(self, stage_id: int, req_id: Any):
        """Context manager for measuring and recording stage postprocessing time.

        Usage:
            with metrics.stage_postprocess_timer(stage_id, request_id):
                next_inputs = next_stage.process_engine_inputs(...)
        """
        _t0 = time.perf_counter()
        try:
            yield
        finally:
            _postproc_ms = (time.perf_counter() - _t0) * 1000.0
            self.record_stage_postprocess_time(stage_id, req_id, _postproc_ms)

    def accumulate_diffusion_metrics(self, stage_type: str, req_id: Any, engine_outputs: Any) -> None:
        """Accumulate diffusion metrics for a request.

        Handles extraction and accumulation of diffusion stage metrics.

        Args:
            req_id: Request ID
            engine_outputs: Engine output object containing metrics
        """
        if stage_type != "diffusion":
            return
        engine_output = engine_outputs[0] if isinstance(engine_outputs, list) and engine_outputs else engine_outputs
        diffusion_metrics: dict = getattr(engine_output, "metrics", {})
        if isinstance(diffusion_metrics, list):
            diffusion_metrics = diffusion_metrics[0]
        if diffusion_metrics:
            for key, value in diffusion_metrics.items():
                self.diffusion_metrics[req_id][key] += value

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
        self.record_transfer_tx(
            from_stage=from_stage,
            to_stage=to_stage,
            request_id=req_id,
            size_bytes=size_bytes,
            tx_time_ms=tx_ms,
            used_shm=used_shm,
        )

    def on_finalize_request(
        self,
        stage_id: int,
        req_id: Any,
        req_start_ts: float,
    ) -> None:
        rid_key = str(req_id)
        if rid_key in self.e2e_done:
            return  # Already finalized
        _t0 = float(req_start_ts)
        _t1 = time.time()
        # Update last output time for this stage
        prev_last = self.stage_last_ts[stage_id]
        self.stage_last_ts[stage_id] = _t1 if prev_last is None else max(prev_last, _t1)
        self.last_finish_ts = max(self.last_finish_ts, _t1)
        e2e_ms = (_t1 - _t0) * 1000.0

        # Sum tokens from all stages for this request
        # Include input tokens from stage 0 + output tokens from all stages
        total_tokens = 0
        if rid_key in self.stage_events:
            for evt in self.stage_events[rid_key]:
                if evt.stage_id == 0:
                    total_tokens += int(evt.num_tokens_in)
                total_tokens += int(evt.num_tokens_out)

        self.e2e_total_ms += e2e_ms
        self.e2e_total_tokens += total_tokens
        self.e2e_count += 1
        self.e2e_done.add(rid_key)
        per_req_record = RequestE2EStats(
            request_id=rid_key,
            e2e_total_ms=e2e_ms,
            e2e_total_tokens=total_tokens,
            transfers_total_time_ms=float(
                sum(evt.total_time_ms for evt in self.transfer_events.values() if evt.request_id == rid_key)
            ),
            transfers_total_bytes=int(
                sum(evt.size_bytes for evt in self.transfer_events.values() if evt.request_id == rid_key)
            ),
        )
        self.e2e_events.append(per_req_record)

    def build_and_log_summary(self) -> dict[str, Any]:
        if not self.log_stats:
            return {}
        wall_time_ms = max(0.0, (self.last_finish_ts - self.wall_start_ts) * 1000.0)
        e2e_avg_req = (wall_time_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        e2e_avg_tok = (self.e2e_total_tokens * 1000.0 / wall_time_ms) if wall_time_ms > 0 else 0.0

        if isinstance(self.final_stage_id_for_e2e, int):
            final_stage_id_map: dict[str, int] = {"*": int(self.final_stage_id_for_e2e)}
        else:
            final_stage_id_map = self.final_stage_id_for_e2e

        stage_wall_time_ms = [
            ((self.stage_last_ts[i] - self.stage_first_ts[i]) * 1000.0)
            if (self.stage_first_ts[i] is not None and self.stage_last_ts[i] is not None)
            else 0.0
            for i in range(self.num_stages)
        ]

        overall_summary = {
            "e2e_requests": int(self.e2e_count),
            "e2e_wall_time_ms": float(wall_time_ms),
            "e2e_total_tokens": int(self.e2e_total_tokens),
            "e2e_avg_time_per_request_ms": float(e2e_avg_req),
            "e2e_avg_tokens_per_s": float(e2e_avg_tok),
        }
        # Add stage_wall_time_ms as separate fields for each stage
        for idx, wall_time in enumerate(stage_wall_time_ms):
            overall_summary[f"e2e_stage_{idx}_wall_time_ms"] = wall_time

        # Print overall summary
        # filter out all-zero fields for logging
        overall_fields = []
        for k in OVERALL_FIELDS or list(overall_summary.keys()):
            v = overall_summary.get(k, None)
            if v not in (0, 0.0, 0.000, None, ""):
                overall_fields.append(k)
        if overall_fields:
            logger.info(
                "\n%s",
                _format_table("Overall Summary", overall_summary, overall_fields),
            )

        all_request_ids = sorted(set(self.stage_events.keys()) | {e.request_id for e in self.e2e_events})

        result_stage_table = []
        result_trans_table = []
        result_e2e_table = []

        for rid in all_request_ids:
            # === E2E table (single column) ===
            e2e_evt = next((e for e in self.e2e_events if e.request_id == rid), None)
            if e2e_evt:
                e2e_data = _build_row(e2e_evt, E2E_FIELDS)
                result_e2e_table.append({"request_id": rid, **e2e_data})

                # filter out all-zero fields for logging
                nonzero_e2e_fields = set()
                for k, v in e2e_data.items():
                    if v not in (0, 0.000, None, ""):
                        nonzero_e2e_fields.add(k)
                value_fields_e2e = sorted(nonzero_e2e_fields)

                if value_fields_e2e:
                    logger.info(
                        "\n%s",
                        _format_table(
                            f"RequestE2EStats [request_id={rid}]",
                            e2e_data,
                            value_fields=value_fields_e2e,
                        ),
                    )

            # === Stage table (columns = stage_id) ===
            stage_evts = sorted(
                self.stage_events.get(rid, []),
                key=lambda e: e.stage_id if e.stage_id is not None else -1,
            )
            # if any stage has diffusion_metrics, remove postprocess_time_ms field
            # because it is already included in diffusion_metrics
            local_exclude = STAGE_EXCLUDE.copy()
            has_diffusion_metrics = any(getattr(evt, "diffusion_metrics", None) for evt in stage_evts)
            if has_diffusion_metrics:
                local_exclude.add("postprocess_time_ms")
            local_stage_fields = _build_field_defs(StageRequestStats, local_exclude, FIELD_TRANSFORMS)

            # if diffusion_metrics is present, expand it into multiple columns
            # then remove diffusion_metrics from the table
            stage_rows = []
            for evt in stage_evts:
                row = {"stage_id": evt.stage_id, **_build_row(evt, local_stage_fields)}
                if evt.diffusion_metrics:
                    row.update(evt.diffusion_metrics)
                row.pop("diffusion_metrics", None)  # Remove the dict itself
                stage_rows.append(row)

            result_stage_table.append({"request_id": rid, "stages": stage_rows})

            if stage_rows:
                # filter out all-zero fields for logging
                all_value_fields = set()
                for row in stage_rows:
                    for k in row.keys():
                        if k != "stage_id":
                            all_value_fields.add(k)
                value_fields_list = []
                for field in sorted(all_value_fields):
                    all_zero = True
                    for row in stage_rows:
                        v = row.get(field, None)
                        if v not in (0, 0.0, 0.000, None, ""):
                            all_zero = False
                            break
                    if not all_zero:
                        value_fields_list.append(field)

                if value_fields_list:
                    logger.info(
                        "\n%s",
                        _format_table(
                            f"StageRequestStats [request_id={rid}]",
                            stage_rows,
                            column_key="stage_id",
                            value_fields=value_fields_list,
                        ),
                    )

            # === Transfer table (columns = edge) ===
            transfer_evts = sorted(
                [e for e in self.transfer_events.values() if e.request_id == rid],
                key=lambda e: (e.from_stage, e.to_stage),
            )
            transfer_rows = [
                {"edge": f"{evt.from_stage}->{evt.to_stage}", **_build_row(evt, TRANSFER_FIELDS)}
                for evt in transfer_evts
            ]
            result_trans_table.append({"request_id": rid, "transfers": transfer_rows})

            if transfer_rows:
                # filter out all-zero fields for logging
                all_value_fields = set()
                for row in transfer_rows:
                    for k in row.keys():
                        if k != "edge":
                            all_value_fields.add(k)
                value_fields_list = []
                for field in sorted(all_value_fields):
                    all_zero = True
                    for row in transfer_rows:
                        v = row.get(field, None)
                        if v not in (0, 0.0, 0.000, None, ""):
                            all_zero = False
                            break
                    if not all_zero:
                        value_fields_list.append(field)

                if value_fields_list:
                    logger.info(
                        "\n%s",
                        _format_table(
                            f"TransferEdgeStats [request_id={rid}]",
                            transfer_rows,
                            column_key="edge",
                            value_fields=value_fields_list,
                        ),
                    )

        return {
            "final_stage_id": final_stage_id_map,
            "overall_summary": overall_summary,
            "stage_table": result_stage_table,
            "trans_table": result_trans_table,
            "e2e_table": result_e2e_table,
        }
