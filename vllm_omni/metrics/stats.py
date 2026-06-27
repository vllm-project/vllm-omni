from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.utils import _build_field_defs, _build_row, _format_table

if TYPE_CHECKING:
    from vllm_omni.metrics.transfer import OmniTransferMetrics

logger = init_logger(__name__)


@dataclass
class StageStats:
    total_token: int = 0
    total_gen_time_ms: float = 0.0

    @property
    def avg_tokens_per_s(self) -> float:
        return (self.total_token * 1000.0 / self.total_gen_time_ms) if self.total_gen_time_ms > 0 else 0.0


@dataclass
class PipelineTimings:
    """Orchestrator-level pipeline timing metrics (all in milliseconds)."""

    queue_wait_ms: float = 0.0
    preprocess_ms: float = 0.0
    ar2diffusion_ms: float = 0.0


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
    replica_id: int | None = None
    final_output_type: str | None = None
    request_id: str | None = None
    postprocess_time_ms: float = 0.0
    diffusion_metrics: dict[str, float] = None
    audio_generated_frames: int = 0
    audio_sample_rate: int = 0
    audio_duration_s: float = 0.0
    audio_rtf: float = 0.0
    image_pixels: int = 0
    denoise_step_latency_ms: float = 0.0
    pipeline_timings: PipelineTimings | None = None
    output_unit_type: str | None = None
    output_unit_count: int = 0
    serving_time_to_first_output_ms: float = 0.0
    time_per_output_unit_ms: float = 0.0
    inter_output_latency_ms: float = 0.0
    inter_output_latencies_ms: list[float] | None = None
    vllm_ttft_ms: float = 0.0
    vllm_tpot_ms: float = 0.0
    vllm_itl_ms: float = 0.0
    vllm_itls_ms: list[float] | None = None

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
    "pipeline_timings",
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
        *,
        transfer_emitter: OmniTransferMetrics | None = None,
        replica_resolver: Callable[[int, str], int | None] | None = None,
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
        # Emit per-physical-transfer Histogram observations to Prometheus
        # alongside the existing TransferEdgeStats accumulation. Both deps
        # are optional so OrchestratorAggregator stays usable in contexts
        # that don't have a Prometheus registry (e.g. unit tests).
        self._transfer_emitter = transfer_emitter
        self._replica_resolver = replica_resolver
        # Snapshot of (stage_id -> replica_id) per request, captured at
        # ``record_transfer_rx`` time from the receiving stage's own
        # ``StageRequestStats.replica_id``. Survives pool-side
        # ``release_binding`` so the transfer emit can resolve label values
        # even when the downstream stage's binding has already been released
        # by the orchestrator's cleanup path (e.g. final-stage cleanup races
        # the omni_base finalize emit).
        self._replica_cache: dict[tuple[int, str], int] = {}

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
            # Emit per-physical-transfer Histogram observations to Prometheus.
            self._emit_transfer_tx(
                from_stage=int(from_stage),
                to_stage=int(to_stage),
                request_id=str(request_id),
                size_bytes=int(size_bytes),
                tx_time_ms=float(tx_time_ms),
            )
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
            # Snapshot the receiving (stage, replica) binding before the
            # pool may release it on cleanup. Used by _resolve_edge_replicas
            # as a fallback when get_bound_replica_id has already cleared.
            if stats.replica_id is not None:
                self._replica_cache[(to_stage, rid_key)] = int(stats.replica_id)
            evt = self._get_or_create_transfer_event(from_stage, to_stage, rid_key)
            # Accumulate rx metrics
            if evt.size_bytes == 0:
                # size_bytes has been recorded in tx phase
                evt.size_bytes = int(stats.rx_transfer_bytes)
            evt.rx_decode_time_ms += float(stats.rx_decode_time_ms)
            evt.in_flight_time_ms += float(stats.rx_in_flight_time_ms)
            # Emit per-physical-receive Histogram observations to Prometheus.
            self._emit_transfer_rx(
                from_stage=from_stage,
                to_stage=to_stage,
                request_id=rid_key,
                rx_decode_time_ms=float(stats.rx_decode_time_ms),
                in_flight_time_ms=float(stats.rx_in_flight_time_ms),
            )
            return evt
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prometheus emit hooks. Both helpers are no-ops when transfer_emitter
    # or replica_resolver is None, or when the resolver cannot find a
    # (stage_id, request_id) -> replica_id mapping. We deliberately
    # fail-safe (skip) rather than emit a series with a wrong or invented
    # replica label.
    # ------------------------------------------------------------------

    def _resolve_edge_replicas(self, from_stage: int, to_stage: int, request_id: str) -> tuple[int, int] | None:
        # Prefer the per-request replica snapshot captured by
        # record_transfer_rx (survives pool binding release); fall back to the
        # resolver callback (pool.get_bound_replica_id) when the cache
        # misses, e.g. for the TX side that fires before the receiving stage
        # has recorded its stats.
        from_r = self._replica_cache.get((from_stage, request_id))
        to_r = self._replica_cache.get((to_stage, request_id))
        if (from_r is None or to_r is None) and self._replica_resolver is not None:
            if from_r is None:
                from_r = self._replica_resolver(from_stage, request_id)
            if to_r is None:
                to_r = self._replica_resolver(to_stage, request_id)
        if from_r is None or to_r is None:
            return None
        return int(from_r), int(to_r)

    def _emit_transfer_tx(
        self,
        *,
        from_stage: int,
        to_stage: int,
        request_id: str,
        size_bytes: int,
        tx_time_ms: float,
    ) -> None:
        if self._transfer_emitter is None:
            return
        replicas = self._resolve_edge_replicas(from_stage, to_stage, request_id)
        if replicas is None:
            return
        from_r, to_r = replicas
        self._transfer_emitter.observe_size(from_stage, from_r, to_stage, to_r, size_bytes)
        self._transfer_emitter.observe_tx_time(from_stage, from_r, to_stage, to_r, tx_time_ms / 1000.0)

    def _emit_transfer_rx(
        self,
        *,
        from_stage: int,
        to_stage: int,
        request_id: str,
        rx_decode_time_ms: float,
        in_flight_time_ms: float,
    ) -> None:
        if self._transfer_emitter is None:
            return
        replicas = self._resolve_edge_replicas(from_stage, to_stage, request_id)
        if replicas is None:
            return
        from_r, to_r = replicas
        self._transfer_emitter.observe_rx_time(from_stage, from_r, to_stage, to_r, rx_decode_time_ms / 1000.0)
        self._transfer_emitter.observe_in_flight_time(from_stage, from_r, to_stage, to_r, in_flight_time_ms / 1000.0)

    def record_audio_generated_frames(
        self,
        output_to_yield: Any,
        stage_id: int,
        request_id: str,
    ) -> None:
        try:
            if (
                output_to_yield.final_output_type == "audio"
                and (multimodal_output := output_to_yield.multimodal_output.get("audio")) is not None
                and len(multimodal_output) > 0
            ):
                nframes = sum(
                    int(t.shape[-1]) if t.ndim > 0 else 1
                    for t in (multimodal_output if isinstance(multimodal_output, list) else [multimodal_output])
                )
                stage_events_for_req = self.stage_events.get(request_id, [])
                if stage_events_for_req:
                    for stage_event in stage_events_for_req:
                        if stage_event.stage_id == stage_id:
                            if stage_event.audio_generated_frames <= 0:
                                stage_event.audio_generated_frames = nframes
                            break
                else:
                    logger.warning(
                        "Failed to record audio generated frames for request %s at stage %s: no stage event found",
                        request_id,
                        stage_id,
                    )
        except Exception:
            logger.debug(
                "Failed to record audio frames for request %s",
                request_id,
                exc_info=True,
            )

    def process_stage_metrics(
        self,
        *,
        result: dict[str, Any],
        stage_type: str,
        stage_id: int,
        req_id: str,
        engine_outputs: Any,
        finished: bool,
        final_output_type: str | None,
        output_to_yield: Any | None,
    ) -> None:
        """Process and record stage metrics.

        Args:
            result: Result dict containing metrics from stage
            stage_type: Type of the stage (e.g., 'llm', 'diffusion')
            stage_id: Stage identifier
            req_id: Request identifier
            engine_outputs: Engine output object
            finished: Whether stage processing is finished
            final_output_type: Type of final output (e.g., 'text', 'audio')
            output_to_yield: Output object to attach metrics to
        """
        try:
            _m: StageRequestStats | None = result.get("metrics")

            # 1. Accumulate metrics from stage stats
            if _m is not None:
                self.accumulated_gen_time_ms[req_id][stage_id] += _m.stage_gen_time_ms
                self.accumulate_diffusion_metrics(stage_type, req_id, engine_outputs)
                if finished:
                    self.on_stage_metrics(stage_id, req_id, _m, final_output_type)

            # 2. No output to yield, nothing more to do
            if output_to_yield is None:
                return

            rid_key = str(req_id)
            stage_snapshot = self._build_stage_metrics_snapshot(rid_key)

            # 3. Not finished yet — expose incremental per-stage snapshot for streaming clients.
            if not finished:
                if stage_snapshot:
                    output_to_yield.metrics = {"stage_metrics": stage_snapshot}
                else:
                    output_to_yield.metrics = {}
                return

            # 4. Finished with output: always attach per-stage snapshot; keep legacy top-level
            # token fields only for text stages (OpenAI-style completion token accounting).
            output_to_yield.metrics = {"stage_metrics": stage_snapshot}
            stage_event = next(
                (evt for evt in reversed(self.stage_events.get(rid_key, [])) if evt.stage_id == stage_id),
                None,
            )
            if stage_event is not None and stage_event.final_output_type == "text":
                output_to_yield.metrics.update(
                    {
                        defs.NUM_TOKENS_IN: stage_event.num_tokens_in,
                        defs.NUM_TOKENS_OUT: stage_event.num_tokens_out,
                        "stage_id": stage_event.stage_id,
                        "final_output_type": stage_event.final_output_type,
                    }
                )
            elif stage_event is not None:
                output_to_yield.metrics.update(
                    {
                        "stage_id": stage_event.stage_id,
                        "final_output_type": stage_event.final_output_type,
                    }
                )

            # 5. Finished: record audio generated frames
            self.record_audio_generated_frames(output_to_yield, stage_id, req_id)

        except Exception:
            logger.exception(
                "Failed to process metrics for stage %s, req %s",
                stage_id,
                req_id,
            )

    @staticmethod
    def _merge_stage_metric_event(
        current: dict[str, Any] | None,
        evt: StageRequestStats,
    ) -> dict[str, Any]:
        sid = int(evt.stage_id) if evt.stage_id is not None else -1
        if current is None:
            current = {
                "stage_id": sid,
                "final_output_type": evt.final_output_type,
                defs.NUM_TOKENS_IN: int(evt.num_tokens_in),
                defs.NUM_TOKENS_OUT: int(evt.num_tokens_out),
                defs.STAGE_GEN_TIME_MS: float(evt.stage_gen_time_ms),
                defs.POSTPROCESS_TIME_MS: float(evt.postprocess_time_ms),
                defs.AUDIO_FRAMES: int(evt.audio_generated_frames),
                defs.AUDIO_SAMPLE_RATE: int(evt.audio_sample_rate),
                f"{defs.AUDIO_DURATION}_s": float(evt.audio_duration_s),
                defs.AUDIO_RTF: float(evt.audio_rtf),
                defs.IMAGE_PIXELS: int(evt.image_pixels),
                defs.DENOISE_STEP_LATENCY_MS: float(evt.denoise_step_latency_ms),
                "output_unit_type": evt.output_unit_type,
                defs.OUTPUT_UNIT_COUNT: int(evt.output_unit_count),
                defs.SERVING_TIME_TO_FIRST_OUTPUT_MS: float(evt.serving_time_to_first_output_ms),
                defs.TIME_PER_OUTPUT_UNIT_MS: float(evt.time_per_output_unit_ms),
                defs.INTER_OUTPUT_LATENCY_MS: float(evt.inter_output_latency_ms),
                defs.INTER_OUTPUT_LATENCIES_MS: list(evt.inter_output_latencies_ms or []),
                defs.VLLM_TTFT_MS: float(evt.vllm_ttft_ms),
                defs.VLLM_TPOT_MS: float(evt.vllm_tpot_ms),
                defs.VLLM_ITL_MS: float(evt.vllm_itl_ms),
                defs.VLLM_ITLS_MS: list(evt.vllm_itls_ms or []),
            }
            return current

        current[defs.NUM_TOKENS_IN] = int(current.get(defs.NUM_TOKENS_IN, 0)) + int(evt.num_tokens_in)
        current[defs.NUM_TOKENS_OUT] = int(current.get(defs.NUM_TOKENS_OUT, 0)) + int(evt.num_tokens_out)
        current[defs.STAGE_GEN_TIME_MS] = float(current.get(defs.STAGE_GEN_TIME_MS, 0.0)) + float(evt.stage_gen_time_ms)
        current[defs.POSTPROCESS_TIME_MS] = float(current.get(defs.POSTPROCESS_TIME_MS, 0.0)) + float(
            evt.postprocess_time_ms
        )
        current[defs.AUDIO_FRAMES] = int(current.get(defs.AUDIO_FRAMES, 0)) + int(evt.audio_generated_frames)
        if int(current.get(defs.AUDIO_SAMPLE_RATE, 0)) <= 0 and int(evt.audio_sample_rate) > 0:
            current[defs.AUDIO_SAMPLE_RATE] = int(evt.audio_sample_rate)
        current[f"{defs.AUDIO_DURATION}_s"] = float(current.get(f"{defs.AUDIO_DURATION}_s", 0.0)) + float(
            evt.audio_duration_s
        )
        current[defs.IMAGE_PIXELS] = int(current.get(defs.IMAGE_PIXELS, 0)) + int(evt.image_pixels)
        denoise_step_latency_ms = float(evt.denoise_step_latency_ms)
        if denoise_step_latency_ms > 0:
            current[defs.DENOISE_STEP_LATENCY_MS] = denoise_step_latency_ms
        current[defs.OUTPUT_UNIT_COUNT] = int(current.get(defs.OUTPUT_UNIT_COUNT, 0)) + int(evt.output_unit_count)

        first_output_ms = float(evt.serving_time_to_first_output_ms)
        current_first_output_ms = float(current.get(defs.SERVING_TIME_TO_FIRST_OUTPUT_MS, 0.0))
        if current_first_output_ms <= 0 < first_output_ms:
            current[defs.SERVING_TIME_TO_FIRST_OUTPUT_MS] = first_output_ms

        if evt.output_unit_type:
            current["output_unit_type"] = evt.output_unit_type
        if evt.final_output_type:
            current["final_output_type"] = evt.final_output_type

        inter_latencies = list(current.get(defs.INTER_OUTPUT_LATENCIES_MS) or [])
        inter_latencies.extend(list(evt.inter_output_latencies_ms or []))
        current[defs.INTER_OUTPUT_LATENCIES_MS] = inter_latencies
        current[defs.INTER_OUTPUT_LATENCY_MS] = (
            sum(inter_latencies) / float(len(inter_latencies)) if inter_latencies else 0.0
        )

        vllm_ttft_ms = float(evt.vllm_ttft_ms)
        current_vllm_ttft_ms = float(current.get(defs.VLLM_TTFT_MS, 0.0))
        if current_vllm_ttft_ms <= 0 < vllm_ttft_ms:
            current[defs.VLLM_TTFT_MS] = vllm_ttft_ms
        vllm_tpot_ms = float(evt.vllm_tpot_ms)
        if vllm_tpot_ms > 0:
            current[defs.VLLM_TPOT_MS] = vllm_tpot_ms
        vllm_itls = list(current.get(defs.VLLM_ITLS_MS) or [])
        vllm_itls.extend(list(evt.vllm_itls_ms or []))
        current[defs.VLLM_ITLS_MS] = vllm_itls
        current[defs.VLLM_ITL_MS] = sum(vllm_itls) / float(len(vllm_itls)) if vllm_itls else 0.0

        output_count = int(current.get(defs.OUTPUT_UNIT_COUNT, 0))
        remaining_ms = max(
            float(current.get(defs.STAGE_GEN_TIME_MS, 0.0))
            - float(current.get(defs.SERVING_TIME_TO_FIRST_OUTPUT_MS, 0.0)),
            0.0,
        )
        current[defs.TIME_PER_OUTPUT_UNIT_MS] = remaining_ms / float(output_count - 1) if output_count > 1 else 0.0
        duration_s = float(current.get(f"{defs.AUDIO_DURATION}_s", 0.0))
        current[defs.AUDIO_RTF] = defs.compute_audio_rtf(
            float(current.get(defs.STAGE_GEN_TIME_MS, 0.0)) / 1000.0, duration_s
        )
        return current

    def _build_stage_metrics_snapshot(self, req_id: str) -> dict[str, dict[str, Any]]:
        """Aggregate per-stage metrics for ``req_id`` (string key), for streaming/benchmark clients."""
        snapshot: dict[str, dict[str, Any]] = {}
        for evt in self.stage_events.get(req_id, []):
            sid = int(evt.stage_id) if evt.stage_id is not None else -1
            if sid < 0:
                continue
            sid_key = str(sid)
            snapshot[sid_key] = self._merge_stage_metric_event(snapshot.get(sid_key), evt)
        return snapshot

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
        if final_output_type is not None:
            stats.final_output_type = final_output_type
        stats.diffusion_metrics = (
            {k: float(v) for k, v in self.diffusion_metrics.pop(req_id, {}).items()}
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
        # Drop per-request replica snapshots once the request is finalized.
        if self._replica_cache:
            for cached_key in [k for k in self._replica_cache if k[1] == rid_key]:
                self._replica_cache.pop(cached_key, None)

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
        # Add average input preprocess time across all requests
        preprocess_times = []
        for _rid, evts in self.stage_events.items():
            for evt in evts:
                if evt.pipeline_timings and evt.pipeline_timings.preprocess_ms > 0:
                    preprocess_times.append(evt.pipeline_timings.preprocess_ms)
                    break  # only once per request
        if preprocess_times:
            overall_summary["input_preprocess_time_ms"] = sum(preprocess_times) / len(preprocess_times)
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

            # === [OmniTiming] concise per-request summary ===
            stage_evts = sorted(
                self.stage_events.get(rid, []),
                key=lambda e: e.stage_id if e.stage_id is not None else -1,
            )
            pt = PipelineTimings()
            if stage_evts:
                pt = stage_evts[-1].pipeline_timings or PipelineTimings()
            if e2e_evt or (stage_evts and stage_evts[-1].pipeline_timings):
                parts = [f"req={rid}"]
                if e2e_evt:
                    parts.append(f"total={e2e_evt.e2e_total_ms / 1000.0:.2f}s")
                if pt.preprocess_ms > 0:
                    parts.append(f"preprocess={pt.preprocess_ms / 1000.0:.2f}s")
                if e2e_evt:
                    engine_ms = e2e_evt.e2e_total_ms - pt.preprocess_ms
                    parts.append(f"engine={engine_ms / 1000.0:.2f}s")
                stage_parts = []
                for evt in stage_evts:
                    sid = evt.stage_id if evt.stage_id is not None else "?"
                    t = evt.stage_gen_time_ms / 1000.0
                    stage_parts.append(f"{sid}:{t:.2f}s")
                if stage_parts:
                    parts.append(f"stages=[{','.join(stage_parts)}]")
                transfer_parts = []
                for te in self.transfer_events.values():
                    if te.request_id == rid:
                        transfer_parts.append(f"{te.from_stage}->{te.to_stage}={te.tx_time_ms:.2f}ms")
                if transfer_parts:
                    parts.append(f"transfers=[{','.join(transfer_parts)}]")
                if pt.ar2diffusion_ms > 0:
                    parts.append(f"ar2diffusion={pt.ar2diffusion_ms:.2f}ms")
                logger.info("[OmniTiming] %s", " ".join(parts))

            # === Stage table (columns = stage_id) ===
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
