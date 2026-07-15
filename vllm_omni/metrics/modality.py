"""OmniModalityMetrics — per-modality Prometheus families (audio path only).

7 audio business-semantic metric families. Text-path metrics (TTFT / ITL /
TPOT / e2e) are NOT here — they come from the upstream
``vllm:*{stage="thinker", ...}`` families exposed via the
``OmniPrometheusStatLogger`` wrap.

Contents:
- Audio family declarations (Histograms + Counters)
- ``OmniModalityMetrics``: label-bound observe API for the audio family
- ``observe_modality_at_finalize``: dispatcher called from omni_base's e2e
  finalize hook; currently handles the audio path only.
- ``observe_audio_first_packet``: TTFP emit from the streaming SSE first
  audio packet.
- ``observe_audio_streaming_finalize``: emits ``audio_underrun_s`` +
  ``audio_continuity_ok_total`` at SSE close using accumulated per-chunk
  arrival timestamps.
- ``extract_mm_output`` / ``count_audio_frames`` (from ``metrics.utils``):
  shape-tolerant helpers for the heterogeneous multimodal_output payloads
  emitted by different audio pipelines.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Histogram

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.utils import count_audio_frames, extract_mm_output

_stage_labels = list(defs.STAGE_LABELS)


# ----------------------------------------------------------------------------
# Audio family
# ----------------------------------------------------------------------------
_audio_ttfp_family = Histogram(
    defs.AUDIO_TTFP_S,
    "Time from request arrival to first audio packet/frame, in seconds.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_audio_duration_family = Histogram(
    defs.AUDIO_DURATION_S,
    "Generated audio content duration in seconds (audio_frames / sample_rate).",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_audio_rtf_family = Histogram(
    defs.AUDIO_RTF_METRIC,
    "Audio real-time factor (stage_gen_time_s / audio_duration_s); streaming TTS requires < 1.",
    labelnames=_stage_labels,
    buckets=defs.RTF_BUCKETS,
)
_audio_frames_family = Counter(
    defs.AUDIO_FRAMES_METRIC,
    "Cumulative audio frame count; per-model rate (not summable across models). Throughput recovered via rate().",
    labelnames=_stage_labels,
)
_audio_underrun_family = Histogram(
    defs.AUDIO_UNDERRUN_S,
    "Per-request worst-case player-deficit in seconds (max time the player "
    "ran out of buffered audio). > 0 indicates listener experienced silent gaps.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_audio_continuity_ok_family = Counter(
    defs.AUDIO_CONTINUITY_OK_METRIC,
    "Incremented when the request's worst underrun stayed below threshold_ms. "
    "Pair with requests_success_total to compute streaming-continuity health rate.",
    labelnames=list(defs.AUDIO_CONTINUITY_LABELS),
)
_audio_skipped_family = Counter(
    defs.AUDIO_SKIPPED_REQUESTS_METRIC,
    "Silent-loss counter — code2wav rejected malformed codec input and returned 200 OK with empty audio.",
    labelnames=list(defs.AUDIO_SKIPPED_LABELS),
)


# ----------------------------------------------------------------------------
# Diffusion family
# ----------------------------------------------------------------------------
_diffusion_exec_family = Histogram(
    defs.DIFFUSION_EXEC_S,
    "DiT forward pass execution time per request in seconds.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_diffusion_exec_per_step_family = Histogram(
    defs.DIFFUSION_EXEC_PER_STEP_S,
    "DiT forward pass execution time per denoising step in seconds.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_diffusion_preprocess_family = Histogram(
    defs.DIFFUSION_PREPROCESS_S,
    "Diffusion input preprocessing time per request in seconds.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_diffusion_postprocess_family = Histogram(
    defs.DIFFUSION_POSTPROCESS_S,
    "Diffusion output postprocessing (VAE decode) time per request in seconds.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_FAST_BUCKETS,
)


class OmniModalityMetrics:
    """Per-modality observe API. Stage/replica are passed at observe time
    because a single OmniModalityMetrics instance per pipeline serves all
    stage+replica combinations.
    """

    def __init__(self, model_name: str, log_stats: bool = True) -> None:
        self._model_name = model_name
        self._log_stats = log_stats

    # ---- Audio ------------------------------------------------------------

    def observe_audio_ttfp(self, stage: str, replica: str, ttfp_seconds: float) -> None:
        if not self._log_stats:
            return
        _audio_ttfp_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(ttfp_seconds)

    def observe_audio_duration(self, stage: str, replica: str, duration_seconds: float) -> None:
        if not self._log_stats:
            return
        _audio_duration_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(
            duration_seconds
        )

    def observe_audio_rtf(self, stage: str, replica: str, rtf: float) -> None:
        if not self._log_stats:
            return
        _audio_rtf_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(rtf)

    def inc_audio_frames(self, stage: str, replica: str, n_frames: int) -> None:
        if not self._log_stats or n_frames <= 0:
            return
        _audio_frames_family.labels(model_name=self._model_name, stage=stage, replica=replica).inc(n_frames)

    def observe_audio_underrun(self, stage: str, replica: str, underrun_s: float) -> None:
        if not self._log_stats:
            return
        _audio_underrun_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(
            max(underrun_s, 0.0)
        )

    def inc_audio_continuity_ok(self, stage: str, replica: str, threshold_ms: int) -> None:
        if not self._log_stats:
            return
        _audio_continuity_ok_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
            threshold_ms=str(int(threshold_ms)),
        ).inc()

    def inc_audio_skipped(self, stage: str, replica: str, reason: str) -> None:
        if not self._log_stats:
            return
        _audio_skipped_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
            reason=reason or "unknown",
        ).inc()

    # ---- Diffusion --------------------------------------------------------

    def observe_diffusion_exec(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats:
            return
        _diffusion_exec_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
        ).observe(seconds)

    def observe_diffusion_exec_per_step(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats:
            return
        _diffusion_exec_per_step_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
        ).observe(seconds)

    def observe_diffusion_preprocess(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats:
            return
        _diffusion_preprocess_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
        ).observe(seconds)

    def observe_diffusion_postprocess(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats:
            return
        _diffusion_postprocess_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
        ).observe(seconds)


def observe_modality_at_finalize(
    mod_metrics: OmniModalityMetrics,
    *,
    output_type: str | None,
    stage_id: int,
    replica_id: int | None,
    stage_metrics: Any,
    engine_outputs: Any,
) -> None:
    """Route audio-path observations for a finalized request.

    Used by ``omni_base._process_single_result`` inside the e2e_done finalize
    guard so it fires once per request. Skips text path (covered by upstream
    ``vllm:*{stage="thinker", ...}``) and any case where required inputs are
    missing — caller should not need to pre-validate.

    audio_ttfp is intentionally NOT observed here; it's emitted by the
    streaming hook at first-packet time, not at finalize.
    """
    if replica_id is None or stage_metrics is None or output_type is None:
        return

    stage_label = str(stage_id)
    replica_label = str(replica_id)
    gen_time_s = float(getattr(stage_metrics, "stage_gen_time_ms", 0.0)) / 1000.0
    mm_out = extract_mm_output(engine_outputs)

    sample_rate = defs.resolve_audio_sample_rate(mm_out)
    # `stage_metrics.audio_generated_frames` is the legacy per-chunk
    # accumulator field on StageRequestStats. No production path currently
    # fills it, so the fallback below is the live source — but we leave the
    # field lookup in place in case the accumulator gets re-wired upstream.
    n_frames = int(getattr(stage_metrics, "audio_generated_frames", 0) or 0)
    if n_frames == 0:
        n_frames = count_audio_frames(mm_out)
    mod_metrics.inc_audio_frames(stage_label, replica_label, n_frames)
    duration_s = n_frames / sample_rate if sample_rate > 0 else 0.0
    if duration_s > 0:
        mod_metrics.observe_audio_duration(stage_label, replica_label, duration_s)
        mod_metrics.observe_audio_rtf(
            stage_label,
            replica_label,
            defs.compute_audio_rtf(gen_time_s, duration_s),
        )
    else:
        # Request completed (finish_reason ∈ {stop, length} — error paths
        # don't reach finalize) but no audio samples were produced. Covers
        # silent `return None` skips in the talker→code2wav stage
        # processors and the `parsed.append((0,0))` malformed-length path
        # in qwen3-tts code2wav. raise-paths surface via the upstream
        # vllm:request_success_total{finished_reason="error"} channel and
        # never reach this branch.
        mod_metrics.inc_audio_skipped(stage_label, replica_label, "no_audio_data")
    # audio_underrun / continuity are emitted from the streaming path in
    # observe_audio_streaming_finalize; finalize is too late for the
    # per-chunk timeline they need.

    dm = getattr(stage_metrics, "diffusion_metrics", None)
    if dm:
        _key_map = {
            "diffusion_engine_exec_time_s": mod_metrics.observe_diffusion_exec,
            "preprocess_time_s": mod_metrics.observe_diffusion_preprocess,
            "postprocess_time_s": mod_metrics.observe_diffusion_postprocess,
        }
        for key, observe_fn in _key_map.items():
            val = dm.get(key)
            if val is not None:
                observe_fn(stage_label, replica_label, float(val))

        exec_time = dm.get("diffusion_engine_exec_time_s")
        num_steps = dm.get("num_inference_steps")
        if exec_time is not None and num_steps and num_steps > 0:
            mod_metrics.observe_diffusion_exec_per_step(
                stage_label,
                replica_label,
                float(exec_time) / int(num_steps),
            )


def observe_audio_first_packet(
    mod_metrics: OmniModalityMetrics,
    *,
    stage_id: int,
    replica_id: int | None,
    arrival_ts: float,
    now_ts: float,
) -> None:
    """Observe audio_ttfp_s on a request's first audio packet.

    Caller is responsible for the once-per-request guard (e.g. checking
    ``ClientRequestState.first_audio_ts is None``) so this function fires at
    most once per request_id. Defensive-skips when ``replica_id`` or
    ``arrival_ts`` is insufficient — both can legitimately be missing in error
    paths and we'd rather drop the sample than emit a wrong (stage, replica).
    """
    if replica_id is None or arrival_ts <= 0:
        return
    ttfp = max(now_ts - arrival_ts, 0.0)
    mod_metrics.observe_audio_ttfp(str(stage_id), str(replica_id), ttfp)


def observe_audio_streaming_finalize(
    mod_metrics: OmniModalityMetrics,
    *,
    stage_id: int,
    replica_id: int | None,
    chunk_arrival_times_s: list[float],
    chunk_bytes: list[int],
    sample_rate: int,
    threshold_s: float = defs.AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S,
) -> None:
    """Emit audio_underrun_s + audio_continuity_ok_total at request end.

    Reuses the math from ``vllm_omni.benchmarks.audio_continuity`` so the
    server-side and bench-side definitions stay aligned. Caller is responsible
    for collecting per-chunk arrival timestamps and byte sizes during the
    streaming response.
    """
    if replica_id is None or not chunk_arrival_times_s:
        return
    # Local import to keep the bench module optional at import time.
    from vllm_omni.benchmarks.audio_continuity import compute_continuity_stats

    stats = compute_continuity_stats(
        chunk_arrival_times_s=chunk_arrival_times_s,
        chunk_bytes=chunk_bytes,
        sample_rate=sample_rate,
        threshold_s=threshold_s,
    )
    stage_label = str(stage_id)
    replica_label = str(replica_id)
    mod_metrics.observe_audio_underrun(stage_label, replica_label, stats.max_underrun_s)
    if stats.is_continuous:
        mod_metrics.inc_audio_continuity_ok(stage_label, replica_label, int(threshold_s * 1000))
