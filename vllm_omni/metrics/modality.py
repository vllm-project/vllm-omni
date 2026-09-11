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
from vllm_omni.metrics.utils import observe_audio_finalize, observe_diffusion_finalize

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
_vae_decode_family = Histogram(
    defs.VAE_DECODE_S,
    "VAE decode latency in seconds (latents -> pixels/audio/video).",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_diffusion_forward_family = Histogram(
    defs.DIFFUSION_FORWARD_S,
    "Diffusion forward-only latency in seconds (denoise loop; excludes "
    "preprocess / postprocess / VAE decode / KV load). Absent when the "
    "pipeline profiler is off.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_diffusion_kv_load_family = Histogram(
    defs.DIFFUSION_KV_LOAD_S,
    "Diffusion KV-recv latency in seconds (AR→diffusion KV fetch; absent when "
    "the stage has no upstream KV to receive).",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_image_ttfp_family = Histogram(
    defs.IMAGE_TTFP_S,
    "Image time-to-first-output in seconds (stage submit → first image materialized; non-streaming single-image). "
    "Stage-level, not e2e — excludes API queue and inter-stage transfer before the image stage.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_denoise_step_latency_family = Histogram(
    defs.DENOISE_STEP_LATENCY_S,
    "Mean per-step denoise forward latency in seconds (forward_time / num_inference_steps).",
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

    def observe_vae_decode(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats or seconds < 0:
            return
        _vae_decode_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(seconds)

    def observe_diffusion_forward(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats or seconds < 0:
            return
        _diffusion_forward_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(seconds)

    def observe_diffusion_kv_load(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats or seconds < 0:
            return
        _diffusion_kv_load_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(seconds)

    def observe_image_ttfp(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats or seconds < 0:
            return
        _image_ttfp_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(seconds)

    def observe_denoise_step_latency(self, stage: str, replica: str, seconds: float) -> None:
        if not self._log_stats or seconds <= 0:
            return
        _denoise_step_latency_family.labels(model_name=self._model_name, stage=stage, replica=replica).observe(seconds)


def observe_modality_at_finalize(
    mod_metrics: OmniModalityMetrics,
    *,
    output_type: str | None,
    stage_id: int,
    replica_id: int | None,
    stage_metrics: Any,
    engine_outputs: Any,
) -> None:
    """Route per-modality observations for a finalized request.

    Used by ``omni_base._process_single_result`` inside the e2e_done finalize
    guard so it fires once per request. Text path falls through — covered by
    upstream ``vllm:*{stage="thinker", ...}``. Caller should not need to
    pre-validate; missing inputs are silently skipped.

    audio_ttfp is intentionally NOT observed here; it's emitted by the
    streaming hook at first-packet time, not at finalize.
    """
    if replica_id is None or stage_metrics is None:
        return

    observe_diffusion_finalize(
        mod_metrics,
        stage_id=stage_id,
        replica_id=replica_id,
        stage_metrics=stage_metrics,
    )

    if output_type == "audio":
        observe_audio_finalize(
            mod_metrics,
            stage_id=stage_id,
            replica_id=replica_id,
            stage_metrics=stage_metrics,
            engine_outputs=engine_outputs,
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
