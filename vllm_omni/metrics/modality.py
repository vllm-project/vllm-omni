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
- ``_extract_mm_output`` / ``_count_audio_frames``: shape-tolerant helpers
  for the heterogeneous multimodal_output payloads emitted by different
  audio pipelines.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Histogram

from vllm_omni.metrics import definitions as defs

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


def _extract_mm_output(engine_outputs: Any) -> dict[str, Any]:
    """Return the multimodal_output dict regardless of where it's nested.

    Three shapes seen in the wild:
      * ``engine_outputs.multimodal_output`` — synthesized on OmniRequestOutput
        for some pipelines (often empty for AR audio)
      * ``engine_outputs.outputs[0].multimodal_output`` — vllm CompletionOutput
        nesting (where actual qwen3-omni audio data lives)
      * neither — returns ``{}``
    """
    mm = getattr(engine_outputs, "multimodal_output", None)
    if isinstance(mm, dict) and mm:
        return mm
    outs = getattr(engine_outputs, "outputs", None)
    if outs:
        nested = getattr(outs[0], "multimodal_output", None)
        if isinstance(nested, dict):
            return nested
    return {}


def _count_audio_frames(mm_out: dict[str, Any]) -> int:
    """Sum the per-tensor sample count of audio chunks in mm_out["audio"].

    Returns the total number of audio frames (samples) across all chunks.
    For multi-dim tensors (e.g. shape [channels, samples]) the last axis is
    treated as the sample dim; for 1-D tensors the only axis is the sample
    dim; scalars count as 1.
    """
    audio_chunks = mm_out.get("audio") if isinstance(mm_out, dict) else None
    if audio_chunks is None:
        return 0
    chunks = audio_chunks if isinstance(audio_chunks, list) else [audio_chunks]
    n = 0
    for t in chunks:
        try:
            ndim = getattr(t, "ndim", 0)
            shape = getattr(t, "shape", None)
            if ndim == 0 or shape is None or len(shape) == 0:
                n += 1
            else:
                n += int(shape[-1])
        except Exception:
            continue
    return n


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
    if output_type != "audio":
        return

    stage_label = str(stage_id)
    replica_label = str(replica_id)
    gen_time_s = float(getattr(stage_metrics, "stage_gen_time_ms", 0.0)) / 1000.0
    mm_out = _extract_mm_output(engine_outputs)

    sample_rate = defs.resolve_audio_sample_rate(mm_out)
    # `stage_metrics.audio_generated_frames` is the legacy per-chunk
    # accumulator field on StageRequestStats. No production path currently
    # fills it, so the fallback below is the live source — but we leave the
    # field lookup in place in case the accumulator gets re-wired upstream.
    n_frames = int(getattr(stage_metrics, "audio_generated_frames", 0) or 0)
    if n_frames == 0:
        n_frames = _count_audio_frames(mm_out)
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
