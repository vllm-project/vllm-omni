"""OmniModalityMetrics — per-modality Prometheus families.

Audio / image / video business-semantic metric families. Text-path metrics
(TTFT / ITL / TPOT / e2e) are NOT here — they come from the upstream
``vllm:*{stage="thinker", ...}`` families exposed by ``OmniPrometheusStatLogger``
(G7 wrap).

Family count and layout match the locked RFC
https://github.com/vllm-project/vllm-omni/issues/3545 ("vllm_omni:* complete
set"): 7 audio + 6 visual business / generation families.
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
    "Audio real-time factor (stage_gen_time_s / audio_duration_s); "
    "streaming TTS requires < 1.",
    labelnames=_stage_labels,
    buckets=defs.RTF_BUCKETS,
)
_audio_frames_family = Counter(
    defs.AUDIO_FRAMES_METRIC,
    "Cumulative audio frame count; per-model rate (not summable across models). "
    "Throughput recovered via rate().",
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
    defs.AUDIO_CONTINUITY_OK,
    "Incremented when the request's worst underrun stayed below threshold_ms. "
    "Pair with requests_success_total to compute streaming-continuity health rate.",
    labelnames=list(defs.AUDIO_CONTINUITY_LABELS),
)
_audio_skipped_family = Counter(
    defs.AUDIO_SKIPPED_REQUESTS,
    "Silent-loss counter — code2wav rejected malformed codec input and "
    "returned 200 OK with empty audio. Refs RFC §3.2.3.",
    labelnames=list(defs.AUDIO_SKIPPED_LABELS),
)


# ----------------------------------------------------------------------------
# Visual family — business semantics (image counts + per-request stage gen
# times + video RTF). Diffusion-internal timings (preprocess / exec /
# postprocess) live in prometheus.py because they are sourced from the engine
# outputs dict, not from a finalize-time hook.
# ----------------------------------------------------------------------------
_image_num_family = Counter(
    defs.IMAGE_NUM_METRIC,
    "Cumulative image count; throughput recovered via rate().",
    labelnames=_stage_labels,
)
_image_generation_family = Histogram(
    defs.IMAGE_GENERATION_S,
    "Per-request total image-generation stage time in seconds. Image has no "
    "RTF (no content duration) so generation time is exposed directly.",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_video_duration_family = Histogram(
    defs.VIDEO_DURATION_S,
    "Video content duration in seconds (num_frames / fps).",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)
_video_rtf_family = Histogram(
    defs.VIDEO_RTF_METRIC,
    "Video real-time factor (stage_gen_time_s / video_duration_s).",
    labelnames=_stage_labels,
    buckets=defs.RTF_BUCKETS,
)
_video_generation_family = Histogram(
    defs.VIDEO_GENERATION_S,
    "Per-request total video-generation stage time in seconds. Complements "
    "video_rtf (reverse-computing from RTF is imprecise).",
    labelnames=_stage_labels,
    buckets=defs.SECONDS_BUCKETS,
)


class OmniModalityMetrics:
    """Per-modality observe API. Stage/replica are passed at observe time
    because a single OmniModalityMetrics instance per pipeline serves all
    stage+replica combinations.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    # ---- Audio ------------------------------------------------------------

    def observe_audio_ttfp(self, stage: str, replica: str, ttfp_seconds: float) -> None:
        _audio_ttfp_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(ttfp_seconds)

    def observe_audio_duration(self, stage: str, replica: str, duration_seconds: float) -> None:
        _audio_duration_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(duration_seconds)

    def observe_audio_rtf(self, stage: str, replica: str, rtf: float) -> None:
        _audio_rtf_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(rtf)

    def inc_audio_frames(self, stage: str, replica: str, n_frames: int) -> None:
        if n_frames <= 0:
            return
        _audio_frames_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).inc(n_frames)

    def observe_audio_underrun(self, stage: str, replica: str, underrun_s: float) -> None:
        _audio_underrun_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(max(underrun_s, 0.0))

    def inc_audio_continuity_ok(
        self, stage: str, replica: str, threshold_ms: int
    ) -> None:
        _audio_continuity_ok_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
            threshold_ms=str(int(threshold_ms)),
        ).inc()

    def inc_audio_skipped(self, stage: str, replica: str, reason: str) -> None:
        _audio_skipped_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
            reason=reason or "unknown",
        ).inc()

    # ---- Image ------------------------------------------------------------

    def inc_image_num(self, stage: str, replica: str, n_images: int) -> None:
        if n_images <= 0:
            return
        _image_num_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).inc(n_images)

    def observe_image_generation(
        self, stage: str, replica: str, gen_time_seconds: float
    ) -> None:
        _image_generation_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(gen_time_seconds)

    # ---- Video ------------------------------------------------------------

    def observe_video_duration(self, stage: str, replica: str, duration_s: float) -> None:
        _video_duration_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(duration_s)

    def observe_video_rtf(self, stage: str, replica: str, rtf: float) -> None:
        _video_rtf_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(rtf)

    def observe_video_generation(
        self, stage: str, replica: str, gen_time_seconds: float
    ) -> None:
        _video_generation_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(gen_time_seconds)


def _resolve_video_duration_seconds(
    engine_outputs: Any,
    multimodal_output: dict[str, Any],
) -> float:
    """Best-effort `num_frames / fps` extraction from heterogeneous video pipelines.

    Diffusion video stages expose num_frames + fps via multimodal_output or
    engine_outputs attributes in shapes that differ across pipelines (i2v /
    t2v / hunyuan / wan). Returns 0.0 when either signal is missing — caller
    skips the observation rather than emitting a wrong sample.
    """
    video_meta = multimodal_output.get("video") if multimodal_output else None
    if isinstance(video_meta, dict):
        num_frames = video_meta.get("num_frames") or video_meta.get("frames")
        fps = video_meta.get("fps") or video_meta.get("frame_rate")
    else:
        num_frames = getattr(engine_outputs, "num_frames", None)
        fps = getattr(engine_outputs, "fps", None) or getattr(engine_outputs, "frame_rate", None)
    try:
        n = float(num_frames or 0)
        f = float(fps or 0)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0 or f <= 0:
        return 0.0
    return n / f


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
    request_arrival_ts: float,
    finalize_ts: float,
) -> None:
    """Route per-modality observations for a finalized request.

    Used by ``omni_base._process_single_result`` inside the e2e_done finalize
    guard so it fires once per request. Skips text path (covered by upstream
    ``vllm:*{stage="thinker", ...}``) and any case where required inputs are
    missing — caller should not need to pre-validate.

    audio_ttfp is intentionally NOT observed here; it's emitted by the
    streaming hook at first-packet time, not at finalize.
    """
    if replica_id is None or stage_metrics is None or output_type is None:
        return
    if output_type not in ("audio", "image", "video"):
        return

    stage_label = str(stage_id)
    replica_label = str(replica_id)
    gen_time_s = float(getattr(stage_metrics, "stage_gen_time_ms", 0.0)) / 1000.0
    mm_out = _extract_mm_output(engine_outputs)

    if output_type == "audio":
        sample_rate = defs.resolve_audio_sample_rate(mm_out)
        # Prefer the accumulator on stage_metrics (kept by
        # OrchestratorAggregator.record_audio_generated_frames where the
        # per-chunk wiring is active); fall back to deriving from the
        # multimodal_output payload directly so the audio family series fire
        # even when the per-chunk accumulator path didn't run.
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
        # audio_underrun / continuity are emitted from the streaming
        # path in observe_audio_streaming_finalize; finalize is too late for
        # the per-chunk timeline they need.
    elif output_type == "image":
        n_images = len(getattr(engine_outputs, "images", []) or [])
        mod_metrics.inc_image_num(stage_label, replica_label, n_images)
        mod_metrics.observe_image_generation(stage_label, replica_label, gen_time_s)
    else:  # video
        mod_metrics.observe_video_generation(stage_label, replica_label, gen_time_s)
        duration_s = _resolve_video_duration_seconds(engine_outputs, mm_out)
        if duration_s > 0:
            mod_metrics.observe_video_duration(stage_label, replica_label, duration_s)
            mod_metrics.observe_video_rtf(
                stage_label,
                replica_label,
                defs.compute_video_rtf(gen_time_s, duration_s),
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
    server-side and bench-side definitions stay aligned (RFC G4). Caller is
    responsible for collecting per-chunk arrival timestamps and byte sizes
    during the streaming response.
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
        mod_metrics.inc_audio_continuity_ok(
            stage_label, replica_label, int(threshold_s * 1000)
        )
