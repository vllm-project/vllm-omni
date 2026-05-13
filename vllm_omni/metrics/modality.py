"""OmniModalityMetrics — per-modality Prometheus families (RFC G1/G2).

Audio / image / video business-semantic metric families with
``{model_name, stage, replica}`` labels. Text-path metrics (TTFT/ITL/TPOT)
are NOT here — they come from the upstream ``vllm:*{stage="thinker", ...}``
families exposed by ``OmniPrometheusStatLogger`` (Phase 2 wrap).

Phase 3 covers 8 of the 10 RFC families:
- audio: ttfp, duration, rtf, frames
- image: ttfp, num, generation_time
- video: generation_time

``video_duration_seconds`` and ``video_rtf`` are intentionally deferred —
diffusion video pipelines (i2v / t2v / cogvideo / hunyuan / wan) expose
num_frames + fps in heterogeneous shapes and a clean abstraction is out of
scope for this iteration.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Histogram

from vllm_omni.metrics import definitions as defs

_labelnames = list(defs.STAGE_LABELS)


# ----------------------------------------------------------------------------
# Audio family (G1) — observed at finalize except for ttfp which is observed
# at the streaming hook (first audio packet emerges).
# ----------------------------------------------------------------------------
_audio_ttfp_family = Histogram(
    defs.AUDIO_TTFP_SECONDS,
    "Time from request arrival to first audio packet, in seconds.",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)
_audio_duration_family = Histogram(
    defs.AUDIO_DURATION_SECONDS,
    "Generated audio content duration, in seconds (audio_frames / sample_rate).",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)
_audio_rtf_family = Histogram(
    defs.AUDIO_RTF_METRIC,
    "Audio real-time factor (stage_gen_time_s / audio_duration_s); SLO red line < 1.",
    labelnames=_labelnames,
    buckets=defs.RTF_BUCKETS,
)
_audio_frames_family = Counter(
    defs.AUDIO_FRAMES_METRIC,
    "Total audio frames generated; throughput recovered via rate().",
    labelnames=_labelnames,
)


# ----------------------------------------------------------------------------
# Image family (G2)
# ----------------------------------------------------------------------------
_image_ttfp_family = Histogram(
    defs.IMAGE_TTFP_SECONDS,
    "Time from request arrival to first image (or only image) emitted, in seconds.",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)
_image_num_family = Counter(
    defs.IMAGE_NUM_METRIC,
    "Total images generated; throughput recovered via rate().",
    labelnames=_labelnames,
)
_image_generation_time_family = Histogram(
    defs.IMAGE_GENERATION_TIME_SECONDS,
    "Per-request image stage generation time, in seconds. Image has no RTF "
    "(no content duration), so generation time is exposed independently.",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)


# ----------------------------------------------------------------------------
# Video family (G2) — only generation_time this iteration; duration/rtf
# require num_frames + fps from heterogeneous diffusion pipelines.
# ----------------------------------------------------------------------------
_video_generation_time_family = Histogram(
    defs.VIDEO_GENERATION_TIME_SECONDS,
    "Per-request video stage generation time, in seconds.",
    labelnames=_labelnames,
    buckets=defs.SECONDS_BUCKETS,
)


class OmniModalityMetrics:
    """Per-modality observe API. Stage/replica are passed at observe time
    because a single OmniModalityMetrics instance per pipeline serves all
    stage+replica combinations.

    See RFC §3.2.6.
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

    # ---- Image ------------------------------------------------------------

    def observe_image_ttfp(self, stage: str, replica: str, ttfp_seconds: float) -> None:
        _image_ttfp_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(ttfp_seconds)

    def inc_image_num(self, stage: str, replica: str, n_images: int) -> None:
        if n_images <= 0:
            return
        _image_num_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).inc(n_images)

    def observe_image_generation_time(
        self, stage: str, replica: str, gen_time_seconds: float
    ) -> None:
        _image_generation_time_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(gen_time_seconds)

    # ---- Video ------------------------------------------------------------

    def observe_video_generation_time(
        self, stage: str, replica: str, gen_time_seconds: float
    ) -> None:
        _video_generation_time_family.labels(
            model_name=self._model_name, stage=stage, replica=replica
        ).observe(gen_time_seconds)


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
    streaming hook (Phase 3.3) at first-packet time, not at finalize.
    """
    if replica_id is None or stage_metrics is None or output_type is None:
        return
    if output_type not in ("audio", "image", "video"):
        return

    stage_label = str(stage_id)
    replica_label = str(replica_id)
    gen_time_s = float(getattr(stage_metrics, "stage_gen_time_ms", 0.0)) / 1000.0

    if output_type == "audio":
        mm_out = getattr(engine_outputs, "multimodal_output", None) or {}
        sample_rate = defs.resolve_audio_sample_rate(mm_out)
        n_frames = int(getattr(stage_metrics, "audio_generated_frames", 0) or 0)
        mod_metrics.inc_audio_frames(stage_label, replica_label, n_frames)
        duration_s = n_frames / sample_rate if sample_rate > 0 else 0.0
        if duration_s > 0:
            mod_metrics.observe_audio_duration(stage_label, replica_label, duration_s)
            mod_metrics.observe_audio_rtf(
                stage_label,
                replica_label,
                defs.compute_audio_rtf(gen_time_s, duration_s),
            )
    elif output_type == "image":
        n_images = len(getattr(engine_outputs, "images", []) or [])
        mod_metrics.inc_image_num(stage_label, replica_label, n_images)
        mod_metrics.observe_image_generation_time(
            stage_label, replica_label, gen_time_s
        )
        image_ttfp_s = max(finalize_ts - request_arrival_ts, 0.0)
        mod_metrics.observe_image_ttfp(stage_label, replica_label, image_ttfp_s)
    else:  # video
        mod_metrics.observe_video_generation_time(
            stage_label, replica_label, gen_time_s
        )


def observe_audio_first_packet(
    mod_metrics: OmniModalityMetrics,
    *,
    stage_id: int,
    replica_id: int | None,
    arrival_ts: float,
    now_ts: float,
) -> None:
    """Observe audio_ttfp_seconds on a request's first audio packet.

    Caller is responsible for the once-per-request guard (e.g. checking
    ``ClientRequestState.first_audio_ts is None``) so this function fires at
    most once per request_id. Defensive-skips when ``replica_id`` or
    ``arrival_ts`` is insufficient — both can legitimately be missing in error
    paths and we'd rather drop the sample than emit a wrong (stage, replica).

    Phase 3.3 — companion to ``observe_modality_at_finalize`` which handles the
    other 7 modality families at finalize time.
    """
    if replica_id is None or arrival_ts <= 0:
        return
    ttfp = max(now_ts - arrival_ts, 0.0)
    mod_metrics.observe_audio_ttfp(str(stage_id), str(replica_id), ttfp)
