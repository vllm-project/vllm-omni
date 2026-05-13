"""Single source of truth for vLLM-Omni Prometheus + bench CLI metric naming.

Consumed by:
- vllm_omni.metrics.prometheus (server-side /metrics families)
- vllm_omni.benchmarks.metrics.metrics (bench CLI MultiModalsBenchmarkMetrics)

RFC: vLLM-Omni Prometheus 多模态语义、跨 stage Transfer (G4/G5).
"""

# vllm:omni_ avoids upstream's unregister_vllm_metrics() stripping; matches PR #3362.
METRIC_PREFIX = "vllm:omni_"


# ============================================================================
# Bench-side stems (also used as RequestFuncOutput attribute names)
# ============================================================================
AUDIO_TTFP = "audio_ttfp"
AUDIO_DURATION = "audio_duration"
AUDIO_RTF = "audio_rtf"
AUDIO_FRAMES = "audio_frames"

IMAGE_TTFP = "image_ttfp"
IMAGE_NUM = "image_num"
IMAGE_GENERATION_TIME = "image_generation_time"

VIDEO_DURATION = "video_duration"
VIDEO_RTF = "video_rtf"
VIDEO_GENERATION_TIME = "video_generation_time"


# ============================================================================
# Pipeline-level metric families (PR #3362 + G6)
# ============================================================================
NUM_REQUESTS_RUNNING = METRIC_PREFIX + "num_requests_running"
NUM_REQUESTS_WAITING = METRIC_PREFIX + "num_requests_waiting"
NUM_REQUESTS_SUCCESS = METRIC_PREFIX + "num_requests_success"
NUM_REQUESTS_FAIL = METRIC_PREFIX + "num_requests_fail"
E2E_REQUEST_LATENCY_SECONDS = METRIC_PREFIX + "e2e_request_latency_seconds"
REQUEST_QUEUE_TIME_SECONDS = METRIC_PREFIX + "request_queue_time_seconds"

# G6: requests_success_total{finished_reason} — Pipeline 全局 Counter
REQUESTS_SUCCESS_TOTAL = METRIC_PREFIX + "requests_success_total"


# ============================================================================
# Audio family (G1)
# ============================================================================
AUDIO_TTFP_SECONDS = METRIC_PREFIX + AUDIO_TTFP + "_seconds"
AUDIO_DURATION_SECONDS = METRIC_PREFIX + AUDIO_DURATION + "_seconds"
AUDIO_RTF_METRIC = METRIC_PREFIX + AUDIO_RTF
AUDIO_FRAMES_METRIC = METRIC_PREFIX + AUDIO_FRAMES


# ============================================================================
# Image / Video family (G2)
# ============================================================================
IMAGE_TTFP_SECONDS = METRIC_PREFIX + IMAGE_TTFP + "_seconds"
IMAGE_NUM_METRIC = METRIC_PREFIX + IMAGE_NUM
IMAGE_GENERATION_TIME_SECONDS = METRIC_PREFIX + IMAGE_GENERATION_TIME + "_seconds"

VIDEO_DURATION_SECONDS = METRIC_PREFIX + VIDEO_DURATION + "_seconds"
VIDEO_RTF_METRIC = METRIC_PREFIX + VIDEO_RTF
VIDEO_GENERATION_TIME_SECONDS = METRIC_PREFIX + VIDEO_GENERATION_TIME + "_seconds"


# ============================================================================
# Diffusion ms-level timing (PR #3362)
# ============================================================================
DIFFUSION_PREPROCESS_TIME_MS = METRIC_PREFIX + "diffusion_preprocess_time_ms"
DIFFUSION_EXEC_TIME_MS = METRIC_PREFIX + "diffusion_exec_time_ms"
DIFFUSION_POSTPROCESS_TIME_MS = METRIC_PREFIX + "diffusion_postprocess_time_ms"
DIFFUSION_STEP_TIME_MS = METRIC_PREFIX + "diffusion_step_time_ms"


# ============================================================================
# Cross-stage Transfer family (G3)
# ============================================================================
TRANSFER_SIZE_BYTES = METRIC_PREFIX + "transfer_size_bytes"
TRANSFER_TX_TIME_MS = METRIC_PREFIX + "transfer_tx_time_ms"
TRANSFER_RX_DECODE_TIME_MS = METRIC_PREFIX + "transfer_rx_decode_time_ms"
TRANSFER_IN_FLIGHT_TIME_MS = METRIC_PREFIX + "transfer_in_flight_time_ms"


# ============================================================================
# Label sets
# ============================================================================
PIPELINE_LABELS = ("model_name",)
SUCCESS_LABELS = ("model_name", "finished_reason")

# Per-stage / per-replica label set used by audio/image/video families and by
# the OmniPrometheusStatLogger wrap (G7) which relabels upstream `engine` into
# `stage` + `replica`.
STAGE_LABELS = ("model_name", "stage", "replica")

# Cross-stage transfer label set (G3). Field names match TransferEdgeStats.
TRANSFER_LABELS = ("from_stage", "from_replica", "to_stage", "to_replica")


# ============================================================================
# Histogram buckets
# ============================================================================
# Seconds bucket for TTFP / duration / generation time families.
SECONDS_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0,
)

# Milliseconds bucket for transfer tx / rx / in-flight times.
MS_BUCKETS = (
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
)

# RTF SLO red line is 1.0 (TTS must generate faster than playback).
RTF_BUCKETS = (
    0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 5.0, 10.0,
)

# Bytes bucket for transfer payload size.
BYTES_BUCKETS = (
    1024, 4096, 16384, 65536, 262144, 1048576,
    4194304, 16777216, 67108864, 268435456,
)


# ============================================================================
# Formula helpers (shared by server-side observe and bench-side calculation)
# ============================================================================
def compute_audio_rtf(stage_gen_time_s: float, audio_duration_s: float) -> float:
    """RTF = stage_gen_time / audio_content_duration.

    SLO red line < 1 — must generate faster than content plays back to stream.
    Returns 0.0 when audio_duration_s is non-positive (caller decides whether
    to observe; we don't want to divide by zero or emit negative samples).
    """
    if audio_duration_s <= 0:
        return 0.0
    return stage_gen_time_s / audio_duration_s


def compute_video_rtf(stage_gen_time_s: float, video_duration_s: float) -> float:
    """Same definition as audio RTF."""
    if video_duration_s <= 0:
        return 0.0
    return stage_gen_time_s / video_duration_s
