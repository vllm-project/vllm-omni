"""Single source of truth for vLLM-Omni Prometheus + bench CLI metric naming.

Consumed by:
- vllm_omni.metrics.prometheus (server-side /metrics pipeline families)
- vllm_omni.metrics.modality (audio families)
- vllm_omni.metrics.transfer (cross-stage transfer families)
- vllm_omni.benchmarks.metrics.metrics (bench CLI MultiModalsBenchmarkMetrics)

Naming conventions for the ``vllm_omni:*`` families exposed here:
time-bearing metrics use the ``_s`` suffix (values in seconds), counters use
``_total`` (auto-suffixed by the prometheus client), sizes use ``_bytes``.
"""

# vllm:omni_ avoids upstream's unregister_vllm_metrics() stripping, which
# removes every collector whose ``_name`` does not start with ``vllm``.
METRIC_PREFIX = "vllm:omni_"


# ============================================================================
# Bench-side stems (also used as RequestFuncOutput attribute names)
# ============================================================================
AUDIO_TTFP = "audio_ttfp"
AUDIO_DURATION = "audio_duration"
AUDIO_RTF = "audio_rtf"
AUDIO_FRAMES = "audio_frames"
AUDIO_UNDERRUN = "audio_underrun"
AUDIO_CONTINUITY_OK = "audio_continuity_ok"
AUDIO_SKIPPED_REQUESTS = "audio_skipped_requests"


# ============================================================================
# Pipeline-level metric families (request counts + e2e latency)
# ============================================================================
NUM_REQUESTS_RUNNING = METRIC_PREFIX + "num_requests_running"
NUM_REQUESTS_WAITING = METRIC_PREFIX + "num_requests_waiting"
E2E_REQUEST_LATENCY_S = METRIC_PREFIX + "e2e_request_latency_s"

# Per-finished_reason Counter; finished_reason ∈ {stop, length, abort, ...}.
# Aborts include client disconnect / cancellation paths. Counter auto-suffixes
# ``_total`` at exposition time.
REQUESTS_SUCCESS = METRIC_PREFIX + "requests_success"


# ============================================================================
# Audio family (per-stage + per-replica audio path metrics)
# ============================================================================
AUDIO_TTFP_S = METRIC_PREFIX + AUDIO_TTFP + "_s"
AUDIO_DURATION_S = METRIC_PREFIX + AUDIO_DURATION + "_s"
AUDIO_RTF_METRIC = METRIC_PREFIX + AUDIO_RTF
AUDIO_FRAMES_METRIC = METRIC_PREFIX + AUDIO_FRAMES
AUDIO_UNDERRUN_S = METRIC_PREFIX + AUDIO_UNDERRUN + "_s"
AUDIO_CONTINUITY_OK_METRIC = METRIC_PREFIX + AUDIO_CONTINUITY_OK
AUDIO_SKIPPED_REQUESTS_METRIC = METRIC_PREFIX + AUDIO_SKIPPED_REQUESTS


# ============================================================================
# Cross-stage Transfer family (per-physical-hop TX/RX/in-flight timings)
# ============================================================================
TRANSFER_SIZE_BYTES = METRIC_PREFIX + "transfer_size_bytes"
TRANSFER_TX_S = METRIC_PREFIX + "transfer_tx_s"
TRANSFER_RX_S = METRIC_PREFIX + "transfer_rx_s"
TRANSFER_IN_FLIGHT_S = METRIC_PREFIX + "transfer_in_flight_s"


# ============================================================================
# Label sets
# ============================================================================
PIPELINE_LABELS = ("model_name",)
SUCCESS_LABELS = ("model_name", "finished_reason")

# Per-stage / per-replica label set used by the audio families and by the
# OmniPrometheusStatLogger wrap which relabels upstream ``engine`` into
# ``stage`` + ``replica``.
STAGE_LABELS = ("model_name", "stage", "replica")

# Audio continuity Counter carries an extra ``threshold_ms`` label so multiple
# threshold buckets can be tracked simultaneously. The ``_ms`` suffix names a
# numeric threshold *value* in ms, not a time-bearing metric.
AUDIO_CONTINUITY_LABELS = ("model_name", "stage", "replica", "threshold_ms")

# Audio skipped-requests Counter carries a `reason` label so the silent-loss
# path (e.g. code2wav rejecting malformed codec input) can be distinguished
# from other "200 OK + empty audio" cases.
AUDIO_SKIPPED_LABELS = ("model_name", "stage", "replica", "reason")

# Cross-stage transfer label set. Each observation is one physical hop from
# (from_stage, from_replica) to (to_stage, to_replica).
TRANSFER_LABELS = (
    "model_name",
    "from_stage",
    "from_replica",
    "to_stage",
    "to_replica",
)


# ============================================================================
# Histogram buckets
# ============================================================================
# Seconds bucket for e2e / generation / TTFP-style metrics that range from
# ~10 ms to several minutes.
SECONDS_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

# Seconds bucket for fine-grained metrics (cross-stage transfer + audio
# underrun) that need millisecond-level resolution.
SECONDS_FAST_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    60.0,
)

# RTF SLO red line is 1.0 — TTS must generate faster than playback.
RTF_BUCKETS = (
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
    1.25,
    1.5,
    2.0,
    5.0,
    10.0,
)

# Bytes bucket for transfer payload size.
BYTES_BUCKETS = (
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
    268435456,
)


# ============================================================================
# Audio-continuity defaults
# ============================================================================
# Default underrun threshold — kept aligned with the bench-side default and
# the commonly-cited "audible gap" threshold for streaming TTS.
AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S = 0.1


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


# ============================================================================
# Audio sample-rate resolution
# ============================================================================
# Most common across vllm-omni talker variants (cosyvoice3, omnivoice,
# qwen3_tts, mimo_audio). voxcpm2 uses 48000, stable_audio 44100,
# ming_flash 16000 — these models populate multimodal_output["audio_sample_rate"]
# at runtime so this default only kicks in when the field is missing.
DEFAULT_AUDIO_SAMPLE_RATE = 24000

_SAMPLE_RATE_KEYS = ("audio_sample_rate", "sample_rate", "sampling_rate", "sr")


def resolve_audio_sample_rate(multimodal_output: dict | None) -> int:
    """Extract audio sample_rate from a multimodal_output dict, with fallbacks.

    Tries the same key chain as serving_chat.py's audio response path so
    /metrics audio_duration_s = audio_frames / sample_rate stays consistent
    with what the OpenAI streaming endpoint reports back to clients.
    Returns DEFAULT_AUDIO_SAMPLE_RATE when no usable value is present.
    """
    if not multimodal_output:
        return DEFAULT_AUDIO_SAMPLE_RATE
    for key in _SAMPLE_RATE_KEYS:
        raw = multimodal_output.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return DEFAULT_AUDIO_SAMPLE_RATE
