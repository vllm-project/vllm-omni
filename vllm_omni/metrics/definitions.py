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

import logging
import os
from collections.abc import Mapping, Sequence

from vllm_omni.metrics.utils import resolve_int_by_sequential_keys

# vllm_omni: namespace for omni-specific Prometheus families, distinct from
# the upstream vllm:* families.
METRIC_PREFIX = "vllm_omni:"
logger = logging.getLogger(__name__)


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

# Bench-side aggregate field names. Keep these centralized so new benchmark
# metrics reuse the same vocabulary instead of inventing parallel spellings.
MEAN_AUDIO_TTFP_MS = f"mean_{AUDIO_TTFP}_ms"
MEDIAN_AUDIO_TTFP_MS = f"median_{AUDIO_TTFP}_ms"
STD_AUDIO_TTFP_MS = f"std_{AUDIO_TTFP}_ms"
PERCENTILES_AUDIO_TTFP_MS = f"percentiles_{AUDIO_TTFP}_ms"
TOTAL_AUDIO_DURATION_S = f"total_{AUDIO_DURATION}_s"
TOTAL_AUDIO_FRAMES = f"total_{AUDIO_FRAMES}"
AUDIO_THROUGHPUT = "audio_throughput"
MEAN_AUDIO_RTF = f"mean_{AUDIO_RTF}"
MEDIAN_AUDIO_RTF = f"median_{AUDIO_RTF}"
STD_AUDIO_RTF = f"std_{AUDIO_RTF}"
PERCENTILES_AUDIO_RTF = f"percentiles_{AUDIO_RTF}"
MEAN_AUDIO_DURATION_S = f"mean_{AUDIO_DURATION}_s"
MEDIAN_AUDIO_DURATION_S = f"median_{AUDIO_DURATION}_s"
STD_AUDIO_DURATION_S = f"std_{AUDIO_DURATION}_s"
PERCENTILES_AUDIO_DURATION_S = f"percentiles_{AUDIO_DURATION}_s"
MEAN_AUDIO_UNDERRUN_S = f"mean_{AUDIO_UNDERRUN}_s"
MEDIAN_AUDIO_UNDERRUN_S = f"median_{AUDIO_UNDERRUN}_s"
STD_AUDIO_UNDERRUN_S = f"std_{AUDIO_UNDERRUN}_s"
PERCENTILES_AUDIO_UNDERRUN_S = f"percentiles_{AUDIO_UNDERRUN}_s"
AUDIO_CONTINUITY_OK_RATE = f"{AUDIO_CONTINUITY_OK}_rate"

IMAGE_COUNT = "image_count"
IMAGE_GENERATION = "image_generation"
IMAGE_GENERATION_TIME_MS = f"{IMAGE_GENERATION}_time_ms"
IMAGE_PIXELS = "image_pixels"
TOTAL_IMAGES = "total_images"
IMAGE_THROUGHPUT = "image_throughput"
AVERAGE_PIXELS_PER_IMAGE = "average_pixels_per_image"
DENOISE_STEP_LATENCY = "denoise_step_latency"
DENOISE_STEP_LATENCY_MS = f"{DENOISE_STEP_LATENCY}_ms"
MEAN_DENOISE_STEP_LATENCY_MS = f"mean_{DENOISE_STEP_LATENCY}_ms"
MEAN_IMAGE_GENERATION_MS = f"mean_{IMAGE_GENERATION}_ms"
MEDIAN_IMAGE_GENERATION_MS = f"median_{IMAGE_GENERATION}_ms"
STD_IMAGE_GENERATION_MS = f"std_{IMAGE_GENERATION}_ms"
PERCENTILES_IMAGE_GENERATION_MS = f"percentiles_{IMAGE_GENERATION}_ms"

# Stage snapshot / StageBenchmarkMetrics field names.
TOTAL_OUTPUT = "total_output"
TTFTS = "ttfts"
TPOTS = "tpots"
ITLS = "itls"
VLLM_TTFTS = "vllm_ttfts"
VLLM_TPOTS = "vllm_tpots"
VLLM_ITLS = "vllm_itls"
AUDIO_TTFPS = "audio_ttfps"
AUDIO_DURATIONS = "audio_durations"
MISSING_AUDIO_DURATION_COUNT = "missing_audio_duration_count"
STAGE_GEN_TIME = "stage_gen_time"
STAGE_GEN_TIME_MS = f"{STAGE_GEN_TIME}_ms"
STAGE_GEN_TIMES_MS = f"{STAGE_GEN_TIME}s_ms"
POSTPROCESS_TIME = "postprocess_time"
POSTPROCESS_TIME_MS = f"{POSTPROCESS_TIME}_ms"
POSTPROCESS_TIMES_MS = f"{POSTPROCESS_TIME}s_ms"
OUTPUT_UNIT_COUNT = "output_unit_count"
SERVING_TIME_TO_FIRST_OUTPUT_MS = "serving_time_to_first_output_ms"
SERVING_TIME_TO_FIRST_OUTPUTS_MS = "serving_time_to_first_outputs_ms"
TIME_PER_OUTPUT_UNIT_MS = "time_per_output_unit_ms"
TIME_PER_OUTPUT_UNITS_MS = "time_per_output_units_ms"
INTER_OUTPUT_LATENCY_MS = "inter_output_latency_ms"
INTER_OUTPUT_LATENCIES_MS = "inter_output_latencies_ms"
VLLM_TTFT_MS = "vllm_ttft_ms"
VLLM_TPOT_MS = "vllm_tpot_ms"
VLLM_ITL_MS = "vllm_itl_ms"
VLLM_ITLS_MS = "vllm_itls_ms"
NUM_TOKENS_IN = "num_tokens_in"
NUM_TOKENS_OUT = "num_tokens_out"
AUDIO_SAMPLE_RATE = "audio_sample_rate"


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

# Token counters — aggregated across all pipeline stages per request.
PROMPT_TOKENS = METRIC_PREFIX + "prompt_tokens"
GENERATION_TOKENS = METRIC_PREFIX + "generation_tokens"


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
# Diffusion family (per-stage + per-replica diffusion timing breakdowns)
# ============================================================================
DIFFUSION_EXEC_S = METRIC_PREFIX + "diffusion_exec_s"
DIFFUSION_PREPROCESS_S = METRIC_PREFIX + "diffusion_preprocess_s"
DIFFUSION_POSTPROCESS_S = METRIC_PREFIX + "diffusion_postprocess_s"


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
# Audio defaults (shared by server-side observe and bench-side calculation)
# ============================================================================
# Most common across vllm-omni talker variants (cosyvoice3, omnivoice,
# qwen3_tts, mimo_audio). voxcpm2 uses 48000, stable_audio 44100,
# ming_flash 16000 — these models surface a rate at runtime on one of
# ``_SAMPLE_RATE_KEYS``, so DEFAULT_AUDIO_SAMPLE_RATE only applies when no
# usable positive rate is found (and as the bench PCM default when env is unset).
# Streamed OpenAI ``response_format=pcm`` is PCM_16 (s16le); sample width is fixed.
DEFAULT_AUDIO_SAMPLE_RATE = 24000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_AUDIO_SAMPLE_WIDTH = 2  # PCM s16le bytes per sample
AUDIO_SAMPLE_RATE_ENV = "VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE"
AUDIO_CHANNELS_ENV = "VLLM_OMNI_BENCH_AUDIO_CHANNELS"

# Default underrun threshold — kept aligned with the bench-side default and
# the commonly-cited "audible gap" threshold for streaming TTS.
AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S = 0.1


# ============================================================================
# Formula helpers (shared by server-side observe and bench-side calculation)
# ============================================================================
def compute_audio_rtf(audio_generation_latency_s: float, audio_duration_s: float) -> float:
    """RTF = audio_generation_latency / audio_content_duration.

    SLO red line < 1 — must generate faster than content plays back to stream.
    Returns 0.0 when audio_duration_s is non-positive (caller decides whether
    to observe; we don't want to divide by zero or emit negative samples).
    """
    if audio_duration_s <= 0:
        return 0.0
    return audio_generation_latency_s / audio_duration_s


def compute_audio_frames(
    pcm_nbytes: int,
    *,
    sample_width: int = DEFAULT_AUDIO_SAMPLE_WIDTH,
    channels: int = DEFAULT_AUDIO_CHANNELS,
) -> int:
    """Convert a PCM byte length to sample/frame count.

    ``pcm_nbytes`` is the raw byte size of interleaved PCM (e.g.
    ``total_pcm_bytes`` or ``len(wav_pcm_buffer)``). Defaults match the
    streamed s16le mono contract
    (``DEFAULT_AUDIO_SAMPLE_WIDTH``, ``DEFAULT_AUDIO_CHANNELS``);
    WAV callers should pass header values.

    Returns 0 when the byte count or frame width (``sample_width * channels``)
    is non-positive.
    """
    frame_width = sample_width * channels
    if pcm_nbytes <= 0 or frame_width <= 0:
        return 0
    return pcm_nbytes // frame_width


def compute_denoise_step_latency(stage_gen_time: float, num_inference_steps: int) -> float:
    """Mean denoise step latency = image stage generation time / step count.

    The returned value uses the same time unit as ``stage_gen_time``.
    """
    if num_inference_steps <= 0:
        return 0.0
    return stage_gen_time / float(num_inference_steps)


# ============================================================================
# Audio sample-rate resolution
# ============================================================================
_SAMPLE_RATE_KEYS = ("output_sample_rate", "audio_sample_rate", "sample_rate", "sampling_rate", "sr")


def resolve_audio_sample_rate_or_none(
    source: Sequence[Mapping[str, object] | object | None] | Mapping[str, object] | object | None,
) -> int | None:
    """Extract an explicitly configured audio sample rate, or ``None`` when absent.

    ``source`` may be:

    - a Mapping (e.g. ``multimodal_output`` / config dict)
    - an object exposing the same fields as attributes
    - a Sequence of the above (``str`` / ``bytes`` excluded); the first
      positive rate found across items wins
    """
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        for item in source:
            rate = resolve_int_by_sequential_keys(item, _SAMPLE_RATE_KEYS)
            if rate is not None:
                return rate
        return None
    return resolve_int_by_sequential_keys(source, _SAMPLE_RATE_KEYS)


def resolve_audio_sample_rate(
    source: Sequence[Mapping[str, object] | object | None] | Mapping[str, object] | object | None,
    default: int = DEFAULT_AUDIO_SAMPLE_RATE,
) -> int:
    """Extract audio sample rate from a dict, config object, or list thereof.

    Delegates to :func:`resolve_audio_sample_rate_or_none` (same key chain and
    coerce rules). Returns ``default`` (``DEFAULT_AUDIO_SAMPLE_RATE`` unless
    overridden) when no usable positive rate is present.

    Used so /metrics ``audio_duration_s = audio_frames / sample_rate`` stays
    consistent with rates surfaced on multimodal outputs and stage configs.
    """
    rate = resolve_audio_sample_rate_or_none(source)
    return rate if rate is not None else default


def stream_pcm_format_from_env(
    *,
    default_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    default_channels: int = DEFAULT_AUDIO_CHANNELS,
) -> tuple[int, int]:
    """Return the sample rate and channel count for streamed raw PCM."""
    sample_rate = default_sample_rate
    channels = default_channels
    raw_sr = os.environ.get(AUDIO_SAMPLE_RATE_ENV)
    if raw_sr:
        try:
            sample_rate = int(raw_sr)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %d", AUDIO_SAMPLE_RATE_ENV, raw_sr, sample_rate)
    raw_ch = os.environ.get(AUDIO_CHANNELS_ENV)
    if raw_ch:
        try:
            channels = int(raw_ch)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %d", AUDIO_CHANNELS_ENV, raw_ch, channels)
    return max(sample_rate, 1), max(channels, 1)
