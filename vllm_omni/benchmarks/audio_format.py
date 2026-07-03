import logging
import os

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE_ENV = "VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE"
AUDIO_CHANNELS_ENV = "VLLM_OMNI_BENCH_AUDIO_CHANNELS"


def stream_pcm_format_from_env(
    *,
    default_sample_rate: int = 24000,
    default_channels: int = 1,
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
