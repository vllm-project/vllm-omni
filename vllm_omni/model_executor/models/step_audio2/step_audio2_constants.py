# SPDX-License-Identifier: Apache-2.0
"""Step-Audio2 configuration constants - Single Source of Truth."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StepAudio2TokenConfig:
    """
    Step-Audio2 token ID ranges and vocabulary configuration.

    Token ID Layout (based on vocab_size=158720):
    - Text tokens: 0 - 151688 (standard Qwen tokenizer)
    - Special tokens: 151689 - 151695 (reserved)
    - Audio tokens: 151696 - 158257 (vocab size 6562)

    These values are architecture constants for Step-Audio2 and do not
    change between model variants (mini/7B).
    """

    # Text token range
    text_max: int = 151688
    """Maximum text token ID (inclusive)"""

    # Audio token range (absolute IDs in full vocabulary)
    audio_start: int = 151696
    """First audio token ID in absolute vocabulary"""

    audio_vocab_size: int = 6562
    """Total number of audio tokens"""

    audio_eos: int = 6561
    """Audio EOS token ID (relative to audio_start, used for padding)"""

    # Special tokens
    audio_patch_token_id: int = 151690
    """<audio_patch> placeholder token ID"""

    @property
    def audio_end(self) -> int:
        """Last audio token ID in absolute vocabulary"""
        return self.audio_start + self.audio_vocab_size - 1


@dataclass(frozen=True)
class StepAudio2EncoderConfig:
    """
    Step-Audio2 audio encoder architecture configuration.

    These values define the encoder structure and should match your model weights.
    """

    # Mel spectrogram parameters
    n_mels: int = 128
    """Number of mel frequency bins"""

    # Encoder architecture
    n_audio_ctx: int = 1500
    """Audio context length (max sequence length)"""

    n_audio_state: int = 512
    """Audio encoder hidden state dimension"""

    n_audio_head: int = 8
    """Number of attention heads in encoder"""

    n_audio_layer: int = 6
    """Number of encoder layers"""

    # Adapter parameters
    kernel_size: int = 3
    """Convolution kernel size for adapter"""

    adapter_stride: int = 2
    """Stride for adapter convolution (downsampling factor)"""


@dataclass(frozen=True)
class StepAudio2Token2WavConfig:
    """Token2Wav synthesis configuration."""

    n_timesteps: int = 10
    """Number of diffusion timesteps for flow model inference"""

    sample_rate: int = 24000
    """Output audio sample rate"""


@dataclass(frozen=True)
class StepAudio2StreamConfig:
    """Streaming (chunked) inference configuration for Token2Wav."""

    chunk_size: int = 25
    """Number of audio tokens consumed per chunk."""

    pre_lookahead_len: int = 3
    """Conformer encoder lookahead tokens beyond the chunk boundary."""

    up_rate: int = 2
    """Token-to-mel upsampling rate inside the flow model."""

    mel_cache_len: int = 8
    """HiFT overlap mel frames (160 ms at 50 Hz mel rate)."""

    n_timesteps: int = 10
    """Flow model ODE solver steps."""

    estimator_cache_keep: int = 100
    """Attention-cache pruning window for the flow estimator."""


@dataclass(frozen=True)
class StepAudio2ModelConfig:
    """
    Step-Audio2 complete model configuration - Single Source of Truth.

    All architecture constants are defined here. The stage YAML and model config.json
    can override these values if needed, but this serves as the default.
    """

    # LLM configuration (Qwen2)
    hidden_size: int = 4096
    """LLM hidden dimension (Qwen2-7B: 4096, Qwen2-1.5B: 1536, Qwen2-0.5B: 896)"""

    # Token configuration
    text_max: int = 151688
    audio_start: int = 151696
    audio_vocab_size: int = 6562
    audio_eos: int = 6561
    audio_patch_token_id: int = 151690

    # Encoder configuration
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 512
    n_audio_head: int = 8
    n_audio_layer: int = 6
    kernel_size: int = 3
    adapter_stride: int = 2


# Default configuration instances
DEFAULT_TOKEN_CONFIG = StepAudio2TokenConfig()
DEFAULT_ENCODER_CONFIG = StepAudio2EncoderConfig()
DEFAULT_TOKEN2WAV_CONFIG = StepAudio2Token2WavConfig()
DEFAULT_STREAM_CONFIG = StepAudio2StreamConfig()
DEFAULT_MODEL_CONFIG = StepAudio2ModelConfig()

STREAM_SOURCE_CACHE_LEN = DEFAULT_STREAM_CONFIG.mel_cache_len * 480  # 3840 samples

# Export constants for backward compatibility
STEP_AUDIO2_TEXT_MAX = DEFAULT_TOKEN_CONFIG.text_max
STEP_AUDIO2_AUDIO_START = DEFAULT_TOKEN_CONFIG.audio_start
STEP_AUDIO2_AUDIO_VOCAB_SIZE = DEFAULT_TOKEN_CONFIG.audio_vocab_size
STEP_AUDIO2_AUDIO_EOS = DEFAULT_TOKEN_CONFIG.audio_eos
STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID = DEFAULT_TOKEN_CONFIG.audio_patch_token_id
STEP_AUDIO2_AUDIO_END = DEFAULT_TOKEN_CONFIG.audio_end

# Default prompt wav path for Token2Wav (can be overridden by env var)
STEP_AUDIO2_DEFAULT_PROMPT_WAV = "default_female.wav"


__all__ = [
    "StepAudio2TokenConfig",
    "StepAudio2EncoderConfig",
    "StepAudio2Token2WavConfig",
    "StepAudio2StreamConfig",
    "StepAudio2ModelConfig",
    "DEFAULT_TOKEN_CONFIG",
    "DEFAULT_ENCODER_CONFIG",
    "DEFAULT_TOKEN2WAV_CONFIG",
    "DEFAULT_STREAM_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "STREAM_SOURCE_CACHE_LEN",
    "STEP_AUDIO2_TEXT_MAX",
    "STEP_AUDIO2_AUDIO_START",
    "STEP_AUDIO2_AUDIO_VOCAB_SIZE",
    "STEP_AUDIO2_AUDIO_EOS",
    "STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID",
    "STEP_AUDIO2_AUDIO_END",
]
