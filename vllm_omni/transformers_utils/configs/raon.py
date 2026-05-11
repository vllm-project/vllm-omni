# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import CONFIG_MAPPING, AutoConfig, Qwen3Config
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from vllm.logger import init_logger

# DEPRECATED: hardcoded audio-token fallbacks for configs that omit these
# fields. New callers should resolve live tokenizer IDs instead of silently
# relying on the placeholder defaults.
_AUDIO_OUTPUT_PLACEHOLDER_ID = 151675
_AUDIO_INPUT_PLACEHOLDER_ID = 151676

logger = init_logger(__name__)

AUDIO_SAMPLE_RATE = 24000
TARGET_ENCODER_SAMPLE_RATE = 16000
TTS_MAX_TOKENS_HARD_CAP = 4096

# Worker hook: strip these keys / key prefixes from per-request ``additional_information``
# when a request finishes.
REQUEST_STATE_CLEANUP_KEYS: tuple[str, ...] = (
    "codec_codes",
    "codec_queue",
    "output_codes",
    "output_code_queue",
    "mimi_cache",
    "mimi_state",
)
REQUEST_STATE_CLEANUP_PREFIXES: tuple[str, ...] = (
    "codec_",
    "mimi_",
    "codec_queue",
)


@dataclass(frozen=True)
class RaonEnvConfig:
    """Frozen at import time; later env-var changes do not take effect."""

    # TTS sampling
    tts_temperature: float
    tts_top_k: int
    tts_top_p: float
    tts_seed_mode: str
    tts_prompt_style: str
    tts_append_audio_start: bool
    tts_fixed_max_tokens: int | None
    # Code predictor
    cp_compile_mode: str
    ras_enabled: bool
    ras_window_size: int
    ras_repetition_threshold: float
    # Stage pipeline
    stage1_max_prompt_tokens: int
    async_chunk_interval: int
    # Continuation silence
    continuation_silence_frames: int = 2
    max_audio_duration_s: int = 0  # 0 = no limit
    tts_long_mode: str = "rolling_icl"
    tts_long_word_threshold: int = 90
    tts_long_max_sentences_per_chunk: int = 1
    tts_long_anchor_reset_every_chunks: int = 5
    tts_long_enable_stitch_trim: bool = True
    tts_long_ref_text_mode: str = "full_prev_chunk"
    tts_long_pause_ms_period: int = 100
    tts_long_pause_ms_clause: int = 30
    tts_long_min_ref_audio_s: float = 1.0
    tts_long_keep_original_speaker_anchor: bool = True
    tts_long_eos_suppress_grace_steps: int = 2
    tts_long_enable_final_best_of_k: bool = False
    tts_long_final_best_of_k: int = 5
    tts_long_final_best_of_k_early_exit_ratio: float = 1.15
    tts_long_final_best_of_k_expected_wps: float = 2.8
    tts_long_final_best_of_k_score_mode: str = "duration"
    tts_long_enable_final_eos_min_gate: bool = True
    tts_long_final_eos_min_duration_ratio: float = 0.55
    tts_long_final_eos_gate_frame_rate_hz: float = 24.0
    tts_long_enable_final_eos_retry: bool = False
    tts_long_final_eos_retry_duration_ratio: float = 0.95


def _log_long_tts_config(config: RaonEnvConfig) -> None:
    logger.info(
        "Raon long-TTS config: tts_long_mode=%s tts_long_max_sentences_per_chunk=%d "
        "tts_long_enable_final_best_of_k=%s tts_long_enable_final_eos_min_gate=%s "
        "tts_long_final_eos_min_duration_ratio=%.2f tts_long_enable_final_eos_retry=%s "
        "tts_long_final_eos_gate_frame_rate_hz=%.2f",
        config.tts_long_mode,
        config.tts_long_max_sentences_per_chunk,
        config.tts_long_enable_final_best_of_k,
        config.tts_long_enable_final_eos_min_gate,
        config.tts_long_final_eos_min_duration_ratio,
        config.tts_long_enable_final_eos_retry,
        config.tts_long_final_eos_gate_frame_rate_hz,
    )


def _load_raon_env_config() -> RaonEnvConfig:
    """Build ``ENV`` from env vars. Called once at import."""
    config = RaonEnvConfig(
        tts_temperature=float(os.getenv("RAON_TTS_TEMPERATURE", "1.0")),
        tts_top_k=int(os.getenv("RAON_TTS_TOP_K", "0")),
        tts_top_p=float(os.getenv("RAON_TTS_TOP_P", "1.0")),
        tts_seed_mode=os.getenv("RAON_TTS_SEED_MODE", "none").strip().lower(),
        tts_prompt_style=os.getenv("RAON_TTS_PROMPT_STYLE", "instruction").strip().lower(),
        tts_append_audio_start=os.getenv("RAON_TTS_APPEND_AUDIO_START", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        tts_fixed_max_tokens=int(v) if (v := os.getenv("RAON_TTS_FIXED_MAX_TOKENS")) else None,
        cp_compile_mode=os.getenv("RAON_CP_COMPILE_MODE", "reduce-overhead"),
        ras_enabled=os.getenv("RAON_RAS_ENABLED", "1") != "0",
        ras_window_size=int(os.getenv("RAON_RAS_WINDOW_SIZE", "50")),
        ras_repetition_threshold=float(os.getenv("RAON_RAS_REPETITION_THRESHOLD", "0.5")),
        stage1_max_prompt_tokens=int(os.getenv("RAON_STAGE1_MAX_PROMPT_TOKENS", "8000")),
        async_chunk_interval=int(os.getenv("RAON_ASYNC_CHUNK_INTERVAL", "25")),
        continuation_silence_frames=int(os.getenv("RAON_CONTINUATION_SILENCE_FRAMES", "2")),
        max_audio_duration_s=int(os.getenv("RAON_MAX_AUDIO_DURATION_S", "0")),
        tts_long_mode=os.getenv("RAON_TTS_LONG_MODE", "rolling_icl").strip().lower(),
        tts_long_word_threshold=int(os.getenv("RAON_TTS_LONG_WORD_THRESHOLD", "90")),
        tts_long_max_sentences_per_chunk=int(os.getenv("RAON_TTS_LONG_MAX_SENTENCES_PER_CHUNK", "1")),
        tts_long_anchor_reset_every_chunks=int(os.getenv("RAON_TTS_LONG_ANCHOR_RESET_EVERY_CHUNKS", "5")),
        tts_long_enable_stitch_trim=os.getenv("RAON_TTS_LONG_ENABLE_STITCH_TRIM", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        tts_long_ref_text_mode=os.getenv("RAON_TTS_LONG_REF_TEXT_MODE", "full_prev_chunk").strip().lower(),
        tts_long_pause_ms_period=int(os.getenv("RAON_TTS_LONG_PAUSE_MS_PERIOD", "100")),
        tts_long_pause_ms_clause=int(os.getenv("RAON_TTS_LONG_PAUSE_MS_CLAUSE", "30")),
        tts_long_min_ref_audio_s=float(os.getenv("RAON_TTS_LONG_MIN_REF_AUDIO_S", "1.0")),
        tts_long_keep_original_speaker_anchor=os.getenv("RAON_TTS_LONG_KEEP_ORIGINAL_SPEAKER_ANCHOR", "1")
        .strip()
        .lower()
        not in ("0", "false", "no", "off"),
        tts_long_eos_suppress_grace_steps=int(os.getenv("RAON_TTS_LONG_EOS_SUPPRESS_GRACE_STEPS", "2")),
        tts_long_enable_final_best_of_k=os.getenv("RAON_TTS_LONG_ENABLE_FINAL_BEST_OF_K", "0").strip().lower()
        not in ("0", "false", "no", "off"),
        tts_long_final_best_of_k=int(os.getenv("RAON_TTS_LONG_FINAL_BEST_OF_K", "5")),
        tts_long_final_best_of_k_early_exit_ratio=float(
            os.getenv("RAON_TTS_LONG_FINAL_BEST_OF_K_EARLY_EXIT_RATIO", "1.15")
        ),
        tts_long_final_best_of_k_expected_wps=float(os.getenv("RAON_TTS_LONG_FINAL_BEST_OF_K_EXPECTED_WPS", "2.8")),
        tts_long_final_best_of_k_score_mode=os.getenv("RAON_TTS_LONG_FINAL_BEST_OF_K_SCORE_MODE", "duration")
        .strip()
        .lower(),
        tts_long_enable_final_eos_min_gate=os.getenv("RAON_TTS_LONG_ENABLE_FINAL_EOS_MIN_GATE", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        tts_long_final_eos_min_duration_ratio=float(os.getenv("RAON_TTS_LONG_FINAL_EOS_MIN_DURATION_RATIO", "0.55")),
        tts_long_final_eos_gate_frame_rate_hz=float(os.getenv("RAON_TTS_LONG_FINAL_EOS_GATE_FRAME_RATE_HZ", "24.0")),
        tts_long_enable_final_eos_retry=os.getenv("RAON_TTS_LONG_ENABLE_FINAL_EOS_RETRY", "0").strip().lower()
        not in ("0", "false", "no", "off"),
        tts_long_final_eos_retry_duration_ratio=float(
            os.getenv("RAON_TTS_LONG_FINAL_EOS_RETRY_DURATION_RATIO", "0.95")
        ),
    )
    _log_long_tts_config(config)
    return config


ENV = _load_raon_env_config()


class EmbeddingAdaptorConfig(PretrainedConfig):
    model_type = "embedding_adaptor"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = TTS_MAX_TOKENS_HARD_CAP,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: dict[str, Any] | Qwen3Config | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_post_norm = use_post_norm
        self.norm_eps = norm_eps

        if isinstance(decoder_config, dict):
            decoder_config = Qwen3Config(**decoder_config)
        self.decoder_config = decoder_config


class SpeakerEncoderConfig(PretrainedConfig):
    model_type = "speaker_encoder"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = TTS_MAX_TOKENS_HARD_CAP,
        num_heads: int = 8,
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        frame_rate: float = 12.5,
        encoder_type: str = "from_scratch",
        pretrained_model_id: str | None = None,
        pretrained_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.frame_rate = frame_rate
        self.encoder_type = encoder_type
        self.pretrained_model_id = pretrained_model_id
        self.pretrained_dim = pretrained_dim


def get_mimi_frame_rate(audio_tokenizer_config: PretrainedConfig | Mapping[str, Any]) -> float:
    """Read Mimi codec frame rate (``_frame_rate``) from dict or ``PretrainedConfig``."""
    if isinstance(audio_tokenizer_config, Mapping):
        raw = audio_tokenizer_config.get("_frame_rate")
    else:
        raw = getattr(audio_tokenizer_config, "_frame_rate", None)
    if raw is None:
        raise ValueError("Mimi audio_tokenizer_config is missing _frame_rate.")
    return float(raw)


def coerce_speaker_encoder_config(
    speaker_cfg: SpeakerEncoderConfig | PretrainedConfig | dict[str, Any] | None,
) -> SpeakerEncoderConfig | None:
    """Best-effort ``SpeakerEncoderConfig`` from Raon sub-config (dict or generic HF config)."""
    if speaker_cfg is None:
        return None
    if isinstance(speaker_cfg, SpeakerEncoderConfig):
        return speaker_cfg
    if isinstance(speaker_cfg, dict):
        return SpeakerEncoderConfig(**speaker_cfg)
    if isinstance(speaker_cfg, PretrainedConfig):
        try:
            return SpeakerEncoderConfig(**speaker_cfg.to_dict())
        except Exception:
            return None
    return None


# Sub-config classes defined locally (not registered in HF CONFIG_MAPPING).
_RAON_SUBCONFIG_CLASSES: tuple[type[PretrainedConfig], ...] = (
    EmbeddingAdaptorConfig,
    SpeakerEncoderConfig,
)


def _build_subconfig(
    config: PretrainedConfig | dict[str, Any] | None,
    *,
    default_model_type: str | None = None,
) -> PretrainedConfig | None:
    if config is None or isinstance(config, PretrainedConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError(f"Expected sub-config dict or PretrainedConfig, got {type(config)}")

    data = dict(config)
    model_type = data.get("model_type", default_model_type)
    if model_type is None:
        return PretrainedConfig(**data)

    for cls in _RAON_SUBCONFIG_CLASSES:
        if cls.model_type == model_type:
            return cls(**data)

    # Some transformers builds omit this model_type from CONFIG_MAPPING.
    if model_type == "qwen3_omni_moe_audio_encoder":
        try:
            return Qwen3OmniMoeAudioEncoderConfig(**data)
        except Exception:
            logger.debug("Config class %s not found in CONFIG_MAPPING", model_type)

    try:
        return CONFIG_MAPPING[model_type](**data)
    except Exception:
        return PretrainedConfig(**data)


def _build_audio_encoder_config(
    config: PretrainedConfig | dict[str, Any] | None,
) -> Qwen3OmniMoeAudioEncoderConfig | None:
    """Parse audio encoder sub-config strictly as ``Qwen3OmniMoeAudioEncoderConfig``."""
    expected = Qwen3OmniMoeAudioEncoderConfig.model_type
    if config is None:
        return None
    if isinstance(config, Qwen3OmniMoeAudioEncoderConfig):
        return config
    if isinstance(config, dict):
        data = dict(config)
        mt = data.get("model_type", expected)
        if mt != expected:
            raise ValueError(f"Raon audio_encoder_config.model_type must be {expected!r} (got {mt!r})")
        return Qwen3OmniMoeAudioEncoderConfig(**data)
    if isinstance(config, PretrainedConfig):
        mt = getattr(config, "model_type", None)
        if mt != expected:
            raise TypeError(
                "Raon audio_encoder_config must be "
                f"Qwen3OmniMoeAudioEncoderConfig with model_type={expected!r} "
                f"(got {type(config).__name__} with model_type={mt!r})"
            )
        return Qwen3OmniMoeAudioEncoderConfig(**config.to_dict())
    raise TypeError(f"Unsupported audio_encoder_config type: {type(config)}")


class RaonConfig(PretrainedConfig):
    model_type = "raon"
    has_no_defaults_at_init = True
    sub_configs = {
        "text_model_config": PretrainedConfig,
        "talker_config": PretrainedConfig,
        "audio_encoder_config": Qwen3OmniMoeAudioEncoderConfig,
        "audio_tokenizer_config": PretrainedConfig,
        "input_adaptor_config": EmbeddingAdaptorConfig,
        "output_adaptor_config": EmbeddingAdaptorConfig,
        "code_predictor_config": PretrainedConfig,
        "speaker_encoder_config": SpeakerEncoderConfig,
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> RaonConfig:
        """Load config with compatibility shims used in Raon model."""
        config_dict = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)[0]

        # Some exports omit tokenizer config; Raon model falls back to encoder config.
        if "audio_tokenizer_config" not in config_dict and "audio_encoder_config" in config_dict:
            config_dict["audio_tokenizer_config"] = config_dict["audio_encoder_config"]

        return cls(**config_dict)

    def __init__(
        self,
        text_model_config: PretrainedConfig | dict[str, Any] | None = None,
        talker_config: PretrainedConfig | dict[str, Any] | None = None,
        audio_encoder_config: Qwen3OmniMoeAudioEncoderConfig | dict[str, Any] | None = None,
        audio_tokenizer_config: PretrainedConfig | dict[str, Any] | None = None,
        input_adaptor_config: PretrainedConfig | dict[str, Any] | None = None,
        output_adaptor_config: PretrainedConfig | dict[str, Any] | None = None,
        code_predictor_config: PretrainedConfig | dict[str, Any] | None = None,
        speaker_encoder_config: PretrainedConfig | dict[str, Any] | None = None,
        audio_output_token_id: int | None = None,
        audio_token_id: int | None = None,
        audio_input_token_id: int | None = None,
        speaker_token_id: int | None = None,
        speaker_embedding_to_code_predictor: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.text_model_config = _build_subconfig(
            text_model_config,
            default_model_type="qwen3",
        )
        self.talker_config = _build_subconfig(
            talker_config,
            default_model_type="qwen3",
        )
        self.audio_encoder_config = _build_audio_encoder_config(audio_encoder_config)
        self.audio_tokenizer_config = _build_subconfig(
            audio_tokenizer_config,
            default_model_type="mimi",
        )
        self.input_adaptor_config = _build_subconfig(
            input_adaptor_config,
            default_model_type=EmbeddingAdaptorConfig.model_type,
        )
        self.output_adaptor_config = _build_subconfig(
            output_adaptor_config,
            default_model_type=EmbeddingAdaptorConfig.model_type,
        )
        self.code_predictor_config = _build_subconfig(code_predictor_config)
        self.speaker_encoder_config = _build_subconfig(
            speaker_encoder_config,
            default_model_type=SpeakerEncoderConfig.model_type,
        )

        self.audio_token_id = audio_token_id
        self.audio_output_token_id = self._resolve_audio_output_token_id(
            audio_output_token_id=audio_output_token_id,
            audio_token_id=audio_token_id,
        )
        self.audio_input_token_id = self._resolve_audio_input_token_id(
            audio_input_token_id=audio_input_token_id,
        )
        self.speaker_token_id = speaker_token_id
        self.speaker_embedding_to_code_predictor = speaker_embedding_to_code_predictor
        self.text_config = self.text_model_config

        super().__init__(**kwargs)

    def _resolve_audio_output_token_id(
        self,
        *,
        audio_output_token_id: int | None,
        audio_token_id: int | None,
    ) -> int:
        if audio_output_token_id is not None:
            return int(audio_output_token_id)
        if audio_token_id is not None:
            return int(audio_token_id)
        logger.warning(
            "RaonConfig.audio_output_token_id omitted; falling back to hardcoded default %d.",
            _AUDIO_OUTPUT_PLACEHOLDER_ID,
        )
        return _AUDIO_OUTPUT_PLACEHOLDER_ID

    def _resolve_audio_input_token_id(
        self,
        *,
        audio_input_token_id: int | None,
    ) -> int:
        if audio_input_token_id is not None:
            return int(audio_input_token_id)
        logger.warning(
            "RaonConfig.audio_input_token_id omitted; falling back to hardcoded default %d.",
            _AUDIO_INPUT_PLACEHOLDER_ID,
        )
        return _AUDIO_INPUT_PLACEHOLDER_ID

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        if self.text_config is None:
            raise ValueError("RaonConfig is missing `text_config`.")
        return self.text_config

    def accept_hidden_layer_resolved(self) -> int:
        """Resolve ``accept_hidden_layer`` (supports negative indices like ``-1`` = last layer)."""
        text_cfg = self.text_model_config
        if text_cfg is None:
            raise ValueError("RaonConfig is missing `text_model_config`.")
        n = int(text_cfg.num_hidden_layers)
        layer = int(getattr(self, "accept_hidden_layer", -1))
        if layer < 0:
            layer = n + layer
        return layer


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


class TaskType(enum.Enum):
    TEXT_QA = "text_qa"
    STT = "stt"
    TTS = "tts"
    TTS_ICL = "tts_icl"
    SPOKEN_QA = "spoken_qa"
    SPEECH_QA = "speech_qa"


@dataclass(frozen=True)
class TaskConfig:
    input_modalities: tuple[str, ...]
    output_mode: str
    sampling_defaults: dict[str, float]
    requires_ref_audio: bool = False
    append_audio_start: bool = False


TASK_REGISTRY: dict[TaskType, TaskConfig] = {
    TaskType.TEXT_QA: TaskConfig(("text",), "text_only", {}),
    TaskType.STT: TaskConfig(("audio",), "text_only", {"temperature": 0.2, "max_tokens": 512}),
    TaskType.TTS: TaskConfig(
        ("text",),
        "audio_only",
        {"temperature": 1.2, "top_p": 0.8, "top_k": 50, "max_tokens": 2048},
        append_audio_start=True,
    ),
    TaskType.TTS_ICL: TaskConfig(
        ("text", "audio"),
        "audio_only",
        {"temperature": 1.2, "top_p": 0.8, "top_k": 50},
        requires_ref_audio=True,
    ),
    TaskType.SPOKEN_QA: TaskConfig(
        ("audio",),
        "text_only",
        {"temperature": 0.7, "repetition_penalty": 1.1, "max_tokens": 2048},
    ),
    TaskType.SPEECH_QA: TaskConfig(
        ("text", "audio"),
        "text_only",
        {"temperature": 0.7, "repetition_penalty": 1.1, "max_tokens": 2048},
    ),
}


__all__ = [
    "AUDIO_SAMPLE_RATE",
    "REQUEST_STATE_CLEANUP_KEYS",
    "REQUEST_STATE_CLEANUP_PREFIXES",
    "EmbeddingAdaptorConfig",
    "ENV",
    "RaonConfig",
    "RaonEnvConfig",
    "SpeakerEncoderConfig",
    "TARGET_ENCODER_SAMPLE_RATE",
    "TASK_REGISTRY",
    "TTS_MAX_TOKENS_HARD_CAP",
    "TaskConfig",
    "TaskType",
    "coerce_speaker_encoder_config",
    "get_mimi_frame_rate",
]

AutoConfig.register(RaonConfig.model_type, RaonConfig)
