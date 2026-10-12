# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Hugging Face configuration for the released Lychee-FD checkpoint.

The public checkpoint uses the historical ``step_audio_2_full_duplex`` model
type, but its four decoder branches form a Lychee-specific execution graph.
This module normalizes those nested configs without changing the checkpoint
on disk and rejects topology drift before vLLM allocates model or KV state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .contract import LycheeConfigError, LycheeFDContract


def _released_decoder_config(num_hidden_layers: int) -> dict[str, Any]:
    return {
        "model_type": "qwen2",
        "vocab_size": 158_363,
        "hidden_size": 3_584,
        "intermediate_size": 18_944,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "max_position_embeddings": 16_384,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "use_sliding_window": False,
        "sliding_window": 2_048,
    }


def _as_qwen2_config(value: Qwen2Config | Mapping[str, Any] | None, *, layers: int) -> Qwen2Config:
    if value is None:
        value = _released_decoder_config(layers)
    if isinstance(value, Qwen2Config):
        return value
    if not isinstance(value, Mapping):
        raise LycheeConfigError(f"Expected a Qwen2 config mapping, got {type(value).__name__}")
    values = dict(value)
    values.pop("model_type", None)
    return Qwen2Config(**values)


class LycheeAudioEncoderConfig(PretrainedConfig):
    """Configuration of Lychee's Whisper-derived audio encoder and adaptor."""

    model_type = "step_audio_2_encoder"

    def __init__(
        self,
        n_mels: int = 128,
        n_audio_ctx: int = 1_500,
        n_audio_state: int = 1_280,
        n_audio_head: int = 20,
        n_audio_layer: int = 32,
        n_codebook_size: int = 4_096,
        llm_dim: int = 3_584,
        kernel_size: int = 3,
        adapter_stride: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_codebook_size = n_codebook_size
        self.llm_dim = llm_dim
        self.kernel_size = kernel_size
        self.adapter_stride = adapter_stride


def _as_audio_encoder_config(
    value: LycheeAudioEncoderConfig | Mapping[str, Any] | None,
) -> LycheeAudioEncoderConfig:
    if value is None:
        return LycheeAudioEncoderConfig()
    if isinstance(value, LycheeAudioEncoderConfig):
        return value
    if not isinstance(value, Mapping):
        raise LycheeConfigError(f"Expected an audio encoder config mapping, got {type(value).__name__}")
    values = dict(value)
    values.pop("model_type", None)
    return LycheeAudioEncoderConfig(**values)


class LycheeFDConfig(PretrainedConfig):
    """Strict config for the released 28/4/4/4 Lychee-FD model."""

    model_type = "step_audio_2_full_duplex"
    is_composition = True
    keys_to_ignore_at_inference = ("past_key_values",)

    RELEASED_LAYER_COUNTS = (28, 4, 4, 4)
    RELEASED_CONTROL_BRANCH_INDEX = 20
    RELEASED_HIDDEN_SIZE = 3_584
    RELEASED_VOCAB_SIZE = 158_363

    def __init__(
        self,
        text_config: Qwen2Config | Mapping[str, Any] | None = None,
        stoken_layer_config: Qwen2Config | Mapping[str, Any] | None = None,
        control_layer_config: Qwen2Config | Mapping[str, Any] | None = None,
        merge_layer_config: Qwen2Config | Mapping[str, Any] | None = None,
        audio_encoder_config: LycheeAudioEncoderConfig | Mapping[str, Any] | None = None,
        control_branch_layer: int = 8,
        control_token_chunk_size: int = 10,
        stoken_delay_num: int = 10,
        stoken_token_ids_min: int = 151_694,
        stoken_token_ids_max: int = 158_352,
        control_token_ids_min: int = 158_352,
        control_token_ids_max: int = 158_356,
        start_speaking_token_id: int = 158_352,
        start_listening_token_id: int = 158_353,
        keep_listening_token_id: int = 158_354,
        keep_speaking_token_id: int = 158_355,
        start_bc_token_id: int = 158_362,
        keep_bc_token_id: int = 158_362,
        end_bc_token_id: int = 158_353,
        detect_token_id: int = 158_356,
        sleep_token_id: int = 158_357,
        text_pad_token_id: int = 158_358,
        stoken_pad_token_id: int = 158_359,
        audio_patch_token_id: int = 151_690,
        audio_pad_token_id: int = 158_360,
        stoken_delay_token_id: int = 158_361,
        tts_start_token_id: int = 151_693,
        tts_end_token_id: int = 151_694,
        tts_pad_token_id: int = 151_695,
        stoken_audio_token_id_min: int = 151_696,
        stoken_codec_vocab_size: int = 6_561,
        stoken_do_sample: bool = True,
        stoken_temperature: float = 0.7,
        stoken_top_k: int = 0,
        stoken_top_p: float = 1.0,
        stoken_no_repeat_ngram_size: int = 4,
        stoken_max_tokens: int = 1_000,
        eos_token_id: int = 151_665,
        pad_token_id: int = 151_643,
        input_sample_rate_hz: int = 16_000,
        output_sample_rate_hz: int = 24_000,
        inference_window_ms: int = 400,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            architectures=architectures or ["LycheeFullDuplexForConditionalGeneration"],
            **kwargs,
        )
        self.text_config = _as_qwen2_config(text_config, layers=28)
        self.stoken_layer_config = _as_qwen2_config(stoken_layer_config, layers=4)
        self.control_layer_config = _as_qwen2_config(control_layer_config, layers=4)
        self.merge_layer_config = _as_qwen2_config(merge_layer_config, layers=4)
        self.audio_encoder_config = _as_audio_encoder_config(audio_encoder_config)

        self.control_branch_layer = control_branch_layer
        self.control_token_chunk_size = control_token_chunk_size
        self.stoken_delay_num = stoken_delay_num
        self.stoken_token_ids_min = stoken_token_ids_min
        self.stoken_token_ids_max = stoken_token_ids_max
        self.control_token_ids_min = control_token_ids_min
        self.control_token_ids_max = control_token_ids_max
        self.start_speaking_token_id = start_speaking_token_id
        self.start_listening_token_id = start_listening_token_id
        self.keep_listening_token_id = keep_listening_token_id
        self.keep_speaking_token_id = keep_speaking_token_id
        self.start_bc_token_id = start_bc_token_id
        self.keep_bc_token_id = keep_bc_token_id
        self.end_bc_token_id = end_bc_token_id
        self.detect_token_id = detect_token_id
        self.sleep_token_id = sleep_token_id
        self.text_pad_token_id = text_pad_token_id
        self.stoken_pad_token_id = stoken_pad_token_id
        self.audio_patch_token_id = audio_patch_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.stoken_delay_token_id = stoken_delay_token_id
        self.tts_start_token_id = tts_start_token_id
        self.tts_end_token_id = tts_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.stoken_audio_token_id_min = stoken_audio_token_id_min
        self.stoken_codec_vocab_size = stoken_codec_vocab_size
        self.stoken_do_sample = stoken_do_sample
        self.stoken_temperature = stoken_temperature
        self.stoken_top_k = stoken_top_k
        self.stoken_top_p = stoken_top_p
        self.stoken_no_repeat_ngram_size = stoken_no_repeat_ngram_size
        self.stoken_max_tokens = stoken_max_tokens
        self.input_sample_rate_hz = input_sample_rate_hz
        self.output_sample_rate_hz = output_sample_rate_hz
        self.inference_window_ms = inference_window_ms

        self.validate_lychee_contract()

    def validate_lychee_contract(self) -> LycheeFDContract:
        contract = LycheeFDContract.from_config(
            self,
            input_sample_rate_hz=self.input_sample_rate_hz,
            output_sample_rate_hz=self.output_sample_rate_hz,
            inference_window_ms=self.inference_window_ms,
        )
        layout = contract.layout
        counts = (layout.main_layers, layout.stoken_layers, layout.control_layers, layout.merge_layers)
        if counts != self.RELEASED_LAYER_COUNTS:
            raise LycheeConfigError(
                "Production Lychee-FD requires decoder layers 28/4/4/4; "
                f"got {counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}"
            )
        if layout.control_branch_index != self.RELEASED_CONTROL_BRANCH_INDEX:
            raise LycheeConfigError(
                f"Production Lychee-FD control branch must start at H{self.RELEASED_CONTROL_BRANCH_INDEX}; "
                f"got H{layout.control_branch_index}"
            )
        if layout.hidden_size != self.RELEASED_HIDDEN_SIZE:
            raise LycheeConfigError(
                f"Production Lychee-FD hidden_size must be {self.RELEASED_HIDDEN_SIZE}; got {layout.hidden_size}"
            )
        if layout.vocab_size != self.RELEASED_VOCAB_SIZE:
            raise LycheeConfigError(
                f"Production Lychee-FD vocab_size must be {self.RELEASED_VOCAB_SIZE}; got {layout.vocab_size}"
            )
        if self.audio_encoder_config.llm_dim != layout.hidden_size:
            raise LycheeConfigError(
                "audio_encoder_config.llm_dim must match decoder hidden_size; "
                f"got {self.audio_encoder_config.llm_dim} and {layout.hidden_size}"
            )
        if not 0 <= self.audio_patch_token_id < layout.vocab_size:
            raise LycheeConfigError(
                f"audio_patch_token_id={self.audio_patch_token_id} is outside vocab_size={layout.vocab_size}"
            )
        if not 0 <= self.audio_pad_token_id < layout.vocab_size:
            raise LycheeConfigError(
                f"audio_pad_token_id={self.audio_pad_token_id} is outside vocab_size={layout.vocab_size}"
            )
        if self.control_token_chunk_size != 10:
            raise LycheeConfigError(
                f"Production Lychee-FD requires control_token_chunk_size=10; got {self.control_token_chunk_size}"
            )
        if not (
            self.tts_start_token_id
            < self.tts_end_token_id
            < self.tts_pad_token_id
            < self.stoken_audio_token_id_min
            < self.stoken_token_ids_max
        ):
            raise LycheeConfigError(
                "Invalid Lychee speech-token layout: expected tts_start < tts_end < tts_pad < audio_min < stoken_max"
            )
        if type(self.stoken_codec_vocab_size) is not int or self.stoken_codec_vocab_size <= 0:
            raise ValueError("stoken_codec_vocab_size must be a positive integer")
        if self.stoken_temperature < 0:
            raise LycheeConfigError(f"stoken_temperature must be non-negative; got {self.stoken_temperature}")
        if self.stoken_top_k < 0:
            raise LycheeConfigError(f"stoken_top_k must be non-negative; got {self.stoken_top_k}")
        if not 0 < self.stoken_top_p <= 1:
            raise LycheeConfigError(f"stoken_top_p must be in (0, 1]; got {self.stoken_top_p}")
        if self.stoken_no_repeat_ngram_size < 0:
            raise LycheeConfigError(
                f"stoken_no_repeat_ngram_size must be non-negative; got {self.stoken_no_repeat_ngram_size}"
            )
        if self.stoken_max_tokens <= 0:
            raise LycheeConfigError(f"stoken_max_tokens must be positive; got {self.stoken_max_tokens}")
        return contract


__all__ = ["LycheeAudioEncoderConfig", "LycheeFDConfig"]
