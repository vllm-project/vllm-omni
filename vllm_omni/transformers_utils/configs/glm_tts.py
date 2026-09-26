# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS config registration with transformers AutoConfig.

Registers GLMTTSConfig (model_type="glm_tts") so that
``AutoConfig.from_pretrained("path/to/glm-tts")`` returns the correct config class.

Note: GLM-TTS uses a Llama backbone, but we register a custom config
to handle the special token IDs and flow model parameters.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class GLMTTSConfig(PretrainedConfig):
    """Llama-based AR model for text-to-speech token generation.

    Special token IDs are loaded dynamically from the tokenizer at init time.
    """

    model_type: str = "glm_tts"

    def __init__(
        self,
        vocab_size: int = 98304,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        audio_token_start: int = -1,
        audio_token_end: int = -1,
        boa_token_id: int = -1,
        eoa_token_id: int = -1,
        pad_token_id: int = -1,
        bos_token_id: int = -1,
        speech_token_vocab_size: int = 32768,
        speech_token_dim: int = 512,
        spk_embed_dim: int = 192,
        mel_dim: int = 80,
        input_frame_rate: float = 25.0,
        mel_framerate: int = 50,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
        sample_method: str = "ras",
        ras_top_p: float = 0.8,
        ras_top_k: int = 25,
        ras_win_size: int = 10,
        ras_tau_r: float = 0.1,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.audio_token_start = audio_token_start
        self.audio_token_end = audio_token_end
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.speech_token_vocab_size = speech_token_vocab_size
        self.speech_token_dim = speech_token_dim
        self.spk_embed_dim = spk_embed_dim
        self.mel_dim = mel_dim
        self.input_frame_rate = input_frame_rate
        self.mel_framerate = mel_framerate
        self.max_token_text_ratio = max_token_text_ratio
        self.min_token_text_ratio = min_token_text_ratio
        self.sample_method = sample_method
        self.ras_top_p = ras_top_p
        self.ras_top_k = ras_top_k
        self.ras_win_size = ras_win_size
        self.ras_tau_r = ras_tau_r
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)


AutoConfig.register("glm_tts", GLMTTSConfig)

__all__ = [
    "GLMTTSConfig",
]
