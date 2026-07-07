# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Miso TTS config registration with transformers AutoConfig."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class MisoTTSConfig(PretrainedConfig):
    model_type: str = "miso_tts"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(getattr(self, "hidden_size", 4096))
        self.num_hidden_layers = int(getattr(self, "num_hidden_layers", 32))
        self.num_attention_heads = int(getattr(self, "num_attention_heads", 32))
        self.num_key_value_heads = int(getattr(self, "num_key_value_heads", 8))
        self.vocab_size = int(getattr(self, "vocab_size", 128256))
        self.max_position_embeddings = int(getattr(self, "max_position_embeddings", 2048))
        self.intermediate_size = int(getattr(self, "intermediate_size", 14336))
        self.audio_vocab_size = int(getattr(self, "audio_vocab_size", 2051))
        self.audio_num_codebooks = int(getattr(self, "audio_num_codebooks", 32))
        self.sample_rate = int(getattr(self, "sample_rate", 24000))
        self.speculative_config = None

    def get_text_config(self, **kwargs):
        return self


AutoConfig.register("miso_tts", MisoTTSConfig)
