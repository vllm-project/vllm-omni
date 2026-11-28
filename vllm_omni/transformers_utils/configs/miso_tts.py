# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Miso TTS config registration with transformers AutoConfig."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig

MISO_MAX_SEQ_LEN = 2048
MISO_DEFAULT_MAX_FRAMES = 1125
MISO_AUDIO_VOCAB_SIZE = 2051
# The MisoLabs/MisoTTS repo tokenizer is invalid; the model uses Llama 3.2's.
MISO_TEXT_TOKENIZER = "meta-llama/Llama-3.2-1B"


class MisoTTSConfig(PretrainedConfig):
    model_type: str = "miso_tts"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(getattr(self, "hidden_size", 4096))
        self.num_hidden_layers = int(getattr(self, "num_hidden_layers", 32))
        self.num_attention_heads = int(getattr(self, "num_attention_heads", 32))
        self.num_key_value_heads = int(getattr(self, "num_key_value_heads", 8))
        self.vocab_size = int(getattr(self, "vocab_size", 128256))
        self.max_position_embeddings = int(getattr(self, "max_position_embeddings", MISO_MAX_SEQ_LEN))
        self.intermediate_size = int(getattr(self, "intermediate_size", 14336))
        self.audio_vocab_size = int(getattr(self, "audio_vocab_size", MISO_AUDIO_VOCAB_SIZE))
        self.audio_num_codebooks = int(getattr(self, "audio_num_codebooks", 32))
        self.sample_rate = int(getattr(self, "sample_rate", 24000))
        self.speculative_config = None

    def get_text_config(self, **kwargs):
        return self


AutoConfig.register("miso_tts", MisoTTSConfig)
