# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for the released Breeze-TTS-2 checkpoint."""

from transformers import LlamaConfig, PretrainedConfig, Qwen3Config
from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig


class BreezeConfig(PretrainedConfig):
    model_type = "breeze"

    def __init__(
        self,
        backbone_config: dict | None = None,
        depth_decoder_config: dict | None = None,
        text_encoder_config: dict | None = None,
        num_codebooks: int = 16,
        audio_vocab_size: int = 2051,
        audio_embed_size: int = 2048,
        max_position_embeddings: int = 2048,
        rope_scaling: dict | None = None,
        rope_theta: float = 500000.0,
        **kwargs: object,
    ) -> None:
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = {"rope_theta": rope_theta, **(rope_scaling or {"rope_type": "default"})}
        self.num_codebooks = num_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.audio_embed_size = audio_embed_size
        self.backbone_config = Qwen3Config(**(backbone_config or {}))
        self.depth_decoder_config = LlamaConfig(**(depth_decoder_config or {}))
        self.text_encoder_config = T5Gemma2TextConfig(**(text_encoder_config or {}))
        # The scheduler samples continuation/EOS tokens. Text encoder IDs are
        # carried in the prompt payload and are outside this vocabulary.
        self.vocab_size = audio_vocab_size + 1
        self.eos_token_id = audio_vocab_size
        self.backbone_config.vocab_size = self.vocab_size
        self.backbone_config.bos_token_id = 0
        self.backbone_config.pad_token_id = 0
        self.backbone_config.eos_token_id = self.eos_token_id
        self.backbone_config.tie_word_embeddings = False
        super().__init__(**kwargs)
        self.vocab_size = audio_vocab_size + 1
        self.eos_token_id = audio_vocab_size

    def get_text_config(self, decoder: bool = False) -> Qwen3Config:
        return self.backbone_config
