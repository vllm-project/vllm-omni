# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for MOSS-TTS-Nano in vLLM-Omni single-stage pipeline."""

from transformers.configuration_utils import PretrainedConfig


class MossTTSNanoConfig(PretrainedConfig):
    """Config for MOSS-TTS-Nano (OpenMOSS-Team/MOSS-TTS-Nano).

    The model is a Global-Local GPT2-based AR LM (0.1B) paired with the
    MOSS-Audio-Tokenizer-Nano codec.  Both components run in a single
    vLLM-Omni generation stage.

    Relevant fields from config.json:
      gpt2_config  – backbone config (n_layer=12, n_embd=768, n_head=12, ...)
      n_vq         – number of RVQ codebooks (16)
      audio_vocab_size – per-codebook vocabulary size (1024)
      audio_tokenizer_pretrained_name_or_path – HF hub path for codec model
    """

    model_type = "moss_tts_nano"

    def __init__(self, **kwargs):
        gpt2_cfg = kwargs.pop("gpt2_config", None) or {}
        if hasattr(gpt2_cfg, "to_dict"):
            gpt2_cfg = gpt2_cfg.to_dict()

        super().__init__(**kwargs)

        # --- GPT2 backbone parameters (exposed at top level for vLLM) ---
        self.hidden_size = gpt2_cfg.get("n_embd", 768)
        self.num_hidden_layers = gpt2_cfg.get("n_layer", 12)
        self.num_attention_heads = gpt2_cfg.get("n_head", 12)
        self.num_key_value_heads = self.num_attention_heads  # no GQA in GPT2
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.vocab_size = gpt2_cfg.get("vocab_size", 16384)
        self.max_position_embeddings = gpt2_cfg.get("n_positions", 32768)
        self.intermediate_size = gpt2_cfg.get("n_inner", self.hidden_size * 4)

        # --- Audio codec parameters ---
        self.n_vq: int = getattr(self, "n_vq", 16)
        self.audio_vocab_size: int = getattr(self, "audio_vocab_size", 1024)
        self.audio_start_token_id: int = getattr(self, "audio_start_token_id", 6)
        self.audio_end_token_id: int = getattr(self, "audio_end_token_id", 7)
        self.audio_user_slot_token_id: int = getattr(self, "audio_user_slot_token_id", 8)
        self.audio_assistant_slot_token_id: int = getattr(self, "audio_assistant_slot_token_id", 9)
        self.audio_tokenizer_pretrained_name_or_path: str = getattr(
            self,
            "audio_tokenizer_pretrained_name_or_path",
            "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        )
        self.audio_tokenizer_sample_rate: int = getattr(self, "audio_tokenizer_sample_rate", 48000)

        # vLLM requires speculative_config to be absent or None
        self.speculative_config = None

    def get_text_config(self, **kwargs):
        """Return self so vLLM uses our top-level config."""
        return self
