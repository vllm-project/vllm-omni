# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for SongGen in vLLM-Omni single-stage pipeline."""

from transformers.configuration_utils import PretrainedConfig


class SongGenVllmConfig(PretrainedConfig):
    """Config for SongGen (LiuZH-19/SongGen_mixed_pro).

    SongGen is a 1.3B single-stage auto-regressive transformer for
    text-to-song generation. Both the AR LM and the X-Codec decoder run
    inside ``SongGenForGeneration.forward()`` using the VoxCPM-style
    generator pattern.

    Relevant config.json fields (from HF repo):
      decoder      - nested decoder config (hidden_size, num_hidden_layers, ...)
      sampling_rate - output audio sample rate (16000 Hz)
      architectures - ["SongGenMixedForConditionalGeneration"] or
                      ["SongGenDualTrackForConditionalGeneration"]
    """

    model_type = "songgen"

    def __init__(self, **kwargs):
        decoder_cfg = kwargs.pop("decoder", None) or {}
        if hasattr(decoder_cfg, "to_dict"):
            decoder_cfg = decoder_cfg.to_dict()

        # Pop sub-configs that PretrainedConfig.__init__ would choke on.
        kwargs.pop("text_encoder", None)

        super().__init__(**kwargs)

        # --- Decoder backbone params (used by vLLM memory profiling) ---
        self.hidden_size: int = decoder_cfg.get("hidden_size", 1024)
        self.num_hidden_layers: int = decoder_cfg.get("num_hidden_layers", 24)
        self.num_attention_heads: int = decoder_cfg.get("num_attention_heads", 16)
        self.num_key_value_heads: int = self.num_attention_heads
        self.head_dim: int = self.hidden_size // self.num_attention_heads
        # decoder vocab_size covers 1024 codec tokens + 64 special tokens
        self.vocab_size: int = decoder_cfg.get("vocab_size", 1088)
        self.max_position_embeddings: int = decoder_cfg.get("max_position_embeddings", 6547)
        self.intermediate_size: int = decoder_cfg.get("ffn_dim", self.hidden_size * 4)

        # --- Audio codec params ---
        self.sampling_rate: int = getattr(self, "sampling_rate", 16000)
        self.num_codebooks: int = decoder_cfg.get("num_codebooks", 8)

        # vLLM requires speculative_config to be absent or None.
        self.speculative_config = None

    def get_text_config(self, **kwargs):
        """Return self so vLLM uses our top-level config."""
        return self
