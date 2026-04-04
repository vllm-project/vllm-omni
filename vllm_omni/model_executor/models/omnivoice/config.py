# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniVoice configuration for vLLM-Omni two-stage pipeline."""

from transformers.configuration_utils import PretrainedConfig


class OmniVoiceConfig(PretrainedConfig):
    """Configuration for OmniVoice model in vLLM-Omni.

    This mirrors the HuggingFace OmniVoiceConfig but adds fields needed
    for the two-stage serving pipeline.
    """

    model_type = "omnivoice"

    def get_text_config(self, **kwargs):
        """Return self so vLLM uses our top-level config (which has
        num_attention_heads etc.) instead of trying to extract a sub-config."""
        return self

    def __init__(self, **kwargs):
        # HF repos (e.g. k2-fsa/OmniVoice) may nest generation hyperparameters.
        gen_cfg = kwargs.pop("generation_config", None)
        if isinstance(gen_cfg, dict):
            for k, v in gen_cfg.items():
                kwargs.setdefault(k, v)

        super().__init__(**kwargs)

        # Audio codec params (prefer values set by PretrainedConfig from config.json)
        self.audio_vocab_size = getattr(self, "audio_vocab_size", 1025)
        self.audio_mask_id = getattr(self, "audio_mask_id", 1024)
        self.num_audio_codebook = getattr(self, "num_audio_codebook", 8)
        self.audio_codebook_weights = getattr(
            self,
            "audio_codebook_weights",
            [8, 8, 6, 6, 4, 4, 2, 2],
        )

        # LLM backbone params (Qwen3-0.6B defaults from HF config)
        llm_config = getattr(self, "llm_config", None) or {}
        if isinstance(llm_config, PretrainedConfig):
            llm_config = llm_config.to_dict()
        elif not isinstance(llm_config, dict):
            llm_config = {}
        self.llm_hidden_size = llm_config.get("hidden_size", 1024)
        self.llm_num_hidden_layers = llm_config.get("num_hidden_layers", 28)
        self.llm_num_attention_heads = llm_config.get("num_attention_heads", 16)
        self.llm_num_key_value_heads = llm_config.get("num_key_value_heads", 8)
        self.llm_intermediate_size = llm_config.get("intermediate_size", 3072)
        self.llm_vocab_size = llm_config.get("vocab_size", 151676)
        self.llm_max_position_embeddings = llm_config.get("max_position_embeddings", 40960)
        self.llm_rope_theta = llm_config.get("rope_theta", 1000000.0)
        self.llm_rms_norm_eps = llm_config.get("rms_norm_eps", 1e-6)
        self.llm_head_dim = llm_config.get("head_dim", self.llm_hidden_size // self.llm_num_attention_heads)

        # Expose LLM params at top level for vLLM ModelConfig compatibility
        # (vLLM expects num_attention_heads, hidden_size, etc. on the config)
        self.num_attention_heads = self.llm_num_attention_heads
        self.num_key_value_heads = self.llm_num_key_value_heads
        self.num_hidden_layers = self.llm_num_hidden_layers
        self.hidden_size = self.llm_hidden_size
        self.head_dim = self.llm_head_dim
        if not hasattr(self, "vocab_size"):
            self.vocab_size = self.llm_vocab_size

        # Generation params (defaults from OmniVoiceGenerationConfig)
        self.num_step = getattr(self, "num_step", 32)
        self.guidance_scale = getattr(self, "guidance_scale", 2.0)
        self.t_shift = getattr(self, "t_shift", 0.1)
        self.layer_penalty_factor = getattr(self, "layer_penalty_factor", 5.0)
        self.position_temperature = getattr(self, "position_temperature", 5.0)
        self.class_temperature = getattr(self, "class_temperature", 0.0)

        # Audio output
        self.sample_rate = getattr(self, "sample_rate", 24000)
        self.frame_rate = getattr(self, "frame_rate", 25)

        # Serving
        self.speculative_config = None
