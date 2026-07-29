# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Vevo2 in vLLM-Omni single-stage pipeline.

The published checkpoint at https://huggingface.co/RMSnow/Vevo2 ships a
multi-component layout (one ``config.json`` per sub-component) rather than
a single root ``config.json``. To make it loadable through the standard
vLLM-Omni model loader we synthesise a minimal :class:`Vevo2Config` whose
``model_type`` is ``vevo2``; the checkpoint's actual component configs are
loaded lazily by :class:`Vevo2InferencePipeline` inside
``modeling_vevo2.load_weights``.

The ``hidden_size`` / ``num_hidden_layers`` / ``vocab_size`` defaults below
are taken from the checkpoint's
``contentstyle_modeling/posttrained/config.json`` (Qwen2.5-0.5B with vocab
extended to 168565 to fit the 16384 ``<|content_style_*|>`` and 512
``<|prosody_*|>`` special tokens). They are advisory only — the model
itself is wrapped via the upstream pipeline class.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class Vevo2Config(PretrainedConfig):
    """Minimal HF config wrapper for Vevo2 in vLLM-Omni.

    The real Vevo2 model is a four-component pipeline (prosody tokenizer,
    content-style tokenizer, AR Qwen2.5-0.5B LM, flow-matching transformer,
    Vocos vocoder) loaded via Amphion's ``Vevo2InferencePipeline``. This
    class exists so vLLM-Omni's :class:`OmniProcessor` and the stage-config
    factory can dispatch on ``model_type == "vevo2"``.
    """

    model_type = "vevo2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # AR backbone — Qwen2.5-0.5B with extended vocab. These are exposed
        # at the top level so vLLM's profile-time helpers (which read
        # ``hidden_size``, ``vocab_size`` etc.) don't crash before
        # ``load_weights`` populates the real upstream model.
        self.hidden_size: int = getattr(self, "hidden_size", 896)
        self.num_hidden_layers: int = getattr(self, "num_hidden_layers", 24)
        self.num_attention_heads: int = getattr(self, "num_attention_heads", 14)
        self.num_key_value_heads: int = getattr(self, "num_key_value_heads", 2)
        self.head_dim: int = getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size: int = getattr(self, "intermediate_size", 4864)
        self.max_position_embeddings: int = getattr(self, "max_position_embeddings", 32768)
        self.vocab_size: int = getattr(self, "vocab_size", 168565)
        self.rope_theta: float = getattr(self, "rope_theta", 1000000.0)
        self.tie_word_embeddings: bool = getattr(self, "tie_word_embeddings", True)

        # Audio output sample rate — Vevo2 emits 24 kHz waveforms via Vocos.
        self.audio_sample_rate: int = getattr(self, "audio_sample_rate", 24000)

        # Sub-checkpoint relative paths (mirrors the layout of RMSnow/Vevo2
        # on HuggingFace). The model class joins these with the model root
        # at load time.
        self.prosody_tokenizer_subdir: str = getattr(
            self, "prosody_tokenizer_subdir", "tokenizer/prosody_fvq512_6.25hz"
        )
        self.content_style_tokenizer_subdir: str = getattr(
            self, "content_style_tokenizer_subdir", "tokenizer/contentstyle_fvq16384_12.5hz"
        )
        self.ar_subdir: str = getattr(self, "ar_subdir", "contentstyle_modeling/posttrained")
        self.ar_config_filename: str = getattr(self, "ar_config_filename", "amphion_config.json")
        self.fmt_subdir: str = getattr(self, "fmt_subdir", "acoustic_modeling/fm_emilia101k_singnet7k_repa")
        self.fmt_config_filename: str = getattr(self, "fmt_config_filename", "config.json")
        self.vocoder_subdir: str = getattr(self, "vocoder_subdir", "vocoder")
        self.vocoder_config_filename: str = getattr(self, "vocoder_config_filename", "config.json")

        # vLLM requires speculative_config to be absent or None.
        self.speculative_config = None

    def get_text_config(self, **kwargs):
        """Return self so vLLM uses our top-level config."""
        return self
