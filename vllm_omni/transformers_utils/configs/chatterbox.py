# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox TTS configuration (Turbo defaults).

The Hub repo ships no ``config.json``, so every constant lives here. Values
come from upstream ``chatterbox`` 0.1.7: ``GPT2_MEDIUM_CONFIG`` in
``models/t3/llama_configs.py``, the ``T3Config`` overrides in
``ChatterboxTurboTTS.from_local``, and ``models/s3gen``.
"""

from transformers.configuration_utils import PretrainedConfig


class ChatterboxConfig(PretrainedConfig):
    """Constants for both Chatterbox stages."""

    model_type = "chatterbox"

    def __init__(self, variant: str = "turbo", **kwargs):
        if variant not in {"turbo", "original"}:
            raise ValueError("Chatterbox variant must be 'turbo' or 'original'")
        kwargs.setdefault("eos_token_id", 6562)
        super().__init__(**kwargs)
        self.variant = variant

        # The engine sizes its sampler by ``vocab_size``. No tokenizer is
        # loaded for these stages, so that is the speech head's vocabulary,
        # not the text one; the text vocabulary only sizes the text
        # embedding and the backbone's idle ``wte``.
        self.vocab_size = 6563
        self.speech_vocab_size = 6563
        self.text_vocab_size = 50276
        # Ids at or above this are control tokens the decoder must never see.
        self.speech_token_limit = 6561
        self.start_speech_token = 6561
        self.stop_speech_token = 6562

        # T3 backbone (GPT-2 medium).
        self.hidden_size = 1024
        self.num_hidden_layers = 24
        self.num_attention_heads = 16
        self.num_key_value_heads = 16
        self.intermediate_size = 4096
        self.max_position_embeddings = 8196
        self.layer_norm_epsilon = 1e-5
        self.activation_function = "gelu_new"
        self.max_new_tokens = 1000

        # Reference conditioning.
        self.speaker_embed_size = 256
        self.cond_prompt_len = 375
        self.enc_cond_seconds = 15
        self.dec_cond_seconds = 10
        self.min_ref_seconds = 5.0
        self.loudness_target_lufs: float | None = -27.0
        self.s3_tokenizer_name = "speech_tokenizer_v2_25hz"

        # Speech tokens and audio.
        self.s3_sample_rate = 16000
        self.sample_rate = 24000
        self.token_rate = 25
        self.token_mel_ratio = 2
        self.pre_lookahead_len = 3
        self.mel = {
            "n_fft": 1920,
            "num_mels": 80,
            "sampling_rate": 24000,
            "hop_size": 480,
            "win_size": 1920,
            "fmin": 0,
            "fmax": 8000,
            "center": False,
        }
        self.silence_token = 4299
        self.n_silence_tokens = 3
        self.meanflow = True
        self.n_cfm_timesteps = 2

        # Checkpoint files inside the Hub repo.
        self.t3_weights = "t3_turbo_v1.safetensors"
        self.s3gen_weights = "s3gen_meanflow.safetensors"
        self.ve_weights = "ve.safetensors"
        if variant == "original":
            self.vocab_size = self.speech_vocab_size = 8194
            self.text_vocab_size = 704
            self.num_hidden_layers = 30
            self.max_position_embeddings = 131072
            self.activation_function = "silu"
            self.cond_prompt_len = 150
            self.enc_cond_seconds = 6
            self.min_ref_seconds = 0.0
            self.loudness_target_lufs = None
            self.n_silence_tokens = 0
            self.meanflow = False
            self.n_cfm_timesteps = 10
            self.t3_weights = "t3_cfg.safetensors"
            self.s3gen_weights = "s3gen.safetensors"
