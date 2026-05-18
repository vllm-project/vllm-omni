# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for higgs-audio v2 in vllm-omni.

Mirrors `bosonai/higgs-audio-v2-generation-3B-base/config.json` and exposes the
Stage-1 codec decoder reference (sample rate, codebook layout, stream-special
token ids) so the rest of the integration can read a single typed object.

The text-LM backbone constants match `Llama-3.2-3B` (hidden=3072, 28 layers,
GQA 24Q/8KV, head_dim=128, vocab=128256, max_position_embeddings=2048, RoPE
``llama3`` scaling with factor=32). The audio side adds 8 codebooks of size
1026 (real codes ``[0..1023]``, stream specials ``audio_stream_bos_id=1024``
and ``audio_stream_eos_id=1025``) plus the DualFFN audio expert.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

__all__ = ["HiggsAudioV2Config"]


class HiggsAudioV2Config(PretrainedConfig):
    """Typed wrapper around the upstream `higgs_audio_v2` config.

    The field set here is a strict subset of the upstream
    ``transformers.models.higgs_audio_v2.HiggsAudioV2Config``: only the
    knobs that vllm-omni reads directly are surfaced. Anything extra coming
    from the HF config dict is preserved via ``PretrainedConfig`` so
    ``AutoConfig.from_pretrained(...)`` round-trips correctly.
    """

    model_type: str = "higgs_audio_v2"
    keys_to_ignore_at_inference = ("past_key_values",)

    # Architecture identifier persisted in config.json and used by both
    # vllm_omni.config.pipeline_registry and vllm_omni.model_executor.models.registry.
    DEFAULT_ARCHITECTURES = ("HiggsAudioV2ForConditionalGeneration",)

    def __init__(
        self,
        # --- Text LM (Llama-3.2-3B) ---
        vocab_size: int = 128256,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_parameters: dict[str, Any] | None = None,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        pretraining_tp: int = 1,
        # --- Audio side ---
        num_codebooks: int = 8,
        codebook_size: int = 1026,
        audio_token_id: int = 128016,
        audio_bos_token_id: int = 128013,
        audio_eos_token_id: int = 128012,
        audio_delay_token_id: int = 128014,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        sample_rate: int = 24000,
        frame_rate: int = 25,
        # --- Standard LM specials ---
        bos_token_id: int = 1,
        eos_token_id: int = 128009,
        pad_token_id: int = 128001,
        # --- DualFFN ---
        # Upstream `HiggsAudioV2DecoderLayer` always carries a parallel `audio_mlp`
        # expert; this flag is kept for forward-compat in case a future config
        # disables it (no-op for the boson-ai checkpoint).
        use_audio_dual_ffn: bool = True,
        # --- Stage-1 tokenizer / codec decoder ---
        # Path to the audio-tokenizer subdir (relative to the model directory).
        # Empty string means the model_dir itself contains config.json +
        # model.safetensors (the layout used by the standalone
        # ``bosonai/higgs-audio-v2-tokenizer`` HF repo). Non-empty means the
        # tokenizer files live in ``<model_dir>/<subdir>/`` (the layout used by
        # the 3B Stage-0 checkpoint which bundles its tokenizer under
        # ``audio_tokenizer/``).
        audio_tokenizer_subdir: str = "",
        # HF repo id (or local path) of a STANDALONE audio tokenizer. When set,
        # Stage-1 resolves this via ``huggingface_hub.snapshot_download`` and
        # loads the codec from there, ignoring ``audio_tokenizer_subdir``. This
        # matches the boson-ai release layout where the codec lives in a
        # separate repo (``bosonai/higgs-audio-v2-tokenizer``).
        audio_tokenizer_id: str | None = "bosonai/higgs-audio-v2-tokenizer",
        # --- Misc ---
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        if rope_parameters is None:
            rope_parameters = {
                "factor": 32.0,
                "high_freq_factor": 0.5,
                "low_freq_factor": 0.125,
                "original_max_position_embeddings": 1024,
                "rope_theta": 500000.0,
                "rope_type": "llama3",
            }

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = dict(rope_parameters)
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.pretraining_tp = pretraining_tp

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_token_id = audio_token_id
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_delay_token_id = audio_delay_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

        self.use_audio_dual_ffn = use_audio_dual_ffn
        self.audio_tokenizer_subdir = audio_tokenizer_subdir
        self.audio_tokenizer_id = audio_tokenizer_id
        self.use_cache = use_cache

    # Codes [0..codebook_size-1] include 2 stream specials at indices
    # `audio_stream_bos_id` and `audio_stream_eos_id`. Real codes ready for the
    # codec decoder are the strict prefix [0..audio_stream_bos_id).
    @property
    def num_real_codes(self) -> int:
        return self.audio_stream_bos_id  # i.e. 1024 for the boson-ai checkpoint

    # The full vocab size of each per-codebook output head.
    @property
    def codebook_output_size(self) -> int:
        return self.codebook_size  # i.e. 1026 for the boson-ai checkpoint

    # Canonical MusicGen-style delay pattern used by
    # `HiggsAudioV2DelayPatternLogitsProcessor`: codebook k starts emitting
    # real codes only after k frames. Kept as a function so subclasses can
    # override if upstream ever publishes a different convention.
    def default_delay_pattern(self) -> list[int]:
        return list(range(self.num_codebooks))
