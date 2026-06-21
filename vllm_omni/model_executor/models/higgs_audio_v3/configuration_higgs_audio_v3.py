# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for higgs-audio v3 (HiggsMultimodalQwen3) in vllm-omni.

``HiggsAudioV3Config.from_pretrained(model_path)`` returns a config with
``tts_token_id``, ``text_token_id``, ``audio_continuation_id``, and
``eos_token_id`` already resolved from the checkpoint tokenizer. If the
tokenizer is unavailable or missing required specials, the load raises.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import transformers
from transformers import AutoConfig, PretrainedConfig

__all__ = ["HiggsAudioV3Config"]

logger = logging.getLogger(__name__)

_QWEN3_ROPE_THETA = 1_000_000

_REQUIRED_SPECIALS = ("<|tts|>", "<|text|>", "<|audio|>")


def _build_text_config(raw: Any) -> PretrainedConfig:
    if isinstance(raw, PretrainedConfig):
        return raw
    cfg = dict(raw or {})
    model_type = cfg.get("model_type", "qwen3")
    if model_type == "qwen3" and cfg.get("rope_theta") is None:
        cfg["rope_theta"] = _QWEN3_ROPE_THETA
    try:
        cfg_cls = transformers.CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown text backbone model_type {model_type!r}") from exc
    return cfg_cls(**cfg)


def _resolve_model_dir(pretrained_model_name_or_path: str) -> str | None:
    """Resolve a model name/path to a local directory containing tokenizer files."""
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=pretrained_model_name_or_path, filename="tokenizer_config.json")
        if isinstance(cached, str) and os.path.isfile(cached):
            return os.path.dirname(cached)
    except Exception:
        pass

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        safe = pretrained_model_name_or_path.replace("/", "--")
        snapshots = os.path.join(HF_HUB_CACHE, f"models--{safe}", "snapshots")
        if os.path.isdir(snapshots):
            for rev in os.listdir(snapshots):
                candidate = os.path.join(snapshots, rev)
                if os.path.isfile(os.path.join(candidate, "tokenizer_config.json")):
                    return candidate
    except Exception:
        pass
    return None


class HiggsAudioV3Config(PretrainedConfig):
    """Typed config for higgs-audio v3 (HiggsMultimodalQwen3).

    ``from_pretrained()`` automatically resolves ``<|tts|>``, ``<|text|>``,
    ``<|audio|>`` and ``eos_token_id`` from the checkpoint tokenizer.
    """

    model_type: str = "higgs_multimodal_qwen3"
    is_composition = True

    def __init__(
        self,
        audio_encoder_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_token_id: int = -100,
        mel_per_sample: int = 8,
        num_codebooks: int = 8,
        codebook_size: int = 1026,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        sample_rate: int = 24000,
        frame_rate: int = 25,
        tts_token_id: int | None = None,
        text_token_id: int | None = None,
        audio_continuation_id: int | None = None,
        enable_flashinfer_api_unwrap: bool = True,
        enable_mlp_cudagraph: bool = True,
        **kwargs: Any,
    ) -> None:
        # Legacy perf knob removed: Higgs v3 scheduler tokens now come from
        # SamplerOutput.sampled_token_ids only.
        kwargs.pop("enable_cpu_token_override", None)
        self.audio_token_id = audio_token_id
        self.mel_per_sample = mel_per_sample

        if audio_encoder_config is None:
            audio_encoder_config = {
                "encoder_type": "discrete",
                "num_codebooks": num_codebooks,
                "vocab_size": codebook_size,
                "tie_word_embeddings": True,
            }
        self.audio_encoder_config = audio_encoder_config

        self.num_codebooks = int(audio_encoder_config.get("num_codebooks", num_codebooks))
        if self.num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be > 0, got {self.num_codebooks}")
        self.codebook_size = int(audio_encoder_config.get("vocab_size", codebook_size))
        self.tie_modality_embeddings = bool(audio_encoder_config.get("tie_word_embeddings", True))

        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

        self.text_config = _build_text_config(text_config)
        self.audio_hidden_size = int(audio_encoder_config.get("out_dim", self.text_config.hidden_size))

        self.tts_token_id = tts_token_id
        self.text_token_id = text_token_id
        self.audio_continuation_id = audio_continuation_id
        self.enable_flashinfer_api_unwrap = bool(enable_flashinfer_api_unwrap)
        self.enable_mlp_cudagraph = bool(enable_mlp_cudagraph)

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> HiggsAudioV3Config:
        """Load config and resolve special token IDs from the checkpoint tokenizer.

        Passes the original ``pretrained_model_name_or_path`` (local dir or
        HF repo ID) directly to ``AutoTokenizer.from_pretrained()`` so it can
        handle cache hits, downloads, and local paths uniformly. Raises if the
        tokenizer is missing required specials.
        """
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.resolve_special_tokens(pretrained_model_name_or_path)
        return config

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.text_config

    @property
    def num_real_codes(self) -> int:
        return self.audio_stream_bos_id

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    def resolve_special_tokens(self, model_path: str) -> None:
        """Resolve <|tts|>, <|text|>, <|audio|> and eos from the HF tokenizer.

        Raises ``ValueError`` if any of the 3 required specials is missing
        from the tokenizer's added vocabulary.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        vocab = dict(tokenizer.get_added_vocab())

        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(
                f"Tokenizer at {model_path} is missing required Higgs TTS v3 "
                f"special tokens: {missing}. Available added tokens: "
                f"{sorted(vocab.keys())[:20]}"
            )

        self.tts_token_id = vocab["<|tts|>"]
        self.text_token_id = vocab["<|text|>"]
        self.audio_continuation_id = vocab["<|audio|>"]

        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.eos_token_id = int(tokenizer.eos_token_id)


AutoConfig.register("higgs_multimodal_qwen3", HiggsAudioV3Config)
