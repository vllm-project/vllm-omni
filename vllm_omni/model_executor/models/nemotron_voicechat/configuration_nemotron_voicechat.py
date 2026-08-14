# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF-compatible configs synthesized from the NeMo-style NemotronVoiceChat checkpoint.

The ``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`` checkpoint ships a NeMo training
config (top-level ``data`` / ``model`` keys) instead of a transformers config:
there is no ``model_type`` or ``architectures``. vLLM-Omni's empty-config patching
(:meth:`OmniEngineArgs._patch_empty_hf_config`) injects
``model_type="nemotron_voicechat"`` and routes the raw dict here, where it is
parsed into per-stage sub-configs:

* ``get_text_config()`` -> the NemotronH backbone config (thinker stage; the AR
  runner reads ``hidden_size`` etc. from it). Resolved from the checkpoint's
  ``model.stt.model.pretrained_llm`` HF id via the local HF cache, or from the
  ``NEMOTRON_VOICECHAT_LLM_PATH`` env var pointing at a local snapshot.
* ``talker_config`` -> Gemma3-based EAR-TTS backbone dims (talker stage).
* ``code2wav_config`` -> RVQ-VAE codec dims (code2wav stage).

The raw NeMo sections are preserved on the config (``stt_cfg`` / ``tts_cfg`` /
``nemo_data``) because the stage models consume them verbatim (perception layout,
fusion weights, TTS generation parameters, codec architecture).
"""

from __future__ import annotations

import os
from typing import Any

from transformers import AutoConfig, PretrainedConfig

_EXPECTED_STRUCTURE = (
    "Expected a NemotronVoiceChat checkpoint config.json with the NeMo layout: "
    "top-level 'model' containing both 'stt' (with 'model.stt.model.perception' "
    "and 'model.stt.model.pretrained_llm') and 'speech_generation' (with "
    "'model.speech_generation.model.tts_config' and '...codec_config')."
)

# Default HF id of the thinker text backbone; the checkpoint records the same id
# under model.stt.model.pretrained_llm.
_DEFAULT_LLM_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

# Env override for air-gapped runs: local directory containing the Nemotron
# backbone's config.json (and optionally tokenizer files).
_LLM_PATH_ENV = "NEMOTRON_VOICECHAT_LLM_PATH"


class NemotronVoiceChatTalkerConfig(PretrainedConfig):
    """Stage config for the EAR-TTS talker (28-layer Gemma3-style backbone)."""

    model_type = "nemotron_voicechat_talker"

    def __init__(self, **kwargs: Any) -> None:
        backbone: dict[str, Any] = kwargs.pop("backbone_config", None) or {}
        super().__init__(**kwargs)
        self.backbone_config = backbone
        self.hidden_size = int(backbone.get("hidden_size", 1152))
        self.num_hidden_layers = int(backbone.get("num_hidden_layers", 28))
        self.num_attention_heads = int(backbone.get("num_attention_heads", 16))
        self.head_dim = int(backbone.get("head_dim", 72))
        self.intermediate_size = int(backbone.get("intermediate_size", 4608))
        self.sliding_window = int(backbone.get("sliding_window", 7500))
        # The talker consumes/produces per-frame code stacks; vocab is nominal
        # (codes are sampled by the vendored MOG head, not a vLLM sampler).
        self.vocab_size = int(kwargs.get("vocab_size", 1024))
        self.max_position_embeddings = int(backbone.get("max_position_embeddings", 32768))

    def get_text_config(self, **_: Any) -> NemotronVoiceChatTalkerConfig:
        return self


class NemotronVoiceChatCode2WavConfig(PretrainedConfig):
    """Stage config for the RVQ-VAE codec decoder."""

    model_type = "nemotron_voicechat_code2wav"

    def __init__(self, **kwargs: Any) -> None:
        codec: dict[str, Any] = kwargs.pop("codec_config", None) or {}
        super().__init__(**kwargs)
        self.codec_config = codec
        self.num_quantizers = int(codec.get("num_quantizers", 31))
        self.codebook_size = int(codec.get("codebook_size", 1024))
        self.latent_size = int(codec.get("latent_size", 512))
        self.wav_to_token_ratio = int(codec.get("wav_to_token_ratio", 1764))
        self.sample_rate = int(kwargs.get("sample_rate", 22050))
        # The generation runner sizes buffers from hidden_size.
        self.hidden_size = self.latent_size
        self.num_attention_heads = 1
        self.num_hidden_layers = 1
        self.vocab_size = self.codebook_size
        self.max_position_embeddings = int(kwargs.get("max_position_embeddings", 65536))

    def get_text_config(self, **_: Any) -> NemotronVoiceChatCode2WavConfig:
        return self


class NemotronVoiceChatConfig(PretrainedConfig):
    """Unified config: parses the raw NeMo dict into per-stage sub-configs."""

    model_type = "nemotron_voicechat"

    def __init__(self, **kwargs: Any) -> None:
        nemo_model = kwargs.pop("model", None)
        nemo_data = kwargs.pop("data", None)
        thinker_text_config = kwargs.pop("thinker_text_config", None)
        talker_config = kwargs.pop("talker_config", None)
        code2wav_config = kwargs.pop("code2wav_config", None)
        stt_cfg = kwargs.pop("stt_cfg", None)
        tts_cfg = kwargs.pop("tts_cfg", None)
        # Drop remaining NeMo training-only top-level sections.
        for training_key in ("exp_manager", "trainer", "hf_export_dir", "name"):
            kwargs.pop(training_key, None)
        # transformers may call get_text_config() during super().__init__;
        # give it a safe placeholder until the real sub-config is resolved.
        self.thinker_text_config = None
        super().__init__(**kwargs)

        if nemo_model is not None:
            stt = nemo_model.get("stt") or {}
            speech_generation = nemo_model.get("speech_generation") or {}
            if not (isinstance(stt, dict) and isinstance(stt.get("model"), dict)) or not (
                isinstance(speech_generation, dict) and isinstance(speech_generation.get("model"), dict)
            ):
                raise ValueError(
                    "config.json is missing the NemotronVoiceChat NeMo signature keys "
                    "'model.stt.model' and/or 'model.speech_generation.model'. " + _EXPECTED_STRUCTURE
                )
            stt_cfg = stt["model"]
            tts_cfg = speech_generation["model"]
            self.inference_speaker_name = nemo_model.get("inference_speaker_name") or "Aria"
            self.tts_data = speech_generation.get("data") or {}
        elif stt_cfg is None or tts_cfg is None:
            # Bare construction (e.g. transformers plumbing); leave a skeleton
            # that fails loudly if a stage model tries to use it.
            self.inference_speaker_name = kwargs.get("inference_speaker_name", "Aria")
            self.tts_data = {}
        if stt_cfg is None or tts_cfg is None:
            self.stt_cfg: dict[str, Any] = {}
            self.tts_cfg: dict[str, Any] = {}
        else:
            if "perception" not in stt_cfg:
                raise ValueError("'model.stt.model.perception' is missing from config.json. " + _EXPECTED_STRUCTURE)
            if "tts_config" not in tts_cfg or "codec_config" not in tts_cfg:
                raise ValueError(
                    "'model.speech_generation.model.tts_config'/'codec_config' are missing "
                    "from config.json. " + _EXPECTED_STRUCTURE
                )
            self.stt_cfg = dict(stt_cfg)
            self.tts_cfg = dict(tts_cfg)
        self.nemo_data = dict(nemo_data) if isinstance(nemo_data, dict) else {}
        self.frame_length = float(self.nemo_data.get("frame_length", 0.08))
        self.source_sample_rate = int(self.nemo_data.get("source_sample_rate", 16000))
        self.target_sample_rate = int(self.nemo_data.get("target_sample_rate", 22050))

        self.thinker_text_config = self._resolve_thinker_text_config(thinker_text_config)
        self.talker_config = self._coerce(
            talker_config,
            NemotronVoiceChatTalkerConfig,
            defaults={"backbone_config": (self.tts_cfg.get("tts_config") or {}).get("backbone_config") or {}},
        )
        self.code2wav_config = self._coerce(
            code2wav_config,
            NemotronVoiceChatCode2WavConfig,
            defaults={
                "codec_config": self.tts_cfg.get("codec_config") or {},
                "sample_rate": self.target_sample_rate,
            },
        )

        if not getattr(self, "architectures", None):
            self.architectures = ["NemotronVoiceChatThinkerForConditionalGeneration"]

    @staticmethod
    def _coerce(value: Any, cls: type[PretrainedConfig], defaults: dict[str, Any]) -> PretrainedConfig:
        if isinstance(value, cls):
            return value
        if isinstance(value, PretrainedConfig):
            value = value.to_dict()
        if isinstance(value, dict):
            merged = dict(defaults)
            merged.update(value)
            return cls(**merged)
        return cls(**defaults)

    def _resolve_thinker_text_config(self, provided: Any) -> PretrainedConfig:
        """NemotronH backbone config: explicit > env-local > HF cache/hub."""
        if isinstance(provided, PretrainedConfig):
            return provided
        if isinstance(provided, dict) and provided:
            try:
                return AutoConfig.for_model(**dict(provided))
            except (KeyError, ValueError):
                return PretrainedConfig(**provided)
        if not self.stt_cfg:
            # Skeleton config (bare construction); resolved lazily by the stage.
            return PretrainedConfig()
        llm_ref = os.environ.get(_LLM_PATH_ENV) or self.stt_cfg.get("pretrained_llm") or _DEFAULT_LLM_ID
        try:
            # nemotron_h is a native transformers model type; trust_remote_code=True
            # would chase the auto_map .py modules and break air-gapped dirs that
            # ship only config.json.
            return AutoConfig.from_pretrained(llm_ref, trust_remote_code=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the NemotronVoiceChat thinker backbone config from {llm_ref!r}. "
                f"Make sure the HF cache contains 'nvidia/NVIDIA-Nemotron-Nano-9B-v2' or set "
                f"{_LLM_PATH_ENV} to a local directory with its config.json."
            ) from exc

    def get_text_config(self, **_: Any) -> PretrainedConfig:
        text_config = self.__dict__.get("thinker_text_config")
        return text_config if text_config is not None else self
