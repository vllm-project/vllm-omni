from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any

import numpy as np
from vllm.multimodal.media import MediaConnector

from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSessionConfig
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    ServingRuntimeConfigError,
)


class MiniCPMO45ClientRuntimeConfigError(ServingRuntimeConfigError):
    pass


class MiniCPMO45NativeDuplexServingAdapter:
    """Serving-side MiniCPM-o 4.5 native duplex session preparation.

    The generic duplex WebSocket handler should not let client-supplied local
    paths reach workers.  This adapter follows the existing media connector
    boundary: serving resolves client media URIs, then workers receive only
    normalized PCM payloads.  Server-owned model assets remain local paths here.
    """

    PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
        {
            "duplex_stage_sampling_params",
            "duplex_stage_max_tokens",
            "duplex_stage0_max_tokens",
            "duplex_scheduler_token_id",
            "duplex_first_append_context_tokens",
            "ref_audio_data",
            "ref_audio_format",
            "ref_audio_sample_rate_hz",
        }
    )

    @classmethod
    def is_enabled(cls, config: DuplexSessionConfig) -> bool:
        return config.extra_body.get("minicpmo45_native_duplex") is True

    @classmethod
    def validate_client_config(cls, config: DuplexSessionConfig) -> None:
        cls.validate_client_extra_body(config.extra_body)

    @classmethod
    def validate_client_extra_body(cls, extra_body: object) -> None:
        if not isinstance(extra_body, dict):
            return
        private_keys = sorted(cls.PRIVATE_RUNTIME_CONFIG_KEYS.intersection(extra_body))
        if private_keys:
            raise MiniCPMO45ClientRuntimeConfigError(
                "native duplex runtime configuration is server-owned: " + ", ".join(private_keys)
            )

    @classmethod
    def runtime_config_for_update(
        cls,
        config: DuplexSessionConfig,
        current: object,
    ) -> dict[str, object]:
        runtime_config = deepcopy(dict(current)) if isinstance(current, dict) else {}
        runtime_config["instructions"] = config.instructions
        stage_max_tokens = runtime_config.get("duplex_stage_max_tokens")
        stage_max_tokens = deepcopy(stage_max_tokens) if isinstance(stage_max_tokens, dict) else {}
        stage_max_tokens["0"] = (
            config.max_tokens if isinstance(config.max_tokens, int) and config.max_tokens > 0 else 20
        )
        stage_max_tokens.setdefault("1", 8192)
        runtime_config["duplex_stage_max_tokens"] = stage_max_tokens

        stage_sampling = runtime_config.get("duplex_stage_sampling_params")
        stage_sampling = deepcopy(stage_sampling) if isinstance(stage_sampling, dict) else {}
        stage0 = stage_sampling.get("0")
        stage0 = deepcopy(stage0) if isinstance(stage0, dict) else {}
        stage0["temperature"] = config.temperature if config.temperature is not None else 0.7
        stage_sampling["0"] = stage0
        runtime_config["duplex_stage_sampling_params"] = stage_sampling
        return runtime_config

    @classmethod
    async def prepare_runtime_config(cls, config: DuplexSessionConfig, *, model_config: Any) -> dict[str, object]:
        extra_body = dict(config.extra_body)
        if any(key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")):
            raise ValueError("ref_audio_path is not accepted by native duplex; use ref_audio URI instead")
        cls.validate_client_config(config)
        runtime_config: dict[str, object] = {"instructions": config.instructions}
        cls._apply_default_scheduler_policy(runtime_config, config=config, model_config=model_config)

        ref_audio = config.ref_audio
        if ref_audio is None and isinstance(extra_body.get("ref_audio"), str):
            ref_audio = extra_body.pop("ref_audio")
        if ref_audio is None and isinstance(extra_body.get("tts_ref_audio"), str):
            ref_audio = extra_body.pop("tts_ref_audio")

        if ref_audio is None:
            if any(str(modality).lower() == "audio" for modality in config.modalities):
                raise MiniCPMO45ClientRuntimeConfigError(
                    "MiniCPM-o native duplex audio output requires ref_audio",
                    code="ref_audio_required",
                )
            cls._apply_first_append_context_tokens(
                runtime_config,
                model_config=model_config,
                instructions=config.instructions,
                ref_sample_count=None,
            )
            config.extra_body = extra_body
            return runtime_config
        else:
            wav_np, sr = await cls.resolve_ref_audio(ref_audio, model_config=model_config)

        wav_np = cls.normalize_ref_audio(wav_np, int(sr), target_sr=16000)
        # Trim to a whole number of pooled audio embeddings (100 ms frames) so
        # the first-append scheduler reserve can count them exactly.
        usable = (len(wav_np) // MiniCPMO45DuplexPolicy.SAMPLES_PER_AUDIO_TOKEN) * (
            MiniCPMO45DuplexPolicy.SAMPLES_PER_AUDIO_TOKEN
        )
        wav_np = wav_np[:usable]
        ref_audio_bytes = np.ascontiguousarray(wav_np, dtype=np.float32).tobytes()
        runtime_config["ref_audio_data"] = base64.b64encode(ref_audio_bytes).decode("ascii")
        runtime_config["ref_audio_format"] = "pcm_f32le"
        runtime_config["ref_audio_sample_rate_hz"] = 16000
        cls._apply_first_append_context_tokens(
            runtime_config,
            model_config=model_config,
            instructions=config.instructions,
            ref_sample_count=len(wav_np),
        )
        config.extra_body = extra_body
        config.ref_audio = None
        return runtime_config

    @classmethod
    def _apply_default_scheduler_policy(
        cls,
        runtime_config: dict[str, object],
        *,
        config: DuplexSessionConfig,
        model_config: Any,
    ) -> None:
        stage0_max_tokens = config.max_tokens if isinstance(config.max_tokens, int) and config.max_tokens > 0 else 20
        runtime_config["duplex_stage_max_tokens"] = {"0": stage0_max_tokens, "1": 8192}
        stage0_params: dict[str, object] = {
            "temperature": config.temperature if config.temperature is not None else 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
        }
        stop_token_ids = cls._native_stage0_stop_token_ids(model_config)
        if stop_token_ids:
            stage0_params["stop_token_ids"] = stop_token_ids
        runtime_config["duplex_stage_sampling_params"] = {"0": stage0_params}
        scheduler_token_id = cls._native_scheduler_token_id(model_config)
        if scheduler_token_id is not None:
            runtime_config["duplex_scheduler_token_id"] = scheduler_token_id

    @classmethod
    def _apply_first_append_context_tokens(
        cls,
        runtime_config: dict[str, object],
        *,
        model_config: Any,
        instructions: object,
        ref_sample_count: int | None,
    ) -> None:
        """Precompute the exact session-context token count for the engine.

        The first data-plane append carries the system template and optional
        reference-audio embeddings ahead of the first unit. The engine
        reserves scheduler slots from this count; an inexact count turns into
        pad embeddings inside the model KV (surplus) or truncated context
        (deficit), so it is computed with the same template and pooling math
        the worker uses.
        """
        if "duplex_first_append_context_tokens" in runtime_config:
            return
        tokenizer = cls._load_native_tokenizer(model_config)
        if tokenizer is None:
            return
        prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
            instructions,
            ref_sample_count is not None,
        )
        try:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        except Exception:
            return
        ref_tokens = MiniCPMO45DuplexPolicy.audio_token_count(ref_sample_count or 0)
        runtime_config["duplex_first_append_context_tokens"] = len(prefix_ids) + ref_tokens + len(suffix_ids)

    @staticmethod
    def _native_stage0_stop_token_ids(model_config: Any) -> list[int]:
        tokenizer = MiniCPMO45NativeDuplexServingAdapter._load_native_tokenizer(model_config)
        if tokenizer is None:
            return []
        out: list[int] = []
        stop_token_fields = (
            "chunk_eos_token_id",
            "chunk_tts_eos_token_id",
            "listen_token_id",
            "turn_eos_token_id",
        )
        for field in stop_token_fields:
            token = MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS[field]
            token_id = MiniCPMO45NativeDuplexServingAdapter._convert_token_to_id(tokenizer, token)
            if token_id is not None and token_id not in out:
                out.append(token_id)
        return out

    @staticmethod
    def _native_scheduler_token_id(model_config: Any) -> int | None:
        tokenizer = MiniCPMO45NativeDuplexServingAdapter._load_native_tokenizer(model_config)
        if tokenizer is None:
            return None
        scheduler_tokens = (
            MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS["unit_token_id"],
            MiniCPMO45DuplexPolicy.OPTIONAL_TOKEN_FIELDS["audio_placeholder_token_id"],
        )
        for token in scheduler_tokens:
            token_id = MiniCPMO45NativeDuplexServingAdapter._convert_token_to_id(tokenizer, token)
            if token_id is not None:
                return token_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
        try:
            return int(eos_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _load_native_tokenizer(model_config: Any) -> Any | None:
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            return None

    @staticmethod
    def _convert_token_to_id(tokenizer: Any, token: str) -> int | None:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        value = None
        if callable(convert):
            value = convert(token)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            token_id = -1
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id >= 0 and token_id != unk_token_id:
            return token_id
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                ids = list(encode(token, add_special_tokens=False))
            except TypeError:
                ids = list(encode(token))
            if len(ids) == 1:
                try:
                    token_id = int(ids[0])
                except (TypeError, ValueError):
                    token_id = -1
                if token_id >= 0 and token_id != unk_token_id:
                    return token_id
        return None

    @staticmethod
    async def resolve_ref_audio(ref_audio: str, *, model_config: Any) -> tuple[np.ndarray, int]:
        connector = MediaConnector(
            allowed_local_media_path=getattr(model_config, "allowed_local_media_path", None),
            allowed_media_domains=getattr(model_config, "allowed_media_domains", None),
        )
        wav_np, sr = await connector.fetch_audio_async(ref_audio)
        return np.asarray(wav_np, dtype=np.float32), int(sr)

    @staticmethod
    def normalize_ref_audio(wav_np: np.ndarray, sample_rate: int, *, target_sr: int) -> np.ndarray:
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        wav_np = wav_np.reshape(-1)
        if sample_rate <= 0 or sample_rate == target_sr or wav_np.size == 0:
            return wav_np.astype(np.float32, copy=False)
        import torch
        import torchaudio

        audio = torch.from_numpy(wav_np).to(dtype=torch.float32).unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, int(sample_rate), int(target_sr))
        return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
