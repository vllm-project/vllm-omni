# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.base.audio_utils import (
    load_native_tokenizer,
    normalize_ref_audio,
    resolve_ref_audio,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSessionConfig
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    ServingRuntimeConfigError,
)
from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy


class Qwen3OmniClientRuntimeConfigError(ServingRuntimeConfigError):
    pass


class Qwen3OmniNativeDuplexServingAdapter:
    PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
        {
            "duplex_stage_sampling_params",
            "duplex_stage_max_tokens",
            "duplex_scheduler_token_id",
            "duplex_context_prefix_tokens",
            "duplex_context_suffix_tokens",
            "ref_audio_data",
            "ref_audio_format",
            "ref_audio_sample_rate_hz",
        }
    )

    @classmethod
    def is_enabled(cls, config: DuplexSessionConfig) -> bool:
        return True

    @classmethod
    def validate_client_extra_body(cls, extra_body: object) -> None:
        if not isinstance(extra_body, dict):
            return
        private_keys = sorted(cls.PRIVATE_RUNTIME_CONFIG_KEYS.intersection(extra_body))
        if private_keys:
            raise Qwen3OmniClientRuntimeConfigError(
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
            config.max_tokens if isinstance(config.max_tokens, int) and config.max_tokens > 0 else 512
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
        cls.validate_client_extra_body(extra_body)
        runtime_config: dict[str, object] = {"instructions": config.instructions}
        cls._apply_default_scheduler_policy(runtime_config, config=config, model_config=model_config)

        ref_audio = config.ref_audio
        if ref_audio is None and isinstance(extra_body.get("ref_audio"), str):
            ref_audio = extra_body.pop("ref_audio")

        ref_sample_count: int | None = None
        if ref_audio is not None:
            wav_np, sr = await resolve_ref_audio(ref_audio, model_config=model_config)
            wav_np = normalize_ref_audio(wav_np, int(sr), target_sr=16000)
            usable = (len(wav_np) // Qwen3OmniDuplexPolicy.SAMPLES_PER_AUDIO_TOKEN) * (
                Qwen3OmniDuplexPolicy.SAMPLES_PER_AUDIO_TOKEN
            )
            wav_np = wav_np[:usable]
            ref_sample_count = len(wav_np)
            ref_audio_bytes = np.ascontiguousarray(wav_np, dtype=np.float32).tobytes()
            runtime_config["ref_audio_data"] = base64.b64encode(ref_audio_bytes).decode("ascii")
            runtime_config["ref_audio_format"] = "pcm_f32le"
            runtime_config["ref_audio_sample_rate_hz"] = 16000
            config.ref_audio = None

        cls._apply_first_append_context_tokens(
            runtime_config,
            model_config=model_config,
            instructions=config.instructions,
            ref_sample_count=ref_sample_count,
        )
        config.extra_body = extra_body
        return runtime_config

    @classmethod
    def _apply_default_scheduler_policy(
        cls,
        runtime_config: dict[str, object],
        *,
        config: DuplexSessionConfig,
        model_config: Any,
    ) -> None:
        stage0_max_tokens = config.max_tokens if isinstance(config.max_tokens, int) and config.max_tokens > 0 else 512
        runtime_config["duplex_stage_max_tokens"] = {"0": stage0_max_tokens, "1": 8192}
        stage0_params: dict[str, object] = {
            "temperature": config.temperature if config.temperature is not None else 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
        }
        stage1_params: dict[str, object] = {
            "temperature": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
        }
        runtime_config["duplex_stage_sampling_params"] = {"0": stage0_params, "1": stage1_params}
        scheduler_token_id = cls._scheduler_token_id(model_config)
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
        if "duplex_context_prefix_tokens" in runtime_config:
            return
        tokenizer = load_native_tokenizer(model_config)
        if tokenizer is None:
            return
        prefix, suffix = Qwen3OmniDuplexPolicy.session_context_texts(instructions)
        try:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        except Exception:
            return
        ref_tokens = Qwen3OmniDuplexPolicy.audio_token_count(ref_sample_count or 0)
        runtime_config["duplex_context_prefix_tokens"] = len(prefix_ids) + ref_tokens + 1
        runtime_config["duplex_context_suffix_tokens"] = 1 + len(suffix_ids)

    @staticmethod
    def _thinker_stop_token_ids(model_config: Any) -> list[int]:
        tokenizer = load_native_tokenizer(model_config)
        if tokenizer is None:
            return []
        return Qwen3OmniDuplexPolicy.thinker_stop_token_ids(tokenizer)

    @staticmethod
    def _scheduler_token_id(model_config: Any) -> int | None:
        tokenizer = load_native_tokenizer(model_config)
        if tokenizer is None:
            return None
        return Qwen3OmniDuplexPolicy.scheduler_token_id(tokenizer)
