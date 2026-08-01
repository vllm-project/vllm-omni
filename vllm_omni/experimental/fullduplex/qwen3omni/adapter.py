# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-Omni duplex serving policy: config validation and capabilities.

Boundary invariant carried over from MiniCPM's adapter: the serving layer
owns client-facing validation and normalization; workers receive only
already-normalized values. Nothing here may resolve URIs or filesystem
paths on a worker's behalf.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import ServingRuntimeConfigError
from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

#: Default stage-0 token bound for one spoken reply. A backstop only -- the
#: real terminator is ``<|im_end|>`` in ``stop_token_ids`` below. Kept in the
#: same range as the half-duplex realtime path
#: (``qwen3_omni.py: realtime_max_tokens = 64``), since a spoken turn should
#: be short; every generated token becomes synthesized speech.
_DEFAULT_STAGE0_MAX_TOKENS = 128

#: Used when the client sets no `instructions`. The thinker needs a system
#: turn for the chat template to be well-formed.
_DEFAULT_INSTRUCTIONS = (
    "You are a helpful voice assistant. Reply in the same language the user "
    "speaks. Keep replies short and conversational -- one or two sentences."
)
_DEFAULT_TALKER_MAX_TOKENS = 8192


#: Checkpoint tokenizer, cached when a session prepares its runtime config.
#:
#: Scaffolding is pre-tokenized at session open (``_scaffolding_token_ids``),
#: but a client text turn arrives later and has to be encoded on the spot. Both
#: run in the API server process, so the tokenizer built there is reused rather
#: than constructed per message. Keyed by model so a mixed-model server cannot
#: encode with the wrong vocabulary.
_TOKENIZER_CACHE: dict[str, Any] = {}


def _tokenizer_cache_key(model_config: Any) -> str:
    return str(getattr(model_config, "model", None) or getattr(model_config, "served_model_name", "") or "default")


def encode_duplex_text(text: str, *, model_key: str | None = None) -> list[int]:
    """Encode a client text turn with the checkpoint tokenizer.

    Returns ``[]`` when no tokenizer has been cached yet, which the caller must
    treat as "cannot accept text" rather than "empty turn" -- reserving zero
    tokens for a turn the user did type would prompt the model to answer
    nothing, which reads to them as the model ignoring them.
    """
    if not text:
        return []
    tokenizer = _TOKENIZER_CACHE.get(model_key) if model_key else None
    if tokenizer is None:
        if len(_TOKENIZER_CACHE) != 1:
            return []
        tokenizer = next(iter(_TOKENIZER_CACHE.values()))
    return [int(token_id) for token_id in tokenizer.encode(text)]


class Qwen3OmniClientRuntimeConfigError(ServingRuntimeConfigError):
    """A client tried to set server-owned Qwen3-Omni duplex configuration."""


class Qwen3OmniNativeDuplexServingAdapter:
    """Stateless serving policy for the Qwen3-Omni duplex path."""

    PRIVATE_RUNTIME_CONFIG_KEYS = Qwen3OmniDuplexPolicy.PRIVATE_RUNTIME_CONFIG_KEYS

    @staticmethod
    def is_enabled(config: object) -> bool:
        extra_body = getattr(config, "extra_body", None)
        if not isinstance(extra_body, Mapping):
            return False
        return extra_body.get(Qwen3OmniDuplexPolicy.ENABLE_FLAG) is True

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
    def capabilities(cls, *, max_sessions: int = 1) -> DuplexCapabilities:
        """Advertise what this integration actually supports.

        Deliberately conservative. In particular
        ``supports_model_native_turn_policy=False`` -- Qwen3-Omni has no
        learned listen/speak control token, so turn boundaries come from the
        client, not the model. Claiming otherwise would make the endpoint
        advertise MiniCPM's behavior and mislead clients.
        """
        supports_multi_session = max_sessions > 1
        return DuplexCapabilities(
            # Qwen3-Omni owns no turn policy; the client signals boundaries.
            supports_model_native_turn_policy=False,
            supports_barge_in=True,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_turn_commit_only=True,
            supports_kv_lease=False,
            supports_core_kv_lease=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_scheduler_native_append=False,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=supports_multi_session,
            supports_multi_session_same_replica=supports_multi_session,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_audio_truncate=True,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            implementation_level="model_native_duplex",
            adapter_patterns=["scheduler_data_plane"],
            input_modes=["append_audio_chunk", "turn_commit_only", "append_tokens"],
            signal_sources=["client_event", "server_policy"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=Qwen3OmniDuplexPolicy.CHUNK_PERIOD_MS,
            target_barge_in_latency_ms=None,
        )

    @staticmethod
    def _scaffolding_token_ids(instructions: object, *, model_config: Any) -> dict[str, object]:
        """Pre-tokenize the conversation scaffolding for the worker.

        The thinker needs chat-template framing around the audio embeddings;
        without it there is nothing instructing the model to reply and it
        emits EOS on the first token.

        Tokenizing here rather than in the worker means the engine's slot
        reservation and the worker's produced-embedding count are derived
        from the same token ids, which is the invariant the model runner
        silently violates on mismatch.
        """
        try:
            from vllm.tokenizers import cached_tokenizer_from_config

            tokenizer = cached_tokenizer_from_config(model_config)
        except Exception:  # noqa: BLE001 - no tokenizer, fall back to no scaffolding
            return {}

        # Keep it for client text turns, which cannot be pre-tokenized here.
        _TOKENIZER_CACHE[_tokenizer_cache_key(model_config)] = tokenizer

        def encode(text: str) -> list[int]:
            return [int(token_id) for token_id in tokenizer.encode(text)]

        session_prefix = Qwen3OmniDuplexPolicy.SESSION_PREFIX_TEMPLATE.format(
            instructions=instructions if isinstance(instructions, str) and instructions else _DEFAULT_INSTRUCTIONS
        )
        return {
            Qwen3OmniDuplexPolicy.SESSION_PREFIX_IDS_KEY: encode(session_prefix),
            Qwen3OmniDuplexPolicy.TURN_PREFIX_IDS_KEY: encode(Qwen3OmniDuplexPolicy.TURN_PREFIX),
            Qwen3OmniDuplexPolicy.TURN_SUFFIX_IDS_KEY: encode(Qwen3OmniDuplexPolicy.TURN_SUFFIX),
            Qwen3OmniDuplexPolicy.NEWLINE_IDS_KEY: encode("\n"),
        }

    @classmethod
    async def prepare_runtime_config(cls, config: object, *, model_config: Any) -> dict[str, object]:
        """Build the server-owned runtime config shipped to the engine."""
        cls.validate_client_extra_body(getattr(config, "extra_body", None))

        max_tokens = _positive_int(getattr(config, "max_tokens", None)) or _DEFAULT_STAGE0_MAX_TOKENS
        temperature = getattr(config, "temperature", None)
        instructions = getattr(config, "instructions", None)

        # Stop on the thinker's own EOS. Without this the model answers and
        # then keeps going to the token cap, and every extra token is spoken.
        stage0_sampling: dict[str, object] = {
            "stop_token_ids": [Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID],
        }
        if isinstance(temperature, (int, float)):
            stage0_sampling["temperature"] = float(temperature)

        runtime_config: dict[str, object] = {
            "duplex_stage_max_tokens": {
                "0": max_tokens,
                "1": _DEFAULT_TALKER_MAX_TOKENS,
            },
            "duplex_chunk_period_ms": Qwen3OmniDuplexPolicy.CHUNK_PERIOD_MS,
        }
        runtime_config.update(cls._scaffolding_token_ids(instructions, model_config=model_config))
        if instructions:
            runtime_config["instructions"] = instructions
        if stage0_sampling:
            runtime_config["duplex_stage_sampling_params"] = {"0": stage0_sampling}
        return runtime_config

    @classmethod
    def runtime_config_for_update(cls, config: object, current: Mapping[str, object]) -> dict[str, object]:
        """Apply a ``session.update`` without discarding server-owned state.

        Only the fields a client may change are overwritten; everything else
        in ``current`` is preserved.
        """
        updated: dict[str, object] = deepcopy(dict(current)) if isinstance(current, Mapping) else {}

        instructions = getattr(config, "instructions", None)
        if instructions is not None:
            updated["instructions"] = instructions

        max_tokens = _positive_int(getattr(config, "max_tokens", None))
        if max_tokens is not None:
            stage_max_tokens = updated.get("duplex_stage_max_tokens")
            stage_max_tokens = dict(stage_max_tokens) if isinstance(stage_max_tokens, Mapping) else {}
            stage_max_tokens["0"] = max_tokens
            stage_max_tokens.setdefault("1", _DEFAULT_TALKER_MAX_TOKENS)
            updated["duplex_stage_max_tokens"] = stage_max_tokens

        temperature = getattr(config, "temperature", None)
        if isinstance(temperature, (int, float)):
            sampling = updated.get("duplex_stage_sampling_params")
            sampling = dict(sampling) if isinstance(sampling, Mapping) else {}
            stage0 = dict(sampling.get("0")) if isinstance(sampling.get("0"), Mapping) else {}
            stage0.setdefault("stop_token_ids", [Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID])
            stage0["temperature"] = float(temperature)
            sampling["0"] = stage0
            updated["duplex_stage_sampling_params"] = sampling

        return updated


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
