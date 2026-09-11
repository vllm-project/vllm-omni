# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 full-duplex model plugin: engine policy and session policy in one class.

Engine policy (scheduler token budget, resumable Stage0 prompt, listen
decision) and session policy (client config validation, reference-audio
resolution, per-stage sampling knobs) both run engine-side, so one class owns
them. Client-supplied media URIs are resolved here through the media
connector; workers only ever receive normalized PCM payloads.
"""

from __future__ import annotations

import asyncio
import base64
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from vllm.multimodal.media import MediaConnector
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    EncodeAudio,
    reject_changed_runtime_value,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import (
    minicpmo45_native_capabilities,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import (
    MiniCPMO45ServingSessionState,
)

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase
    from vllm.config import ModelConfig

_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600
# <image> + 64 resampler embeddings + </image> per frame (max_slice_nums=1),
# matching MiniCPMO45DuplexPolicy.VISION_TOKENS_PER_FRAME.
_DUPLEX_VISION_TOKENS_PER_FRAME = 66
# Official stacked pair uses max_slice_nums=[2, 1]: the current frame is HD
# sliced (1 source + 2 patches on 960x540) and the composite is not.
_DUPLEX_HD_SLICES_PER_BASE_FRAME = 3

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
        "initial_user_text",
    }
)


class MiniCPMO45ClientRuntimeConfigError(DuplexRuntimeConfigError):
    pass


# ---- engine policy helpers: scheduler token budget ----


def _duplex_frame_count(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    frames = payload.get("video_frames")
    if not isinstance(frames, list):
        return 0
    return sum(1 for frame in frames if isinstance(frame, str) and frame)


def _duplex_vision_tokens(payload: object) -> int:
    """Scheduler slots for this append's camera track.

    Audio is never stacked: a unit still carries one second of soundtrack.
    ``stack_frames`` only adds a second *image*. Official HD on that pair is
    ``[2, 1]``, so the base frame reserves three 66-token blocks and every
    extra frame reserves one.
    """
    count = _duplex_frame_count(payload)
    if count <= 0:
        return 0
    if count >= 2:
        return (_DUPLEX_HD_SLICES_PER_BASE_FRAME + (count - 1)) * _DUPLEX_VISION_TOKENS_PER_FRAME
    return count * _DUPLEX_VISION_TOKENS_PER_FRAME


def _duplex_pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def duplex_payload_is_exact_chunks(payload: object) -> bool:
    sample_count = _duplex_pcm_sample_count(payload)
    return sample_count is not None and sample_count != 0 and sample_count % _DUPLEX_CHUNK_SAMPLES == 0


def duplex_first_append_unit_count(payload: object) -> int | None:
    sample_count = _duplex_pcm_sample_count(payload)
    if not sample_count or sample_count % _DUPLEX_CHUNK_SAMPLES != 0:
        return None
    return max(1, sample_count // _DUPLEX_CHUNK_SAMPLES - 1)


def duplex_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    vision_tokens = _duplex_vision_tokens(payload)
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default)) + vision_tokens
    sample_count = max(1, sample_count)
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (2 + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN) + vision_tokens
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)) + vision_tokens


def duplex_first_append_context_reserve(runtime_config: object) -> int:
    if not isinstance(runtime_config, dict):
        return 48
    exact = runtime_config.get("duplex_first_append_context_tokens")
    if isinstance(exact, int) and exact >= 0:
        return exact
    reserve = 48
    ref = runtime_config.get("ref_audio_data")
    if isinstance(ref, str) and ref:
        try:
            raw = b64decode(ref, validate=True)
        except (BinasciiError, ValueError):
            raw = b""
        if raw:
            reserve += max(0, (len(raw) // 4) // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)
    return reserve


def _duplex_force_listen_count(extra_body: object) -> int:
    raw = extra_body.get("force_listen_count") if isinstance(extra_body, dict) else None
    try:
        return 0 if raw is None else max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def build_duplex_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, object],
    runtime_config: dict[str, object],
    seq: int,
    turn_seq: int,
    payload: object,
    final: bool,
) -> dict[str, object]:
    token_budget = duplex_scheduler_token_budget(payload)
    if seq <= 1:
        context_reserve = duplex_first_append_context_reserve(runtime_config)
        token_budget += context_reserve
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = context_reserve + first_units * 12 - 1 + _duplex_vision_tokens(payload)
    if seq > 1 and duplex_payload_is_exact_chunks(payload):
        token_budget += 1
    if final and duplex_payload_is_exact_chunks(payload):
        token_budget += 12
    extra_body = session_config.get("extra_body")
    raw_token_id = runtime_config.get("duplex_scheduler_token_id")
    try:
        token_id = 0 if raw_token_id is None else max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0
    force_listen_count = _duplex_force_listen_count(extra_body)
    if (
        force_listen_count > 0
        and turn_seq <= force_listen_count
        and isinstance(payload, dict)
        and payload.get("force_listen") is not True
    ):
        payload = {**payload, "force_listen": True}
    return {
        "prompt_token_ids": [token_id] * token_budget,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "fence": fence,
                "session_id": fence.session_id,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "turn_seq": turn_seq,
                "mode": "append_audio_chunk",
                "payload": payload,
                "final": final,
                "data_plane": True,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "scheduler_token_budget": token_budget,
                "scheduler_token_id": token_id,
            },
        },
    }


# ---- engine policy helpers: listen decision ----


def _coerce_int(value: object) -> int | None:
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            flat: torch.Tensor = detach().cpu().reshape(-1)
            if flat.numel() == 0:
                return None
            value = flat[0].item()
        except Exception:
            return None
    try:
        return int(cast(Any, value))  # Any: duck-typed scalar (int/float/str/tensor item)
    except (TypeError, ValueError):
        return None


def _coerce_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1).tolist()
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [token_id for item in value if (token_id := _coerce_int(item)) is not None]


def _first_completion(output: object) -> object | None:
    outputs = getattr(output, "outputs", None)
    return outputs[0] if isinstance(outputs, list) and outputs else None


def _multimodal_output(output: object, completion: object | None) -> dict[str, object]:
    metadata = getattr(output, "multimodal_output", None)
    if isinstance(metadata, dict):
        return metadata
    metadata = getattr(completion, "multimodal_output", None) if completion is not None else None
    return metadata if isinstance(metadata, dict) else {}


def _special_token_ids(metadata: dict[str, object]) -> dict[str, int]:
    sources: list[object] = [metadata.get("special_token_ids"), metadata.get("meta")]
    sources.append(
        {
            key.removeprefix("meta."): value
            for key, value in metadata.items()
            if isinstance(key, str) and key.startswith("meta.")
        }
    )
    token_ids: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            token_id = _coerce_int(value)
            if isinstance(key, str) and token_id is not None and token_id >= 0:
                token_ids[key] = token_id
    return token_ids


def _completion_token_ids(completion: object | None) -> list[int]:
    if completion is None:
        return []
    for attribute in ("token_ids", "cumulative_token_ids"):
        token_ids = _coerce_int_list(getattr(completion, attribute, None))
        if token_ids:
            return token_ids
    return []


def _stage_config_value(runtime_config: dict[str, object], key: str, stage_id: int) -> object | None:
    raw = runtime_config.get(key)
    if isinstance(raw, dict):
        value = raw.get(stage_id)
        return raw.get(str(stage_id)) if value is None else value
    if isinstance(raw, (list, tuple)) and stage_id < len(raw):
        return raw[stage_id]
    return None


# ---- session policy helpers: tokenizer / reference audio ----


def _load_tokenizer(model_config: ModelConfig | None) -> PreTrainedTokenizerBase | None:
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


def _convert_token_to_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    value = None
    if callable(convert):
        value = convert(token)
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else None
    if value is None:
        token_id = -1
    else:
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


def _stage0_stop_token_ids(tokenizer: PreTrainedTokenizerBase | None) -> list[int]:
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
        token_id = _convert_token_to_id(tokenizer, token)
        if token_id is not None and token_id not in out:
            out.append(token_id)
    return out


def _scheduler_token_id(tokenizer: PreTrainedTokenizerBase | None) -> int | None:
    if tokenizer is None:
        return None
    scheduler_tokens = (
        MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS["unit_token_id"],
        MiniCPMO45DuplexPolicy.OPTIONAL_TOKEN_FIELDS["audio_placeholder_token_id"],
    )
    for token in scheduler_tokens:
        token_id = _convert_token_to_id(tokenizer, token)
        if token_id is not None:
            return token_id
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return None
    try:
        return int(eos_id)
    except (TypeError, ValueError):
        return None


async def resolve_ref_audio(ref_audio: str, *, model_config: ModelConfig | None) -> tuple[NDArray[np.float32], int]:
    connector = MediaConnector(
        allowed_local_media_path=getattr(model_config, "allowed_local_media_path", None),
        allowed_media_domains=getattr(model_config, "allowed_media_domains", None),
    )
    wav_np, sr = await connector.fetch_audio_async(ref_audio)
    return np.asarray(wav_np, dtype=np.float32), int(sr)


def normalize_ref_audio(wav_np: NDArray[np.float32], sample_rate: int, *, target_sr: int) -> NDArray[np.float32]:
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


def _apply_first_append_context_tokens(
    runtime_config: dict[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase | None,
    instructions: object,
    initial_user_text: object,
    ref_sample_count: int | None,
) -> None:
    """Precompute the exact session-context token count for the engine.

    The first data-plane append carries the system template and optional
    reference-audio embeddings ahead of the first unit. The engine reserves
    scheduler slots from this count; an inexact count turns into pad
    embeddings inside the model KV (surplus) or truncated context (deficit),
    so it is computed with the same template and pooling math the worker uses.
    """
    if "duplex_first_append_context_tokens" in runtime_config or tokenizer is None:
        return
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
        instructions,
        ref_sample_count is not None,
        initial_user_text,
    )
    try:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    except Exception:
        return
    ref_tokens = MiniCPMO45DuplexPolicy.audio_token_count(ref_sample_count or 0)
    runtime_config["duplex_first_append_context_tokens"] = len(prefix_ids) + ref_tokens + len(suffix_ids)


def _apply_default_scheduler_policy(
    runtime_config: dict[str, object],
    *,
    config: DuplexSessionConfig,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    stage0_max_tokens = config.max_tokens if isinstance(config.max_tokens, int) and config.max_tokens > 0 else 20
    runtime_config["duplex_stage_max_tokens"] = {"0": stage0_max_tokens, "1": 8192}
    stage0_params: dict[str, object] = {
        "temperature": config.temperature if config.temperature is not None else 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }
    stop_token_ids = _stage0_stop_token_ids(tokenizer)
    if stop_token_ids:
        stage0_params["stop_token_ids"] = stop_token_ids
    # Stage 1 keeps the deploy YAML's codec knobs, minus upstream's
    # min_new_token=50: that floor is for whole-utterance chat TTS, while a
    # duplex chunk is 26 codec samples, so check_stop would never release
    # the Talker and Thinker would stall on the next model turn.
    runtime_config["duplex_stage_sampling_params"] = {"0": stage0_params, "1": {"min_tokens": 0}}
    scheduler_token_id = _scheduler_token_id(tokenizer)
    if scheduler_token_id is not None:
        runtime_config["duplex_scheduler_token_id"] = scheduler_token_id


class MiniCPMO45DuplexPlugin(DuplexModelPlugin):
    """MiniCPM-owned sampling policy, append planning, session state and output projection."""

    plugin_id = "minicpmo45"
    private_runtime_config_keys = PRIVATE_RUNTIME_CONFIG_KEYS
    silence_continuation_samples = 16000

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = MiniCPMO45DataPlaneSession(encode_audio)
        # The tokenizer is only needed for special-token ids and the session
        # context token count; it is the same for every session, so load it
        # once per model path, off the orchestrator loop.
        self._tokenizers: dict[str, PreTrainedTokenizerBase | None] = {}
        self._tokenizer_lock = asyncio.Lock()

    async def _tokenizer_for(self, model_config: ModelConfig | None) -> PreTrainedTokenizerBase | None:
        model_path = getattr(model_config, "model", None)
        key = model_path if isinstance(model_path, str) else ""
        if key in self._tokenizers:
            return self._tokenizers[key]
        async with self._tokenizer_lock:
            if key not in self._tokenizers:
                self._tokenizers[key] = await asyncio.to_thread(_load_tokenizer, model_config)
        return self._tokenizers[key]

    # ---- engine policy (the resumable Stage0 request) ----

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        configured: list[object] = []
        for stage_id, default in enumerate(defaults):
            max_tokens = _coerce_int(_stage_config_value(runtime_config, "duplex_stage_max_tokens", stage_id))
            raw_overrides = _stage_config_value(runtime_config, "duplex_stage_sampling_params", stage_id)
            overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
            if not isinstance(default, SamplingParams) or (not overrides and (max_tokens is None or max_tokens <= 0)):
                configured.append(default)
                continue
            params = default.clone()
            if max_tokens is not None and max_tokens > 0:
                params.max_tokens = max_tokens
            for name, value in overrides.items():
                if not hasattr(params, name):
                    continue
                setattr(params, name, value)
                if name == "stop_token_ids":
                    all_stop_token_ids = getattr(params, "_all_stop_token_ids", None)
                    if isinstance(all_stop_token_ids, set):
                        all_stop_token_ids.update(int(token_id) for token_id in value)
            configured.append(params)
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        return DuplexAppendPlan(
            prompt=build_duplex_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload=payload,
                final=final,
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        if stage_id >= final_stage_id or not segment_finished:
            return None

        completion = _first_completion(output)
        output_metadata = _multimodal_output(output, completion)
        special_token_ids = _special_token_ids(segment_output_metadata)
        special_token_ids.update(_special_token_ids(output_metadata))
        listen_id = special_token_ids.get("listen_token_id")
        if listen_id is None:
            return None

        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
        token_ids = _completion_token_ids(completion) or list(segment_token_ids)
        if _coerce_int(stop_reason) != listen_id and (not token_ids or token_ids[-1] != listen_id):
            return None

        metadata = dict(output_metadata)
        for key, value in special_token_ids.items():
            metadata.setdefault(f"meta.{key}", value)
        metadata.update(
            {
                "duplex_direct_response": True,
                "duplex_native_decision": "listen",
                "model_listen": True,
                "listen_source": "model_listen",
            }
        )
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
        )

    # ---- session policy ----

    def create_session_state(self) -> MiniCPMO45ServingSessionState:
        return MiniCPMO45ServingSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return minicpmo45_native_capabilities(max_sessions=max_sessions)

    def validate_client_extra_body(self, extra_body: object) -> None:
        if not isinstance(extra_body, dict):
            return
        private_keys = sorted(PRIVATE_RUNTIME_CONFIG_KEYS.intersection(extra_body))
        if private_keys:
            raise MiniCPMO45ClientRuntimeConfigError(
                "duplex runtime configuration is server-owned: " + ", ".join(private_keys)
            )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        extra_body = dict(config.extra_body)
        if any(key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")):
            raise MiniCPMO45ClientRuntimeConfigError(
                "ref_audio_path is not accepted by duplex sessions; use ref_audio URI instead",
                code="unsupported_ref_audio_path",
            )
        runtime_config: dict[str, object] = {"instructions": config.instructions}
        initial_user_text = extra_body.pop("duplex_initial_user_text", None)
        if isinstance(initial_user_text, str) and initial_user_text:
            runtime_config["initial_user_text"] = initial_user_text
        tokenizer = await self._tokenizer_for(model_config)
        _apply_default_scheduler_policy(runtime_config, config=config, tokenizer=tokenizer)

        ref_audio = config.ref_audio
        extra_ref_audio = extra_body.get("ref_audio")
        if ref_audio is None and isinstance(extra_ref_audio, str):
            extra_body.pop("ref_audio")
            ref_audio = extra_ref_audio
        extra_tts_ref_audio = extra_body.get("tts_ref_audio")
        if ref_audio is None and isinstance(extra_tts_ref_audio, str):
            extra_body.pop("tts_ref_audio")
            ref_audio = extra_tts_ref_audio

        if ref_audio is None:
            if any(str(modality).lower() == "audio" for modality in config.modalities):
                raise MiniCPMO45ClientRuntimeConfigError(
                    "MiniCPM-o duplex audio output requires ref_audio",
                    code="ref_audio_required",
                )
            _apply_first_append_context_tokens(
                runtime_config,
                tokenizer=tokenizer,
                instructions=config.instructions,
                initial_user_text=initial_user_text,
                ref_sample_count=None,
            )
            config.extra_body = extra_body
            return runtime_config

        wav_np, sr = await resolve_ref_audio(ref_audio, model_config=model_config)
        # torchaudio resampling is CPU work: keep it off the orchestrator loop.
        wav_np = await asyncio.to_thread(normalize_ref_audio, wav_np, int(sr), target_sr=16000)
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
        _apply_first_append_context_tokens(
            runtime_config,
            tokenizer=tokenizer,
            instructions=config.instructions,
            initial_user_text=initial_user_text,
            ref_sample_count=len(wav_np),
        )
        config.extra_body = extra_body
        config.ref_audio = None
        return runtime_config

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        runtime_config = deepcopy(dict(current))
        reject_changed_runtime_value(
            config.instructions,
            runtime_config.get("instructions"),
            message="instructions cannot be changed after the session is created",
            code="instructions_update_unsupported",
            error_cls=MiniCPMO45ClientRuntimeConfigError,
        )
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

    def data_plane_context(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> MiniCPMO45DataPlaneContext:
        return MiniCPMO45DataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )


__all__ = [
    "PRIVATE_RUNTIME_CONFIG_KEYS",
    "MiniCPMO45ClientRuntimeConfigError",
    "MiniCPMO45DuplexPlugin",
    "build_duplex_data_plane_prompt",
    "duplex_first_append_context_reserve",
    "duplex_first_append_unit_count",
    "duplex_payload_is_exact_chunks",
    "duplex_scheduler_token_budget",
    "normalize_ref_audio",
    "resolve_ref_audio",
]
