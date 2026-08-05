# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import binascii
import io
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.personaplex.policy import (
    AUDIO_SILENCE_FRAME_CNT,
    SILENCE_TOKENS,
    SINE_TOKENS,
    ZERO_TEXT_TOKEN,
    wrap_with_system_tags,
)

_FRAME_SAMPLES = 1920


@dataclass(slots=True)
class PersonaPlexStage0PreparedAppend:
    input_ids: Any
    inputs_embeds: Any
    user_codes: Any
    info_update: dict[str, Any]
    prefill_applied: bool
    prompt_offset: int


@dataclass(slots=True)
class PersonaPlexStage0SessionState:
    session_id: str
    incarnation: int
    user_codes: Any | None = None
    last_text_token: Any | None = None
    last_agent_codes: Any | None = None
    prefill_slots: int = 0
    prepared_identity: tuple[int, int] | None = None
    sampled_identity: tuple[int, int] | None = None
    prepared: PersonaPlexStage0PreparedAppend | None = None
    last_seq: int = 0
    request_ids: set[str] = field(default_factory=set)
    codec: Any | None = None


def _tokenizer_path(model_path: str) -> Path:
    path = Path(model_path) / "tokenizer_spm_32k_3.model"
    if not path.is_file():
        raise FileNotFoundError(f"PersonaPlex tokenizer not found: {path}")
    return path


def load_personaplex_tokenizer(model_path: str):
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(str(_tokenizer_path(model_path)))
    return lambda text: list(tokenizer.encode(text))


def load_personaplex_voice_state(model_path: str, voice: str) -> dict[str, Any]:
    import torch

    root = Path(model_path)
    for candidate in (root / "voices" / voice, root / voice):
        if candidate.is_file():
            state = torch.load(candidate, map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                return state
            raise ValueError(f"PersonaPlex voice bundle is not a mapping: {candidate}")

    archive = root / "voices.tgz"
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as tar:
            member = next(
                (item for item in tar.getmembers() if item.isfile() and Path(item.name).name == voice),
                None,
            )
            if member is not None:
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"PersonaPlex voice prompt {voice!r} is unreadable")
                state = torch.load(io.BytesIO(extracted.read()), map_location="cpu", weights_only=True)
                if isinstance(state, dict):
                    return state
                raise ValueError(f"PersonaPlex voice bundle {voice!r} is not a mapping")
    raise FileNotFoundError(f"PersonaPlex bundled voice prompt {voice!r} was not found under {model_path!r}")


def personaplex_prefill_slots(model_path: str, voice: str, persona: str) -> int:
    state = load_personaplex_voice_state(model_path, voice)
    embeddings = state.get("embeddings")
    if not hasattr(embeddings, "shape") or len(embeddings.shape) < 1:
        raise ValueError(f"PersonaPlex voice prompt {voice!r} has no embeddings")
    tokenizer = load_personaplex_tokenizer(model_path)
    persona_tokens = tokenizer(wrap_with_system_tags(persona)) if persona else []
    return int(embeddings.shape[0]) + 2 * AUDIO_SILENCE_FRAME_CNT + len(persona_tokens)


class PersonaPlexStage0DuplexRuntime:
    """Own per-session streaming Mimi encoders and first-append prefill."""

    def __init__(
        self,
        stage_model: Any,
        *,
        model_path: str,
        device: str,
        codec: Any | None = None,
        codec_factory: Callable[[], Any] | None = None,
        max_sessions: int = 1,
        tokenizer=None,
        voice_loader=None,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError("PersonaPlex Stage 0 max_sessions must be positive")
        self.stage_model = stage_model
        self.model_path = model_path
        self.device = device
        self.max_sessions = max_sessions
        self._codec_factory = codec_factory
        self._free_codecs: list[Any] = []
        self._tokenizer = tokenizer
        self._voice_loader = voice_loader
        self.sessions: dict[tuple[str, int], PersonaPlexStage0SessionState] = {}
        self.request_sessions: dict[str, tuple[str, int]] = {}
        if codec is not None:
            codec.streaming_init(1)
            self._free_codecs.append(codec)

    def prepare_append(
        self,
        duplex: dict[str, Any],
        *,
        prompt_len: int,
        request_id: str | None = None,
    ) -> PersonaPlexStage0PreparedAppend:
        import torch

        session_id = duplex.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("PersonaPlex duplex append requires session_id")
        incarnation = _coerce_non_negative_int(duplex.get("incarnation"), "incarnation")
        epoch = _coerce_non_negative_int(duplex.get("epoch"), "epoch")
        seq = _coerce_positive_int(duplex.get("seq"), "seq")
        identity = (epoch, seq)
        key = (session_id, incarnation)

        state = self.sessions.get(key)
        if state is None:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(f"PersonaPlex Stage 0 session capacity {self.max_sessions} is exhausted")
            state = PersonaPlexStage0SessionState(
                session_id=session_id,
                incarnation=incarnation,
                codec=self._acquire_codec(),
            )
            self.sessions[key] = state
        if request_id:
            state.request_ids.add(request_id)
            self.request_sessions[request_id] = key
        if state.prepared_identity == identity and state.prepared is not None:
            return state.prepared
        if seq <= state.last_seq:
            raise ValueError(f"PersonaPlex duplex append seq must increase: last={state.last_seq}, got={seq}")

        pcm = self._decode_pcm(duplex.get("payload"))
        codec = state.codec
        if codec is None:
            raise RuntimeError("PersonaPlex Stage 0 session has no streaming encoder")
        encoded = codec.encode_frame(torch.from_numpy(pcm).reshape(1, _FRAME_SAMPLES))
        user_frame = encoded.detach().to(dtype=torch.long, device="cpu").reshape(1, -1)
        if user_frame.shape[1] < 8:
            raise RuntimeError(
                f"PersonaPlex Mimi encoder returned {user_frame.shape[1]} codebooks, expected at least 8"
            )
        user_frame = user_frame[:, :8].contiguous()
        state.user_codes = user_frame if state.user_codes is None else torch.cat([state.user_codes, user_frame], dim=0)

        runtime_config = duplex.get("runtime_config")
        runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}
        first_append = state.last_seq == 0
        device, dtype = self._model_device_dtype()
        silence = torch.tensor(SILENCE_TOKENS, dtype=torch.long, device=device)
        sine = torch.tensor(SINE_TOKENS, dtype=torch.long, device=device)
        text_token = state.last_text_token
        if text_token is None:
            text_token = torch.tensor([ZERO_TEXT_TOKEN], dtype=torch.long, device=device)
        last_agent = state.last_agent_codes if state.last_agent_codes is not None else silence
        user_frame_count = state.user_codes.shape[0]
        # Match the native lockstep ring: the temporal input sees the previous
        # effective agent frame. User cb0 trails the frame being appended by one
        # tick and user cb1..7 trail it by two ticks. Before live user frames
        # fill those slots, text-prompt prefill leaves encoded sine on user rows.
        user_d0 = state.user_codes[-2].to(device) if user_frame_count > 1 else sine
        user_d1 = state.user_codes[-3].to(device) if user_frame_count > 2 else sine
        depformer_audio_tokens = torch.cat(
            [
                silence,
                user_frame[0, :1].to(device),
                user_d0[1:8],
            ]
        )
        depformer_audio_provided = torch.tensor(
            [
                False,
                *([first_append] * 7),
                *([True] * 8),
            ],
            dtype=torch.bool,
            device=device,
        )
        live_embed = self.stage_model._build_frame_embed(
            text_token,
            last_agent,
            last_agent,
            device,
            user_d0=user_d0,
            user_d1=user_d1,
        )
        if first_append:
            voice = runtime_config.get("personaplex_voice_prompt", "NATF2.pt")
            persona = runtime_config.get("personaplex_persona", "")
            if not isinstance(voice, str) or not voice:
                raise ValueError("PersonaPlex runtime voice prompt is invalid")
            if not isinstance(persona, str):
                raise ValueError("PersonaPlex runtime persona is invalid")
            voice_state = self._load_voice(voice)
            voice_embeddings = voice_state.get("embeddings")
            if not isinstance(voice_embeddings, torch.Tensor) or voice_embeddings.numel() == 0:
                raise ValueError(f"PersonaPlex voice prompt {voice!r} has no embeddings")
            voice_embeddings = voice_embeddings.reshape(-1, voice_embeddings.shape[-1]).to(
                device=device,
                dtype=dtype,
            )
            tokenizer = self._load_tokenizer()
            persona_tokens = tokenizer(wrap_with_system_tags(persona)) if persona else []
            prefill_tokens = torch.tensor(
                [
                    *([ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT),
                    *persona_tokens,
                    *([ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT),
                ],
                dtype=torch.long,
                device=device,
            )
            token_prefill = self.stage_model._build_prefill_embed(
                prefill_tokens,
                0,
                int(prefill_tokens.numel()),
                device,
                silence,
                user_sine=sine,
            )
            prefill_embeds = torch.cat(
                [
                    voice_embeddings,
                    token_prefill.to(dtype=dtype),
                ],
                dim=0,
            )
            state.prefill_slots = int(prefill_embeds.shape[0])
            full_embeds = torch.cat([prefill_embeds, live_embed.to(dtype=dtype)], dim=0)
        else:
            full_embeds = live_embed.to(dtype=dtype)

        prepared_len = int(full_embeds.shape[0])
        prompt_offset = int(prompt_len) - prepared_len
        if prompt_offset < 0:
            raise ValueError(
                f"PersonaPlex scheduler prompt reservation mismatch: reserved={prompt_len}, prepared={prepared_len}"
            )
        input_ids = torch.zeros((prepared_len,), dtype=torch.long, device=device)
        info_update = {
            "pplex_user_codes": state.user_codes,
            "pplex_silence_codes": silence.detach().cpu(),
            "pplex_depformer_audio_tokens": depformer_audio_tokens.detach().cpu(),
            "pplex_depformer_audio_provided": depformer_audio_provided.detach().cpu(),
            "meta": {
                "pplex_frame": state.prefill_slots + int(state.user_codes.shape[0]),
                "pplex_prefill_len": state.prefill_slots,
            },
            "duplex": {
                "stage0_prepared": True,
                "prefill_applied": first_append,
                "session_id": session_id,
                "incarnation": incarnation,
                "epoch": epoch,
                "seq": seq,
            },
        }
        prepared = PersonaPlexStage0PreparedAppend(
            input_ids=input_ids,
            inputs_embeds=full_embeds,
            user_codes=state.user_codes,
            info_update=info_update,
            prefill_applied=first_append,
            prompt_offset=prompt_offset,
        )
        state.prepared_identity = identity
        state.prepared = prepared
        state.last_seq = seq
        return prepared

    def record_sample(
        self,
        *,
        request_id: str,
        text_token: Any,
        agent_codes: Any,
    ) -> None:
        """Commit one sampled temporal frame for the next live append."""
        import torch

        key = self.request_sessions.get(request_id)
        if key is None:
            raise KeyError(f"PersonaPlex Stage 0 request is not attached to a live session: {request_id}")
        state = self.sessions.get(key)
        if state is None or state.prepared_identity is None:
            raise RuntimeError(f"PersonaPlex Stage 0 request has no prepared append: {request_id}")
        if state.sampled_identity == state.prepared_identity:
            return

        text = torch.as_tensor(text_token, device=self._model_device_dtype()[0], dtype=torch.long).reshape(-1)
        codes = torch.as_tensor(agent_codes, device=text.device, dtype=torch.long).reshape(-1)
        if text.numel() != 1:
            raise ValueError(f"PersonaPlex Stage 0 expected one sampled text token, got {text.numel()}")
        if codes.numel() < 8:
            raise ValueError(f"PersonaPlex Stage 0 expected at least 8 agent codes, got {codes.numel()}")

        effective_codes = codes[:8].clone()
        prepared = state.prepared
        if prepared is None:
            raise RuntimeError(f"PersonaPlex Stage 0 request has no prepared payload: {request_id}")
        target = prepared.info_update.get("pplex_depformer_audio_tokens")
        provided = prepared.info_update.get("pplex_depformer_audio_provided")
        if not isinstance(target, torch.Tensor) or not isinstance(provided, torch.Tensor):
            raise RuntimeError("PersonaPlex Stage 0 prepared payload has no depformer teacher-forcing state")
        target = target.reshape(-1).to(device=effective_codes.device, dtype=torch.long)
        provided = provided.reshape(-1).to(device=effective_codes.device, dtype=torch.bool)
        if target.numel() < 8 or provided.numel() < 8:
            raise RuntimeError("PersonaPlex Stage 0 depformer teacher-forcing state is shorter than one agent frame")
        effective_codes = torch.where(provided[:8], target[:8], effective_codes)
        state.last_agent_codes = effective_codes.detach()
        state.last_text_token = text.detach()
        state.sampled_identity = state.prepared_identity

    def close_request(self, request_id: str) -> None:
        key = self.request_sessions.pop(request_id, None)
        if key is None:
            return
        state = self.sessions.get(key)
        if state is None:
            return
        state.request_ids.discard(request_id)
        if not state.request_ids:
            self.close_session(*key)

    def close_session(self, session_id: str, incarnation: int) -> None:
        key = (session_id, incarnation)
        state = self.sessions.pop(key, None)
        if state is None:
            return
        for request_id in state.request_ids:
            self.request_sessions.pop(request_id, None)
        if state.codec is not None:
            state.codec.reset_streaming()
            self._free_codecs.append(state.codec)
            state.codec = None

    def _acquire_codec(self):
        if self._free_codecs:
            return self._free_codecs.pop()
        if self._codec_factory is not None:
            codec = self._codec_factory()
            codec.streaming_init(1)
            return codec
        from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
            PersonaPlexMimiCodec,
        )

        checkpoint = Path(self.model_path) / "tokenizer-e351c8d8-checkpoint125.safetensors"
        codec = PersonaPlexMimiCodec(
            checkpoint=str(checkpoint) if checkpoint.is_file() else None,
            device=self.device,
        )
        codec.streaming_init(1)
        return codec

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_personaplex_tokenizer(self.model_path)
        return self._tokenizer

    def _load_voice(self, voice: str) -> dict[str, Any]:
        if self._voice_loader is not None:
            state = self._voice_loader(voice)
        else:
            state = load_personaplex_voice_state(self.model_path, voice)
        if not isinstance(state, dict):
            raise ValueError(f"PersonaPlex voice prompt {voice!r} is not a mapping")
        return state

    def _model_device_dtype(self):
        import torch

        try:
            parameter = next(self.stage_model.parameters())
            return parameter.device, parameter.dtype
        except Exception:
            device = getattr(self.stage_model, "device", torch.device(self.device))
            dtype = getattr(self.stage_model, "dtype", torch.float32)
            return torch.device(device), dtype

    @staticmethod
    def _decode_pcm(payload: object) -> np.ndarray:
        if not isinstance(payload, dict):
            raise ValueError("PersonaPlex duplex payload must be a mapping")
        if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != 24000:
            raise ValueError("PersonaPlex Stage 0 requires 24 kHz pcm_f32le")
        audio = payload.get("audio")
        if not isinstance(audio, str):
            raise ValueError("PersonaPlex Stage 0 requires base64 audio")
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("PersonaPlex Stage 0 audio is not valid base64") from exc
        samples = np.frombuffer(raw, dtype="<f4")
        if samples.size != _FRAME_SAMPLES:
            raise ValueError(f"PersonaPlex Stage 0 requires {_FRAME_SAMPLES} samples per append")
        if not np.isfinite(samples).all():
            raise ValueError("PersonaPlex Stage 0 samples must be finite")
        return np.ascontiguousarray(samples, dtype=np.float32).copy()


def _coerce_non_negative_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PersonaPlex duplex {name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"PersonaPlex duplex {name} must be non-negative")
    return result


def _coerce_positive_int(value: object, name: str) -> int:
    result = _coerce_non_negative_int(value, name)
    if result <= 0:
        raise ValueError(f"PersonaPlex duplex {name} must be positive")
    return result


__all__ = [
    "PersonaPlexStage0DuplexRuntime",
    "PersonaPlexStage0PreparedAppend",
    "PersonaPlexStage0SessionState",
    "load_personaplex_tokenizer",
    "load_personaplex_voice_state",
    "personaplex_prefill_slots",
]
