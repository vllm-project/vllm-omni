# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import io
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.model_executor.common.audio.pcm import pcm_f32le_samples
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.models.personaplex.duplex.config import FRAME_SIZE, SAMPLE_RATE
from vllm_omni.model_executor.models.personaplex.duplex.policy import (
    AUDIO_SILENCE_FRAME_CNT,
    SILENCE_TOKENS,
    SINE_TOKENS,
    ZERO_TEXT_TOKEN,
    wrap_with_system_tags,
)

if TYPE_CHECKING:
    import torch

_FRAME_SAMPLES = FRAME_SIZE

logger = init_logger(__name__)


class PersonaPlexStage0CapacityError(RuntimeError):
    """Every streaming encoder row is leased by a live session."""


class PersonaPlexStage0StaleEpochError(RuntimeError):
    """The append belongs to an epoch that a newer epoch of the same session has superseded."""


@dataclass(slots=True)
class PersonaPlexStage0PreparedAppend:
    input_ids: Any
    inputs_embeds: Any
    user_codes: Any
    info_update: dict[str, Any]
    prefill_applied: bool
    prompt_offset: int
    # Device copies of the depformer teacher-forcing state (batched path only).
    depformer_audio_tokens_device: torch.Tensor | None = None
    depformer_audio_provided_device: torch.Tensor | None = None


@dataclass(slots=True)
class PersonaPlexStage0SessionState:
    """Lockstep state of one (session, epoch): a new epoch is a new Stage 0 request with fresh KV."""

    session_id: str
    epoch: int
    user_codes: Any | None = None
    last_text_token: Any | None = None
    last_agent_codes: Any | None = None
    prefill_slots: int = 0
    prepared_identity: tuple[int, int] | None = None
    sampled_identity: tuple[int, int] | None = None
    prepared: PersonaPlexStage0PreparedAppend | None = None
    last_seq: int = 0
    request_ids: set[str] = field(default_factory=set)
    slot: int | None = None
    encoded_identity: tuple[int, int] | None = None
    encoded_frame: Any | None = None
    # Device copies of the encoded frame and of the last two consumed user
    # frames (newest last), so live appends are built without host round trips.
    encoded_frame_device: torch.Tensor | None = None
    user_history_device: list[torch.Tensor] = field(default_factory=list)


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


@lru_cache(maxsize=4)
def _cached_tokenizer(model_path: str):
    return load_personaplex_tokenizer(model_path)


@lru_cache(maxsize=16)
def _cached_voice_embedding_rows(model_path: str, voice: str) -> int:
    state = load_personaplex_voice_state(model_path, voice)
    embeddings = state.get("embeddings")
    if not hasattr(embeddings, "shape") or len(embeddings.shape) < 1:
        raise ValueError(f"PersonaPlex voice prompt {voice!r} has no embeddings")
    return int(embeddings.shape[0])


def personaplex_prefill_slots(model_path: str, voice: str, persona: str) -> int:
    """Scheduler slots the first append of a session needs for the voice + persona prefill.

    The voice bundle row count and the tokenizer are cached per model path:
    they are constant, and this runs on every session open.
    """
    voice_rows = _cached_voice_embedding_rows(model_path, voice)
    persona_tokens = _cached_tokenizer(model_path)(wrap_with_system_tags(persona)) if persona else []
    return voice_rows + 2 * AUDIO_SILENCE_FRAME_CNT + len(persona_tokens)


class PersonaPlexStage0DuplexRuntime:
    """Own the shared streaming Mimi encoder and each session's first-append prefill.

    One encoder holds ``max_sessions`` streaming rows; a live ``(session, epoch)``
    leases one row for its lifetime. ``encode_appends`` encodes every new append
    of a scheduler step in one batched call (rows without a new append are
    inactive and keep their state), and ``prepare_append`` consumes the result.
    """

    def __init__(
        self,
        stage_model: Any,
        *,
        model_path: str,
        device: str,
        codec: Any | None = None,
        codec_factory: Callable[[], Any] | None = None,
        max_sessions: int = 1,
        codec_cuda_graphs: bool = False,
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
        self._codec_cuda_graphs = codec_cuda_graphs
        self._codec: Any | None = None
        self._free_slots: list[int] = list(reversed(range(max_sessions)))
        self._tokenizer = tokenizer
        self._voice_loader = voice_loader
        self.sessions: dict[tuple[str, int], PersonaPlexStage0SessionState] = {}
        self.request_sessions: dict[str, tuple[str, int]] = {}
        # Requests of a superseded epoch that still reached this step; they are
        # finished by the engine and must not lease a row or record a sample.
        self._stale_requests: set[str] = set()
        self._constants_cache: dict[str, torch.Tensor] | None = None
        if codec is not None:
            codec.streaming_init(max_sessions)
            self._codec = codec

    def encode_appends(self, appends: list[dict[str, Any]]) -> None:
        """Encode the new user frame of each append in one batched encoder call.

        Called once per scheduler step before the per-request ``prepare_append``
        calls. An append whose ``(epoch, seq)`` is already encoded or prepared
        (a chunked first prefill spans several steps) is skipped, so a row's
        streaming state advances exactly once per frame. Appends that cannot be
        admitted are left for ``prepare_append`` to reject.
        """
        parsed: list[tuple[str, int, int, dict[str, Any]]] = []
        for duplex in appends:
            try:
                parsed.append((*_append_identity(duplex), duplex))
            except ValueError:
                continue
        # A cancel can put a session's aborted epoch and its restarted epoch in
        # one step. Only the newest epoch is live: admitting it closes the old
        # one, so the old append must neither be encoded nor re-leased.
        newest: dict[str, int] = {}
        for session_id, epoch, _, _ in parsed:
            newest[session_id] = max(epoch, newest.get(session_id, epoch))
        for session_id, epoch in self.sessions:
            if session_id in newest:
                newest[session_id] = max(epoch, newest[session_id])
        rows: list[tuple[PersonaPlexStage0SessionState, tuple[int, int], np.ndarray]] = []
        for session_id, epoch, seq, duplex in parsed:
            if epoch != newest[session_id]:
                continue
            try:
                state = self._session_state(session_id, epoch)
            except PersonaPlexStage0CapacityError:
                # Left for prepare_append to reject. Any other error (e.g. the
                # shared codec failing to initialize) propagates immediately.
                continue
            identity = (epoch, seq)
            if identity in (state.prepared_identity, state.encoded_identity) or seq <= state.last_seq:
                continue
            if any(row[0] is state for row in rows):
                continue
            rows.append((state, identity, self._decode_pcm(duplex.get("payload"))))
        if rows:
            self._encode_rows(rows)

    def prepare_append(
        self,
        duplex: dict[str, Any],
        *,
        prompt_len: int,
        request_id: str | None = None,
    ) -> PersonaPlexStage0PreparedAppend:
        import torch

        session_id, epoch, seq = _append_identity(duplex)
        identity = (epoch, seq)
        key = (session_id, epoch)
        try:
            state = self._session_state(session_id, epoch)
        except PersonaPlexStage0StaleEpochError:
            if request_id:
                self._stale_requests.add(request_id)
            raise
        if request_id:
            state.request_ids.add(request_id)
            self.request_sessions[request_id] = key
        if state.prepared_identity == identity and state.prepared is not None:
            prepared = state.prepared
            if prepared.prompt_offset < 0:
                prepared.prompt_offset = int(prompt_len) - int(prepared.inputs_embeds.shape[0])
                if prepared.prompt_offset < 0:
                    raise ValueError(
                        "PersonaPlex scheduler prompt reservation mismatch: "
                        f"reserved={prompt_len}, prepared={prepared.inputs_embeds.shape[0]}"
                    )
            return prepared
        if seq <= state.last_seq:
            raise ValueError(f"PersonaPlex duplex append seq must increase: last={state.last_seq}, got={seq}")

        if state.encoded_identity != identity:
            self._encode_rows([(state, identity, self._decode_pcm(duplex.get("payload")))])
        user_frame = state.encoded_frame
        self._push_user_history(state)
        state.encoded_identity = None
        state.encoded_frame = None
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

    @staticmethod
    def _push_user_history(state: PersonaPlexStage0SessionState) -> None:
        """Move the consumed encoded frame into the device history (last two frames)."""
        if state.encoded_frame_device is None:
            raise RuntimeError("PersonaPlex Stage 0 append has no encoded user frame")
        state.user_history_device.append(state.encoded_frame_device)
        del state.user_history_device[:-2]
        state.encoded_frame_device = None

    def _constants(self) -> dict[str, torch.Tensor]:
        """Host and device copies of the live-append constant frames, built once."""
        if self._constants_cache is not None:
            return self._constants_cache
        import torch

        device, _ = self._model_device_dtype()
        silence = torch.tensor(SILENCE_TOKENS, dtype=torch.long)
        sine = torch.tensor(SINE_TOKENS, dtype=torch.long)
        live_provided = torch.tensor([False] * 8 + [True] * 8, dtype=torch.bool)
        self._constants_cache = {
            "silence": silence,
            "sine": sine,
            "live_provided": live_provided,
            "silence_device": silence.to(device),
            "sine_device": sine.to(device),
            "live_provided_device": live_provided.to(device),
            "zero_text_device": torch.tensor([ZERO_TEXT_TOKEN], dtype=torch.long, device=device),
        }
        return self._constants_cache

    def prepare_live_appends(self, appends: list[dict[str, Any]]) -> int:
        """Prepare every one-frame live append of this step with one batched embedding call.

        Only appends whose frame was encoded this step and that are not the
        first append of their epoch (which carries the voice/persona prefill)
        take this path; the rest are left to :meth:`prepare_append`. Results are
        stored exactly as :meth:`prepare_append` stores them, so its per-request
        call returns the cached preparation. Returns the number prepared.
        """
        import torch

        batch: dict[tuple[str, int], tuple[PersonaPlexStage0SessionState, int]] = {}
        for duplex in appends:
            try:
                session_id, epoch, seq = _append_identity(duplex)
            except ValueError:
                continue
            key = (session_id, epoch)
            state = self.sessions.get(key)
            # Only an append encoded in this step and not yet consumed: its seq
            # is past last_seq, so it has not been prepared either.
            if state is None or state.last_seq == 0 or seq <= state.last_seq or key in batch:
                continue
            if state.encoded_identity != (epoch, seq):
                continue
            batch[key] = (state, seq)
        if not batch:
            return 0
        c = self._constants()
        device, dtype = self._model_device_dtype()
        sine_d, silence_d = c["sine_device"], c["silence_device"]
        texts, agents, d0s, d1s, currents = [], [], [], [], []
        for state, _ in batch.values():
            history = state.user_history_device  # consumed frames, newest last
            texts.append(
                state.last_text_token.reshape(1) if state.last_text_token is not None else c["zero_text_device"]
            )
            agents.append(state.last_agent_codes[:8] if state.last_agent_codes is not None else silence_d)
            # The frame being appended is not in the history yet: user cb0 reads
            # the previous frame and user cb1..7 the one before it.
            d0s.append(history[-1] if len(history) >= 1 else sine_d)
            d1s.append(history[-2] if len(history) >= 2 else sine_d)
            currents.append(state.encoded_frame_device)
        n = len(batch)
        text = torch.stack(texts)
        agent = torch.stack(agents)
        user_d0 = torch.stack(d0s)
        user_d1 = torch.stack(d1s)
        current = torch.stack(currents)
        stack = torch.cat([text, agent, user_d0[:, :1], user_d1[:, 1:8]], dim=1).unsqueeze(-1)
        embeds = self.stage_model.input_embeddings(stack).reshape(n, 1, -1).to(dtype=dtype)
        dep_tokens_device = torch.cat([silence_d.expand(n, 8), current[:, :1], user_d0[:, 1:8]], dim=1)
        input_ids = torch.zeros((n, 1), dtype=torch.long, device=device)
        for row, (state, seq) in enumerate(batch.values()):
            user_frame = state.encoded_frame
            self._push_user_history(state)
            state.encoded_identity = None
            state.encoded_frame = None
            state.user_codes = torch.cat([state.user_codes, user_frame], dim=0)
            user_d0_host = state.user_codes[-2] if state.user_codes.shape[0] > 1 else c["sine"]
            dep_tokens = torch.cat([c["silence"], user_frame[0, :1], user_d0_host[1:8]])
            info_update = {
                "pplex_user_codes": state.user_codes,
                "pplex_silence_codes": c["silence"],
                "pplex_depformer_audio_tokens": dep_tokens,
                "pplex_depformer_audio_provided": c["live_provided"],
                "meta": {
                    "pplex_frame": state.prefill_slots + int(state.user_codes.shape[0]),
                    "pplex_prefill_len": state.prefill_slots,
                },
                "duplex": {
                    "stage0_prepared": True,
                    "prefill_applied": False,
                    "session_id": state.session_id,
                    "epoch": state.epoch,
                    "seq": seq,
                },
            }
            state.prepared = PersonaPlexStage0PreparedAppend(
                input_ids=input_ids[row],
                inputs_embeds=embeds[row],
                user_codes=state.user_codes,
                info_update=info_update,
                prefill_applied=False,
                # Resolved by prepare_append from the scheduler's prompt length.
                prompt_offset=-1,
                depformer_audio_tokens_device=dep_tokens_device[row],
                depformer_audio_provided_device=c["live_provided_device"],
            )
            state.prepared_identity = (state.epoch, seq)
            state.last_seq = seq
        return n

    def prepared_depformer_state(self, request_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Device teacher-forcing tensors of the request's current append, when prepared on the device."""
        key = self.request_sessions.get(request_id)
        state = self.sessions.get(key) if key is not None else None
        prepared = state.prepared if state is not None else None
        if prepared is None or prepared.depformer_audio_tokens_device is None:
            return None
        return prepared.depformer_audio_tokens_device, prepared.depformer_audio_provided_device

    def record_sample(
        self,
        *,
        request_id: str,
        text_token: Any,
        effective_codes: Any,
    ) -> None:
        """Commit one sampled temporal frame for the next live append.

        ``effective_codes`` are the frame's eight agent codes with the append's
        depformer teacher forcing already applied, which the next append reads.
        """
        import torch

        if request_id in self._stale_requests:
            return
        key = self.request_sessions.get(request_id)
        if key is None:
            raise KeyError(f"PersonaPlex Stage 0 request is not attached to a live session: {request_id}")
        state = self.sessions.get(key)
        if state is None or state.prepared_identity is None:
            raise RuntimeError(f"PersonaPlex Stage 0 request has no prepared append: {request_id}")
        if state.sampled_identity == state.prepared_identity:
            return

        text = torch.as_tensor(text_token, device=self._model_device_dtype()[0], dtype=torch.long).reshape(-1)
        codes = torch.as_tensor(effective_codes, device=text.device, dtype=torch.long).reshape(-1)
        if text.numel() != 1:
            raise ValueError(f"PersonaPlex Stage 0 expected one sampled text token, got {text.numel()}")
        if codes.numel() != 8:
            raise ValueError(f"PersonaPlex Stage 0 expected 8 effective agent codes, got {codes.numel()}")
        state.last_agent_codes = codes.detach()
        state.last_text_token = text.detach()
        state.sampled_identity = state.prepared_identity

    def close_request(self, request_id: str) -> None:
        self._stale_requests.discard(request_id)
        key = self.request_sessions.pop(request_id, None)
        if key is None:
            return
        state = self.sessions.get(key)
        if state is None:
            return
        state.request_ids.discard(request_id)
        if not state.request_ids:
            self.close_session(*key)

    def close_session(self, session_id: str, epoch: int) -> None:
        key = (session_id, epoch)
        state = self.sessions.pop(key, None)
        if state is None:
            return
        for request_id in state.request_ids:
            self.request_sessions.pop(request_id, None)
        if state.slot is not None:
            self._shared_codec().reset_slot(state.slot)
            self._free_slots.append(state.slot)
            state.slot = None

    def _session_state(self, session_id: str, epoch: int) -> PersonaPlexStage0SessionState:
        key = (session_id, epoch)
        state = self.sessions.get(key)
        if state is not None:
            return state
        # A newer epoch supersedes the session's earlier lockstep state: the
        # engine aborted that request, but its finish notification may still
        # be in flight, so release it here rather than let the two epochs share
        # the encoder budget.
        # The reverse also holds: once a newer epoch is live, an append of an
        # older epoch is a leftover of the aborted request and must not re-lease
        # a row (at capacity it would fail the whole step).
        newer = [k[1] for k in self.sessions if k[0] == session_id and k[1] > epoch]
        if newer:
            raise PersonaPlexStage0StaleEpochError(
                f"PersonaPlex Stage 0 epoch {epoch} of session {session_id} is superseded by epoch {max(newer)}"
            )
        for stale_key in [k for k in self.sessions if k[0] == session_id and k[1] < epoch]:
            self.close_session(*stale_key)
        if len(self.sessions) >= self.max_sessions or not self._free_slots:
            raise PersonaPlexStage0CapacityError(
                f"PersonaPlex Stage 0 session capacity {self.max_sessions} is exhausted"
            )
        self._shared_codec()
        state = PersonaPlexStage0SessionState(session_id=session_id, epoch=epoch, slot=self._free_slots.pop())
        self.sessions[key] = state
        return state

    def _encode_rows(self, rows: list[tuple[PersonaPlexStage0SessionState, tuple[int, int], np.ndarray]]) -> None:
        import torch

        codec = self._shared_codec()
        pcm = torch.zeros((self.max_sessions, _FRAME_SAMPLES), dtype=torch.float32)
        active = torch.zeros((self.max_sessions,), dtype=torch.bool)
        for state, _, samples in rows:
            assert state.slot is not None
            pcm[state.slot] = torch.from_numpy(samples)
            active[state.slot] = True
        encoded = codec.encode_frame(pcm, active)
        codes_device = encoded.detach()[:, :8].to(dtype=torch.long)
        # One device-to-host copy per step, whatever the number of sessions.
        codes = encoded.detach().to(dtype=torch.long, device="cpu")
        if codes.shape[-1] < 8:
            raise RuntimeError(f"PersonaPlex Mimi encoder returned {codes.shape[-1]} codebooks, expected at least 8")
        for state, identity, _ in rows:
            state.encoded_frame = codes[state.slot : state.slot + 1, :8].clone()
            state.encoded_frame_device = codes_device[state.slot]
            state.encoded_identity = identity

    def _shared_codec(self):
        if self._codec is not None:
            return self._codec
        if self._codec_factory is not None:
            codec = self._codec_factory()
        else:
            from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
                PersonaPlexMimiCodec,
            )

            checkpoint = Path(self.model_path) / "tokenizer-e351c8d8-checkpoint125.safetensors"
            codec = PersonaPlexMimiCodec(
                checkpoint=str(checkpoint) if checkpoint.is_file() else None,
                device=self.device,
            )
        codec.streaming_init(self.max_sessions)
        if self._codec_cuda_graphs:
            # Rows are leased and recycled in place, so one graph at
            # max_sessions with the active mask as input serves every step.
            captured = codec.capture_cuda_graphs(encode=True, decode_frame_counts=())
            logger.info(
                "PersonaPlex Stage 0 Mimi encoder %s at %d rows",
                "replays a CUDA graph" if captured else "runs eagerly",
                self.max_sessions,
            )
        self._codec = codec
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
        raw = decode_pcm_f32le_payload(
            payload,
            sample_rate_hz=SAMPLE_RATE,
            exact_samples=_FRAME_SAMPLES,
            model="PersonaPlex Stage 0",
        )
        return pcm_f32le_samples(raw)


def _append_identity(duplex: dict[str, Any]) -> tuple[str, int, int]:
    session_id = duplex.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("PersonaPlex duplex append requires session_id")
    epoch = _coerce_non_negative_int(duplex.get("epoch"), "epoch")
    seq = _coerce_positive_int(duplex.get("seq"), "seq")
    return session_id, epoch, seq


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
