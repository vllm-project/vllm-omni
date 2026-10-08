# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import io
import tarfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.model_executor.common.audio.pcm import check_pcm_f32le_finite
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.models.personaplex.duplex.config import (
    DEFAULT_PERSONA,
    DEFAULT_VOICE,
    FRAME_SIZE,
    SAMPLE_RATE,
)
from vllm_omni.model_executor.models.personaplex.duplex.policy import (
    AUDIO_SILENCE_FRAME_CNT,
    SILENCE_TOKENS,
    SINE_TOKENS,
    ZERO_TEXT_TOKEN,
    wrap_with_system_tags,
)

_FRAME_SAMPLES = FRAME_SIZE
# Per-runtime LRU sizes of the first-append caches: device voice embeddings per
# voice, and complete voice + persona prefill embeddings per (voice, persona).
_VOICE_EMBEDDINGS_CACHE_SIZE = 16
_PREFILL_EMBEDS_CACHE_SIZE = 64


class PersonaPlexStage0CapacityError(RuntimeError):
    """Every streaming encoder row is leased by a live session."""


class PersonaPlexStage0StaleEpochError(RuntimeError):
    """The append belongs to an epoch that a newer epoch of the same session has superseded."""


@dataclass(slots=True)
class PersonaPlexStage0PreparedAppend:
    input_ids: Any
    inputs_embeds: Any
    user_frame: Any
    info_update: dict[str, Any]
    prefill_applied: bool
    prompt_offset: int


@dataclass(slots=True)
class PersonaPlexStage0SessionState:
    """Lockstep state of one (session, epoch): a new epoch is a new Stage 0 request with fresh KV.

    Only host bookkeeping: the frame-to-frame tensors live in the runtime's
    per-slot device buffers, indexed by ``slot``.
    """

    session_id: str
    epoch: int
    user_frames: int = 0
    prefill_slots: int = 0
    prepared_identity: tuple[int, int] | None = None
    sampled_identity: tuple[int, int] | None = None
    last_seq: int = 0
    request_ids: set[str] = field(default_factory=set)
    slot: int | None = None
    encoded_identity: tuple[int, int] | None = None
    encoded_frame: tuple[Any, int] | None = None
    live_identity: tuple[int, int] | None = None
    live_embed: tuple[Any, int] | None = None
    live_prepared: tuple[Any, ...] | None = None
    _prepared: PersonaPlexStage0PreparedAppend | None = None

    @property
    def prepared(self) -> PersonaPlexStage0PreparedAppend | None:
        if self.live_prepared is not None:
            input_ids, embed, frame, prompt_offset, info = self.live_prepared
            self._prepared = PersonaPlexStage0PreparedAppend(
                input_ids=input_ids,
                inputs_embeds=_row(embed),
                user_frame=_row(frame),
                info_update=_info_update(*info, first_append=False),
                prefill_applied=False,
                prompt_offset=prompt_offset,
            )
            self.live_prepared = None
        return self._prepared

    @prepared.setter
    def prepared(self, prepared: PersonaPlexStage0PreparedAppend | None) -> None:
        self._prepared = prepared
        self.live_prepared = None


def _row(batch: tuple[Any, int]):
    """The ``[1, ...]`` row view of a ``(batch, row)`` pair."""
    tensor, row = batch
    return tensor[row : row + 1]


def _info_update(
    silence_cpu: Any,
    frame: int,
    prefill_len: int,
    session_id: str,
    epoch: int,
    seq: int,
    *,
    first_append: bool,
) -> dict[str, Any]:
    # No device tensor: the runner would move each one to the host, one
    # request at a time.
    return {
        "pplex_silence_codes": silence_cpu,
        "meta": {"pplex_frame": frame, "pplex_prefill_len": prefill_len},
        "duplex": {
            "stage0_prepared": True,
            "prefill_applied": first_append,
            "session_id": session_id,
            "epoch": epoch,
            "seq": seq,
        },
    }


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
        data = _voice_archive_members(str(archive)).get(voice)
        if data is not None:
            state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                return state
            raise ValueError(f"PersonaPlex voice bundle {voice!r} is not a mapping")
    raise FileNotFoundError(f"PersonaPlex bundled voice prompt {voice!r} was not found under {model_path!r}")


@lru_cache(maxsize=2)
def _voice_archive_members(archive: str) -> dict[str, bytes]:
    """Every file of a ``voices.tgz`` bundle by base name, read in one pass.

    A gzip stream has no random access, so each lookup would decompress the
    archive from the start; the bundle is small (about 8 MB unpacked), so read it
    once and keep the members. The first file with a given base name wins.
    """
    members: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for item in tar:
            if not item.isfile():
                continue
            extracted = tar.extractfile(item)
            if extracted is not None:
                members.setdefault(Path(item.name).name, extracted.read())
    return members


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
    inactive and keep their state) and builds their frame inputs in one batched
    pass, and ``prepare_append`` consumes the result. The state a row carries to
    its next frame stays in device buffers indexed by its slot.
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
        tokenizer=None,
        voice_loader=None,
        codec_cuda_graphs: bool = False,
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
        self._voice_embeddings_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._prefill_embeds_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self.sessions: dict[tuple[str, int], PersonaPlexStage0SessionState] = {}
        self.request_sessions: dict[str, tuple[str, int]] = {}
        # Requests of a superseded epoch that still reached this step; they are
        # finished by the engine and must not lease a row or record a sample.
        self._stale_requests: set[str] = set()
        # Per-slot device state, allocated on first use (see _slot_buffers).
        self._slot_device = self._last_text = self._last_agent = self._user_history = None
        self._teacher_tokens = self._teacher_provided = self._silence = self._sine = None
        self._live_provided = self._silence_cpu = self._zero_ids = self._live_embeds = None
        if codec is not None:
            codec.streaming_init(max_sessions, decode=False)
            self._codec = codec

    def encode_appends(self, appends: list[dict[str, Any]]) -> None:
        """Encode the new user frame of each append and build its frame inputs, batched.

        Called once per scheduler step before the per-request ``prepare_append``
        calls: one encoder call for every new frame of the step, then one
        embedding pass for every row still to be prepared. An append whose
        ``(epoch, seq)`` is already encoded or prepared (a chunked first prefill
        spans several steps) is not encoded again, so a row's streaming state
        advances exactly once per frame. Appends that cannot be admitted are
        left for ``prepare_append`` to reject.
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
        rows: list[tuple[PersonaPlexStage0SessionState, tuple[int, int]]] = []
        seen: set[int] = set()
        new_frames: list[tuple[PersonaPlexStage0SessionState, tuple[int, int]]] = []
        payloads: list[object] = []
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
            if seq <= state.last_seq:
                continue
            if id(state) in seen:
                continue
            seen.add(id(state))
            rows.append((state, identity))
            if identity != state.encoded_identity:
                new_frames.append((state, identity))
                payloads.append(duplex.get("payload"))
        if new_frames:
            self._encode_rows(new_frames, self._decode_pcm_rows(payloads))
        unbuilt = [row for row in rows if row[0].live_identity != row[1]]
        if unbuilt:
            self._build_live_rows(unbuilt)

    def prepare_live_appends(
        self,
        appends: list[tuple[str, dict[str, Any], int]],
    ) -> tuple[list[int], Any]:
        """``prepare_append`` for the live one-frame appends of a step, at once.

        ``appends`` holds ``(request_id, duplex, prompt_len)`` for requests whose
        one scheduled token is the last prompt slot. An append is handled here
        when its session already has its prefill and its frame was encoded and
        built in the latest ``encode_appends`` batch; everything else is left
        for ``prepare_append`` (which also raises the errors).

        Returns the handled positions in ``appends`` and their ``[rows, hidden]``
        frame embeddings, in that order. The session state ends as
        ``prepare_append`` leaves it, except that ``prepared`` is only built when
        read. No tensor op runs per append.
        """
        handled: list[int] = []
        batch_rows: list[int] = []
        batch = self._live_embeds
        dtype = input_ids = None
        for position, (request_id, duplex, prompt_len) in enumerate(appends):
            try:
                session_id, epoch, seq = _append_identity(duplex)
            except ValueError:
                continue
            key = (session_id, epoch)
            state = self.sessions.get(key)
            identity = (epoch, seq)
            if (
                state is None
                or request_id in self._stale_requests
                or state.last_seq == 0
                or seq <= state.last_seq
                or state.encoded_identity != identity
                or state.live_identity != identity
                or state.live_embed is None
                or state.live_embed[0] is not batch
                or prompt_len < 1
            ):
                continue
            if dtype is None:
                dtype = self._model_device_dtype()[1]
                input_ids = self._zero_input_ids(1)
            state.request_ids.add(request_id)
            self.request_sessions[request_id] = key
            batch_rows.append(state.live_embed[1])
            frame = state.prefill_slots + state.user_frames
            info = (self._silence_cpu, frame, state.prefill_slots, session_id, epoch, seq)
            state.prepared = None
            state.live_prepared = (input_ids, state.live_embed, state.encoded_frame, int(prompt_len) - 1, info)
            state.encoded_identity = state.encoded_frame = state.live_identity = state.live_embed = None
            state.prepared_identity = identity
            state.last_seq = seq
            handled.append(position)
        if not handled:
            return handled, None
        embeds = batch
        if batch_rows != list(range(int(embeds.shape[0]))):
            embeds = embeds.index_select(0, self._index(batch_rows))
        return handled, embeds.to(dtype=dtype)

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
            return state.prepared
        if seq <= state.last_seq:
            raise ValueError(f"PersonaPlex duplex append seq must increase: last={state.last_seq}, got={seq}")

        if state.encoded_identity != identity:
            self._encode_rows([(state, identity)], self._decode_pcm_rows([duplex.get("payload")]))
        if state.live_identity != identity:
            self._build_live_rows([(state, identity)])
        live_embed = _row(state.live_embed)
        user_frame = _row(state.encoded_frame)
        state.encoded_identity = state.encoded_frame = state.live_identity = state.live_embed = None

        runtime_config = duplex.get("runtime_config")
        runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}
        first_append = state.last_seq == 0
        device, dtype = self._model_device_dtype()
        if first_append:
            voice = runtime_config.get("personaplex_voice_prompt", DEFAULT_VOICE)
            persona = runtime_config.get("personaplex_persona", "")
            if not isinstance(voice, str) or not voice:
                raise ValueError("PersonaPlex runtime voice prompt is invalid")
            if not isinstance(persona, str):
                raise ValueError("PersonaPlex runtime persona is invalid")
            prefill_embeds = self._prefill_embeds(voice, persona, device, dtype)
            state.prefill_slots = int(prefill_embeds.shape[0])
            # The cached prefill is shared: cat copies it into this append's rows.
            full_embeds = torch.cat([prefill_embeds, live_embed.to(dtype=dtype)], dim=0)
        else:
            full_embeds = live_embed.to(dtype=dtype)

        prepared_len = int(full_embeds.shape[0])
        prompt_offset = int(prompt_len) - prepared_len
        if prompt_offset < 0:
            raise ValueError(
                f"PersonaPlex scheduler prompt reservation mismatch: reserved={prompt_len}, prepared={prepared_len}"
            )
        info = (self._silence_cpu, state.prefill_slots + state.user_frames, state.prefill_slots, session_id, epoch, seq)
        prepared = PersonaPlexStage0PreparedAppend(
            input_ids=self._zero_input_ids(prepared_len),
            inputs_embeds=full_embeds,
            user_frame=user_frame,
            info_update=_info_update(*info, first_append=first_append),
            prefill_applied=first_append,
            prompt_offset=prompt_offset,
        )
        state.prepared_identity = identity
        state.prepared = prepared
        state.last_seq = seq
        return prepared

    def _zero_input_ids(self, length: int):
        """``[length]`` zero placeholder ids: a view of one shared buffer, never written.

        The talker runs on the prepared embeddings, so every append can read the
        same zeros instead of filling its own tensor on the device.
        """
        import torch

        zeros = self._zero_ids
        if zeros is None or zeros.shape[0] < length:
            device = self._model_device_dtype()[0]
            size = max(int(length), 2 * int(zeros.shape[0]) if zeros is not None else 64)
            zeros = torch.zeros((size,), dtype=torch.long, device=device)
            self._zero_ids = zeros
        return zeros[:length]

    def depformer_teacher_forcing(self, request_ids: list[str]) -> tuple[Any, Any]:
        """``[B, 16]`` depformer teacher-forcing tokens and mask for the prepared frames.

        One gather from the slot buffers. A request of a superseded epoch gets
        the neutral row (silence, nothing forced): its output is discarded.
        """
        slots = [
            self.scratch_slot if request_id in self._stale_requests else self._prepared_state(request_id).slot
            for request_id in request_ids
        ]
        return self.teacher_forcing_rows(self._index(slots))

    @property
    def scratch_slot(self) -> int:
        """The slot-buffer row past the live slots.

        Its teacher forcing is neutral and never written; it takes the commits
        that must not reach a live slot.
        """
        return self.max_sessions

    def depformer_rows(self, request_ids: list[str]) -> tuple[list[int], list[int]]:
        """The slots one post-sample step reads and writes, one pair per row.

        Row ``i`` reads its teacher forcing from ``read[i]`` and commits its sample
        to ``write[i]``. A superseded epoch's request, a session's repeated row and
        an already committed frame write to the scratch row, so ``write`` names
        each live slot at most once. Committing frames are marked sampled, as
        ``record_samples`` does.
        """
        scratch = self.scratch_slot
        read: list[int] = []
        write: list[int] = []
        committing: dict[int, PersonaPlexStage0SessionState] = {}
        for request_id in request_ids:
            if request_id in self._stale_requests:
                read.append(scratch)
                write.append(scratch)
                continue
            state = self._prepared_state(request_id)
            read.append(state.slot)
            if state.sampled_identity == state.prepared_identity or id(state) in committing:
                write.append(scratch)
            else:
                write.append(state.slot)
                committing[id(state)] = state
        for state in committing.values():
            state.sampled_identity = state.prepared_identity
        return read, write

    def teacher_forcing_rows(self, slots: Any) -> tuple[Any, Any]:
        """``[B, 16]`` depformer teacher-forcing tokens and mask of the ``slots`` rows."""
        return self._teacher_tokens[slots], self._teacher_provided[slots]

    def commit_rows(self, slots: Any, text_tokens: Any, agent_codes: Any, tokens: Any, provided: Any) -> None:
        """Write each row's effective agent frame (the forced code where ``provided``) and text token.

        ``slots`` must not repeat a live slot. Device-only, so it can run inside a
        CUDA graph.
        """
        import torch

        self._last_agent[slots] = torch.where(provided[:, :8], tokens[:, :8], agent_codes[:, :8])
        self._last_text[slots] = text_tokens

    def _prepared_state(self, request_id: str) -> PersonaPlexStage0SessionState:
        key = self.request_sessions.get(request_id)
        state = self.sessions.get(key) if key is not None else None
        if state is None or state.prepared_identity is None:
            raise ValueError("PersonaPlex duplex depformer teacher-forcing state is missing")
        return state

    def record_samples(
        self,
        *,
        request_ids: list[str],
        text_tokens: Any,
        agent_codes: Any,
    ) -> None:
        """Commit the sampled temporal frame of every row for its next live append.

        ``text_tokens`` is ``[B]`` (or ``[B, 1]``) and ``agent_codes`` ``[B, >=8]``,
        row ``i`` belonging to ``request_ids[i]``; one batched write.
        """
        import torch

        rows: list[int] = []
        states: list[PersonaPlexStage0SessionState] = []
        seen: set[int] = set()
        for row, request_id in enumerate(request_ids):
            if request_id in self._stale_requests:
                continue
            key = self.request_sessions.get(request_id)
            if key is None:
                raise KeyError(f"PersonaPlex Stage 0 request is not attached to a live session: {request_id}")
            state = self.sessions.get(key)
            if state is None or state.prepared_identity is None:
                raise RuntimeError(f"PersonaPlex Stage 0 request has no prepared append: {request_id}")
            if state.sampled_identity == state.prepared_identity or id(state) in seen:
                continue
            seen.add(id(state))
            rows.append(row)
            states.append(state)
        if not rows:
            return

        device = self._slot_buffers()
        text = torch.as_tensor(text_tokens, dtype=torch.long).to(device).reshape(len(request_ids), -1)
        codes = torch.as_tensor(agent_codes, dtype=torch.long).to(device).reshape(len(request_ids), -1)
        if text.shape[1] != 1:
            raise ValueError(f"PersonaPlex Stage 0 expected one sampled text token, got {text.shape[1]}")
        if codes.shape[1] < 8:
            raise ValueError(f"PersonaPlex Stage 0 expected at least 8 agent codes, got {codes.shape[1]}")

        if len(rows) != len(request_ids):
            row_index = self._index(rows)
            text, codes = text[row_index], codes[row_index]
        slots = self._index([state.slot for state in states])
        self.commit_rows(slots, text[:, 0], codes, *self.teacher_forcing_rows(slots))
        for state in states:
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
        self._reset_slot_state(state.slot)
        self.sessions[key] = state
        return state

    def _encode_rows(
        self,
        rows: list[tuple[PersonaPlexStage0SessionState, tuple[int, int]]],
        samples: np.ndarray,
    ) -> None:
        """Encode one new user frame per row; ``samples`` is ``[rows, frame]`` float32."""
        import torch

        codec = self._shared_codec()
        slot_list = [state.slot for state, _ in rows]
        assert None not in slot_list
        pin = self._slot_buffers().type == "cuda"
        pcm = torch.zeros((self.max_sessions, _FRAME_SAMPLES), dtype=torch.float32, pin_memory=pin)
        active = torch.zeros((self.max_sessions,), dtype=torch.bool, pin_memory=pin)
        # One scatter into the pinned staging rows for the whole step.
        pcm.numpy()[slot_list] = samples
        active.numpy()[slot_list] = True
        # The codes stay on the device: they only feed this row's frame inputs.
        encoded = codec.encode_frame(self._upload(pcm), self._upload(active))
        if encoded.shape[-1] < 8:
            raise RuntimeError(f"PersonaPlex Mimi encoder returned {encoded.shape[-1]} codebooks, expected at least 8")
        slots = self._index(slot_list)
        codes = encoded.to(device=slots.device, dtype=torch.long)[slots, :8]
        history = self._user_history[slots]
        # A frame that was encoded but never prepared does not count as a user
        # frame: the newer one replaces it instead of pushing it down.
        fresh = [state.encoded_identity is None for state, _ in rows]
        if all(fresh):
            older = history[:, :2]
        else:
            keep = self._upload(torch.tensor(fresh, dtype=torch.bool))[:, None, None]
            older = torch.where(keep, history[:, :2], history[:, 1:])
        self._user_history[slots] = torch.cat([codes[:, None], older], dim=1)
        for row, ((state, identity), is_fresh) in enumerate(zip(rows, fresh, strict=True)):
            state.encoded_frame = (codes, row)
            state.encoded_identity = identity
            state.user_frames += int(is_fresh)

    def _build_live_rows(self, rows: list[tuple[PersonaPlexStage0SessionState, tuple[int, int]]]) -> None:
        """Build the live-frame embedding and depformer teacher forcing of every row at once."""
        import torch

        slots = self._index([state.slot for state, _ in rows])
        history = self._user_history[slots]
        # Match the native lockstep ring: the temporal input sees the previous
        # effective agent frame. User cb0 trails the frame being appended by one
        # tick and user cb1..7 trail it by two ticks. Before live user frames
        # fill those slots, text-prompt prefill leaves encoded sine on user rows
        # (the history starts as sine).
        embeds = self.stage_model._build_frame_embeds(
            self._last_text[slots],
            self._last_agent[slots],
            user_d0=history[:, 1],
            user_d1=history[:, 2],
        )
        # Agent rows: silence; user rows: this frame's cb0 and the previous
        # frame's cb1..7.
        self._teacher_tokens[slots] = torch.cat(
            [self._silence.expand(len(rows), -1), history[:, 0, :1], history[:, 1, 1:8]],
            dim=1,
        )
        first = [row for row, (state, _) in enumerate(rows) if state.last_seq == 0]
        if first:
            # The first append forces agent cb1..7 to silence as well.
            provided = np.tile(np.array([False] * 8 + [True] * 8), (len(rows), 1))
            provided[first, 1:8] = True
            self._teacher_provided[slots] = self._upload(torch.from_numpy(provided))
        else:
            self._teacher_provided[slots] = self._live_provided.expand(len(rows), -1)
        self._live_embeds = embeds
        for row, (state, identity) in enumerate(rows):
            state.live_embed = (embeds, row)
            state.live_identity = identity

    def _slot_buffers(self):
        """Allocate the per-slot device state on first use; returns its device."""
        if self._slot_device is not None:
            return self._slot_device
        import torch

        device = self._model_device_dtype()[0]
        rows = self.max_sessions + 1
        self._silence = torch.tensor(SILENCE_TOKENS, dtype=torch.long, device=device)
        self._sine = torch.tensor(SINE_TOKENS, dtype=torch.long, device=device)
        self._silence_cpu = torch.tensor(SILENCE_TOKENS, dtype=torch.long)
        self._last_text = torch.full((rows,), ZERO_TEXT_TOKEN, dtype=torch.long, device=device)
        self._last_agent = self._silence.repeat(rows, 1)
        self._user_history = self._sine.repeat(rows, 3, 1)
        self._teacher_tokens = self._silence.repeat(rows, 2)
        self._teacher_provided = torch.zeros((rows, 16), dtype=torch.bool, device=device)
        self._live_provided = torch.tensor([False] * 8 + [True] * 8, dtype=torch.bool, device=device)
        self._slot_device = device
        return device

    def _reset_slot_state(self, slot: int) -> None:
        self._slot_buffers()
        self._last_text[slot].fill_(ZERO_TEXT_TOKEN)
        self._last_agent[slot] = self._silence
        self._user_history[slot] = self._sine

    def _index(self, values: list[int]):
        import torch

        return self._upload(torch.tensor(values, dtype=torch.long))

    def _upload(self, tensor):
        """Host tensor to the slot device without a stream sync (pinned, non-blocking)."""
        device = self._slot_buffers()
        if device.type != "cuda":
            return tensor.to(device)
        if not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=True)

    def load_encoder(self, *, cuda_graph: bool = False) -> None:
        self._slot_buffers()
        codec = self._shared_codec()
        if cuda_graph:
            codec.capture_encode_graph()

    def _shared_codec(self):
        if self._codec is not None:
            return self._codec
        if self._codec_factory is not None:
            codec = self._codec_factory()
        else:
            import torch
            from vllm.utils.torch_utils import set_default_torch_dtype

            from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
                PersonaPlexMimiCodec,
            )

            checkpoint = Path(self.model_path) / "tokenizer-e351c8d8-checkpoint125.safetensors"
            # The encoder runs in fp32. vLLM loads weights with the model dtype
            # as the default, which would otherwise build it in bf16.
            with set_default_torch_dtype(torch.float32):
                codec = PersonaPlexMimiCodec(
                    checkpoint=str(checkpoint) if checkpoint.is_file() else None,
                    device=self.device,
                )
        # Stage 0 only encodes: no decoder rows (half of the codec's per-row state).
        codec.streaming_init(self.max_sessions, decode=False)
        if self._codec_cuda_graphs:
            codec.capture_cuda_graphs(encode=True, decode_frame_counts=())
        self._codec = codec
        return codec

    def warm_prefill(self, voice: str = DEFAULT_VOICE, persona: str = DEFAULT_PERSONA) -> None:
        """Build and cache the first-append prefill of ``(voice, persona)`` once the weights are loaded.

        The defaults are what the duplex plugin sends for a session that names no
        voice or instructions.
        """
        device, dtype = self._model_device_dtype()
        self._prefill_embeds(voice, persona, device, dtype)

    def _prefill_embeds(self, voice: str, persona: str, device: Any, dtype: Any):
        """Voice + persona prefill rows of a session's first append, cached per ``(voice, persona)``.

        Building them loads the voice bundle and runs a per-token embedding loop,
        which would stall the step thread on every session start. The returned
        tensor is shared: never modify it in place.
        """
        import torch

        key = (voice, persona, str(device), dtype)
        cached = self._prefill_embeds_cache.get(key)
        if cached is not None:
            self._prefill_embeds_cache.move_to_end(key)
            return cached
        voice_embeddings = self._voice_embeddings(voice, device, dtype)
        tokenizer = self._load_tokenizer()
        persona_tokens = tokenizer(wrap_with_system_tags(persona)) if persona else []
        prefill_tokens = torch.tensor(
            [ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT + persona_tokens + [ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT,
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            token_prefill = self.stage_model._build_prefill_embed(
                prefill_tokens,
                0,
                int(prefill_tokens.numel()),
                device,
                torch.tensor(SILENCE_TOKENS, dtype=torch.long, device=device),
                user_sine=torch.tensor(SINE_TOKENS, dtype=torch.long, device=device),
            )
            prefill_embeds = torch.cat([voice_embeddings, token_prefill.to(dtype=dtype)], dim=0)
        self._prefill_embeds_cache[key] = prefill_embeds
        if len(self._prefill_embeds_cache) > _PREFILL_EMBEDS_CACHE_SIZE:
            self._prefill_embeds_cache.popitem(last=False)
        return prefill_embeds

    def _voice_embeddings(self, voice: str, device: Any, dtype: Any):
        """The voice bundle's embeddings as ``[rows, hidden]`` on the model device, cached per voice."""
        import torch

        key = (voice, str(device), dtype)
        cached = self._voice_embeddings_cache.get(key)
        if cached is not None:
            self._voice_embeddings_cache.move_to_end(key)
            return cached
        voice_embeddings = self._load_voice(voice).get("embeddings")
        if not isinstance(voice_embeddings, torch.Tensor) or voice_embeddings.numel() == 0:
            raise ValueError(f"PersonaPlex voice prompt {voice!r} has no embeddings")
        voice_embeddings = (
            voice_embeddings.detach().reshape(-1, voice_embeddings.shape[-1]).to(device=device, dtype=dtype)
        )
        self._voice_embeddings_cache[key] = voice_embeddings
        if len(self._voice_embeddings_cache) > _VOICE_EMBEDDINGS_CACHE_SIZE:
            self._voice_embeddings_cache.popitem(last=False)
        return voice_embeddings

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
    def _decode_pcm_rows(payloads: list[object]) -> np.ndarray:
        """The PCM of several appends as one read-only ``[rows, frame]`` float32 array.

        The samples are checked for finiteness once for the whole step, not once
        per append.
        """
        raws = [
            decode_pcm_f32le_payload(
                payload,
                sample_rate_hz=SAMPLE_RATE,
                exact_samples=_FRAME_SAMPLES,
                model="PersonaPlex Stage 0",
                check_finite=False,
            )
            for payload in payloads
        ]
        samples = np.frombuffer(b"".join(raws), dtype="<f4").reshape(len(raws), _FRAME_SAMPLES)
        check_pcm_f32le_finite(samples, model="PersonaPlex Stage 0")
        return samples


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
