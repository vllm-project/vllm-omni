# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex stage 0 for OpenAI Live sessions (``/v1/live/sessions``).

Each Live session is one resumable request; each streaming update is one
80 ms frame and carries, in ``model_intermediate_buffer["personaplex_live"]``,
the frame's user PCM and the text token stage 0 sampled for the previous frame
(stage 0 is a final output, so the API server sees it and feeds it back).

The depformer step of frame ``t`` (its agent codes, conditioned on the frame's
temporal hidden state and sampled text token) runs here, in the ``preprocess``
of frame ``t + 1``: the hidden state is the one ``postprocess`` stored and the
text token comes with the update. Frame ``t``'s codes are emitted to Code2Wav
on that step, then frame ``t + 1``'s input embedding is built from them (Moshi
lockstep: agent codes of ``t``, user codebook 0 of ``t`` and 1..7 of
``t - 1``). Output audio therefore trails the input by one extra frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.model_executor.models.personaplex.prompting import (
    AUDIO_SILENCE_FRAME_CNT,
    FRAME_SIZE,
    SILENCE_TOKENS,
    SINE_TOKENS,
    ZERO_TEXT_TOKEN,
    cached_personaplex_tokenizer,
    load_personaplex_voice_state,
    personaplex_model_dir,
    wrap_with_system_tags,
)

LIVE_KEY = "personaplex_live"
_AGENT_CODEBOOKS = 8


@dataclass
class _LiveSession:
    slot: int
    seq: int = 0
    user_codes: list[torch.Tensor] = field(default_factory=list)
    last_text: int | None = None
    last_agent: torch.Tensor | None = None
    # Teacher forcing of the frame whose depformer step is still pending.
    forced_tokens: torch.Tensor | None = None
    forced_mask: torch.Tensor | None = None
    embeds: torch.Tensor | None = None
    prompt_offset: int = 0
    encoded_seq: int = 0
    encoded: torch.Tensor | None = None
    codes: torch.Tensor | None = None


class PersonaPlexLiveStage0:
    """Per-request Live state of the PersonaPlex talker."""

    def __init__(self, model: Any, *, model_path: str, max_sessions: int) -> None:
        self.model = model
        self.model_path = personaplex_model_dir(model_path)
        self.max_sessions = max_sessions
        self.sessions: dict[str, _LiveSession] = {}
        self._free_slots = list(reversed(range(max_sessions)))
        self._codec: Any = None
        self._voices: dict[str, torch.Tensor] = {}

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.model.parameters())
        return parameter.device, self.model._dtype

    def _shared_codec(self) -> Any:
        if self._codec is None:
            from vllm_omni.model_executor.models.personaplex.personaplex_mimi import PersonaPlexMimiCodec

            checkpoint = Path(self.model_path) / "tokenizer-e351c8d8-checkpoint125.safetensors"
            codec = PersonaPlexMimiCodec(
                checkpoint=str(checkpoint) if checkpoint.is_file() else None,
                device=str(self._device_dtype()[0]),
            )
            codec.streaming_init(self.max_sessions, decode=False)
            self._codec = codec
        return self._codec

    def _session(self, request_id: str) -> _LiveSession:
        session = self.sessions.get(request_id)
        if session is None:
            if not self._free_slots:
                raise RuntimeError(f"PersonaPlex Live stage-0 capacity {self.max_sessions} is exhausted")
            self._shared_codec()
            session = self.sessions[request_id] = _LiveSession(slot=self._free_slots.pop())
        return session

    def close(self, request_id: str) -> None:
        session = self.sessions.pop(request_id, None)
        if session is not None:
            self._shared_codec().reset_slot(session.slot)
            self._free_slots.append(session.slot)

    def encode(self, appends: list[tuple[str, dict[str, Any]]]) -> None:
        """Mimi-encode the new user frame of every Live append of this step."""
        rows: list[tuple[_LiveSession, int, np.ndarray]] = []
        for request_id, live in appends:
            session = self._session(request_id)
            seq = int(live["seq"])
            if seq in (session.seq, session.encoded_seq):
                continue
            rows.append((session, seq, np.frombuffer(live["pcm"], dtype=np.float32)))
        if not rows:
            return
        pcm = torch.zeros((self.max_sessions, FRAME_SIZE), dtype=torch.float32)
        active = torch.zeros((self.max_sessions,), dtype=torch.bool)
        for session, _, samples in rows:
            pcm[session.slot] = torch.from_numpy(samples.copy())
            active[session.slot] = True
        codes = self._shared_codec().encode_frame(pcm, active).detach().to(dtype=torch.long, device="cpu")
        for session, seq, _ in rows:
            session.encoded = codes[session.slot, :_AGENT_CODEBOOKS].clone()
            session.encoded_seq = seq

    def _voice_embeddings(self, voice: str) -> torch.Tensor:
        embeddings = self._voices.get(voice)
        if embeddings is None:
            state = load_personaplex_voice_state(self.model_path, voice)
            raw = state.get("embeddings")
            if not isinstance(raw, torch.Tensor) or raw.numel() == 0:
                raise ValueError(f"PersonaPlex voice prompt {voice!r} has no embeddings")
            embeddings = self._voices[voice] = raw.reshape(-1, raw.shape[-1])
        return embeddings

    def _finish_previous_frame(self, session: _LiveSession, text_token: int, hidden: torch.Tensor) -> None:
        """Depformer step of the previous frame: its agent codes for Code2Wav and
        the effective (teacher-forced) codes the next input embeds."""
        model = self.model
        device, dtype = self._device_dtype()
        text = torch.tensor([text_token], dtype=torch.long, device=device)
        codes = model.depformer(
            text,
            hidden.reshape(1, 1, -1).to(device=device, dtype=dtype),
            audio_tokens=session.forced_tokens.reshape(1, -1).to(device=device),
            audio_provided=session.forced_mask.reshape(1, -1).to(device=device),
            num_steps=model.num_active_codebooks,
        ).to(torch.long)
        forced = session.forced_tokens[:_AGENT_CODEBOOKS].to(device)
        mask = session.forced_mask[:_AGENT_CODEBOOKS].to(device)
        session.last_agent = torch.where(mask, forced, codes[0, :_AGENT_CODEBOOKS]).detach()
        session.last_text = text_token
        session.codes = codes.detach()

    def _prepare(self, request_id: str, session: _LiveSession, live: dict[str, Any], info: dict[str, Any]) -> None:
        model = self.model
        device, dtype = self._device_dtype()
        seq = int(live["seq"])
        first = session.seq == 0
        if not first:
            hidden = (info.get("hidden_states") or {}).get("last")
            previous_text = live.get("prev_text")
            if not isinstance(hidden, torch.Tensor) or previous_text is None:
                raise ValueError(f"PersonaPlex Live frame {seq} of {request_id} lacks the previous frame's sample")
            self._finish_previous_frame(session, int(previous_text), hidden)

        if session.encoded_seq != seq:
            self.encode([(request_id, live)])
        user_frame = session.encoded.to(device)
        session.user_codes = [*session.user_codes[-2:], user_frame]

        silence = torch.tensor(SILENCE_TOKENS, dtype=torch.long, device=device)
        sine = torch.tensor(SINE_TOKENS, dtype=torch.long, device=device)
        user_d0 = session.user_codes[-2] if len(session.user_codes) > 1 else sine
        user_d1 = session.user_codes[-3] if len(session.user_codes) > 2 else sine
        session.forced_tokens = torch.cat([silence, user_frame[:1], user_d0[1:_AGENT_CODEBOOKS]])
        session.forced_mask = torch.tensor(
            [False, *([first] * (_AGENT_CODEBOOKS - 1)), *([True] * _AGENT_CODEBOOKS)],
            dtype=torch.bool,
            device=device,
        )
        text_token = torch.tensor(
            [session.last_text if session.last_text is not None else ZERO_TEXT_TOKEN], dtype=torch.long, device=device
        )
        last_agent = session.last_agent if session.last_agent is not None else silence
        frame = model._build_frame_embed(text_token, last_agent, device, user_d0=user_d0, user_d1=user_d1)
        if first:
            persona = str(live.get("persona") or "")
            persona_tokens = (
                cached_personaplex_tokenizer(self.model_path)(wrap_with_system_tags(persona)) if persona else []
            )
            prefill_tokens = torch.tensor(
                [
                    *([ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT),
                    *persona_tokens,
                    *([ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT),
                ],
                dtype=torch.long,
                device=device,
            )
            prefill = model._build_prefill_embed(
                prefill_tokens, 0, int(prefill_tokens.numel()), device, silence, user_sine=sine
            )
            voice = self._voice_embeddings(str(live["voice"])).to(device=device, dtype=dtype)
            session.embeds = torch.cat([voice, prefill.to(dtype=dtype), frame.to(dtype=dtype)], dim=0)
        else:
            session.embeds = frame.to(dtype=dtype)
        session.seq = seq

    def preprocess(
        self, input_ids: torch.Tensor, live: dict[str, Any], info: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        request_id = str(info["request_id"])
        session = self._session(request_id)
        span = int(input_ids.shape[0])
        if session.seq != int(live["seq"]):
            self._prepare(request_id, session, live, info)
            session.prompt_offset = int(info["_omni_prompt_len"]) - int(session.embeds.shape[0])
        local = int(info["_omni_num_computed_tokens"]) - session.prompt_offset
        embeds = session.embeds[local : local + span]
        if local < 0 or embeds.shape[0] != span:
            raise ValueError(
                f"PersonaPlex Live span [{local}, {local + span}) is outside the prepared frame "
                f"({session.embeds.shape[0]} rows) of {request_id}"
            )
        return input_ids, embeds.to(device=input_ids.device, dtype=self.model._dtype), {}

    def take_codes(self, request_id: str | None) -> torch.Tensor | None:
        session = self.sessions.get(request_id) if request_id is not None else None
        if session is None or session.codes is None:
            return None
        codes, session.codes = session.codes, None
        return codes
