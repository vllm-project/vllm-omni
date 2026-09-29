# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 live appends are prepared per step, not per request.

The runtime and talker hooks of a live one-frame append must leave exactly the
state and inputs the per-request path leaves.
"""

import base64

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.personaplex.duplex.stage0 import PersonaPlexStage0DuplexRuntime
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 4
_PREFILL = 17  # 2 voice rows + 6 silence + 3 persona tokens + 6 silence


class _BatchCodec:
    """Shared streaming encoder without per-row Python: a row's code is its frame count."""

    def __init__(self) -> None:
        self.frames: torch.Tensor | None = None

    def streaming_init(self, batch_size: int) -> None:
        self.frames = torch.zeros(batch_size, dtype=torch.long)

    def encode_frame(self, pcm: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        assert self.frames is not None
        self.frames += active.to(torch.long)
        # A per-row PCM checksum keeps the codes tied to the staged samples.
        return (self.frames + pcm[:, :1].to(torch.long).reshape(-1))[:, None].expand(-1, 8).clone()

    def reset_slot(self, row: int) -> None:
        assert self.frames is not None
        self.frames[row] = 0


class _Talker:
    device = torch.device("cpu")
    dtype = torch.float32

    def _build_prefill_embed(self, tokens, offset, span, device, silence=None, user_sine=None):
        del offset, silence, user_sine
        return tokens[:span].to(device=device, dtype=torch.float32)[:, None].expand(-1, _HIDDEN).contiguous()

    def _build_frame_embeds(self, text_tokens, last_agent, *, user_d0, user_d1):
        return (text_tokens[:, None] * 1000 + last_agent[:, :1] * 100 + user_d0[:, :1] * 10 + user_d1[:, 1:2]).to(
            torch.float32
        ).expand(-1, _HIDDEN) + torch.arange(_HIDDEN, dtype=torch.float32)


def _runtime(max_sessions: int) -> PersonaPlexStage0DuplexRuntime:
    voice_embeddings = torch.arange(2 * _HIDDEN, dtype=torch.float32).reshape(2, 1, 1, _HIDDEN)
    return PersonaPlexStage0DuplexRuntime(
        _Talker(),
        model_path="/unused",
        device="cpu",
        codec=_BatchCodec(),
        max_sessions=max_sessions,
        tokenizer=lambda _text: [7, 8, 9],
        voice_loader=lambda _voice: {"embeddings": voice_embeddings},
    )


def _duplex(session_id: str, seq: int, epoch: int = 0) -> dict:
    pcm = np.full(1920, float(seq + len(session_id)), dtype="<f4")
    return {
        "data_plane": True,
        "session_id": session_id,
        "epoch": epoch,
        "seq": seq,
        "payload": {
            "format": "pcm_f32le",
            "sample_rate_hz": 24000,
            "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
        },
        "runtime_config": {"personaplex_voice_prompt": "NATF2.pt", "personaplex_persona": "Be concise."},
    }


def _sessions(count: int) -> list[str]:
    return [f"s{index:03d}" for index in range(count)]


def _open_and_sample(runtime: PersonaPlexStage0DuplexRuntime, sessions: list[str]) -> None:
    """First append (voice + persona prefill) of every session, then one committed sample."""
    runtime.encode_appends([_duplex(session_id, 1) for session_id in sessions])
    for session_id in sessions:
        runtime.prepare_append(_duplex(session_id, 1), prompt_len=_PREFILL + 1, request_id=session_id)
    count = len(sessions)
    runtime.record_samples(
        request_ids=sessions,
        text_tokens=torch.arange(count) + 5,
        agent_codes=torch.arange(8 * count).reshape(count, 8) % 7,
    )


def _live_step(runtime: PersonaPlexStage0DuplexRuntime, sessions: list[str], seq: int):
    appends = [(session_id, _duplex(session_id, seq), _PREFILL + seq) for session_id in sessions]
    runtime.encode_appends([duplex for _, duplex, _ in appends])
    return runtime.prepare_live_appends(appends)


def _session_snapshot(runtime: PersonaPlexStage0DuplexRuntime) -> dict:
    return {
        key: (
            state.slot,
            state.user_frames,
            state.prefill_slots,
            state.last_seq,
            state.prepared_identity,
            sorted(state.request_ids),
            state.prepared.prompt_offset,
            state.prepared.prefill_applied,
            {key: value for key, value in state.prepared.info_update.items() if key != "pplex_silence_codes"},
            state.prepared.user_frame.tolist(),
            state.prepared.input_ids.tolist(),
        )
        for key, state in runtime.sessions.items()
    }


def test_live_appends_match_the_per_request_prepare() -> None:
    sessions = _sessions(5)
    batched, reference = _runtime(8), _runtime(8)
    for runtime in (batched, reference):
        _open_and_sample(runtime, sessions)

    handled, embeds = _live_step(batched, sessions, 2)
    reference.encode_appends([_duplex(session_id, 2) for session_id in sessions])
    prepared = [
        reference.prepare_append(_duplex(session_id, 2), prompt_len=_PREFILL + 2, request_id=session_id)
        for session_id in sessions
    ]

    assert handled == list(range(len(sessions)))
    assert torch.equal(embeds, torch.cat([item.inputs_embeds for item in prepared]))
    assert _session_snapshot(batched) == _session_snapshot(reference)
    assert batched.request_sessions == reference.request_sessions
    for key in ("_last_text", "_last_agent", "_user_history", "_teacher_tokens", "_teacher_provided"):
        assert torch.equal(getattr(batched, key), getattr(reference, key)), key
    # Nothing is left half-built for the per-request path.
    assert all(state.live_embed is None and state.encoded_frame is None for state in batched.sessions.values())


def _talker(runtime: PersonaPlexStage0DuplexRuntime) -> PersonaPlexTalkerForConditionalGeneration:
    talker = PersonaPlexTalkerForConditionalGeneration.__new__(PersonaPlexTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker._personaplex_duplex_stage0_runtime = runtime
    talker._dtype = torch.float32
    talker.mtp_hidden_size = _HIDDEN
    return talker


def _step_layout(sessions: list[str], seq: int, *, prefill: str | None = None):
    """A step with one live row per session, plus one session's chunked prefill in the middle."""
    req_ids, offsets, scheduled, computed, prompt_lens = [], [], [], [], []
    buffer = {}
    offset = 0
    for index, session_id in enumerate(sessions):
        if prefill is not None and index == len(sessions) // 2:
            req_ids.append(prefill)
            offsets.append(offset)
            scheduled.append(_PREFILL + 1)
            computed.append(0)
            prompt_lens.append(_PREFILL + 1)
            buffer[prefill] = {"duplex": _duplex(prefill, 1)}
            offset += _PREFILL + 1
        req_ids.append(session_id)
        offsets.append(offset)
        scheduled.append(1)
        computed.append(_PREFILL + seq - 1)
        prompt_lens.append(_PREFILL + seq)
        buffer[session_id] = {"duplex": _duplex(session_id, seq)}
        offset += 1
    return req_ids, buffer, offsets, scheduled, computed, prompt_lens, offset


def _batch_preprocess(talker, layout):
    req_ids, buffer, offsets, scheduled, computed, prompt_lens, total = layout
    input_ids = torch.full((total,), 9, dtype=torch.int32)
    inputs_embeds = torch.full((total, _HIDDEN), -1.0)
    handled = talker.preprocess_batch(
        req_ids=req_ids,
        model_intermediate_buffer=buffer,
        device=torch.device("cpu"),
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        token_offsets=offsets,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=computed,
        prompt_lens=prompt_lens,
    )
    return handled, input_ids, inputs_embeds


def test_talker_batch_preprocess_writes_the_rows_the_per_request_path_writes() -> None:
    sessions = _sessions(6)
    batched, reference = _runtime(8), _runtime(8)
    for runtime in (batched, reference):
        _open_and_sample(runtime, sessions)
    layout = _step_layout(sessions, 2, prefill="late")
    req_ids, buffer, offsets, scheduled, computed, prompt_lens, total = layout

    handled, input_ids, inputs_embeds = _batch_preprocess(_talker(batched), layout)

    assert handled == set(sessions)
    reference_talker = _talker(reference)
    reference_talker.preprocess_batch(req_ids=req_ids, model_intermediate_buffer=buffer, device=torch.device("cpu"))
    for index, req_id in enumerate(req_ids):
        start, span = offsets[index], scheduled[index]
        rows = slice(start, start + span)
        if req_id not in handled:
            # The prefill row is untouched, for the per-request preprocess.
            assert torch.equal(inputs_embeds[rows], torch.full((span, _HIDDEN), -1.0))
            assert torch.equal(input_ids[rows], torch.full((span,), 9, dtype=torch.int32))
            continue
        ids, embeds, _ = reference_talker.preprocess(
            torch.zeros(span, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            request_id=req_id,
            duplex_prompt_len=prompt_lens[index],
            duplex_token_offset=computed[index],
            **buffer[req_id],
        )
        assert torch.equal(inputs_embeds[rows], embeds)
        assert torch.equal(input_ids[rows], ids.to(torch.int32))
