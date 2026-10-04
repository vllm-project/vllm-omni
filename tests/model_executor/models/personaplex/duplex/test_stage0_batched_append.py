# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched live-append preparation must match the per-request path exactly."""

import base64
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.personaplex.duplex.policy import (
    SILENCE_TOKENS,
    SINE_TOKENS,
)
from vllm_omni.model_executor.models.personaplex.duplex.stage0 import (
    PersonaPlexStage0DuplexRuntime,
)
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_WIDTH = 17  # text row + 8 agent rows + 8 user rows
_PREFILL_LEN = 18  # 2 voice rows + 2 x 6 silence frames + 3 persona tokens + the live frame


class _FakeCodec:
    """Shared streaming encoder whose codes encode the row's frame count."""

    def __init__(self) -> None:
        self.encode_calls = 0
        self.frames: list[int] = []

    def streaming_init(self, batch_size: int) -> None:
        self.frames = [0] * batch_size

    def encode_frame(self, pcm: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        del pcm
        self.encode_calls += 1
        codes = torch.zeros((len(self.frames), 8), dtype=torch.long)
        for row, is_active in enumerate(active.tolist()):
            if is_active:
                self.frames[row] += 1
                codes[row] = self.frames[row] * 16 + torch.arange(8) + 1
        return codes

    def reset_slot(self, row: int) -> None:
        self.frames[row] = 0


@dataclass(frozen=True)
class _TalkerConfig:
    num_audio_codebooks: int = 16
    audio_vocab_size: int = 2048


class _EmbeddingTalker:
    """Stage model whose input embedding is the token stack itself, so embeds expose the layout."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = _TalkerConfig()

    def input_embeddings(self, stack: torch.Tensor) -> torch.Tensor:
        return stack.squeeze(-1).to(torch.float32)

    def _build_prefill_embed(self, tokens, offset, span, device, silence=None, user_sine=None):
        del offset, silence, user_sine
        values = tokens[:span].to(device=device, dtype=torch.float32)
        return values[:, None].expand(-1, _WIDTH).contiguous()

    _build_frame_embed = PersonaPlexTalkerForConditionalGeneration._build_frame_embed


def _depformer(text_token, hidden, *, audio_tokens, audio_provided, num_steps):
    """Deterministic stand-in whose codes depend on every input it is given."""
    del hidden
    assert audio_provided.dtype == torch.bool
    base = audio_tokens[:, 8 : 8 + num_steps] * 7 + audio_provided[:, :num_steps].long()
    return (base + text_token[:, None] + torch.arange(num_steps)) % 2048


def _talker(max_sessions: int) -> PersonaPlexTalkerForConditionalGeneration:
    runtime = PersonaPlexStage0DuplexRuntime(
        _EmbeddingTalker(),
        model_path="/unused",
        device="cpu",
        codec=_FakeCodec(),
        max_sessions=max_sessions,
        tokenizer=lambda _text: [7, 8, 9],
        voice_loader=lambda _voice: {"embeddings": torch.arange(2 * _WIDTH, dtype=torch.float32).reshape(2, _WIDTH)},
    )
    talker = PersonaPlexTalkerForConditionalGeneration.__new__(PersonaPlexTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker._personaplex_duplex_stage0_runtime = runtime
    talker._dtype = torch.float32
    talker.mtp_hidden_size = _WIDTH
    talker.num_active_codebooks = 8
    talker.depformer = _depformer
    return talker


def _append(session_id: str, seq: int, epoch: int = 0) -> dict:
    pcm = np.zeros(1920, dtype="<f4")
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


@dataclass
class _StepResult:
    embeds: dict[str, torch.Tensor]
    audio_tokens: torch.Tensor
    audio_provided: torch.Tensor
    codes: torch.Tensor
    agents: dict[tuple[str, int], torch.Tensor]


def _run_step(
    talker: PersonaPlexTalkerForConditionalGeneration,
    step: list[tuple[str, dict, int]],
    *,
    batched: bool,
    text_base: int,
) -> _StepResult:
    """One scheduler step: batch preprocess, per-request preprocess, then depformer sampling.

    ``step`` holds ``(request_id, append, prompt_len)``; a prompt length of
    ``_PREFILL_LEN`` marks a first append scheduled as one full prefill span.
    """
    runtime = talker._personaplex_duplex_stage0_runtime
    infos = {}
    spans = {}
    for request_id, duplex, prompt_len in step:
        spans[request_id] = prompt_len if prompt_len == _PREFILL_LEN else 1
        infos[request_id] = {
            "duplex": duplex,
            "request_id": request_id,
            "duplex_prompt_len": prompt_len,
            "duplex_token_offset": prompt_len - spans[request_id],
        }
    if batched:
        talker.preprocess_batch(req_ids=list(infos), model_intermediate_buffer=infos, device=torch.device("cpu"))
    else:
        runtime.encode_appends([info["duplex"] for info in infos.values()])
    embeds, req_infos = {}, []
    for request_id, info in infos.items():
        input_ids = torch.zeros(spans[request_id], dtype=torch.long)
        _, embed, update = talker.preprocess(input_ids, None, _omni_is_prefill=True, **info)
        embeds[request_id] = embed
        req_infos.append(update)
    received = {}

    def depformer(text_token, hidden, **kwargs):
        received.update(kwargs)
        return _depformer(text_token, hidden, **kwargs)

    talker.depformer = depformer
    req_ids = list(infos)
    codes = talker.post_sample_talker_mtp(
        input_ids=torch.arange(len(req_ids)) + text_base,
        hidden_states=torch.zeros((len(req_ids), _WIDTH)),
        req_ids=req_ids,
        req_infos=req_infos,
    )
    return _StepResult(
        embeds=embeds,
        audio_tokens=received["audio_tokens"],
        audio_provided=received["audio_provided"],
        codes=codes,
        agents={key: state.last_agent_codes.clone() for key, state in runtime.sessions.items()},
    )


def _assert_same(batched: _StepResult, reference: _StepResult) -> None:
    assert batched.embeds.keys() == reference.embeds.keys()
    for request_id, embed in batched.embeds.items():
        assert torch.equal(embed, reference.embeds[request_id]), request_id
    assert torch.equal(batched.audio_tokens, reference.audio_tokens)
    assert torch.equal(batched.audio_provided, reference.audio_provided)
    assert torch.equal(batched.codes, reference.codes)
    assert batched.agents.keys() == reference.agents.keys()
    for key, agent in batched.agents.items():
        assert torch.equal(agent, reference.agents[key]), key


def _device_prepared(talker, key: tuple[str, int]) -> bool:
    state = talker._personaplex_duplex_stage0_runtime.sessions[key]
    return state.prepared.depformer_audio_tokens_device is not None


def test_mixed_prefill_and_live_steps_match_the_per_request_path() -> None:
    steps = [
        [("a", _append("a", 1), 18), ("b", _append("b", 1), 18)],
        # A first append joins a step of live appends.
        [("a", _append("a", 2), 19), ("b", _append("b", 2), 19), ("c", _append("c", 1), 18)],
        [("a", _append("a", 3), 20), ("b", _append("b", 3), 20), ("c", _append("c", 2), 19)],
        # Only one session appends; the others keep their state.
        [("c", _append("c", 3), 20)],
        [("a", _append("a", 4), 21), ("c", _append("c", 4), 21)],
    ]
    batched, reference = _talker(max_sessions=4), _talker(max_sessions=4)

    for index, step in enumerate(steps):
        text_base = 100 + 10 * index
        result = _run_step(batched, step, batched=True, text_base=text_base)
        _assert_same(result, _run_step(reference, step, batched=False, text_base=text_base))

    assert _device_prepared(batched, ("a", 0)) and _device_prepared(batched, ("c", 0))
    assert not _device_prepared(reference, ("a", 0))
    runtime = batched._personaplex_duplex_stage0_runtime
    assert runtime._shared_codec().encode_calls == len(steps)
    assert len(runtime.sessions[("a", 0)].user_history_device) == 2
    # Live embeds carry the causal user delay: cb0 from the previous frame, cb1..7 from the one before.
    a4 = result.embeds["a"][0]
    assert a4[9].item() == 3 * 16 + 1
    assert a4[10:17].tolist() == [2 * 16 + k + 1 for k in range(1, 8)]


def test_second_append_reads_sine_for_the_missing_user_history() -> None:
    talker = _talker(max_sessions=1)

    def state_of():
        return talker._personaplex_duplex_stage0_runtime.sessions[("a", 0)]

    first = _run_step(talker, [("a", _append("a", 1), 18)], batched=True, text_base=100)
    # The first append forces agent cb1..7 to silence; cb0 is sampled.
    assert state_of().last_agent_codes.tolist() == [first.codes[0, 0].item(), *SILENCE_TOKENS[1:]]
    result = _run_step(talker, [("a", _append("a", 2), 19)], batched=True, text_base=101)
    # Live appends force only the user codebooks, so the sampled agent frame is kept as is.
    assert torch.equal(state_of().last_agent_codes, result.codes[0])

    embed = result.embeds["a"][0]
    assert embed[0].item() == 100
    assert embed[9].item() == 16 + 1
    assert embed[10:17].tolist() == list(SINE_TOKENS[1:])
    assert result.audio_tokens[0].tolist() == [*SILENCE_TOKENS, 2 * 16 + 1, *(16 + k + 1 for k in range(1, 8))]
    assert result.audio_provided[0].tolist() == [False] * 8 + [True] * 8


def test_cancel_restart_and_recycled_slot_match_the_per_request_path() -> None:
    steps = [
        [("a0", _append("a", 1), 18), ("b", _append("b", 1), 18)],
        [("a0", _append("a", 2), 19), ("b", _append("b", 2), 19)],
        # A cancel restarts session a as epoch 1 while epoch 0's aborted request
        # still has an append in this step, next to b's live append.
        [("a0", _append("a", 3), 20), ("a1", _append("a", 1, epoch=1), 18), ("b", _append("b", 3), 20)],
        [("a1", _append("a", 2, epoch=1), 19), ("b", _append("b", 4), 21)],
    ]
    batched, reference = _talker(max_sessions=2), _talker(max_sessions=2)

    for index, step in enumerate(steps):
        text_base = 100 + 10 * index
        result = _run_step(batched, step, batched=True, text_base=text_base)
        _assert_same(result, _run_step(reference, step, batched=False, text_base=text_base))

    runtime = batched._personaplex_duplex_stage0_runtime
    assert sorted(runtime.sessions) == [("a", 1), ("b", 0)]
    assert _device_prepared(batched, ("a", 1))
    # The restarted epoch starts from an empty history on its recycled row.
    assert result.embeds["a1"][0][10:17].tolist() == list(SINE_TOKENS[1:])

    # A new session on the row of a closed one also starts from a clean history.
    for talker in (batched, reference):
        talker.on_requests_finished({"b"})
    followups = [
        [("c", _append("c", 1), 18)],
        [("c", _append("c", 2), 19), ("a1", _append("a", 3, epoch=1), 20)],
    ]
    for index, step in enumerate(followups):
        text_base = 200 + 10 * index
        result = _run_step(batched, step, batched=True, text_base=text_base)
        _assert_same(result, _run_step(reference, step, batched=False, text_base=text_base))
    assert result.embeds["c"][0][10:17].tolist() == list(SINE_TOKENS[1:])


def test_prepare_live_appends_ignores_empty_first_and_unencoded_appends() -> None:
    talker = _talker(max_sessions=2)
    runtime = talker._personaplex_duplex_stage0_runtime

    assert runtime.prepare_live_appends([]) == 0
    talker.preprocess_batch(req_ids=["x"], model_intermediate_buffer={"x": {}}, device=torch.device("cpu"))
    assert runtime.sessions == {}

    runtime.encode_appends([_append("a", 1)])
    assert runtime.prepare_live_appends([_append("a", 1)]) == 0  # the first append keeps the prefill path
    _run_step(talker, [("a", _append("a", 1), 18)], batched=False, text_base=100)
    # Not encoded in this step: left to prepare_append.
    assert runtime.prepare_live_appends([_append("a", 2)]) == 0
    runtime.encode_appends([_append("a", 2)])
    assert runtime.prepare_live_appends([_append("a", 2), _append("a", 2)]) == 1
    # Already prepared: a repeat in a later step is a no-op.
    assert runtime.prepare_live_appends([_append("a", 2)]) == 0
    assert runtime.prepare_append(_append("a", 2), prompt_len=19, request_id="a").prompt_offset == 18
