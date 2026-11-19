# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 turn-chunked Thinker-to-Talker handoff (``llm2tts_chunked``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.handoff import unpack_handoff_tensor
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    MINICPMO45_TURN_CHUNK_TOKENS,
    llm2tts,
    llm2tts_chunked,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TTS_BOS = 151703
TTS_EOS = 151704


def _row(position: int, token_id: int) -> torch.Tensor:
    # Each Thinker row records the position and token it was computed for.
    return torch.tensor([[float(position), float(token_id)]])


class _Thinker:
    """DELTA-mode Stage-0 outputs: the payload accumulates per-step tensors in lists."""

    def __init__(self, prompt: list[int], request_id: str = "req") -> None:
        self.prompt = list(prompt)
        self.request_id = request_id
        self.output: list[int] = []
        self.latent: list[torch.Tensor] = []
        self.ids: list[torch.Tensor] = []
        self.positions: list[torch.Tensor] = []
        self._forward(list(enumerate(self.prompt)))

    def _forward(self, rows: list[tuple[int, int]]) -> None:
        self.latent.append(torch.cat([_row(position, token_id) for position, token_id in rows]))
        self.ids.append(torch.tensor([token_id for _, token_id in rows]))
        self.positions.append(torch.tensor([position for position, _ in rows]))

    def step(self, accepted: list[int], *, rows: list[tuple[int, int]] | None = None, finished: bool = False):
        """Sample ``accepted``; the step forwarded ``rows`` (default: the previous last token)."""
        if rows is None and self.output:
            rows = [(len(self.prompt) + len(self.output) - 1, self.output[-1])]
        if rows:
            self._forward(rows)
        self.output.extend(accepted)
        return self.snapshot(finished=finished, delta=accepted)

    def snapshot(self, *, finished: bool, delta: list[int] | None = None, consolidated: bool = False):
        payload = {
            "latent": list(self.latent),
            "latent_input_ids": list(self.ids),
            "latent_positions": list(self.positions),
        }
        if consolidated:
            payload = {key: torch.cat(value) for key, value in payload.items()}
        completion = SimpleNamespace(
            token_ids=list(self.output if delta is None else delta),
            cumulative_token_ids=list(self.output),
            text="",
            multimodal_output=None,
        )
        return SimpleNamespace(
            request_id=self.request_id,
            finished=finished,
            prompt_token_ids=list(self.prompt),
            multimodal_output=payload,
            outputs=[completion],
        )


def _context():
    return SimpleNamespace(bridge_states={})


def _handoff(tts_input):
    buffer = tts_input["model_intermediate_buffer"]
    hidden = unpack_handoff_tensor(buffer["hidden_states"]["tts"])
    return list(buffer["ids"]["tts"]), torch.as_tensor(hidden), buffer["meta"]


def _speak(thinker: _Thinker, context, reply: list[int]):
    """Feed one token per step, then the closing <|tts_eos|>; return every handoff."""
    emitted = []
    for token_id in reply:
        emitted.extend(llm2tts_chunked([thinker.step([token_id])], None, False, context))
    final = thinker.step([TTS_EOS], finished=True)
    emitted.extend(llm2tts_chunked([final], None, False, context))
    return emitted, final


def test_reply_goes_out_in_ten_token_blocks_and_closes_on_finish() -> None:
    thinker = _Thinker([5, 6, TTS_BOS])
    context = _context()
    reply = list(range(100, 125))

    emitted, final = _speak(thinker, context, reply)

    assert MINICPMO45_TURN_CHUNK_TOKENS == 10
    blocks = [_handoff(tts_input) for tts_input in emitted]
    assert [ids for ids, _, _ in blocks] == [reply[:10], reply[10:20], reply[20:]]
    for ids, hidden, _ in blocks:
        # One Thinker row per token, taken at that token's position.
        positions = [3 + reply.index(token_id) for token_id in ids]
        assert hidden[:, 0].tolist() == positions
        assert hidden[:, 1].tolist() == ids
    assert [meta["streaming_condition_seq"] for _, _, meta in blocks] == [0, 1, 2]
    assert [meta["turn_chunk_close"] for _, _, meta in blocks] == [False, False, True]
    assert [meta.get("replace_streaming_prompt", False) for _, _, meta in blocks] == [True, False, False]
    assert all(meta["turn_chunk"] is True for _, _, meta in blocks)
    # <audio_bos> closes open blocks; <text_eos><audio_bos> the last one.
    assert [meta["next_stage_prompt_len"] for _, _, meta in blocks] == [11, 11, 7]
    assert [len(tts_input["prompt_token_ids"]) for tts_input in emitted] == [11, 11, 7]

    # The terminal update after the last block reuses it; vLLM ignores its prompt.
    terminal = llm2tts_chunked([final], None, False, context)
    assert len(terminal) == 1 and terminal[0] is emitted[-1]


def test_block_waits_for_the_row_of_its_last_token() -> None:
    thinker = _Thinker([TTS_BOS])
    context = _context()
    for token_id in range(100, 110):
        assert llm2tts_chunked([thinker.step([token_id])], None, False, context) == []
    # Ten tokens are sampled, but the tenth has no Thinker row until the next step.
    emitted = llm2tts_chunked([thinker.step([110])], None, False, context)
    assert [_handoff(tts_input)[0] for tts_input in emitted] == [list(range(100, 110))]


def test_multi_token_step_takes_accepted_rows_not_rejected_drafts() -> None:
    thinker = _Thinker([5, TTS_BOS])
    context = _context()
    thinker.step([100])
    # Verifying drafts [101, 999]: 101 is accepted, 999 rejected, 102 is the bonus token.
    llm2tts_chunked([thinker.step([101, 102], rows=[(2, 100), (3, 101), (4, 999)])], None, False, context)
    emitted = []
    for token_id in range(103, 113):
        emitted.extend(llm2tts_chunked([thinker.step([token_id])], None, False, context))

    ids, hidden, _ = _handoff(emitted[0])
    assert ids == list(range(100, 110))
    # Position 4 was re-forwarded with the accepted token after the rejected draft.
    assert hidden[:, 1].tolist() == ids
    assert hidden[:, 0].tolist() == list(range(2, 12))


def test_rejected_draft_row_holds_the_block_until_the_accepted_row_arrives() -> None:
    thinker = _Thinker([TTS_BOS])
    context = _context()
    thinker.step([100])
    for token_id in range(101, 110):
        llm2tts_chunked([thinker.step([token_id])], None, False, context)
    # The step that sampled 110 and 111 forwarded a rejected draft at 109's position.
    stuck = thinker.step([110, 111], rows=[(10, 777)])
    assert llm2tts_chunked([stuck], None, False, context) == []
    emitted = llm2tts_chunked([thinker.step([112], rows=[(10, 109), (11, 110), (12, 111)])], None, False, context)
    ids, hidden, _ = _handoff(emitted[0])
    assert ids == list(range(100, 110))
    assert hidden[:, 1].tolist() == ids


def test_short_reply_goes_out_whole_when_the_thinker_finishes() -> None:
    thinker = _Thinker([5, TTS_BOS])
    context = _context()
    emitted, _ = _speak(thinker, context, list(range(100, 106)))

    assert len(emitted) == 1
    ids, hidden, meta = _handoff(emitted[0])
    assert ids == list(range(100, 106))
    assert hidden.shape[0] == 6
    assert meta["turn_chunk_close"] is True
    assert meta["streaming_condition_seq"] == 0
    assert meta["replace_streaming_prompt"] is True
    assert meta["next_stage_prompt_len"] == 8


def test_empty_reply_hands_over_an_empty_closing_condition() -> None:
    thinker = _Thinker([5, TTS_BOS])
    context = _context()
    emitted, _ = _speak(thinker, context, [])

    ids, hidden, meta = _handoff(emitted[0])
    assert ids == []
    assert hidden.numel() == 0
    assert meta["turn_chunk_close"] is True
    # The Talker conditions an empty reply on its two boundary embeddings.
    assert meta["next_stage_prompt_len"] == 2


def test_reply_without_tts_bos_is_spoken_whole_from_its_first_token() -> None:
    # The marker could still come later, so the reply waits for the finish.
    thinker = _Thinker([5, 6])
    context = _context()
    reply = list(range(100, 125))
    emitted, _ = _speak(thinker, context, reply)

    blocks = [_handoff(tts_input) for tts_input in emitted]
    assert [ids for ids, _, _ in blocks] == [reply]
    assert blocks[0][1][:, 0].tolist() == list(range(2, 27))
    assert blocks[0][2]["turn_chunk_close"] is True


def test_preamble_before_a_late_tts_bos_is_not_spoken() -> None:
    thinker = _Thinker([5, 6])
    context = _context()
    preamble = list(range(200, 215))
    speech = list(range(100, 125))
    emitted, _ = _speak(thinker, context, [*preamble, TTS_BOS, *speech])

    blocks = [_handoff(tts_input) for tts_input in emitted]
    assert [ids for ids, _, _ in blocks] == [speech[:10], speech[10:20], speech[20:]]
    first_position = 2 + len(preamble) + 1
    assert blocks[0][1][:, 0].tolist() == list(range(first_position, first_position + 10))
    assert [meta["streaming_condition_seq"] for _, _, meta in blocks] == [0, 1, 2]


def test_final_only_output_hands_over_the_whole_reply_once() -> None:
    thinker = _Thinker([5, TTS_BOS])
    for token_id in range(100, 123):
        thinker.step([token_id])
    thinker.step([TTS_EOS])
    final = thinker.snapshot(finished=True, consolidated=True)

    emitted = llm2tts_chunked([final], None, False, _context())

    ids, hidden, meta = _handoff(emitted[0])
    assert ids == list(range(100, 123))
    assert hidden[:, 1].tolist() == ids
    assert meta["turn_chunk_close"] is True


def test_without_request_state_whole_replies_go_through_llm2tts() -> None:
    thinker = _Thinker([5, TTS_BOS])
    for token_id in range(100, 113):
        partial = thinker.step([token_id])
    assert llm2tts_chunked([partial], None, False, None) == []

    thinker.step([TTS_EOS])
    final = thinker.snapshot(finished=True, consolidated=True)
    chunked = llm2tts_chunked([final], None, False, None)
    whole = llm2tts([final], None, False, None)
    assert [_handoff(tts_input)[0] for tts_input in chunked] == [_handoff(tts_input)[0] for tts_input in whole]
    assert "turn_chunk" not in chunked[0]["model_intermediate_buffer"]["meta"]


def test_native_duplex_output_is_rejected() -> None:
    thinker = _Thinker([TTS_BOS])
    output = thinker.step([100], finished=True)
    output.multimodal_output["ids.prompt"] = [TTS_BOS]
    with pytest.raises(ValueError, match="does not serve native duplex"):
        llm2tts_chunked([output], None, False, _context())


def _talker(max_model_len: int = 4096):
    return SimpleNamespace(max_model_len=max_model_len, hf_config_name=None)


def _talker_stage_bridge(max_model_len: int):
    """Call the bridge the way the Talker's stage client does, with the Talker's model config."""
    from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

    client = object.__new__(StageEngineCoreClient)
    client.custom_process_input_func = llm2tts_chunked
    client.requires_multimodal_data = False
    client._stage_hf_config = None
    client.vllm_config = SimpleNamespace(model_config=_talker(max_model_len))
    return lambda outputs, context: client._call_custom_process_input(outputs, None, context)


@pytest.mark.parametrize("max_model_len", [4096, 768])
def test_long_reply_keeps_the_resumable_talker_within_its_context(max_model_len) -> None:
    """Blocks stop once the Talker context bound is full; the closing block takes what still fits.

    Without the bound every 10 tokens add a block, and the resumable Talker
    outgrows max_model_len, which fails the whole request.
    """
    thinker = _Thinker([5, TTS_BOS])
    context = _context()
    bridge = _talker_stage_bridge(max_model_len)
    reply = [100 + index % 50 for index in range(900)]
    emitted = []
    for token_id in reply:
        emitted.extend(bridge([thinker.step([token_id])], context))
    final = thinker.step([TTS_EOS], finished=True)
    emitted.extend(bridge([final], context))

    blocks = [_handoff(tts_input) for tts_input in emitted]
    *opened, (closing_ids, _, closing_meta) = blocks
    assert closing_meta["turn_chunk_close"] is True
    assert all(not meta["turn_chunk_close"] for _, _, meta in opened)
    # Worst case per open block: its condition, its code cap (the Talker's
    # default ceiling without one) and the step that samples its EOS; then
    # the closing condition and one CUDA K-step of samples on top.
    worst = sum(meta["next_stage_prompt_len"] + meta.get("turn_chunk_max_codes", 500) + 16 for _, _, meta in opened)
    worst += closing_meta["next_stage_prompt_len"] + 16
    assert worst <= max_model_len
    spoken = [token_id for ids, _, _ in blocks for token_id in ids]
    assert spoken == reply[: len(spoken)]
    assert 0 < len(closing_ids) and len(spoken) < len(reply)


def test_reply_that_fits_is_spoken_whole_under_the_context_bound() -> None:
    thinker = _Thinker([5, TTS_BOS])
    context = _context()
    bridge = _talker_stage_bridge(4096)
    reply = list(range(100, 267))
    emitted = []
    for token_id in reply:
        emitted.extend(bridge([thinker.step([token_id])], context))
    final = thinker.step([TTS_EOS], finished=True)
    emitted.extend(bridge([final], context))

    blocks = [_handoff(tts_input) for tts_input in emitted]
    assert [token_id for ids, _, _ in blocks for token_id in ids] == reply
    assert all(meta["turn_chunk_max_codes"] == 200 for _, _, meta in blocks[:-1])
    assert "turn_chunk_max_codes" not in blocks[-1][2]


def test_blocks_concatenate_only_the_steps_that_forwarded_their_rows(monkeypatch) -> None:
    from vllm_omni.model_executor.stage_input_processors import minicpmo_4_5_omni

    prompt = [5] * 300 + [TTS_BOS]
    thinker = _Thinker(prompt)
    context = _context()
    seen: list[int] = []
    original = minicpmo_4_5_omni._row_tensor

    def record(item):
        seen.append(item.shape[0])
        return original(item)

    monkeypatch.setattr(minicpmo_4_5_omni, "_row_tensor", record)
    emitted, _ = _speak(thinker, context, list(range(100, 125)))

    assert len(emitted) == 3
    # The 301 prompt rows are never concatenated; each block reads its own steps.
    assert len(prompt) not in seen
    assert sum(seen) == 25
    assert _handoff(emitted[1])[1][:, 0].tolist() == list(range(len(prompt) + 10, len(prompt) + 20))


def test_only_the_chunked_bridge_takes_partial_outputs() -> None:
    assert llm2tts_chunked.accepts_partial_source_outputs is True
    assert not getattr(llm2tts, "accepts_partial_source_outputs", False)
