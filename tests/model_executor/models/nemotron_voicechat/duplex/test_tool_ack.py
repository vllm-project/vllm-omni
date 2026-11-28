# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The client's on-hold message has to reach the talker, or the wait is silent."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin import (
    _MIN_ACK_PAD_TOKENS,
    NemotronVoiceChatDuplexPlugin,
    _ack_token_sequence,
    _tool_ack_phrases,
)
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Tokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.vocab.setdefault(word, 100 + len(self.vocab)) for word in text.split()]


def _plugin() -> NemotronVoiceChatDuplexPlugin:
    return object.__new__(NemotronVoiceChatDuplexPlugin)


def test_ack_tokens_are_bos_phrase_pads_eos() -> None:
    tokenizer = _Tokenizer()
    phrase = "One moment while I generate a number."
    tokens = _ack_token_sequence(tokenizer, phrase, bos_id=1, eos_id=2, pad_id=12)

    body = tokenizer.encode(phrase)
    assert tokens[0] == 1
    assert tokens[1 : 1 + len(body)] == body
    assert tokens[-1] == 2
    pads = tokens[1 + len(body) : -1]
    assert pads and set(pads) == {12}
    assert len(pads) >= _MIN_ACK_PAD_TOKENS


def test_ack_tokens_are_empty_when_the_phrase_tokenises_to_nothing() -> None:
    assert _ack_token_sequence(_Tokenizer(), "   ", bos_id=1, eos_id=2, pad_id=12) == []


def test_phrases_are_read_per_tool() -> None:
    tools = [
        {"name": "generate_random_number", "ack_messages": ["One moment."]},
        {"name": "get_recipe"},
    ]
    assert _tool_ack_phrases(tools) == {"generate_random_number": ["One moment."]}


def test_publishing_a_call_queues_that_tools_phrase() -> None:
    runtime = {
        "nvc_tool_ack_tokens": {"generate_random_number": [[1, 100, 12, 2], [1, 101, 12, 2]]},
    }
    updated = _plugin().note_published_function_call(
        runtime, call_id="call_1", history_len=3, name="generate_random_number"
    )
    assert updated is not None
    batch = updated["nvc_tool_ack_batch"]
    assert updated["nvc_tool_ack_generation"] == 1
    assert batch["generation"] == 1
    assert batch["name"] == "generate_random_number"
    assert batch["tokens"] in ([1, 100, 12, 2], [1, 101, 12, 2])
    # The tokens stay in the runtime: the next call picks from the same table.
    assert updated["nvc_tool_ack_tokens"] == runtime["nvc_tool_ack_tokens"]


def test_publishing_an_unknown_tool_queues_nothing() -> None:
    runtime = {"nvc_tool_ack_tokens": {"generate_random_number": [[1, 100, 12, 2]]}}
    assert _plugin().note_published_function_call(runtime, call_id="call_1", history_len=3, name="get_recipe") is None
    assert _plugin().note_published_function_call(runtime, call_id="call_1", history_len=3, name=None) is None


def test_sync_adopts_each_published_batch_once() -> None:
    session: dict[str, object] = {}
    batch = {"generation": 1, "tokens": [7, 8, 9]}

    NemotronVoiceChatThinkerForConditionalGeneration._sync_tool_ack(session, {"nvc_tool_ack_batch": batch})
    assert session["nvc_tool_ack_generation"] == 1
    assert session["nvc_tool_ack_pending"] == [7, 8, 9]

    # The same generation must not re-queue; a newer one replaces the queue.
    session["nvc_tool_ack_pending"] = []
    NemotronVoiceChatThinkerForConditionalGeneration._sync_tool_ack(session, {"nvc_tool_ack_batch": batch})
    assert session["nvc_tool_ack_pending"] == []
    NemotronVoiceChatThinkerForConditionalGeneration._sync_tool_ack(
        session, {"nvc_tool_ack_batch": {"generation": 2, "tokens": [4]}}
    )
    assert session["nvc_tool_ack_generation"] == 2
    assert session["nvc_tool_ack_pending"] == [4]


def test_sync_rejects_malformed_batches() -> None:
    session: dict[str, object] = {}
    for bad in ({}, {"generation": 1}, {"generation": 1, "tokens": []}, {"generation": "x", "tokens": [1]}):
        NemotronVoiceChatThinkerForConditionalGeneration._sync_tool_ack(session, {"nvc_tool_ack_batch": bad})
    assert "nvc_tool_ack_pending" not in session


def _sampler_model(session: dict[str, object]):
    import torch

    model = object.__new__(NemotronVoiceChatThinkerForConditionalGeneration)
    model._duplex_sampling_rows = {0: "req-1"}
    model._nvc_function_sampling_tokens = torch.tensor([12], dtype=torch.int64)
    model._use_function_head = True
    model._mrv2_duplex_sampling = True
    model._sessions = {"req-1": session}
    model._duplex_previous_text_tokens = {}
    return model


def _sampler_session(pending: list[int] | None) -> dict[str, object]:
    return {
        "sampling_frame_eligible": True,
        "nvc_text_pad_id": 12,
        "nvc_function_sotc_id": 20,
        "nvc_function_eotc_id": 21,
        "function_call_active": True,
        "func_token": 12,
        **({"nvc_tool_ack_pending": list(pending)} if pending is not None else {}),
    }


def test_sampler_speaks_the_ack_token_in_place_of_pad() -> None:
    import torch

    session = _sampler_session([101, 102])
    model = _sampler_model(session)
    logits = torch.zeros(1, 1000)
    logits[0, 999] = 10.0

    out = model.sample(logits, None)
    assert int(out.sampled_token_ids[0, 0].item()) == 101
    assert session["nvc_tool_ack_pending"] == [102]
    assert model._duplex_previous_text_tokens["req-1"] == 101

    import torch as _torch

    model._duplex_sampling_rows = {0: "req-1"}
    model._nvc_function_sampling_tokens = _torch.tensor([12], dtype=_torch.int64)
    out = model.sample(logits, None)
    assert int(out.sampled_token_ids[0, 0].item()) == 102
    assert session["nvc_tool_ack_pending"] == []

    model._duplex_sampling_rows = {0: "req-1"}
    model._nvc_function_sampling_tokens = _torch.tensor([12], dtype=_torch.int64)
    out = model.sample(logits, None)
    # Queue drained: the call is active, so PAD silence returns.
    assert int(out.sampled_token_ids[0, 0].item()) == 12
