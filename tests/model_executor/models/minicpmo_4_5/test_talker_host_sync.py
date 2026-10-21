# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 Talker per-step path: batched, host-sync-free, same results."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    ConditionalChatTTSConfig,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 8
_EOS = 7


@dataclass
class _SamplingMetadata:
    repetition_penalties: torch.Tensor
    prompt_token_ids: torch.Tensor | None = None
    no_penalties: bool = False


def _make_talker(device: str = "cpu") -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = _VOCAB
    talker._codec_eos_id = _EOS
    talker._force_eos_rows = None
    talker._mask_eos_rows = None
    talker._pending_force_eos_rows = None
    talker._penalty_histories = None
    talker._request_audio_states = {}
    talker._request_condition_states = {}
    talker._deferred_cleanup_ids = set()
    talker._tts_config = ConditionalChatTTSConfig()
    # Set by __init__ from the stage-1 speculative config; the single-frame
    # paths this fixture drives read it, so it has to be present even though
    # __init__ is bypassed here.
    talker._k_step_frames = 0
    generator = torch.Generator().manual_seed(0)
    talker.emb_code = nn.ModuleList([nn.Embedding(_VOCAB, 4)])
    talker.head_code = nn.ModuleList([nn.Linear(4, _VOCAB, bias=False)])
    with torch.no_grad():
        talker.emb_code[0].weight.copy_(torch.randn(_VOCAB, 4, generator=generator))
        talker.head_code[0].weight.copy_(torch.randn(_VOCAB, 4, generator=generator))
    return talker.to(device)


def _states() -> dict[str, dict[str, Any]]:
    return {
        # Mid-stream row: forwards its code, EOS still masked by min_tokens.
        "req-live": {"finished": False, "step": 2, "max_tokens": 100, "min_tokens": 5, "recent_codes": [1, 2]},
        # The previous step sampled codec EOS: forward nothing, finish, force EOS.
        "req-eos": {"finished": False, "step": 60, "max_tokens": 100, "recent_codes": [4, 4, 5]},
        # Leftover decode of an already-finished request.
        "req-done": {"finished": True, "step": 3, "max_tokens": 100},
        # Reaches its codec budget on this step.
        "req-cap": {"finished": False, "step": 8, "max_tokens": 10, "recent_codes": list(range(7)) * 3},
    }


def _infos(states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"request_id": request_id, "audio_state": dict(state), "_omni_is_prefill": False}
        for request_id, state in states.items()
    ]


def _step(talker, infos, hidden, mocker, *, batched: bool, input_ids: torch.Tensor, penalties: torch.Tensor):
    if batched:
        returned_ids, embeds, updates = talker.preprocess_decode_batch(input_ids=input_ids, req_infos=infos)
        assert returned_ids is input_ids
    else:
        rows = [talker.preprocess(input_ids[row : row + 1], None, **info) for row, info in enumerate(infos)]
        embeds = torch.cat([row[1] for row in rows])
        updates = [row[2] for row in rows]
    for info, update in zip(infos, updates, strict=True):
        info["codes"] = update["codes"]
    output = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=infos,
        request_token_spans=[(row, row + 1) for row in range(len(infos))],
    )
    logits = talker.compute_logits(output.text_hidden_states)
    captured: dict[str, torch.Tensor] = {}

    def _sampler(logits, sampling_metadata):
        captured["logits"] = logits.clone()
        return SimpleNamespace(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.Sampler",
        return_value=_sampler,
    )
    sampled = talker.sample(logits, _SamplingMetadata(repetition_penalties=penalties))
    return embeds, output, logits, captured["logits"], sampled.sampled_token_ids


def test_batched_decode_matches_scalar_preprocess(mocker) -> None:
    input_ids = torch.tensor([3, _EOS, _EOS, 6], dtype=torch.int32)
    hidden = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))
    penalties = torch.tensor([1.05, 1.2, 1.05, 1.0])
    results = {}
    talkers = {}
    for batched in (False, True):
        talker = _make_talker()
        states = _states()
        talker._request_audio_states = copy.deepcopy(states)
        results[batched] = _step(
            talker, _infos(states), hidden, mocker, batched=batched, input_ids=input_ids, penalties=penalties
        )
        talkers[batched] = talker

    # Embeddings, masked logits, penalized logits and sampled ids.
    for index in (0, 2, 3, 4):
        assert torch.equal(results[False][index], results[True][index])
    for result in results.values():
        output = result[1].multimodal_outputs
        assert [t.tolist() for t in output["codes"]["audio"]] == [[[3]], [], [], [[6]]]
        assert [t.item() for t in output["meta"]["finished"]] == [False, True, True, True]
    assert talkers[True]._request_audio_states == talkers[False]._request_audio_states
    assert talkers[True]._decode_codec_ids == {}
