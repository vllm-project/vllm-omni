# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ACCEPT = 24
HIDDEN = 8
PROMPT = [151644, 872, 5, 6, 151645, 151644, 77091, 198]
EOS = 151645


class _Manager:
    def __init__(self) -> None:
        self.put_req_chunk: defaultdict[str, int] = defaultdict(int)
        self.request_payload: dict[str, dict] = {}
        self.config = SimpleNamespace(
            hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=ACCEPT))
        )

    def process(self, output, request, is_finished=False):
        payload = q3.thinker2talker_async_chunk(self, output, request, is_finished=is_finished)
        if payload is not None:
            self.put_req_chunk[request.external_req_id] += 1
        return payload


def _request(output_token_ids, *, history=True, max_tokens=64):
    # The native data plane keeps token history only while chunk 0 is pending.
    return SimpleNamespace(
        external_req_id="r",
        prompt_token_ids=list(PROMPT) if history else [],
        output_token_ids=list(output_token_ids) if history else [],
        all_token_ids=list(PROMPT) + list(output_token_ids) if history else [],
        output_token_count=len(output_token_ids),
        last_output_token_id=output_token_ids[-1] if output_token_ids else None,
        resumable=False,
        additional_information=None,
        sampling_params=SimpleNamespace(all_stop_token_ids={EOS}, max_tokens=max_tokens),
    )


def _embed(token_id: int) -> torch.Tensor:
    return torch.full((1, HIDDEN), float(token_id), dtype=torch.bfloat16)


def _output(token_ids, sampled=None):
    layers = {
        0: torch.cat([_embed(t) for t in token_ids]),
        ACCEPT: torch.randn(len(token_ids), HIDDEN).to(torch.bfloat16),
    }
    embed = {"tts_bos": _embed(1), "tts_eos": _embed(2), "tts_pad": _embed(3)}
    if sampled is not None:
        embed["sampled"] = sampled
    return {"hidden_states": {"layers": layers}, "embed": embed}


def _run_turn(tokens, *, publish, max_tokens=64, prefill_chunks=(len(PROMPT),)):
    manager = _Manager()
    rows, chunks, start = [], 0, 0
    for i, size in enumerate(prefill_chunks):
        last = i == len(prefill_chunks) - 1
        sampled = _embed(tokens[0]) if last else torch.empty(0, dtype=torch.bfloat16)
        request = _request(tokens[:1] if last else [], max_tokens=max_tokens)
        payload = manager.process(_output(PROMPT[start : start + size], sampled if publish else None), request)
        start += size
        if payload is not None:
            chunks += 1
            rows.extend(payload.embed.prefill[len(PROMPT) :])
    for step in range(1, len(tokens)):
        history = manager.put_req_chunk["r"] == 0
        request = _request(tokens[: step + 1], history=history, max_tokens=max_tokens)
        output = _output([tokens[step - 1]], _embed(tokens[step]) if publish else None)
        payload = manager.process(output, request)
        if payload is None:
            continue
        chunks += 1
        rows.extend(payload.embed.prefill[len(PROMPT) :] if payload.embed.prefill is not None else payload.embed.decode)
    return [int(row[0]) for row in rows], chunks, manager


@pytest.mark.parametrize(
    ("tokens", "max_tokens"),
    [
        ([11, 12, 13, EOS], 64),  # ends on a stop token
        ([11, 12, 13], 3),  # ends on the max_tokens-th token
        ([11, EOS], 64),
    ],
)
def test_sampled_stream_matches_captured_stream(tokens, max_tokens):
    late, _chunks, _manager = _run_turn(tokens, publish=False, max_tokens=max_tokens)
    early, _chunks, manager = _run_turn(tokens, publish=True, max_tokens=max_tokens)
    # The final token is never processed, so neither path feeds it to the Talker.
    assert early == late == tokens[:-1]
    assert q3._SAMPLED_TEXT_STREAM in manager.request_payload["r"]


def test_sampled_stream_sends_each_token_one_step_earlier():
    manager = _Manager()
    chunk0 = manager.process(_output(PROMPT, sampled=_embed(11)), _request([11]))
    assert chunk0 is not None  # at the prefill step, not after the next decode step
    assert torch.equal(chunk0.embed.prefill, torch.cat([_embed(t) for t in [*PROMPT, 11]]))
    assert chunk0.hidden_states.output.shape[0] == len(PROMPT)
    assert chunk0.ids.all == [*PROMPT, 11]
    # Decode step 1 processes 11 and samples 12: it carries 12.
    chunk1 = manager.process(_output([11], sampled=_embed(12)), _request([11, 12], history=False))
    assert torch.equal(chunk1.embed.decode, _embed(12))


def test_chunked_prefill_uses_sample_of_final_chunk_only():
    early, chunks, _manager = _run_turn([11, 12, EOS], publish=True, prefill_chunks=(5, 3))
    assert early == [11, 12] and chunks == 2


def test_terminal_first_token_keeps_captured_stream():
    manager = _Manager()
    assert manager.process(_output(PROMPT, sampled=_embed(EOS)), _request([EOS])) is None
    assert q3._SAMPLED_TEXT_STREAM not in manager.request_payload["r"]


@pytest.mark.parametrize("prefill_rows", [0, 1, 3])
def test_missing_sample_on_sampled_stream(prefill_rows):
    manager = _Manager()
    manager.process(_output(PROMPT, sampled=_embed(11)), _request([11]))
    chunk = manager.process(_output([11], sampled=_embed(12)), _request([11, 12], history=False))
    assert torch.equal(chunk.embed.decode, _embed(12))
    if not prefill_rows:
        with pytest.raises(RuntimeError, match="sample embedding missing"):
            manager.process(_output([12]), _request([11, 12], history=False))
    else:
        before = manager.put_req_chunk["r"]
        for tokens in (PROMPT[:prefill_rows], [11]):
            partial = _output(tokens, sampled=torch.empty(0))
            assert manager.process(partial, _request([11, 12], history=False)) is None
        assert manager.put_req_chunk["r"] == before
        chunk = manager.process(_output([12], sampled=_embed(13)), _request([11, 12, 13], history=False))
        assert torch.equal(chunk.embed.decode, _embed(13))
