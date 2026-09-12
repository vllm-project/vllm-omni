# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Qwen2.5-Omni talker preprocess phase routing and the
stateless decode-span embedding assembly.

Regression tests for the span-length prefill/decode heuristic: a
speculative-decoding verify step schedules num_draft_tokens + 1 decode
positions in one span, and a chunked prefill can end in a 1-token tail —
both misroute under ``span_len > 1``. The decode path must also assemble
one ``thinker_reply`` row per position, statelessly, so a preempt/resume
replay span reconstructs exactly the embeddings the original steps used.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 4


@functools.lru_cache(maxsize=1)
def _qwen2_5_omni_mod():
    """Import lazily so collection does not pull model_executor too early."""
    import vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni as m

    return m


def _fake_embed_input_ids(ids: torch.Tensor) -> torch.Tensor:
    # Deterministic, id-distinguishing stand-in for the codec embedding table.
    return ids.reshape(-1, 1).to(torch.float32).expand(-1, HIDDEN).contiguous()


def _make_minimal_model():
    cls = _qwen2_5_omni_mod().Qwen2_5OmniForConditionalGeneration
    # ``__new__`` skips ``__init__`` (no weights, no config); set only what
    # the decode path touches.
    model = cls.__new__(cls)
    model.talker = SimpleNamespace(embed_input_ids=_fake_embed_input_ids)
    return model


def _thinker_reply(num_rows: int) -> torch.Tensor:
    # Row k filled with 100 * (k + 1) so any offset error is visible.
    return (torch.arange(1, num_rows + 1, dtype=torch.float32).reshape(-1, 1) * 100.0).expand(-1, HIDDEN).contiguous()


def test_spec_verify_span_gets_one_thinker_row_per_position():
    # A verify step with 2 draft tokens schedules a 3-position decode span at
    # generation offset 2 (prompt_len 4, num_computed 6). Every position must
    # get thinker_reply[offset + i] + codec_embedding(token_i).
    model = _make_minimal_model()
    input_ids = torch.tensor([11, 22, 33], dtype=torch.long)
    q = _thinker_reply(8)

    out_ids, out_embeds, update = model.talker_preprocess(
        input_ids,
        None,
        embed={"thinker_reply": q},
        _omni_is_prefill=False,
        _omni_prompt_len=4,
        _omni_num_computed_tokens=6,
    )

    assert torch.equal(out_ids, input_ids)
    expected = _fake_embed_input_ids(input_ids) + q[2:5]
    assert torch.equal(out_embeds, expected)
    # Stateless: the reply buffer must not be consumed or rewritten.
    assert update == {}
    assert torch.equal(q, _thinker_reply(8))


def test_decode_span_is_replayable():
    # Preempt/resume replays already-generated tokens as one multi-token span.
    # Because the reply buffer is never consumed, calling preprocess twice with
    # the same metadata must produce identical embeddings (the stock consuming
    # path would have shifted by one queue row per call).
    model = _make_minimal_model()
    input_ids = torch.tensor([7, 8], dtype=torch.long)
    payload = dict(
        embed={"thinker_reply": _thinker_reply(6)},
        _omni_is_prefill=False,
        _omni_prompt_len=3,
        _omni_num_computed_tokens=4,
    )

    _, first, _ = model.talker_preprocess(input_ids, None, **payload)
    _, second, _ = model.talker_preprocess(input_ids, None, **payload)

    assert torch.equal(first, second)


def test_positions_past_thinker_reply_keep_codec_embedding():
    # Near the end of the thinker text the span can extend past the last
    # reply row; those positions keep the plain codec embedding (the verifier
    # rejects drafts there naturally).
    model = _make_minimal_model()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    q = _thinker_reply(6)

    _, out_embeds, _ = model.talker_preprocess(
        input_ids,
        None,
        embed={"thinker_reply": q},
        _omni_is_prefill=False,
        _omni_prompt_len=2,
        _omni_num_computed_tokens=7,  # offset 5: one reply row left
    )

    base = _fake_embed_input_ids(input_ids)
    assert torch.equal(out_embeds[0], base[0] + q[5])
    assert torch.equal(out_embeds[1:], base[1:])


def test_one_token_prefill_tail_routes_to_prefill():
    # Chunked prefill can end in a span_len == 1 tail. With the runner
    # metadata saying is_prefill, the row must go down the prefill path and
    # must not touch the decode-side reply buffer (the span heuristic would
    # have misrouted it to decode and consumed one row).
    model = _make_minimal_model()
    sentinel = (torch.tensor([0]), torch.zeros(1, HIDDEN), {"via": "prefill"})
    calls = []

    def fake_prefill_process(input_ids, input_embeds, payload):
        calls.append(input_ids)
        return sentinel

    model.thinker_to_talker_process = fake_prefill_process

    out = model.talker_preprocess(
        torch.tensor([5], dtype=torch.long),
        None,
        embed={"thinker_reply": _thinker_reply(4)},
        _omni_is_prefill=True,
        _omni_prompt_len=8,
        _omni_num_computed_tokens=7,
    )

    assert len(calls) == 1
    assert out == sentinel


def test_missing_runner_metadata_falls_back_to_legacy_single_step():
    # Out-of-tree callers that predate the #3662 runner metadata must keep the
    # stock consuming single-step behavior, bit for bit.
    model = _make_minimal_model()
    sentinel = (torch.tensor([9]), torch.ones(1, HIDDEN), {"via": "legacy"})
    calls = []

    def fake_one_step(input_ids, input_embeds, payload):
        calls.append(input_ids)
        return sentinel

    model.thinker_to_talker_decode_one_step = fake_one_step

    out = model.talker_preprocess(
        torch.tensor([9], dtype=torch.long),
        None,
        embed={"thinker_reply": _thinker_reply(4)},
    )

    assert len(calls) == 1
    assert out == sentinel
