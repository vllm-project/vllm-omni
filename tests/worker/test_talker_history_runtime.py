# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU protocol tests for request-local Talker codec replay."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.core.prefix_cache.adapter import PrefixCacheRequestOwner
from vllm_omni.worker.talker_history import (
    bind_talker_history,
    build_request_end_talker_codes,
    collect_pending_talker_primary,
    install_pending_talker_primary,
    preprocess_talker_history,
    validate_local_talker_resume,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(mocker):
    model = SimpleNamespace(
        talker=SimpleNamespace(num_code_groups=3),
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=16),
        ),
    )
    model.talker_replay_inputs = mocker.Mock(
        side_effect=lambda input_ids, input_embeds, **_: (input_ids, input_embeds, {})
    )
    return model


def _state(owner, *, output_tokens=None, max_tokens=8):
    state = SimpleNamespace(
        prompt_token_ids=[100, 101, 102],
        output_token_ids=list(output_tokens or []),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    bind_talker_history(state, owner)
    return state


def _payload():
    return {"hidden_states": {"last": torch.full((4,), 3, dtype=torch.bfloat16)}, "meta": {}}


@pytest.mark.parametrize(
    ("accepted", "history", "expected_generated"),
    [
        ([7, 8, 9], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]),
        ([7, 8, 9], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6]]),
        ([], [], []),
    ],
    ids=["terminal-n-minus-one", "abort-after-recompute-n", "zero-accepted"],
)
def test_request_end_snapshot_uses_terminal_frontier_without_aliasing(mocker, accepted, history, expected_generated):
    owner = PrefixCacheRequestOwner(88)
    state = _state(owner, output_tokens=accepted)
    state.talker_codec_inputs = [torch.tensor(row, dtype=torch.long) for row in history]
    snapshot = build_request_end_talker_codes(_model(mocker), state, owner=owner)

    assert snapshot.shape == (len(state.prompt_token_ids) + len(expected_generated), 3)
    assert snapshot.dtype == torch.long
    assert torch.count_nonzero(snapshot[: len(state.prompt_token_ids)]) == 0
    if expected_generated:
        torch.testing.assert_close(
            snapshot[len(state.prompt_token_ids) :],
            torch.tensor(expected_generated, dtype=torch.long),
            rtol=0,
            atol=0,
        )
    if history and expected_generated:
        snapshot[-1].fill_(99)
        assert state.talker_codec_inputs[len(expected_generated) - 1].tolist() == expected_generated[-1]


@pytest.mark.parametrize(
    ("accepted", "history", "owner", "message"),
    [
        ([7, 8], [], PrefixCacheRequestOwner(88), "missing"),
        ([7], [[1, 2, 3], [4, 5, 6]], PrefixCacheRequestOwner(88), "ahead"),
        ([7], [[1, 2, 3]], PrefixCacheRequestOwner(89), "owner"),
    ],
)
def test_request_end_snapshot_fails_closed_on_invalid_frontier(mocker, accepted, history, owner, message):
    state = _state(PrefixCacheRequestOwner(88), output_tokens=accepted)
    state.talker_codec_inputs = [torch.tensor(row, dtype=torch.long) for row in history]
    with pytest.raises(ValueError, match=message):
        build_request_end_talker_codes(_model(mocker), state, owner=owner)


def test_pending_primary_materializes_before_first_recompute_chunk(monkeypatch, mocker):
    owner = PrefixCacheRequestOwner(10)
    state = _state(owner, output_tokens=[7])
    model = _model(mocker)
    monkeypatch.setattr(torch, "equal", mocker.Mock(side_effect=AssertionError("no full-history GPU compare")))
    monkeypatch.setattr(torch, "cat", mocker.Mock(side_effect=AssertionError("no O(n) history append")))
    pending = collect_pending_talker_primary(
        model,
        state,
        row_start=0,
        input_ids=torch.tensor([100, 101]),
        input_device=torch.device("cpu"),
        input_dtype=torch.bfloat16,
        payload=_payload(),
    )
    assert pending is not None
    install_pending_talker_primary(
        model,
        state,
        pending_position=pending[0],
        codes=torch.tensor([7, 2, 3]),
        embedding=torch.full((4,), 5, dtype=torch.bfloat16),
    )
    ids, embeds, update = preprocess_talker_history(
        model,
        state,
        row_start=0,
        input_ids=torch.tensor([100, 101]),
        input_embeds=torch.zeros(2, 4, dtype=torch.bfloat16),
        payload=_payload(),
    )

    assert ids.tolist() == [100, 101]
    assert embeds.shape == (2, 4)
    assert update == {}
    assert len(state.talker_codec_inputs) == 1
    torch.testing.assert_close(state.talker_codec_inputs[0], torch.tensor([7, 2, 3]))
    assert state.talker_next_input_embedding is None
    model.talker_replay_inputs.assert_called_once()


def test_preprocess_fails_closed_when_pending_primary_was_not_collected(mocker):
    owner = PrefixCacheRequestOwner(19)
    state = _state(owner, output_tokens=[7])
    with pytest.raises(ValueError, match="not materialized"):
        preprocess_talker_history(
            _model(mocker),
            state,
            row_start=0,
            input_ids=torch.tensor([100, 101]),
            input_embeds=torch.zeros(2, 4, dtype=torch.bfloat16),
            payload=_payload(),
        )


def test_pending_primary_rejects_terminal_length_frontier(mocker):
    state = _state(PrefixCacheRequestOwner(20), output_tokens=[7], max_tokens=1)
    with pytest.raises(ValueError, match="max_tokens"):
        collect_pending_talker_primary(
            _model(mocker),
            state,
            row_start=3,
            input_ids=torch.tensor([7]),
            input_device=torch.device("cpu"),
            input_dtype=torch.bfloat16,
            payload=_payload(),
        )
    assert state.talker_codec_inputs == []
    assert state.talker_next_input_embedding is None


def test_steady_decode_consumes_prepared_embedding_without_predictor(mocker):
    owner = PrefixCacheRequestOwner(11)
    state = _state(owner, output_tokens=[7])
    state.talker_codec_inputs = [torch.tensor([7, 2, 3])]
    state.talker_next_input_position = len(state.prompt_token_ids)
    state.talker_next_input_embedding = torch.full((1, 4), 5, dtype=torch.bfloat16)
    payload = _payload()
    payload["meta"] = {"decode_flag": True}
    model = _model(mocker)
    predict = mocker.Mock(side_effect=AssertionError("accepted codec decisions must not be regenerated"))

    preprocess_talker_history(
        model,
        state,
        row_start=len(state.prompt_token_ids),
        input_ids=torch.tensor([7]),
        input_embeds=torch.zeros(1, 4, dtype=torch.bfloat16),
        payload=payload,
    )

    assert state.talker_next_input_embedding is None
    predict.assert_not_called()
    assert model.talker_replay_inputs.call_args.kwargs["codec_embeddings"].shape == (1, 4)


def test_first_decode_handoff_uses_retained_prefill_tail_without_reconstructing(mocker):
    owner = PrefixCacheRequestOwner(111)
    state = _state(owner, output_tokens=[7])
    state.talker_codec_inputs = [torch.tensor([7, 2, 3])]
    state.talker_next_input_position = len(state.prompt_token_ids)
    state.talker_next_input_embedding = torch.full((1, 4), 5, dtype=torch.bfloat16)
    payload = _payload()
    payload["meta"] = {
        "prefill_consumed_text_tokens": 1,
        "num_processed_tokens": len(state.prompt_token_ids),
    }
    model = _model(mocker)
    predict = mocker.Mock(side_effect=AssertionError("accepted codec decisions must not be regenerated"))

    preprocess_talker_history(
        model,
        state,
        row_start=len(state.prompt_token_ids),
        input_ids=torch.tensor([7]),
        input_embeds=torch.zeros(1, 4, dtype=torch.bfloat16),
        payload=payload,
    )

    predict.assert_not_called()
    kwargs = model.talker_replay_inputs.call_args.kwargs
    assert kwargs["restore_text"] is False
    assert kwargs["codec_embeddings"].shape == (1, 4)


@pytest.mark.parametrize(
    "meta",
    [
        {"prefill_consumed_text_tokens": 1, "num_processed_tokens": 2},
        {"num_processed_tokens": 3},
    ],
)
def test_first_decode_handoff_requires_completed_prefill_cursor(mocker, meta):
    owner = PrefixCacheRequestOwner(112)
    state = _state(owner, output_tokens=[7])
    state.talker_codec_inputs = [torch.tensor([7, 2, 3])]
    state.talker_next_input_position = len(state.prompt_token_ids)
    state.talker_next_input_embedding = torch.full((1, 4), 5, dtype=torch.bfloat16)
    payload = _payload()
    payload["meta"] = dict(meta)
    model = _model(mocker)

    preprocess_talker_history(
        model,
        state,
        row_start=len(state.prompt_token_ids),
        input_ids=torch.tensor([7]),
        input_embeds=torch.zeros(1, 4, dtype=torch.bfloat16),
        payload=payload,
    )

    assert model.talker_replay_inputs.call_args.kwargs["restore_text"] is True


@pytest.mark.parametrize(
    ("history", "accepted_len", "num_output_tokens", "message"),
    [
        ([], 2, 2, "missing"),
        ([torch.tensor([7, 2, 3]), torch.tensor([8, 2, 3])], 1, 1, "missing|ahead"),
        ([torch.tensor([7, 2, 3]), torch.tensor([8, 2, 3]), torch.tensor([9, 2, 3])], 3, 2, "rolled-back"),
    ],
)
def test_resume_fails_closed_on_missing_or_rolled_back_local_state(history, accepted_len, num_output_tokens, message):
    owner = PrefixCacheRequestOwner(12)
    state = _state(owner, output_tokens=[7] * accepted_len)
    state.talker_codec_inputs = list(history)
    with pytest.raises(ValueError, match=message):
        validate_local_talker_resume(state, owner, num_output_tokens=num_output_tokens)


def test_resume_rejects_owner_mismatch():
    owner = PrefixCacheRequestOwner(13)
    state = _state(owner, output_tokens=[7])
    state.talker_codec_inputs = [torch.tensor([7, 2, 3])]
    with pytest.raises(ValueError, match="owner"):
        validate_local_talker_resume(state, PrefixCacheRequestOwner(14), num_output_tokens=1)


@pytest.mark.parametrize("output_tokens", [[], [7]])
def test_resume_requires_retained_history_even_at_initial_frontier(output_tokens):
    owner = PrefixCacheRequestOwner(15)
    state = _state(owner, output_tokens=output_tokens)
    del state.talker_codec_inputs
    with pytest.raises(ValueError, match="retained"):
        validate_local_talker_resume(state, owner, num_output_tokens=len(output_tokens))


def test_resume_requires_all_accepted_primary_outputs():
    owner = PrefixCacheRequestOwner(16)
    state = _state(owner)
    with pytest.raises(ValueError, match="accepted primaries"):
        validate_local_talker_resume(state, owner, num_output_tokens=1)


def test_steady_decode_does_not_scan_previous_decisions(mocker):
    class NonIterableList(list):
        def __iter__(self):
            raise AssertionError("steady decode must not scan request history")

    owner = PrefixCacheRequestOwner(17)
    state = _state(owner)
    state.output_token_ids = NonIterableList([7] * 1024)
    state.talker_codec_inputs = NonIterableList([torch.tensor([7, 2, 3])] * 1024)
    position = len(state.prompt_token_ids) + 1023
    state.talker_next_input_position = position
    state.talker_next_input_embedding = torch.ones(1, 4, dtype=torch.bfloat16)
    payload = _payload()
    payload["meta"]["decode_flag"] = True
    predict = mocker.Mock(side_effect=AssertionError("accepted decisions must not be sampled twice"))
    preprocess_talker_history(
        _model(mocker),
        state,
        row_start=position,
        input_ids=torch.tensor([7]),
        input_embeds=torch.zeros(1, 4, dtype=torch.bfloat16),
        payload=payload,
    )
    predict.assert_not_called()


def test_collect_and_install_pending_primary_owns_batched_rows(mocker):
    owner = PrefixCacheRequestOwner(18)
    state = _state(owner, output_tokens=[7])
    model = _model(mocker)
    input_ids = torch.tensor([100, 7])
    input_embeds = torch.zeros(2, 4, dtype=torch.bfloat16)

    pending = collect_pending_talker_primary(
        model,
        state,
        row_start=2,
        input_ids=input_ids,
        payload=_payload(),
        input_device=input_ids.device,
        input_dtype=input_embeds.dtype,
    )

    assert pending is not None
    position, primary, hidden = pending
    assert position == 3
    torch.testing.assert_close(primary, torch.tensor([[7]]))
    assert hidden.shape == (1, 1, 4)

    batch_codes = torch.tensor([[7, 2, 3]], dtype=torch.long)
    batch_embeds = torch.full((1, 4), 5, dtype=torch.bfloat16)
    owned_codes = batch_codes.detach().clone()
    owned_embeds = batch_embeds.detach().clone()
    install_pending_talker_primary(
        model, state, pending_position=position, codes=owned_codes[0], embedding=owned_embeds[0]
    )
    batch_codes.fill_(9)
    batch_embeds.fill_(9)

    torch.testing.assert_close(state.talker_codec_inputs[0], torch.tensor([7, 2, 3]))
    torch.testing.assert_close(state.talker_next_input_embedding, torch.full((1, 4), 5, dtype=torch.bfloat16))
