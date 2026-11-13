# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.zonos3.sampling_state import CONTINUE, EOS, Zonos3ARSamplingState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
def test_independent_eoa_countdown_matches_reference_with_batch_reordering(concurrency):
    state = Zonos3ARSamplingState(concurrency, 1024, 1025, "cpu")
    slots = torch.arange(concurrency)
    state.reset(slots, torch.ones(concurrency, dtype=torch.bool))
    # Host reference from TTSSequence._check_eos: first observed EOA wins,
    # highest matching column determines alignment, and countdown starts at 11.
    eoa = [None] * concurrency
    countdown = [0] * concurrency
    lengths = [0] * concurrency
    for step in range(concurrency + 15):
        active = [i for i in range(concurrency) if eoa[i] is None or countdown[i] > 0]
        if step % 2:
            active.reverse()
        if not active:
            break
        codes = torch.zeros((len(active), 10), dtype=torch.int64)
        for row, i in enumerate(active):
            if step == i + 2:
                codes[row, 0] = codes[row, i % 10] = 1024
            columns = (codes[row] == 1024).nonzero().flatten().tolist()
            if eoa[i] is None and columns:
                eoa[i] = max(0, lengths[i] - max(columns))
                countdown[i] = 11
            if eoa[i] is not None and countdown[i] > 0:
                countdown[i] -= 1
            lengths[i] += 1
        result = state.accept(torch.tensor(active), codes)
        assert result.eoa_frame.tolist() == [-1 if eoa[i] is None else eoa[i] for i in active]
        assert result.logical_token_ids.flatten().tolist() == [
            EOS if eoa[i] is not None and countdown[i] == 0 else CONTINUE for i in active
        ]
        assert state.drain_remaining[active].tolist() == [countdown[i] for i in active]


def test_cold_start_triangle_cannot_end_a_request():
    state = Zonos3ARSamplingState(1, 1024, 1025, "cpu")
    slots = torch.tensor([0])
    state.reset(slots, torch.tensor([False]))
    for position in range(9):
        codes = torch.zeros((1, 10), dtype=torch.int64)
        codes[:, position + 1 :] = 1024
        result = state.accept(slots, codes)
        assert result.eoa_frame.item() == -1
        assert (result.codes[:, position + 1 :] == 1025).all()
        assert result.codebook_valid.sum() == position + 1


def test_slot_reuse_resets_all_stop_and_delay_state():
    state = Zonos3ARSamplingState(3, 1024, 1025, "cpu")
    slots = torch.tensor([2, 0])
    state.reset(slots, torch.tensor([True, True]))
    state.accept(slots, torch.full((2, 10), 1024))
    state.reset(torch.tensor([2]), torch.tensor([False]))
    result = state.accept(slots, torch.zeros((2, 10), dtype=torch.int64))
    assert result.eoa_frame.tolist() == [-1, 0]
    assert state.drain_remaining.tolist() == [9, 0, 0]
    assert (result.codes[0, 1:] == 1025).all()


def test_incomplete_prefill_rows_do_not_advance_eoa_or_drain_state():
    state = Zonos3ARSamplingState(2, 1024, 1025, "cpu")
    slots = torch.tensor([0, 1])
    state.reset(slots, torch.tensor([False, True]))
    eoa = torch.full((2, 10), 1024, dtype=torch.int64)
    first = state.accept(slots, eoa, torch.tensor([False, True]))
    assert first.active.tolist() == [False, True]
    assert state.position.tolist() == [0, 1] and state.eoa_frame.tolist() == [-1, 0]
    assert state.drain_remaining.tolist() == [0, 10]
    previous = state.next_codes.clone()
    for _ in range(10):
        state.accept(slots, eoa, torch.tensor([False, False]))
    assert state.position.tolist() == [0, 1] and state.drain_remaining.tolist() == [0, 10]
    assert torch.equal(state.next_codes, previous)


def test_async_step_after_logical_eos_keeps_device_state_absorbing():
    state = Zonos3ARSamplingState(1, 1024, 1025, "cpu")
    slot = torch.tensor([0])
    state.reset(slot, torch.tensor([True]))
    first = state.accept(slot, torch.full((1, 10), 1024, dtype=torch.int64))
    assert first.logical_token_ids.item() == CONTINUE
    for _ in range(10):
        last = state.accept(slot, torch.zeros((1, 10), dtype=torch.int64))
    assert last.logical_token_ids.item() == EOS
    before = state.next_codes.clone()
    extra = state.accept(slot, torch.full((1, 10), 17, dtype=torch.int64))
    assert extra.logical_token_ids.item() == EOS and not extra.active.item()
    assert state.position.item() == 11 and torch.equal(state.next_codes, before)


def test_codebook0_policy_and_ignore_eos_remain_request_local():
    state = Zonos3ARSamplingState(2, 1024, 1025, "cpu")
    slots = torch.tensor([0, 1])
    state.reset(slots, torch.tensor([True, True]), torch.tensor([False, True]), torch.tensor([True, False]))
    codes = torch.zeros((2, 10), dtype=torch.int64)
    codes[:, 9] = 1024
    step = state.accept(slots, codes)
    assert step.eoa_frame.tolist() == [-1, -1]
    assert not step.codebook_valid[:, 9].any()
    codes[:, 0] = 1024
    step = state.accept(slots, codes)
    assert step.eoa_frame.tolist() == [1, -1]
    for _ in range(10):
        step = state.accept(slots, codes)
    assert step.logical_token_ids[:, 0].tolist() == [EOS, CONTINUE]
