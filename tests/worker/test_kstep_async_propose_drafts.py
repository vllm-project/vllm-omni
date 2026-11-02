"""Async draft-feedback closure for the K-step proposer.

Under async scheduling the sampled values have not reached the host when
``propose_draft_token_ids`` runs (they ride the GPU copy into the next
step's inputs) and the step's spec booking comes back one step later through
``update_draft_token_ids``, so the proposer cannot derive its split from
either. It must propose the constant draft for every row and let the
scheduler trim to what it books -- otherwise every step folds on empty rows,
the scheduler never sees a draft, never books a K-span, and the feedback
loop is severed at birth (the async-arm deadlock).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu.worker import talker_multiframe
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_model():
    """Minimal stand-in satisfying the multi-frame runner contract."""
    return SimpleNamespace(
        supports_multi_frame_decode=True,
        batch_stop_logits=None,
        take_batch_stop_logits=lambda: None,
        set_batch_stop_logits=lambda logits: None,
        merge_frame_outputs=lambda *a, **k: None,
    )


def _runner(*, num_reqs: int, async_sched: bool = True):
    runner = object.__new__(GPUARModelRunner)
    runner.model = _fake_model()
    runner.num_spec_tokens = 7
    runner.use_async_scheduling = async_sched
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=[f"r{i}" for i in range(num_reqs)],
    )
    return runner


def _sched_out(booked: dict | None = None):
    return SimpleNamespace(scheduled_spec_decode_tokens=booked or {})


def test_async_propose_emits_full_constant_drafts_without_sampled_values():
    """Async rows carry no host-readable sampled values: the proposer must
    still emit the full K-1 drafts for every row, or the booking loop dies."""
    runner = _runner(num_reqs=4)
    out = runner.propose_draft_token_ids(_sched_out(), [[-1], [-1], [-1], [-1]])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 7)
    assert torch.all(out == talker_multiframe.CONTINUE_TOKEN_ID)


def test_async_propose_reuses_cached_constant_tensor():
    """Same (rows, drafts) shape proposes the same cached tensor; a shape
    change rebuilds. No host-list build, no per-step device allocation."""
    runner = _runner(num_reqs=4)
    sched_out = _sched_out()
    first = runner.propose_draft_token_ids(sched_out, [[-1]] * 4)
    second = runner.propose_draft_token_ids(sched_out, [[-1]] * 4)
    assert first is second

    grown_runner = _runner(num_reqs=5)
    grown = grown_runner.propose_draft_token_ids(sched_out, [[-1]] * 5)
    assert grown.shape == (5, 7)
    assert grown is not first


def test_async_propose_does_not_fold_on_short_sampled_rows():
    """A 1-wide sampled row under async is a deferred value, not a codec
    stop: the proposer must not read stop-truncation out of row lengths."""
    runner = _runner(num_reqs=2)
    booked = {"r0": [talker_multiframe.CONTINUE_TOKEN_ID] * 7}
    out = runner.propose_draft_token_ids(_sched_out(booked), [[5], [9]])
    assert out.shape == (2, 7)
    assert torch.all(out == talker_multiframe.CONTINUE_TOKEN_ID)


def test_sync_propose_still_folds_a_mid_step_stop():
    """Sync rows carry real rejection-sampler output: a row shorter than its
    booked span is a codec stop and the whole batch folds to zero width."""
    runner = _runner(num_reqs=2, async_sched=False)
    booked = {f"r{i}": [talker_multiframe.CONTINUE_TOKEN_ID] * 7 for i in range(2)}
    out = runner.propose_draft_token_ids(_sched_out(booked), [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 0)


def test_sync_propose_fills_a_mixed_batch_and_caches_the_tensor():
    """Sync mixed batches draft in full (converge to uniform spans) and the
    resulting tensor comes from the shape cache."""
    runner = _runner(num_reqs=2, async_sched=False)
    booked = {"r0": [talker_multiframe.CONTINUE_TOKEN_ID] * 7}
    out = runner.propose_draft_token_ids(_sched_out(booked), [[1, 2, 3, 4, 5, 6, 7, 8], [9]])
    assert out.shape == (2, 7)
    assert torch.all(out == talker_multiframe.CONTINUE_TOKEN_ID)
    again = runner.propose_draft_token_ids(_sched_out(booked), [[1, 2, 3, 4, 5, 6, 7, 8], [9]])
    assert again is out
