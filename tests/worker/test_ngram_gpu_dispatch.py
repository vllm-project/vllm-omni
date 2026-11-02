# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise sample_tokens dispatch; device work and output building are isolated."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU

from vllm_omni.worker import gpu_ar_model_runner as ar

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DispatchCompleteError(Exception):
    """Stop after sampling/drafting, before output tensor materialization."""


def _runner(monkeypatch, *, method, async_scheduling, fits=True, count_event=True, padded=True):
    events = []
    runner = object.__new__(ar.GPUARModelRunner)
    sampled = torch.tensor([[11, -1], [22, 23]], dtype=torch.int32)
    cpu_rows = [] if async_scheduling else [[11], [22, 23]]
    runner.execute_model_state = (
        SimpleNamespace(total_num_scheduled_tokens=3),
        None,
        None,
        None if method == "ngram_gpu" and not fits else object(),
        torch.zeros((3, 2)),
        None,
        torch.zeros((2, 2)),
        None,
        None,
        None,
        {},
        None,
        None,
    )
    runner.input_batch = SimpleNamespace(sampling_metadata=SimpleNamespace(), req_ids=["a", "b"])
    runner.model = SimpleNamespace()
    runner.use_async_scheduling = async_scheduling
    runner.broadcast_pp_output = False
    runner.kv_connector_output = None
    runner.speculative_config = SimpleNamespace(
        use_eagle=lambda: method == "eagle",
        uses_draft_model=lambda: method == "draft",
        uses_extract_hidden_states=lambda: False,
        use_ngram_gpu=lambda: method == "ngram_gpu",
        disable_padded_drafter_batch=not padded,
    )
    runner.device = torch.device("cpu")
    runner.num_spec_tokens = 3
    runner._draft_probs = object()
    runner._draft_prob_req_ids = ["old-request"]
    runner.valid_sampled_token_count_event = object() if count_event else None
    proposer_type = {"ngram_gpu": NgramProposerGPU, "eagle": ar.EagleProposer, "draft": ar.DraftModelProposer}.get(
        method
    )
    runner.drafter = object.__new__(proposer_type) if proposer_type else SimpleNamespace()
    runner.token_ids_gpu_tensor = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    runner.num_tokens_no_spec_gpu = torch.tensor([1, 1], dtype=torch.int32)
    runner.discard_request_mask = SimpleNamespace(gpu=torch.tensor([False, True]))
    runner._sample = lambda *args: SimpleNamespace(sampled_token_ids=sampled)
    runner._update_states_after_model_execute = lambda *args: None
    runner._input_fits_in_drafter = lambda *args: fits

    def post_sample_mtp(**kwargs):
        assert kwargs["sampled_token_ids"] is sampled
        assert sampled.tolist() == [[11, -1], [22, 23]]
        return kwargs["multimodal_outputs"]

    runner._run_post_sample_talker_mtp = post_sample_mtp
    runner.finalize_kv_connector = lambda: events.append(("finalize_kv", None))
    runner.eplb_step = lambda: None

    def propose(_scheduler, tokens, *args):
        events.append(("propose", tokens))
        return [[31], [32]]

    def bookkeep(_scheduler, sampler_output, *args):
        assert sampler_output.sampled_token_ids is sampled
        assert sampled.tolist() == [[11, -1], [22, 23]]
        events.append(("bookkeep", None))
        return 0, None, None, cpu_rows, None, ["a", "b"], {"a": 0, "b": 1}, []

    def copy_drafts(_scheduler, zeros_only=False):
        if zeros_only:
            assert runner._draft_token_ids.shape == (2, runner.num_spec_tokens)
            assert runner._draft_token_ids.dtype == torch.int32
            assert torch.count_nonzero(runner._draft_token_ids).item() == 0
            assert runner._draft_probs is None
            assert runner._draft_prob_req_ids is None
            events.append(("copy_zeros", None))
        else:
            events.append(("copy_drafts", None))

    def update(tokens, batch, token_ids, num_tokens, discard):
        assert tokens is sampled
        assert batch is runner.input_batch
        assert token_ids is runner.token_ids_gpu_tensor
        assert num_tokens is runner.num_tokens_no_spec_gpu
        assert discard is runner.discard_request_mask.gpu
        events.append(("update_ngram", tokens))
        masked = tokens.clone()
        masked[discard] = -1
        return torch.tensor([11, 9]), torch.tensor([1, 0]), masked

    def copy_counts(next_ids, counts):
        assert next_ids.tolist() == [11, 9]
        assert counts.tolist() == [1, 0]
        events.append(("copy_counts", None))

    def finish_dispatch():
        raise _DispatchCompleteError

    runner.propose_draft_token_ids = propose
    runner._copy_draft_token_ids_to_cpu = copy_drafts
    runner._bookkeeping_sync = bookkeep
    runner._copy_valid_sampled_token_count = copy_counts
    runner._should_use_async_omni_output = finish_dispatch
    if method == "ngram_gpu":
        monkeypatch.setattr(runner.drafter, "update_token_ids_ngram", update)
    monkeypatch.setattr(ar, "get_pp_group", lambda: SimpleNamespace(world_size=1, is_last_rank=True))
    return runner, sampled, cpu_rows, events


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_padded_gpu_ngram_uses_current_tensor_before_bookkeeping(monkeypatch, async_scheduling):
    runner, sampled, _, events = _runner(monkeypatch, method="ngram_gpu", async_scheduling=async_scheduling)
    with pytest.raises(_DispatchCompleteError):
        runner.sample_tokens(None)
    assert [event for event, _ in events] == ["propose", "copy_drafts", "bookkeep", "finalize_kv"]
    assert events[0][1] is sampled


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("count_event", [False, True])
@pytest.mark.parametrize("method", ["ngram", "ngram_gpu"])
def test_ngram_over_budget_clears_drafts_before_bookkeeping(monkeypatch, async_scheduling, count_event, method):
    runner, _, _, events = _runner(
        monkeypatch, method=method, async_scheduling=async_scheduling, fits=False, count_event=count_event
    )
    with pytest.raises(_DispatchCompleteError):
        runner.sample_tokens(None)
    expected = (["update_ngram", "copy_counts"] if method == "ngram_gpu" and count_event else []) + [
        "copy_zeros",
        "bookkeep",
        "finalize_kv",
    ]
    assert [event for event, _ in events] == expected
    assert runner._draft_probs is None
    assert runner._draft_prob_req_ids is None


@pytest.mark.parametrize("method,padded", [("ngram", True), ("ngram_gpu", False)])
def test_cpu_list_dispatch_stays_after_bookkeeping(monkeypatch, method, padded):
    runner, _, cpu_rows, events = _runner(monkeypatch, method=method, async_scheduling=False, padded=padded)
    with pytest.raises(_DispatchCompleteError):
        runner.sample_tokens(None)
    assert [event for event, _ in events] == ["bookkeep", "propose", "copy_drafts", "finalize_kv"]
    assert events[1][1] is cpu_rows


@pytest.mark.parametrize("method", ["eagle", "draft"])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_gpu_proposer_false_fit_without_count_event_clears_drafts(monkeypatch, method, async_scheduling):
    runner, _, _, events = _runner(
        monkeypatch, method=method, async_scheduling=async_scheduling, fits=False, count_event=False
    )
    with pytest.raises(_DispatchCompleteError):
        runner.sample_tokens(None)
    assert [event for event, _ in events] == ["copy_zeros", "bookkeep", "finalize_kv"]
