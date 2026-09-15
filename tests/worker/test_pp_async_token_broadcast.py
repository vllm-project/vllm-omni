# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Async scheduling + pipeline parallelism: the last rank must send what earlier ranks wait for.

vLLM's ``GPUModelRunner.sample_tokens`` pairs two collectives on the PP device group:
a non-final rank posts an async broadcast *receive* of the newly sampled token ids
(``_pp_receive_prev_sampled_token_ids_to_input_batch``) and the final rank performs
the matching broadcast *send* (``_pp_broadcast_prev_sampled_token_ids``) right after
``_update_states_after_model_execute``. ``GPUARModelRunner.sample_tokens`` overrides the
upstream method and inherited the receive but not the send. With async scheduling and
PP > 1 the first rank never returned from posting that receive: it is the first
collective on the PP group's communicator, NCCL has to create the communicator with every
rank, and the last rank never joined (observed live on 2x A100: rank 0 inside
``torch.distributed.broadcast``, rank 1 waiting for the next intermediate tensors, engine
dead after the RPC timeout). Had the communicator existed, the same missing send would
have blocked the ``_pp_recv_work.wait()`` in ``_prepare_input_ids`` on the next step.

These tests drive the real Omni override with a recorded ``torch.distributed.broadcast``
ledger and, separately, a real two-process gloo group so the receive is proven to
complete only because the Omni sampling path sent.
"""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker import gpu_model_runner as upstream_runner_module

from vllm_omni.worker import gpu_ar_model_runner as omni_runner_module
from vllm_omni.worker.gpu_ar_model_runner import ExecuteModelState, GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.parallel]


class _StopAtBookkeepingError(Exception):
    """Raised by the bookkeeping double: everything before it is what we assert on."""


def _pp(rank: int, world_size: int, device_group: object = "pp-group") -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        world_size=world_size,
        last_rank=world_size - 1,
        is_last_rank=rank == world_size - 1,
        is_first_rank=rank == 0,
        device_group=device_group,
    )


def _input_batch(num_reqs: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=[f"req-{i}" for i in range(num_reqs)],
        num_tokens_no_spec=np.ones(num_reqs, dtype=np.int64),
        is_token_ids=np.zeros((num_reqs, 8), dtype=bool),
        sampling_metadata=SimpleNamespace(no_penalties=True, logitsprocs=SimpleNamespace(non_argmax_invariant=[])),
        vocab_size=32,
        prev_sampled_token_ids=None,
        prev_req_id_to_index=None,
    )


def _last_rank_runner(
    *,
    sampled: torch.Tensor,
    async_scheduling: bool = True,
    broadcast_pp_output: bool = False,
    all_chunked_prefill: bool = False,
    events: list[str],
) -> GPUARModelRunner:
    """A final-rank runner double that reaches the sampler and the state update."""
    num_reqs = sampled.shape[0]
    hidden = torch.ones(num_reqs, 8)
    runner = object.__new__(GPUARModelRunner)
    runner.use_async_scheduling = async_scheduling
    runner.broadcast_pp_output = broadcast_pp_output
    runner.speculative_config = None
    runner.device = torch.device("cpu")
    runner.input_batch = _input_batch(num_reqs)
    runner.discard_request_mask = SimpleNamespace(np=np.full(num_reqs, all_chunked_prefill, dtype=bool))
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=SimpleNamespace(total_num_scheduled_tokens=num_reqs),
        logits=torch.zeros(num_reqs, 32),
        spec_decode_metadata=None,
        spec_decode_common_attn_metadata=None,
        hidden_states=hidden,
        hidden_states_cpu=None,
        sample_hidden_states=hidden,
        aux_hidden_states=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        multimodal_outputs={},
        slot_mappings=None,
    )
    runner._sample = lambda logits, spec: SimpleNamespace(sampled_token_ids=sampled, logprobs_tensors=None)
    runner._update_states_after_model_execute = lambda *args: events.append("state-update")

    def stop(*args):
        raise _StopAtBookkeepingError

    runner._bookkeeping_sync = stop
    return runner


def _non_last_rank_runner(num_reqs: int) -> GPUARModelRunner:
    runner = object.__new__(GPUARModelRunner)
    runner.use_async_scheduling = True
    runner.execute_model_state = None
    runner.kv_connector_output = None
    runner.device = torch.device("cpu")
    runner._pp_recv_work = None  # as vLLM's __init__ leaves it before any receive is posted
    runner.input_batch = _input_batch(num_reqs)
    runner.discard_request_mask = SimpleNamespace(np=np.zeros(num_reqs, dtype=bool))
    runner.requests = {rid: SimpleNamespace(output_token_ids=[]) for rid in runner.input_batch.req_ids}
    runner.attach_omni_connector_output = lambda output: output
    return runner


class _Ledger(list):
    """Every torch.distributed.broadcast, plus a shared timeline with the state update."""

    def __init__(self):
        super().__init__()
        self.timeline: list[str] = []


@pytest.fixture
def ledger(monkeypatch):
    calls = _Ledger()

    class Pending:
        def wait(self):
            if not any(c["kind"] == "send" for c in calls):
                raise RuntimeError("receive has no matching broadcast")

    def broadcast(tensor, src, group, async_op=False):
        kind = "receive" if async_op else "send"
        calls.append(
            dict(
                kind=kind,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                src=src,
                group=group,
                value=tensor.clone(),
            )
        )
        calls.timeline.append(kind)
        return Pending() if async_op else None

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    return calls


def _use_pp(monkeypatch, pp):
    # The guard lives in the Omni override; the sender/receiver bodies live in vLLM.
    monkeypatch.setattr(omni_runner_module, "get_pp_group", lambda: pp)
    monkeypatch.setattr(upstream_runner_module, "get_pp_group", lambda: pp)


def test_non_last_rank_posts_the_receive_before_the_connector_only_return(monkeypatch, ledger):
    _use_pp(monkeypatch, _pp(rank=0, world_size=2))
    runner = _non_last_rank_runner(num_reqs=2)
    output = runner.sample_tokens(None)
    assert output is not None
    assert [c["kind"] for c in ledger] == ["receive"]
    assert ledger[0]["src"] == 1 and ledger[0]["group"] == "pp-group"
    assert ledger[0]["shape"] == (2, 1) and ledger[0]["dtype"] == torch.int32
    assert runner._pp_recv_work is not None
    assert runner.input_batch.prev_sampled_token_ids.shape == (2, 1)
    assert runner.input_batch.prev_req_id_to_index == {"req-0": 0, "req-1": 1}


def test_last_rank_broadcasts_sampled_tokens_after_the_state_update(monkeypatch, ledger):
    pp = _pp(rank=1, world_size=2)
    _use_pp(monkeypatch, pp)
    sampled = torch.tensor([[4], [7]], dtype=torch.int32)
    runner = _last_rank_runner(sampled=sampled, events=ledger.timeline)
    with pytest.raises(_StopAtBookkeepingError):
        runner.sample_tokens(None)
    assert ledger.timeline == ["state-update", "send"], "the send must follow the state update, not precede it"
    sends = [c for c in ledger if c["kind"] == "send"]
    assert len(sends) == 1
    assert sends[0]["src"] == pp.rank == pp.last_rank
    assert sends[0]["group"] == "pp-group"
    assert sends[0]["shape"] == (2, 1) and sends[0]["dtype"] == torch.int32
    assert torch.equal(sends[0]["value"], sampled)


def test_receive_and_send_form_one_matched_pair(monkeypatch, ledger):
    """The negative control from the handoff probe, now expected to complete."""
    _use_pp(monkeypatch, _pp(rank=0, world_size=2))
    first = _non_last_rank_runner(num_reqs=1)
    first.sample_tokens(None)
    _use_pp(monkeypatch, _pp(rank=1, world_size=2))
    last = _last_rank_runner(sampled=torch.tensor([[4]], dtype=torch.int32), events=[])
    with pytest.raises(_StopAtBookkeepingError):
        last.sample_tokens(None)
    first._pp_recv_work.wait()
    assert [c["kind"] for c in ledger] == ["receive", "send"]
    assert ledger[0]["shape"] == ledger[1]["shape"] and ledger[0]["dtype"] == ledger[1]["dtype"]
    assert ledger[0]["src"] == ledger[1]["src"] and ledger[0]["group"] == ledger[1]["group"]


@pytest.mark.parametrize(
    ("async_scheduling", "broadcast_pp_output", "world_size", "reason"),
    [
        (False, False, 2, "synchronous scheduling feeds tokens through the scheduler"),
        (True, True, 2, "external_launcher already broadcast the logits to every rank"),
        (True, False, 1, "a single rank has nobody waiting"),
    ],
)
def test_last_rank_does_not_send_outside_the_async_pp_case(
    monkeypatch, ledger, async_scheduling, broadcast_pp_output, world_size, reason
):
    _use_pp(monkeypatch, _pp(rank=world_size - 1, world_size=world_size))
    runner = _last_rank_runner(
        sampled=torch.tensor([[4]], dtype=torch.int32),
        async_scheduling=async_scheduling,
        broadcast_pp_output=broadcast_pp_output,
        events=[],
    )
    with pytest.raises(_StopAtBookkeepingError):
        runner.sample_tokens(None)
    assert ledger == [], reason


def test_both_sides_skip_an_all_chunked_prefill_step(monkeypatch, ledger):
    """Dummy tokens from unfinished prefill chunks are discarded, so neither side talks."""
    _use_pp(monkeypatch, _pp(rank=0, world_size=2))
    first = _non_last_rank_runner(num_reqs=1)
    first.discard_request_mask = SimpleNamespace(np=np.ones(1, dtype=bool))
    first.sample_tokens(None)
    assert first._pp_recv_work is None
    _use_pp(monkeypatch, _pp(rank=1, world_size=2))
    last = _last_rank_runner(sampled=torch.tensor([[4]], dtype=torch.int32), all_chunked_prefill=True, events=[])
    with pytest.raises(_StopAtBookkeepingError):
        last.sample_tokens(None)
    assert ledger == []


def test_sender_rejects_multi_token_rows(monkeypatch, ledger):
    """The receive buffer is [num_reqs, 1]; the upstream sender guards that contract."""
    _use_pp(monkeypatch, _pp(rank=1, world_size=2))
    runner = _last_rank_runner(sampled=torch.tensor([[4, 5]], dtype=torch.int32), events=[])
    with pytest.raises(AssertionError, match="num_reqs, 1"):
        runner.sample_tokens(None)
    assert ledger == []


# --- real collective -----------------------------------------------------------------


def _gloo_rank(rank: int, world_size: int, rendezvous: str, result_path: str) -> None:
    """Rank 0 runs the Omni receive branch; rank 1 runs the Omni sampling path.

    Both use the real methods bound to doubles; the process group is a real gloo group,
    so rank 0's wait can only return because rank 1's ``sample_tokens`` broadcast.
    """
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        pp = _pp(rank=rank, world_size=world_size, device_group=dist.group.WORLD)
        omni_runner_module.get_pp_group = lambda: pp
        upstream_runner_module.get_pp_group = lambda: pp
        if rank == world_size - 1:
            runner = _last_rank_runner(sampled=torch.tensor([[4], [7]], dtype=torch.int32), events=[])
            try:
                runner.sample_tokens(None)
            except _StopAtBookkeepingError:
                pass
            received = None
        else:
            runner = _non_last_rank_runner(num_reqs=2)
            runner.sample_tokens(None)
            assert runner._pp_recv_work is not None
            # A missing send surfaces here as a gloo timeout instead of a silent hang.
            runner._pp_recv_work.wait(timeout=timedelta(seconds=30))
            received = runner.input_batch.prev_sampled_token_ids.tolist()
        with open(f"{result_path}.{rank}", "w") as handle:
            handle.write(repr(received))
    finally:
        dist.destroy_process_group()


def test_gloo_pp2_receive_completes_only_because_the_omni_sampling_path_sent(tmp_path):
    import torch.multiprocessing as mp

    rendezvous = str(tmp_path / "rendezvous")
    result_path = str(tmp_path / "received")
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_gloo_rank, args=(rank, 2, rendezvous, result_path), daemon=True) for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=240)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=30)
    exit_codes = [process.exitcode for process in processes]
    assert exit_codes == [0, 0], f"rank exit codes {exit_codes}: an unmatched receive times out on rank 0"
    with open(f"{result_path}.0") as handle:
        assert handle.read() == repr([[4], [7]])
    assert os.path.exists(f"{result_path}.1")
