# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lifecycle of a Noisy KV session, and the runner's use of it."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.cell_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.session import (
    NoisyKVSession,
    NoisyKVSessionRegistry,
    NoisySessionError,
    NoisySessionState,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(request_id: str = "r1", **extra: object) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="a world",
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, extra_args={"session_id": "s1", **extra}),
    )


class _Capability:
    """Cell capability whose context binding does nothing."""

    def bind_ar_diffusion_cell_context(self, ctx: object):
        del ctx
        return nullcontext()


def _runner(mocker, *, execute=None) -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.noisy_kv_cache = object()
    runner._ar_diffusion_cell_capability = _Capability()
    runner._noisy_sessions = NoisyKVSessionRegistry(capacity=1)
    runner._perf_e2e_times = []
    ctx = SimpleNamespace(inflight=[], pending=[])
    mocker.patch.object(ARDiffusionModelRunner, "_make_chunk_context", return_value=ctx)
    if execute is not None:
        mocker.patch.object(DiffusionModelRunner, "execute_model", side_effect=execute)
    return runner


def test_commit_advances_the_continuation_offset():
    registry = NoisyKVSessionRegistry(capacity=1)
    session, generation = registry.admit("s1")
    assert session.committed_chunks == 0
    assert not session.is_resumable

    session.commit_chunks(3, generation=generation)

    assert session.committed_chunks == 3
    assert session.is_resumable


def test_stale_generation_cannot_commit_after_reset():
    registry = NoisyKVSessionRegistry(capacity=1)
    session, generation = registry.admit("s1")
    session.commit_chunks(2, generation=generation)

    registry.reset("s1")

    assert session.committed_chunks == 0
    with pytest.raises(NoisySessionError, match="stale commit"):
        session.commit_chunks(2, generation=generation)


def test_failed_request_invalidates_until_reset():
    registry = NoisyKVSessionRegistry(capacity=1)
    session, _ = registry.admit("s1")

    registry.inactivate("s1", "chunk forward failed")

    assert session.state is NoisySessionState.INVALID
    with pytest.raises(NoisySessionError, match="invalid"):
        registry.admit("s1")

    registry.reset("s1")

    assert session.state is NoisySessionState.ACTIVE
    assert registry.admit("s1")[0] is session


def test_closed_session_rejects_requests_and_reset():
    registry = NoisyKVSessionRegistry(capacity=1)
    session, _ = registry.admit("s1")
    registry.close("s1")

    assert session.state is NoisySessionState.CLOSED
    with pytest.raises(NoisySessionError, match="closed"):
        session.reset()


def test_registry_evicts_the_least_recently_used_session():
    registry = NoisyKVSessionRegistry(capacity=2)
    registry.admit("old")
    registry.admit("keep")
    registry.admit("keep")  # refresh recency

    registry.admit("new")

    assert [s.session_id for s in registry] == ["keep", "new"]


def test_two_sessions_keep_independent_offsets():
    registry = NoisyKVSessionRegistry(capacity=2)
    first, first_generation = registry.admit("a")
    second, second_generation = registry.admit("b")

    first.commit_chunks(4, generation=first_generation)

    assert first.committed_chunks == 4
    assert second.committed_chunks == 0
    second.commit_chunks(1, generation=second_generation)
    assert (first.committed_chunks, second.committed_chunks) == (4, 1)


def test_runner_invalidates_the_session_when_the_chunk_forward_fails(mocker):
    runner = _runner(mocker, execute=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        runner.execute_model(_request())

    session = runner._noisy_sessions.get("s1")
    assert session is not None
    assert session.state is NoisySessionState.INVALID
    # A poisoned session refuses the next request instead of resuming blindly.
    with pytest.raises(NoisySessionError):
        runner.execute_model(_request())


def test_runner_keeps_the_session_active_after_a_successful_chunk_forward(mocker):
    output = DiffusionOutput(output=None)
    runner = _runner(mocker, execute=lambda *_a, **_k: output)

    assert runner.execute_model(_request()) is output

    session = runner._noisy_sessions.get("s1")
    assert session is not None
    assert session.state is NoisySessionState.ACTIVE


def test_runner_closes_the_session_when_the_request_asks(mocker):
    output = DiffusionOutput(output=None)
    runner = _runner(mocker, execute=lambda *_a, **_k: output)

    runner.execute_model(_request(close_session=True))

    assert runner._noisy_sessions.get("s1") is None


class _SeqRecorder(CellAdapter):
    """Records the sequence length each cell's KV view exposes."""

    def __init__(self) -> None:
        self.seq_lens: dict[tuple, int] = {}

    def forward(self, tasks, kv_contexts, *, hidden):
        del hidden
        for (_req, cell), ctx_list in zip(tasks, kv_contexts):
            view = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            self.seq_lens[cell] = int(view.seq_len)
        return torch.zeros(1, 4, 8)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def _cache() -> NoisyKVCache:
    return NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )


def _run_request(state: NoisyKVState, session: NoisyKVSession, req: str, chunk_start: int) -> _SeqRecorder:
    plan = build_cell_plan(
        CellSchedule(
            chunks=2,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )
    )
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=state,
        chunk_start=chunk_start,
        session=session,
    )
    assert ctx.admit(req, plan, chunk_tokens=4, slot=0)
    adapter = _SeqRecorder()
    run_cell_pipeline(ctx=ctx, adapter=adapter)
    return adapter


def test_second_request_continues_from_session_history():
    cache = _cache()
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")

    first = _run_request(state, session, "A", chunk_start=0)

    assert first.seq_lens[(0, 0, 0)] == 4
    assert session.committed_chunks == 2
    assert {chunk for chunk, _block in session.retained} == {1}

    second = _run_request(state, session, "B", chunk_start=session.committed_chunks)

    # The first cell of the second request reads one retained chunk plus its own
    # write slot, so its KV view is twice as long as a cold cell's.
    assert second.seq_lens[(0, 0, 0)] == 8
    assert session.committed_chunks == 4
    # Retention is bounded by H=1, so only the newest committed chunk survives.
    assert {chunk for chunk, _block in session.retained} == {3}


def test_retained_history_is_released_when_the_cache_is_reset():
    cache = _cache()
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")
    _run_request(state, session, "A", chunk_start=0)

    assert state.resident_versions == 1
    state.reset_all()

    assert state.resident_versions == 0
