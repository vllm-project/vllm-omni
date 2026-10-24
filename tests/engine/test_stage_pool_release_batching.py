# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest
from vllm import SamplingParams

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.orchestrator import OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.errors import OmniClientError
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Replica:
    stage_type = "llm"
    final_output = False

    def __init__(self, *, hang: bool = False, fail: bool = False):
        self.calls: list[list[str]] = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        if not hang:
            self.gate.set()
        self.fail = fail
        self.admitted: list[str] = []

    async def add_request_async(self, request, **kwargs):
        self.admitted.append(request.external_req_id)

    async def call_utility_async(self, method, request_ids):
        assert method == "omni_release_request_resources"
        self.calls.append(list(request_ids))
        self.started.set()
        await self.gate.wait()
        if self.fail:
            raise RuntimeError("replica gone")


@pytest.fixture
def make_pool(mocker):
    config = mocker.Mock(model_config=mocker.Mock(async_chunk=True))
    return lambda *replicas: StagePool(0, list(replicas), stage_vllm_config=config)


async def _settle(pool: StagePool) -> None:
    await asyncio.wait_for(asyncio.gather(*pool._release_flushers.values()), timeout=2.0)


async def _submit(pool: StagePool, request_id: str, replica_id: int = 0) -> int:
    params = SamplingParams(max_tokens=2, detokenize=False)
    request = OmniEngineCoreRequest(
        request_id=f"{request_id}-internal",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        external_req_id=request_id,
    )
    state = OrchestratorRequestState(request_id=request_id, sampling_params_list=[params])
    return await pool.submit_initial(request_id, state, request, pinned_replica_id=replica_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("different_replica", [False, True])
async def test_reused_id_requires_queued_release_on_every_replica(make_pool, different_replica):
    old, other = _Replica(hang=True), _Replica()
    pool = make_pool(old, other)
    pool._output_processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    pool.schedule_release_request_resources(["unrelated"])
    await asyncio.wait_for(old.started.wait(), timeout=2)
    pool.schedule_release_request_resources(["reused"])
    try:
        with pytest.raises(OmniClientError, match="release.*pending") as exc:
            await _submit(pool, "reused", replica_id=int(different_replica))
        assert exc.value.status_code == 409
        assert not old.admitted and not other.admitted
        assert not pool.output_processor.request_states
    finally:
        old.gate.set()
        await _settle(pool)
    assert old.calls == [["unrelated"], ["reused"]]
    assert await _submit(pool, "reused", replica_id=int(different_replica)) == int(different_replica)
    assert (other if different_replica else old).admitted == ["reused"]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_unknown_release_blocks_only_its_id_without_retry(make_pool, monkeypatch, timeout):
    replica = _Replica(hang=timeout, fail=not timeout)
    pool = make_pool(replica)
    pool._output_processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    monkeypatch.setattr(pool, "RELEASE_RPC_TIMEOUT_S", 0.02)
    pool.schedule_release_request_resources(["reused"])
    await _settle(pool)
    replica.fail = False
    replica.gate.set()
    with pytest.raises(OmniClientError, match="release.*unknown") as exc:
        await _submit(pool, "reused")
    assert exc.value.status_code == 409
    assert not pool.output_processor.request_states
    assert await _submit(pool, "healthy") == 0
    pool.schedule_release_request_resources(["reused", "healthy"])
    await _settle(pool)
    assert replica.calls == [["reused"], ["healthy"]]
    assert replica.admitted == ["healthy"]


@pytest.mark.asyncio
async def test_reused_id_requires_inflight_release(make_pool):
    replica = _Replica(hang=True)
    pool = make_pool(replica)
    pool._output_processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    pool.schedule_release_request_resources(["reused"])
    await asyncio.wait_for(replica.started.wait(), timeout=2)
    try:
        with pytest.raises(OmniClientError, match="release.*pending") as exc:
            await _submit(pool, "reused")
        assert exc.value.status_code == 409
        assert not pool.output_processor.request_states
    finally:
        replica.gate.set()
        await _settle(pool)
    assert await _submit(pool, "reused") == 0
    assert replica.admitted == ["reused"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "async_chunk,use_v2,native,expected",
    [(False, False, False, False), (True, False, False, True), (False, True, True, True), (False, True, False, False)],
)
async def test_release_includes_native_full_payload_but_preserves_legacy_gate(
    mocker, async_chunk, use_v2, native, expected
) -> None:
    config = mocker.Mock(
        model_config=mocker.Mock(
            async_chunk=async_chunk, use_v2_model_runner=use_v2, supports_native_mrv2_data_plane=native
        )
    )
    replica = _Replica()
    pool = StagePool(0, [replica], stage_vllm_config=config)

    pool.schedule_release_request_resources(["request"])
    await _settle(pool)

    assert replica.calls == ([["request"]] if expected else [])
    assert not pool._pending_releases and not pool._release_flushers


@pytest.mark.asyncio
async def test_hung_replica_does_not_delay_healthy_replica_batches(monkeypatch, make_pool) -> None:
    hung, healthy = _Replica(hang=True), _Replica()
    pool = make_pool(hung, healthy)
    monkeypatch.setattr(pool, "RELEASE_RPC_TIMEOUT_S", 0.2)
    warn = []
    monkeypatch.setattr("vllm_omni.engine.stage_pool.logger.warning", lambda *a: warn.append(a))

    pool.schedule_release_request_resources(["a"])
    pool.schedule_release_request_resources(["a"])
    await asyncio.wait_for(healthy.started.wait(), timeout=2.0)
    healthy.started.clear()
    pool.schedule_release_request_resources(["b"])
    pool.schedule_release_request_resources(["c", "b"])
    await asyncio.wait_for(healthy.started.wait(), timeout=2.0)

    assert healthy.calls == [["a"], ["b", "c"]]
    assert hung.calls == [["a"]]

    await _settle(pool)
    assert hung.calls == [["a"], ["b", "c"]]
    assert len(warn) == 2  # both of the hung replica's RPCs timed out


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [False, True])
async def test_replica_exit_retains_unconfirmed_releases(monkeypatch, make_pool, removed) -> None:
    replica = _Replica(hang=True, fail=not removed)
    pool = make_pool(replica)

    pool.schedule_release_request_resources(["a"])
    await asyncio.wait_for(replica.started.wait(), timeout=2.0)
    pool.schedule_release_request_resources(["b"])
    warn = []
    monkeypatch.setattr("vllm_omni.engine.stage_pool.logger.warning", lambda *a: warn.append(a))
    if removed:
        pool.clients[0] = None
    replica.gate.set()
    await _settle(pool)

    assert replica.calls == ([["a"]] if removed else [["a"], ["b"]])
    assert pool._pending_releases == {0: ({"b": True} if removed else {"a": True, "b": True})}
    assert not pool._release_flushers
    replica.fail = False
    pool.schedule_release_request_resources(["c"])
    await _settle(pool)
    assert len(warn) == (0 if removed else 2)
    if not removed:
        assert replica.calls == [["a"], ["b"], ["c"]]
        assert pool._pending_releases == {0: {"a": True, "b": True}}
