# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coalesced resumable updates: one ADD per replica per loop turn, params sent once per request id.

The scheduler requests a stage engine core builds from what ``StageEngineCoreClient``
sends must equal the ones it builds when every update carries its params in full,
through the real msgpack wire and vLLM's own request preprocessing.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc, _CoalescedAdds
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ADD = EngineCoreRequestType.ADD
_ABORT = EngineCoreRequestType.ABORT
_ENCODER = MsgpackEncoder()
_DECODER = MsgpackDecoder(OmniEngineCoreRequest)


def _params(temperature: float = 0.8) -> SamplingParams:
    return SamplingParams(max_tokens=16, temperature=temperature, top_k=25, seed=7, stop_token_ids=[3, 5])


def _update(request_id: str, seq: int, params: SamplingParams, *, resumable: bool = True) -> OmniEngineCoreRequest:
    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[seq, seq + 1],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=float(seq),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
        external_req_id=request_id,
        model_intermediate_buffer={"duplex": {"seq": seq}},
    )


@pytest.fixture
def record_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each client keeps what it sends: ADDs as their wire frames, at send time."""

    def record(self, request_type, request, engine=None):
        self.sent.append((request_type, _ENCODER.encode(request) if request_type == _ADD else request))
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    monkeypatch.setattr(AsyncMPClient, "_send_input", record)


def _client(*, full_params: bool = False) -> StageEngineCoreClient:
    client = object.__new__(StageEngineCoreClient)
    client.client_index = 1
    client.stage_id = 0
    client.replica_id = 0
    client._ensure_output_queue_task = lambda: None
    client.sent = []  # type: ignore[attr-defined]
    if full_params:
        # The path before references: every update carries its params.
        client._refer_to_held_sampling_params = lambda request: request  # type: ignore[method-assign]
    return client


class _EngineInput:
    """A stage engine core's input thread: decode each ADD and preprocess it as vLLM does."""

    def __init__(self) -> None:
        self.proc = object.__new__(StageEngineCoreProc)
        self.proc.mm_receiver_cache = None
        self.proc.request_block_hasher = None
        self.failed: list[tuple[str, str]] = []
        self.proc._handle_request_preproc_error = self._failed  # type: ignore[method-assign]

    def _failed(self, request: Any) -> None:
        self.failed.append((request.request_id, str(sys.exc_info()[1])))

    def add(self, frames: Any) -> list[Any]:
        request = _DECODER.decode(frames)
        try:
            preprocessed, _wave = self.proc.preprocess_add_request(request)
        except Exception:
            # EngineCoreProc.process_input_sockets
            self.proc._handle_request_preproc_error(request)
            return []
        if isinstance(preprocessed, _CoalescedAdds):
            return [item for item, _wave in preprocessed]
        return [preprocessed]

    def replay(self, sent: list[tuple[EngineCoreRequestType, Any]]) -> list[Any]:
        # Aborts are the busy loop's; the input thread only adds.
        return [request for kind, frames in sent if kind == _ADD for request in self.add(frames)]


def _record(request: Any) -> tuple[Any, ...]:
    """What the scheduler gets from one preprocessed ADD."""
    return (
        request.request_id,
        request.external_req_id,
        request.client_index,
        list(request.prompt_token_ids),
        request.arrival_time,
        request.resumable,
        request.max_tokens,
        request.sampling_params,
        request.kv_transfer_params,
        request.status,
        request.additional_information,
    )


def _wire(sent: list[tuple[EngineCoreRequestType, Any]]) -> list[tuple[Any, ...]]:
    """Which updates carried their params ("full") and which only a reference ("ref")."""
    pattern: list[tuple[Any, ...]] = []
    for kind, frames in sent:
        if kind == _ABORT:
            pattern.append(("abort", *frames))
            continue
        message = _DECODER.decode(frames)
        if message.released_sampling_params_ids:
            pattern.append(("release", *message.released_sampling_params_ids))
        for item in message.coalesced_requests or [message]:
            seq = item.model_intermediate_buffer["duplex"]["seq"]
            pattern.append((item.request_id, seq, "ref" if item.sampling_params is None else "full"))
    return pattern


async def _drive(client: StageEngineCoreClient) -> None:
    params = {"p": _params(), "p_equal": _params(), "q": _params(temperature=0.5)}

    async def turn(*updates: tuple[Any, ...]) -> None:
        for request_id, seq, name, *ends in updates:
            client.add_request_coalesced(_update(request_id, seq, params[name], resumable=not ends))
        await asyncio.sleep(0)

    # Two sessions open; then equal params, the same object or a new one, are referred to.
    await turn(("s0.e0", 1, "p"), ("s1.e0", 1, "p"))
    await turn(("s0.e0", 2, "p"), ("s1.e0", 2, "p_equal"))
    # A config change sends the new params once.
    await turn(("s0.e0", 3, "q"), ("s1.e0", 3, "p"))
    await turn(("s0.e0", 4, "q"))
    # Abort and reopen the same request id.
    await client._send_input(_ABORT, ["s1.e0"])
    await turn(("s1.e0", 1, "p"), ("s0.e0", 5, "q"))
    await turn(("s1.e0", 2, "p"))
    # A new epoch is a new request id; the aborted one's params are released.
    await client._send_input(_ABORT, ["s0.e0"])
    await turn(("s0.e1", 1, "q"))
    await turn(("s0.e1", 2, "q"), ("s1.e0", 3, "p"))
    # Params changed in place are sent again.
    params["p"].temperature = 0.3
    await turn(("s1.e0", 4, "p"), ("s0.e1", 3, "q"))
    await turn(("s1.e0", 5, "p"), ("s0.e1", 4, "q"))
    # A stream end, sent on its own or queued between updates, drops the held params.
    await client.add_request_async(_update("s0.e1", 5, params["q"], resumable=False))
    await turn(("s0.e1", 1, "q"), ("s1.e0", 6, "p"))
    await turn(("s1.e0", 7, "p"), ("s1.e0", 8, "p", "ends"), ("s1.e0", 1, "p"))
    await turn(("s1.e0", 2, "p"), ("s0.e1", 2, "q"))


@pytest.mark.asyncio
@pytest.mark.usefixtures("record_sends")
async def test_the_scheduler_gets_the_requests_full_params_would_give_it() -> None:
    client = _client()
    baseline = _client(full_params=True)
    await _drive(client)
    await _drive(baseline)

    engine = _EngineInput()
    baseline_engine = _EngineInput()
    added = engine.replay(client.sent)
    expected = baseline_engine.replay(baseline.sent)

    assert engine.failed == [] and baseline_engine.failed == []
    assert len(added) == len(expected) == 25
    assert [_record(request) for request in added] == [_record(request) for request in expected]

    assert _wire(client.sent) == [
        ("s0.e0", 1, "full"),
        ("s1.e0", 1, "full"),
        ("s0.e0", 2, "ref"),
        ("s1.e0", 2, "ref"),
        ("s0.e0", 3, "full"),
        ("s1.e0", 3, "ref"),
        ("s0.e0", 4, "ref"),
        ("abort", "s1.e0"),
        ("release", "s1.e0"),
        ("s1.e0", 1, "full"),
        ("s0.e0", 5, "ref"),
        ("s1.e0", 2, "ref"),
        ("abort", "s0.e0"),
        ("release", "s0.e0"),
        ("s0.e1", 1, "full"),
        ("s0.e1", 2, "ref"),
        ("s1.e0", 3, "ref"),
        ("s1.e0", 4, "full"),
        ("s0.e1", 3, "ref"),
        ("s1.e0", 5, "ref"),
        ("s0.e1", 4, "ref"),
        ("s0.e1", 5, "full"),
        ("s0.e1", 1, "full"),
        ("s1.e0", 6, "ref"),
        ("s1.e0", 7, "ref"),
        ("s1.e0", 8, "full"),
        ("s1.e0", 1, "full"),
        ("s1.e0", 2, "ref"),
        ("s0.e1", 2, "ref"),
    ]

    # Both ends agree on what is held, and nothing aborted or ended is left.
    assert {rid: ref for rid, (ref, _) in engine.proc._held_sampling_params.items()} == {
        rid: ref for rid, (ref, _) in client._held_sampling_params.items()
    }
    assert set(client._held_sampling_params) == {"s0.e1", "s1.e0"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("record_sends")
async def test_requests_sent_outside_the_coalesced_path_are_untouched() -> None:
    """MiniCPM-o and every non-duplex stage add requests one by one: full params, no reference."""
    client = _client()
    first = _update("m0", 1, _params())
    await client.add_request_async(first)
    await client.add_request_async(_update("m0", 2, _params()))
    await client._send_input(_ABORT, ["m0"])
    await client.add_request_async(_update("m1", 1, _params(), resumable=False))

    assert client.sent[0] == (_ADD, _ENCODER.encode(first))
    assert first.sampling_params_ref is None
    assert _wire(client.sent) == [("m0", 1, "full"), ("m0", 2, "full"), ("abort", "m0"), ("m1", 1, "full")]
    assert client._held_sampling_params is None
    assert client._released_sampling_params_ids is None

    engine = _EngineInput()
    assert len(engine.replay(client.sent)) == 3
    assert engine.failed == []
    assert engine.proc._held_sampling_params is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("record_sends")
async def test_updates_of_one_turn_go_out_as_one_add_before_any_other_send() -> None:
    client = _client()
    for index in range(3):
        client.add_request_coalesced(_update(f"r{index}", 1, _params()))
    assert client.sent == []  # nothing leaves before the callbacks of this turn ran

    await client._send_input(_ABORT, ["r0"])
    await asyncio.sleep(0)

    assert [kind for kind, _ in client.sent] == [_ADD, _ABORT]
    carrier = _DECODER.decode(client.sent[0][1])
    assert [item.request_id for item in carrier.coalesced_requests] == ["r0", "r1", "r2"]
    assert carrier.client_index == 1
    assert all(item.client_index == 1 for item in carrier.coalesced_requests)


def test_engine_core_handles_each_carried_request_like_a_separate_add(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = object.__new__(StageEngineCoreProc)
    handled: list[tuple[object, object]] = []
    monkeypatch.setattr(
        EngineCoreProc, "_handle_client_request", lambda self, kind, request: handled.append((kind, request))
    )

    proc._handle_client_request(_ADD, (_CoalescedAdds([("r0", 7), ("r1", 7)]), 7))
    proc._handle_client_request(_ADD, ("r2", 7))

    assert handled == [(_ADD, ("r0", 7)), (_ADD, ("r1", 7)), (_ADD, ("r2", 7))]


@pytest.mark.asyncio
@pytest.mark.parametrize("coalesce", [True, False])
async def test_stage_pool_queues_a_coalesced_update_instead_of_sending_it(coalesce: bool) -> None:
    queued: list[object] = []
    client = SimpleNamespace(
        stage_type="llm",
        add_request_coalesced=queued.append,
        add_request_async=AsyncMock(),
    )
    output_processor = SimpleNamespace(add_request=lambda **kwargs: None)
    pool = StagePool(0, [client], output_processor=output_processor)  # type: ignore[arg-type]
    pool._request_bindings["r0"] = 0
    request = _update("r0", 1, _params())
    req_state = SimpleNamespace(sampling_params_list=[request.sampling_params])

    replica_id = await pool.submit_update("r0", req_state, request, coalesce=coalesce)  # type: ignore[arg-type]

    assert replica_id == 0
    if coalesce:
        assert queued == [request]
        client.add_request_async.assert_not_awaited()
    else:
        assert queued == []
        client.add_request_async.assert_awaited_once_with(request)


def test_only_personaplex_coalesces_by_default() -> None:
    from vllm_omni.engine.duplex.plugin import DuplexModelPlugin
    from vllm_omni.model_executor.models.personaplex.duplex.plugin import PersonaPlexDuplexPlugin

    assert DuplexModelPlugin.coalesces_resumable_updates is False
    assert PersonaPlexDuplexPlugin.coalesces_resumable_updates is True
