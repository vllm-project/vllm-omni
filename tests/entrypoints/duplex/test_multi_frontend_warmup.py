# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Frontend-local startup probes use normal admission and release every slot."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.replica_placement import DuplexReplicaPlacement
from vllm_omni.entrypoints.duplex import warmup
from vllm_omni.protocol.duplex import RealtimeInputDefaults

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Plugin:
    def capabilities(self, *, max_sessions):
        return SimpleNamespace(required_input_modalities={"audio"}, supports_core_resumable_request=True)

    def silence_unit_payload(self):
        return {"audio": "AAAAAA==", "format": "pcm_f32le", "sample_rate_hz": 24000}


class _Handle:
    def __init__(self, owner, session_id):
        self.owner = owner
        self.session_id = session_id
        self.events_queue: asyncio.Queue[SimpleNamespace | None] = asyncio.Queue()
        self.appends = []
        self.closed = False

    async def append_audio(self, audio, **kwargs):
        self.appends.append((audio, kwargs))
        if self.owner.failure == "append":
            raise RuntimeError("append failed")
        if self.owner.failure == "blocked":
            await asyncio.Event().wait()
        if self.owner.failure == "silent":
            return
        event_type = "error" if self.owner.failure == "event" else "response.output_audio.delta"
        self.events_queue.put_nowait(SimpleNamespace(type=event_type))

    async def events(self):
        while (event := await self.events_queue.get()) is not None:
            yield event

    async def close(self, *, reason, timeout):
        self.owner.placement.release(self.session_id)
        self.closed = True
        self.events_queue.put_nowait(None)


class _Omni:
    def __init__(self, replicas, frontends, index, budget=1, *, policy="least_sessions", resume=False, failure=None):
        self.placement = DuplexReplicaPlacement(
            stage_replica_counts=[replicas, replicas],
            max_sessions_per_replica=1,
            frontend_count=frontends,
            frontend_index=index,
        )
        self.engine = SimpleNamespace(
            plugin=_Plugin(),
            duplex_frontend_group_count=self.placement.owned_group_count,
            duplex_frontend_max_sessions=budget,
        )
        self.duplex_session_config = SimpleNamespace(replica_policy=policy)
        self.duplex_capabilities = SimpleNamespace(supports_session_resume=resume)
        self.failure = failure
        self.handles = []
        self.configs = []
        self.normalized_configs = []
        self.groups = []
        self.opened = asyncio.Event()

    async def open_session(self, config, *, timeout):
        # A one-session frontend can still warm all of its replica groups.
        assert all(handle.closed for handle in self.handles)
        assert self.engine.duplex_frontend_max_sessions > 0
        normalized = DuplexSessionConfig.from_realtime(config, model="nvidia/personaplex-7b-v1").normalized()
        session_id = f"warmup-{len(self.handles)}"
        reservation = self.placement.reserve(session_id)
        self.groups.append(reservation.group_id)
        self.configs.append(dict(config))
        self.normalized_configs.append(normalized)
        handle = _Handle(self, session_id)
        self.handles.append(handle)
        self.opened.set()
        return handle


def _app(omni, count):
    return SimpleNamespace(
        state=SimpleNamespace(engine_client=omni, api_server_count=count, duplex_warmup_done=asyncio.Event())
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replicas,frontends,index", [(8, 2, 1), (8, 4, 0), (8, 4, 3), (8, 8, 7), (7, 3, 1)])
async def test_local_warmup_covers_only_owned_groups_without_public_port(monkeypatch, replicas, frontends, index):
    def no_http(*args, **kwargs):
        pytest.fail("local warmup must not use /health or the public websocket")

    monkeypatch.setattr(warmup.httpx, "AsyncClient", no_http)
    omni = _Omni(replicas, frontends, index)
    app = _app(omni, frontends)
    # No port/model fields: the local path must not build a public URL. The
    # actual frontend count on app.state takes precedence over legacy args.
    await warmup._warmup_duplex_realtime(app, SimpleNamespace(api_server_count=1), 1)

    assert omni.groups == list(range(index, replicas, frontends))
    assert all(handle.closed for handle in omni.handles)
    assert all(value == 0 for value in omni.placement.snapshot()["reserved"].values())
    assert all(set(config).isdisjoint({"replica_id", "replica_group", "group_id"}) for config in omni.configs)
    assert omni.handles[0].appends == [(b"\0" * 4, {"format": "pcm_f32le", "sample_rate_hz": 24000})]
    assert app.state.duplex_warmup_done.is_set()


@pytest.mark.asyncio
async def test_local_warmup_config_preserves_personaplex_audio_contract():
    omni = _Omni(8, 8, 3)
    await warmup._warmup_duplex_realtime(_app(omni, 8), SimpleNamespace(), 1)

    (config,) = omni.normalized_configs
    defaults = RealtimeInputDefaults().with_session_payload(config.extra_body["realtime_session_payload"])
    assert config.model == "nvidia/personaplex-7b-v1"
    assert defaults.input_audio_format == "pcm_f32le"
    assert defaults.input_sample_rate_hz == 24000
    assert defaults.output_audio_format == "pcm16"
    assert config.response_format == "pcm"
    assert config.extra_body["auto_response"] is True
    assert config.extra_body["realtime_turn_detection"] is None
    assert config.voice is None  # PersonaPlex supplies its bundled default.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,policy,resume", [(0, "least_sessions", False), (1, "legacy", False), (1, "least_sessions", True)]
)
async def test_unavailable_or_unsupported_frontend_skips_local_warmup(budget, policy, resume):
    omni = _Omni(8, 4, 0, budget, policy=policy, resume=resume)
    app = _app(omni, 4)

    await warmup._warmup_duplex_realtime(app, SimpleNamespace(), 1)

    assert omni.handles == []
    assert app.state.duplex_warmup_done.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["append", "event"])
async def test_failed_local_warmup_closes_session_and_releases_admission(failure):
    omni = _Omni(8, 4, 0, failure=failure)
    app = _app(omni, 4)

    await warmup._warmup_duplex_realtime(app, SimpleNamespace(), 1)

    assert len(omni.handles) == 1
    assert omni.handles[0].closed
    assert all(value == 0 for value in omni.placement.snapshot()["reserved"].values())
    assert app.state.duplex_warmup_done.is_set()


@pytest.mark.asyncio
async def test_cancelled_local_warmup_closes_session_and_releases_admission():
    omni = _Omni(8, 4, 0, failure="blocked")
    app = _app(omni, 4)
    task = asyncio.create_task(warmup._warmup_duplex_realtime(app, SimpleNamespace(), 1))
    await omni.opened.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert omni.handles[0].closed
    assert all(value == 0 for value in omni.placement.snapshot()["reserved"].values())
    assert app.state.duplex_warmup_done.is_set()


@pytest.mark.asyncio
async def test_audio_timeout_releases_slot_before_warming_next_owned_group(monkeypatch):
    monkeypatch.setattr(warmup, "DUPLEX_WARMUP_AUDIO_WAIT_S", 0)
    omni = _Omni(8, 4, 1, failure="silent")
    app = _app(omni, 4)

    await warmup._warmup_duplex_realtime(app, SimpleNamespace(), 1)

    assert omni.groups == [1, 5]
    assert all(handle.closed for handle in omni.handles)
    assert all(value == 0 for value in omni.placement.snapshot()["reserved"].values())
    assert app.state.duplex_warmup_done.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 4])
async def test_public_warmup_flag_only_bypasses_single_frontend_gate(count):
    app = _app(None, count)
    websocket = SimpleNamespace(app=app, query_params={"vllm_omni_warmup": "1"})
    task = asyncio.create_task(warmup.wait_for_duplex_warmup(websocket))
    await asyncio.sleep(0)
    assert task.done() is (count == 1)
    app.state.duplex_warmup_done.set()
    await task


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 4])
async def test_warmup_timeout_admits_single_and_multiple_frontend_clients(monkeypatch, count):
    monkeypatch.setattr(warmup, "DUPLEX_WARMUP_CLIENT_WAIT_S", 0.01)
    app = _app(None, count)
    websocket = SimpleNamespace(app=app, query_params={})

    await asyncio.wait_for(warmup.wait_for_duplex_warmup(websocket), timeout=2)

    assert not app.state.duplex_warmup_done.is_set()
