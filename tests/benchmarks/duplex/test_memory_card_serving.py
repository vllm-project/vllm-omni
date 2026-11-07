# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replay uses the public client; only the remote server is controlled here."""

import asyncio
import importlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

import pybase64
import pytest

from vllm_omni.benchmarks.duplex.memory_card_replay import (
    MemoryReplayEvent,
    MemoryReplayVariant,
    load_memory_replay_cases,
    memory_replay_variants,
)
from vllm_omni.clients.duplex import DuplexClient, ReconnectPolicy, SessionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


def api():
    return importlib.import_module("vllm_omni.benchmarks.duplex.memory_card_serving")


def event(kind, **payload):
    return MemoryReplayEvent(kind, MappingProxyType({"type": kind, **payload}))


def message(item_id="fact", text="My tea preference is green.", role="user"):
    return event("message", item_id=item_id, label=item_id, text=text, role=role)


def variant(*events):
    return MemoryReplayVariant("synthetic", 0, "fixed_idle", tuple(events), "Which tea do I prefer?", "0" * 64)


class Socket:
    def __init__(self, server):
        self.server = server
        self.incoming: asyncio.Queue[dict[str, object] | Exception] = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        if self.closed:
            raise ConnectionError("controlled disconnect")
        await self.server.receive(self, json.loads(data))

    async def recv(self):
        data = await self.incoming.get()
        if isinstance(data, Exception):
            raise data
        return json.dumps(data)

    async def close(self):
        if not self.closed:
            self.closed = True
            self.incoming.put_nowait(ConnectionError("controlled disconnect"))


class Server:
    def __init__(self, *, audio=True, hold_item=False, hold_response=False, response_status="completed"):
        self.commands = []
        self.sockets = []
        self.created = 0
        self.seq = 0
        self.audio = audio
        self.hold_item = hold_item
        self.hold_response = hold_response
        self.response_status = response_status
        self.item_received = asyncio.Event()
        self.last_delta = None

    def emit(self, socket, data, *, journal=True):
        if journal:
            self.seq += 1
            data = {**data, "server_event_seq": self.seq}
        socket.incoming.put_nowait(data)
        return data

    async def connect(self, url):
        socket = Socket(self)
        self.sockets.append(socket)
        return socket

    async def receive(self, socket, command):
        self.commands.append(command)
        kind = command["type"]
        if kind == "session.update":
            self.emit(
                socket,
                {"type": "session.created", "session": {"id": "same-session"}, "resume_token": "private-token-1"},
                journal=False,
            )
        elif kind == "session.resume":
            self.emit(
                socket,
                {"type": "session.resumed", "session": {"id": "same-session"}, "resume_token": "private-token-2"},
                journal=False,
            )
            if self.last_delta is not None:
                socket.incoming.put_nowait(self.last_delta)
        elif kind == "conversation.item.create":
            self.item_received.set()
            if not self.hold_item:
                self.emit(socket, {"type": "conversation.item.created", "item": command["item"]})
        elif kind == "conversation.item.delete":
            self.emit(socket, {"type": "conversation.item.deleted", "item_id": command["item_id"]})
        elif kind == "response.create":
            self.created += 1
            rid, iid = f"response-{self.created}", f"assistant-{self.created}"
            self.emit(socket, {"type": "response.created", "response": {"id": rid}})
            if self.hold_response:
                return
            self.emit(
                socket,
                {
                    "type": "response.output_item.added",
                    "response_id": rid,
                    "item": {"id": iid, "type": "message", "role": "assistant"},
                },
            )
            self.last_delta = self.emit(
                socket, {"type": "response.output_text.delta", "response_id": rid, "delta": "Controlled answer."}
            )
            if self.audio:
                self.emit(
                    socket,
                    {
                        "type": "response.output_audio.delta",
                        "response_id": rid,
                        "item_id": iid,
                        "delta": pybase64.b64encode(bytes(19200)).decode(),
                        "sample_rate_hz": 24000,
                    },
                )
            self.emit(socket, {"type": "response.done", "response": {"id": rid, "status": self.response_status}})
        elif kind == "playback.ack":
            self.emit(
                socket,
                {
                    "type": "playback.acknowledged",
                    "event": {
                        "response_id": command["response_id"],
                        "item_id": command["item_id"],
                        "played_ms": command["played_ms"],
                        "committed_ms": command["committed_ms"],
                    },
                },
            )
        elif kind == "conversation.item.truncate":
            self.emit(
                socket,
                {
                    "type": "conversation.item.truncated",
                    "item_id": command["item_id"],
                    "content_index": 0,
                    "audio_end_ms": command["audio_end_ms"],
                },
            )
        elif kind == "session.close":
            self.emit(socket, {"type": "session.closed", "session_id": "same-session"})

    def client(self):
        return DuplexClient(
            "ws://localhost:8000",
            model="test-model",
            config=SessionConfig(auto_response=False),
            connect=self.connect,
            reconnect=ReconnectPolicy(max_attempts=1, backoff_s=(0, 0)),
            heartbeat_interval_s=None,
            handshake_timeout_s=1,
        )


async def run(server, *variants, timeout_s=1):
    return await api().replay_serving_trials(variants, client_factory=server.client, timeout_s=timeout_s)


async def wait_for_item(server, task):
    waiter = asyncio.create_task(server.item_received.wait())
    try:
        done, _ = await asyncio.wait((task, waiter), timeout=1, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
        await asyncio.wait_for(waiter, 1)
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


async def test_trial_waits_for_item_confirmation_before_querying():
    server = Server(hold_item=True)
    task = asyncio.create_task(run(server, variant(message())))
    try:
        await wait_for_item(server, task)
        await asyncio.sleep(0)
        assert server.created == 0
        server.hold_item = False
        server.emit(server.sockets[0], {"type": "conversation.item.created", "item": {"id": "fact"}})
        (result,) = await task
        assert result.status == "completed" and result.output_text == "Controlled answer."
        assert [row["type"] for row in server.commands].count("session.update") == 1
        assert [ack.event_type for ack in result.confirmed_events] == ["conversation.item.created"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_each_independent_query_trial_replays_into_a_fresh_session():
    server = Server()
    first = variant(message())
    results = await run(server, first, replace(first, query_index=1, query_text="A separate question"))
    assert [r.status for r in results] == ["completed", "completed"]
    assert len(server.sockets) == 2
    assert [c["type"] for c in server.commands].count("session.close") == 2
    assert [c["item"]["id"] for c in server.commands if c["type"] == "conversation.item.create"] == [
        "fact",
        "memory-query",
        "fact",
        "memory-query",
    ]


async def test_live_turns_keep_one_session_and_playback_uses_observed_audio():
    server = Server()
    (result,) = await run(
        server,
        variant(
            message(),
            event("live_response", played_audio_ms=100, truncate=True),
            message("next", "What did you say?"),
            event("live_response", played_audio_ms=200, truncate=False),
        ),
    )
    assert result.status == "completed" and len(server.sockets) == 1 and server.created == 3
    acks = [c for c in server.commands if c["type"] == "playback.ack"]
    assert [(c["response_id"], c["played_ms"]) for c in acks] == [("response-1", 100), ("response-2", 200)]
    truncations = [c for c in server.commands if c["type"] == "conversation.item.truncate"]
    assert [(c["item_id"], c["audio_end_ms"]) for c in truncations] == [("assistant-1", 100)]
    # Query metrics do not accidentally include the two prior responses.
    assert result.query_response_id == "response-3"
    assert [r["response_id"] for r in result.query_metrics] == ["response-3"]
    assert result.session_metrics["audio_turn_count"] == 3


@pytest.mark.parametrize("audio,played,reason", [(False, 100, "missing_audio"), (True, 401, "playback_out_of_range")])
async def test_playback_never_acknowledges_missing_or_unobserved_audio(audio, played, reason):
    server = Server(audio=audio)
    (result,) = await run(server, variant(message(), event("live_response", played_audio_ms=played, truncate=True)))
    assert result.status == "failed" and result.error_code == reason
    assert not any(c["type"] in {"playback.ack", "conversation.item.truncate"} for c in server.commands)


@pytest.mark.parametrize("kind", ["edit", "assistant_playback", "unknown"])
async def test_unsupported_cpu_events_are_not_reinterpreted_or_partially_replayed(kind):
    server = Server()
    (result,) = await run(server, variant(message(), event(kind)))
    assert result.status == "unsupported" and result.error_code == "unsupported_event"
    assert not any(c["type"] in {"conversation.item.create", "response.create"} for c in server.commands)


async def test_resume_reuses_public_rotation_and_dedup_without_exporting_credentials():
    server = Server()
    clients = []

    def factory():
        client = server.client()
        clients.append(client)
        return client

    plan = variant(
        message(),
        event("live_response", played_audio_ms=100, truncate=False),
        event("resume_live_session", same_session_object=True),
        message("after", "Continue."),
    )
    (result,) = await api().replay_serving_trials((plan,), client_factory=factory, timeout_s=1)
    assert result.status == "completed" and clients[0].session_id == "same-session"
    assert clients[0].resume_token == "private-token-2" and len(server.sockets) == 2
    collector = clients[0]._event_collector
    assert collector is not None
    assert (
        sum(
            e.get("type") == "response.output_text.delta" and e.get("response_id") == "response-1"
            for e in collector.events
        )
        == 1
    )
    (resume,) = [c for c in server.commands if c["type"] == "session.resume"]
    assert resume["resume_token"] == "private-token-1" and resume["last_received_server_event_seq"] > 0
    exported = json.dumps(asdict(result))
    assert "private-token" not in exported and "resume_token" not in exported
    assert result.output_text == "Controlled answer."


async def test_delete_waits_for_confirmation_and_fixed_idle_does_not_poll_extraction(monkeypatch):
    server = Server()
    sleep = asyncio.sleep
    durations = []

    async def fixed_sleep(seconds):
        if seconds >= 1:
            durations.append(seconds)
        else:
            await sleep(seconds)

    monkeypatch.setattr(api().asyncio, "sleep", fixed_sleep)
    (result,) = await run(server, variant(message(), event("idle", duration_ms=7000), event("delete", item_id="fact")))
    assert result.status == "completed" and durations == [7.0]
    assert [a.event_type for a in result.confirmed_events] == ["conversation.item.created", "conversation.item.deleted"]
    assert not any("memory" in c["type"] or "extraction" in c["type"] for c in server.commands)


@pytest.mark.parametrize("failure", ["item_timeout", "response_timeout", "response_failed"])
async def test_failed_trials_are_retained_with_partial_acknowledgements(failure):
    server = Server(
        hold_item=failure == "item_timeout",
        hold_response=failure == "response_timeout",
        response_status="failed" if failure == "response_failed" else "completed",
    )
    (result,) = await run(server, variant(message()), timeout_s=0.05)
    assert result.status == "failed"
    assert result.error_code == ("response_failed" if failure == "response_failed" else "timeout")
    assert len(result.confirmed_events) == (0 if failure == "item_timeout" else 1)


async def test_server_errors_do_not_expose_message_or_credentials():
    server = Server(hold_item=True)
    task = asyncio.create_task(run(server, variant(message())))
    try:
        await wait_for_item(server, task)
        server.emit(
            server.sockets[0],
            {"type": "error", "error": {"code": "invalid_request", "message": "private-token-and-customer-body"}},
        )
        (result,) = await task
        assert result.status == "failed" and result.error_code == "protocol_error"
        assert "private-token" not in json.dumps(asdict(result))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_manifest_uses_execution_variants_and_identical_timelines_for_all_arms(tmp_path):
    seeds = Path(__file__).parent / "fixtures/memory_card_seeds.jsonl"
    variants = memory_replay_variants(load_memory_replay_cases(seeds))
    paths = {}
    for arm in ("budget_only", "l2", "l2_l3"):
        path = tmp_path / f"{arm}.yaml"
        path.write_text(f"label: {arm}\n", encoding="utf-8")
        paths[arm] = path
    manifest = api().serving_manifest(variants, deployment_configs=paths)
    assert manifest["compatible_variants"] == 8 and manifest["planned_trials"] == 72
    assert [row["supported"] for row in manifest["variants"]].count(False) == 5
    assert len({row["timeline_sha256"] for row in manifest["arms"]}) == 1
    assert all(len(row["deployment_sha256"]) == 64 for row in manifest["arms"])
    assert "gold_json" not in json.dumps(manifest) and "required_facts" not in json.dumps(manifest)


async def test_evaluation_gold_is_not_an_execution_input(tmp_path):
    path = tmp_path / "case.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "no-gold",
                "synthetic": True,
                "events": [
                    {"type": "message", "item_id": "fact", "label": "fact", "role": "user", "text": "Green tea."}
                ],
                "queries": [{"text": "Which tea?", "gold": {"canary": "gold-only-never-on-wire"}}],
                "timing_variants": ["fixed_idle"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    variants = memory_replay_variants(load_memory_replay_cases(path))
    server = Server()
    (result,) = await run(server, *variants)
    assert result.status == "completed" and "gold-only" not in json.dumps(server.commands)


async def test_query_does_not_reuse_a_fixture_item_identity():
    server = Server()
    (result,) = await run(server, variant(message("memory-query")))
    ids = [c["item"]["id"] for c in server.commands if c["type"] == "conversation.item.create"]
    assert result.status == "completed" and len(ids) == len(set(ids))


async def test_cancellation_before_session_open_does_not_leave_a_collector_running():
    server = Server()
    before = asyncio.all_tasks()
    task = asyncio.create_task(run(server, variant(message())))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    leaked = [
        t
        for t in asyncio.all_tasks() - before
        if not t.done() and getattr(t.get_coro(), "__qualname__", None) == "EventCollector.consume"
    ]
    try:
        assert not leaked
    finally:
        for pending in leaked:
            pending.cancel()
        await asyncio.gather(*leaked, return_exceptions=True)


async def test_prepare_command_exports_one_timeline_and_three_launch_commands(tmp_path):
    seeds = Path(__file__).parent / "fixtures/memory_card_seeds.jsonl"
    output = tmp_path / "manifest.json"
    args = [
        "prepare",
        "--cases",
        str(seeds),
        "--url",
        "ws://localhost:8000",
        "--model",
        "test-model",
        "--output",
        str(output),
    ]
    for name in ("budget-only", "l2", "l2-l3"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"label: {name}\n", encoding="utf-8")
        args += [f"--{name}", str(path)]
    assert api().main(args) == 0
    manifest = json.loads(output.read_text())
    assert manifest["planned_trials"] == 72
    assert all(
        "--deploy-config" in arm["launch_command"] and "--arm" in arm["replay_command"] for arm in manifest["arms"]
    )


async def test_run_command_preserves_failed_samples_and_writes_no_resume_token(tmp_path, monkeypatch):
    server = Server(response_status="failed")

    async def connect(client, url):
        return await server.connect(url)

    monkeypatch.setattr(DuplexClient, "_default_connect", connect)
    seeds = tmp_path / "cases.jsonl"
    seeds.write_text(
        json.dumps(
            {
                "case_id": "cli",
                "synthetic": True,
                "events": [
                    {"type": "message", "item_id": "fact", "label": "fact", "role": "user", "text": "Green tea."}
                ],
                "queries": [{"text": "Which tea?", "gold": {}}],
                "timing_variants": ["fixed_idle"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    deploy = tmp_path / "l2_l3.yaml"
    deploy.write_text("label: l2_l3\n", encoding="utf-8")
    output = tmp_path / "result.json"
    code = await asyncio.to_thread(
        api().main,
        [
            "run",
            "--cases",
            str(seeds),
            "--url",
            "ws://localhost:8000",
            "--model",
            "test-model",
            "--arm",
            "l2_l3",
            "--deployment-config",
            str(deploy),
            "--repeats",
            "1",
            "--timeout-s",
            "1",
            "--output",
            str(output),
        ],
    )
    assert code == 1
    result = json.loads(output.read_text())
    assert result["trials"][0]["status"] == "failed" and result["trials"][0]["error_code"] == "response_failed"
    assert "private-token" not in output.read_text() and "resume_token" not in output.read_text()
