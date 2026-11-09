# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
from collections import Counter

import pytest

from vllm_omni.benchmarks.duplex_history import make_cases, next_answer, report, score

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_frozen_case_design_has_disjoint_development_topics_and_expected_corrections():
    cases = make_cases()
    assert Counter(c["split"] for c in cases) == {"dev": 10, "eval": 30}
    assert len({c["id"] for c in cases}) == 40
    dev = {c["turns"][0] for c in cases if c["split"] == "dev"}
    assert not dev.intersection(c["turns"][0] for c in cases if c["split"] == "eval")
    for case in cases:
        assert len(case["turns"]) == 20
        if case["category"] == "correction":
            assert case["accepted"][0] in case["turns"][5]
            assert case["forbidden"][0] in case["turns"][0]


def test_scoring_does_not_accept_negation_or_old_value_by_substring():
    assert score("三千元。", ["三千元"], ["五千元"]) == "correct"
    assert score("不是三千元", ["三千元"], ["五千元"]) == "needs_review"
    assert score("五千元", ["三千元"], ["五千元"]) == "incorrect"
    assert score("", ["三千元"], []) == "incorrect"


def test_report_pairs_scenarios_and_refuses_different_manifests(tmp_path):
    def row(policy, case, result):
        return {
            "policy": policy,
            "case_id": case,
            "score": result,
            "modality": "text",
            "idle_seconds": 3,
            "category": "fact",
            "turns": [{"ttft_ms": 10, "ttfp_ms": None}],
            "manifest_sha256": "same",
        }

    rows = [
        row("bounded_recent", "a", "incorrect"),
        row("incremental", "a", "correct"),
        row("bounded_recent", "b", "correct"),
        row("incremental", "b", "needs_review"),
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    result = report([path])
    assert result["paired"]["text/3"] == {"improved": 1, "unresolved": 1}
    assert result["thresholds"] is None
    rows[1]["manifest_sha256"] = "different"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="manifest"):
        report([path])


@pytest.mark.asyncio
async def test_audio_listening_beats_do_not_finish_the_current_turn():
    from vllm_omni.clients.duplex import DuplexClient, SessionConfig

    client = DuplexClient("ws://unused", model="qwen", config=SessionConfig())
    await client._dispatch({"type": "response.listen"})
    await client._dispatch({"type": "response.listen"})
    await client._dispatch({"type": "response.created", "response": {"id": "answer-1"}})
    await client._dispatch({"type": "response.text.delta", "response_id": "answer-1", "delta": "remembered"})
    handle = await next_answer(client.responses())
    assert handle.response_id == "answer-1"
    assert handle.text == "remembered"


@pytest.fixture
def scripted_clients(monkeypatch):
    import asyncio

    import vllm_omni.clients.duplex as duplex

    modes: list[str] = []

    class ScriptedClient(duplex.DuplexClientBase):
        """Exercise real response/event iteration with a deterministic transport."""

        def __init__(self, url, *, model, config, reconnect):
            super().__init__(model=model, config=config)
            self.mode = modes.pop(0)
            self.turn = 0
            self.torn_down = False
            clients.append(self)

        async def _open(self):
            if self.mode == "connect":
                raise duplex.DuplexConnectionError("connection refused")

        async def _teardown(self):
            self.torn_down = True

        async def responses(self):
            async for handle in super().responses():
                if self.turn == 2 and self.mode == "bug":
                    raise RuntimeError("unexpected benchmark bug")
                yield handle

        async def _send_command(self, payload):
            kind = payload["type"]
            if kind == "session.update":
                await self._dispatch({"type": "session.created", "session": {"id": "fake-session"}})
            elif kind == "session.close":
                await self._dispatch({"type": "session.closed"})
            elif kind == "response.create":
                self.turn += 1
                if self.turn == 2 and self.mode == "protocol":
                    await self._dispatch({"type": "error", "error": {"code": "rejected", "message": "bad request"}})
                    return
                if self.turn == 2 and self.mode == "cancel":
                    raise asyncio.CancelledError
                response_id = f"answer-{self.turn}"
                await self._dispatch({"type": "response.created", "response": {"id": response_id}})
                await self._dispatch({"type": "response.text.delta", "response_id": response_id, "delta": "remembered"})
                if self.turn == 2 and self.mode == "timeout":
                    return
                await self._dispatch({"type": "response.done", "response": {"id": response_id, "status": "completed"}})

    clients: list[ScriptedClient] = []
    monkeypatch.setattr(duplex, "DuplexClient", ScriptedClient)
    return modes, clients


def benchmark_args(tmp_path):
    from argparse import Namespace

    return Namespace(
        url="ws://unused",
        model="qwen",
        max_output_tokens=128,
        modality="text",
        realtime_input=False,
        policy="incremental",
        idle_seconds=0,
        timeout=0.02,
        manifest=tmp_path / "cases.jsonl",
        output=tmp_path / "results.jsonl",
        split="eval",
        limit=0,
    )


@pytest.mark.asyncio
async def test_run_records_operational_failures_and_continues(tmp_path, scripted_clients):
    from vllm_omni.benchmarks.duplex_history import run

    modes, clients = scripted_clients
    modes.extend(["timeout", "protocol", "connect", "success"])
    args = benchmark_args(tmp_path)
    cases = [
        {
            "id": str(i),
            "split": "eval",
            "category": "fact",
            "turns": ["first", "last"],
            "accepted": ["remembered"],
            "forbidden": [],
        }
        for i in range(4)
    ]
    args.manifest.write_text("\n".join(json.dumps(case) for case in cases))
    await run(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert [row["case_id"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["score"] for row in rows] == ["run_error", "run_error", "run_error", "correct"]
    assert [len(row["turns"]) for row in rows] == [1, 1, 0, 2]
    assert rows[0]["failed_turn"]["index"] == 1
    assert rows[0]["failed_turn"]["text"] == "remembered"
    assert rows[0]["failed_turn"]["response_id"] == "answer-2"
    assert rows[0]["errors"][-1]["type"] == "TimeoutError"
    assert rows[1]["errors"][-1]["type"] == "DuplexProtocolError"
    assert rows[1]["errors"][-1]["raw"]["error"]["code"] == "rejected"
    assert rows[2]["failed_turn"] is None
    assert rows[2]["errors"][-1]["type"] == "DuplexConnectionError"
    assert all(client.torn_down and not client._subscribers for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["bug", "cancel"])
async def test_run_does_not_swallow_bugs_or_cancellation(tmp_path, scripted_clients, mode):
    import asyncio

    from vllm_omni.benchmarks.duplex_history import run_case

    modes, clients = scripted_clients
    modes.append(mode)
    case = {"id": "a", "category": "fact", "turns": ["first", "last"], "accepted": ["remembered"], "forbidden": []}
    with pytest.raises(RuntimeError if mode == "bug" else asyncio.CancelledError):
        await run_case(case, benchmark_args(tmp_path))
    assert clients[0].torn_down and not clients[0]._subscribers
