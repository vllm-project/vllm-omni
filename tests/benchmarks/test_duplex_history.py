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
