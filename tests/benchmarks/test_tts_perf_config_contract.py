# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the TTS perf config, without loading models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]
CONFIG = Path(__file__).resolve().parents[2] / "tests/dfx/perf/tests/test_tts.json"


def test_concurrent_tts_perf_cases_send_no_client_seed() -> None:
    # A client seed opts a speech request into seeded isolation
    # (vllm_omni/core/sched/seeded_isolation.py): the request runs alone, so a
    # concurrency sweep or a request-rate case would serve one request at a time.
    seeded = []
    for test in json.loads(CONFIG.read_text(encoding="utf-8")):
        for case in test["benchmark_params"]:
            if case.get("enabled") is False:
                continue
            concurrent = any(c > 1 for c in case.get("max_concurrency") or ()) or bool(case.get("request_rate"))
            if concurrent and "seed" in (case.get("extra_body") or {}):
                seeded.append(f"{test['test_name']} {case['task']} {case['eval_phase']}")
    assert seeded == []
