# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for PD metadata routing in the orchestrator."""

from __future__ import annotations

import logging

import pytest
from vllm import SamplingParams

from vllm_omni.engine.orchestrator import Orchestrator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pd_orchestrator(kv_params: dict | None) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._pd_kv_params = {"req": kv_params} if kv_params is not None else {}
    orchestrator._pd_bootstrap_addr = "http://127.0.0.1:25201"
    orchestrator._pd_prefill_engine_id = "prefill-engine"
    return orchestrator


def test_pd_decode_params_allow_missing_remote_request_id(caplog: pytest.LogCaptureFixture) -> None:
    orchestrator = _make_pd_orchestrator({"kv_ready": True, "connector_metadata": "kept"})

    with caplog.at_level(logging.WARNING):
        result = orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))

    kv_params = result.extra_args["kv_transfer_params"]
    assert kv_params["connector_metadata"] == "kept"
    assert kv_params["remote_bootstrap_addr"] == "http://127.0.0.1:25201"
    assert kv_params["remote_engine_id"] == "prefill-engine"
    assert kv_params["do_remote_prefill"] is True
    assert kv_params["do_remote_decode"] is False
    assert "remote_request_id" not in kv_params
    assert "without field validation" in caplog.text


def test_pd_decode_params_still_require_connector_metadata() -> None:
    orchestrator = _make_pd_orchestrator(None)

    with pytest.raises(RuntimeError, match="Missing prefill kv_transfer_params"):
        orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))
