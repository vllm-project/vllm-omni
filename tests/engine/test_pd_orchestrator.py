# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for PD metadata routing in the orchestrator."""

from __future__ import annotations

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


def test_pd_decode_params_allow_missing_prefill_output() -> None:
    orchestrator = _make_pd_orchestrator(None)
    result = orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))

    kv_params = result.extra_args["kv_transfer_params"]
    assert kv_params["transfer_id"] == "xfer-req"
    assert kv_params["remote_bootstrap_addr"] == "http://127.0.0.1:25201"
    assert kv_params["remote_engine_id"] == "prefill-engine"
    assert kv_params["do_remote_prefill"] is True
    assert kv_params["do_remote_decode"] is False
    assert "remote_request_id" not in kv_params


def test_pd_decode_params_preserve_optional_prefill_output() -> None:
    orchestrator = _make_pd_orchestrator(
        {
            "kv_ready": True,
            "remote_request_id": "prefill-request",
            "connector_metadata": "kept",
        }
    )
    result = orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))

    kv_params = result.extra_args["kv_transfer_params"]
    assert kv_params["remote_request_id"] == "prefill-request"
    assert kv_params["connector_metadata"] == "kept"


@pytest.mark.parametrize(
    ("attribute", "missing_field"),
    [
        ("_pd_prefill_engine_id", "remote_engine_id"),
        ("_pd_bootstrap_addr", "remote_bootstrap_addr"),
    ],
)
def test_pd_decode_params_require_mooncake_routing_fields(attribute: str, missing_field: str) -> None:
    orchestrator = _make_pd_orchestrator(None)
    setattr(orchestrator, attribute, None)

    with pytest.raises(RuntimeError, match=missing_field):
        orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))
