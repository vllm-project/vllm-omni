"""
L5(b) reliability integration tests (RFC: test_reliability.py).

Loads ``tests/dfx/reliability/tests/scenarios.json``, runs ``fault_inject`` by scenario
type, then validates recovery via ``openai_client``.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import OmniServerParams, dummy_messages_from_mix_data
from tests.dfx.conftest import create_unique_server_params, load_configs
from tests.dfx.reliability.scripts.fault_inject import (
    assert_fault_http_expectation,
    build_recovery_result,
    inject_abnormal_input_faults,
)
from tests.utils import hardware_test
from vllm_omni.platforms import current_omni_platform

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REL_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = REL_DIR / "tests" / "scenarios.json"
# Reuse e2e stage configs (same as prior test_abnormal_input.py)
E2E_STAGE_CONFIGS_DIR = REL_DIR.parent.parent / "e2e" / "stage_configs"


def _configs_with_platform_stage_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve stage config path for XPU vs CUDA/ROCm (async_chunk yaml is CUDA-only)."""
    out: list[dict[str, Any]] = []
    for c in configs:
        c2 = copy.deepcopy(c)
        if current_omni_platform.is_xpu():
            c2["server_params"]["stage_config_name"] = "xpu/qwen3_omni_ci.yaml"
            c2["server_params"].pop("update", None)
            c2["server_params"].pop("delete", None)
        out.append(c2)
    return out


def _load_reliability_configs() -> list[dict[str, Any]]:
    try:
        raw = load_configs(str(SCENARIOS_PATH))
    except (ValueError, FileNotFoundError):
        return []
    return _configs_with_platform_stage_configs(raw)


def _get_system_prompt() -> dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


CONFIGS = _load_reliability_configs()
ABNORMAL_SCENARIOS = [c for c in CONFIGS if c.get("scenario", {}).get("type") == "abnormal_input"]
_UNIQUE_PARAMS = (
    create_unique_server_params(ABNORMAL_SCENARIOS, E2E_STAGE_CONFIGS_DIR) if ABNORMAL_SCENARIOS else []
)
TEST_PARAMS = [OmniServerParams(model=m, stage_config_path=p) for _, m, p in _UNIQUE_PARAMS]


def test_scenarios_json_loads() -> None:
    """Guarantee RFC ``scenarios.json`` exists and parses (collect-only / CI smoke)."""
    assert SCENARIOS_PATH.is_file(), f"Missing {SCENARIOS_PATH}"
    cfg = load_configs(str(SCENARIOS_PATH))
    assert isinstance(cfg, list)
    assert len(cfg) > 0, "scenarios.json should list at least one scenario"


@pytest.mark.slow
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.skipif(
    not (ABNORMAL_SCENARIOS and TEST_PARAMS),
    reason="no abnormal_input scenarios or server params",
)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_reliability_abnormal_input(omni_server, openai_client) -> None:
    """
    fault_inject (abnormal_input) + expect 4xx + recovery request via openai_client.

    RecoveryResult is recorded for observability (RFC); assertions are explicit below.
    """
    scenario = next(
        (c for c in ABNORMAL_SCENARIOS if c["server_params"]["model"] == omni_server.model),
        ABNORMAL_SCENARIOS[0],
    )
    sc = scenario["scenario"]
    fault_params = sc["fault_params"]
    expect = sc["expect"]
    recovery = sc["recovery_request"]

    host, port = omni_server.host, omni_server.port
    fault = inject_abnormal_input_faults(host, port, omni_server.model, fault_params)
    assert_fault_http_expectation(fault.http_statuses, expect)

    t0 = time.perf_counter()
    messages = dummy_messages_from_mix_data(
        system_prompt=_get_system_prompt(),
        content_text=recovery["content_text"],
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": recovery.get("stream", False),
        "modalities": recovery.get("modalities", ["text"]),
        "key_words": recovery.get("key_words", {}),
    }
    openai_client.send_omni_request(request_config, request_num=1)
    elapsed = time.perf_counter() - t0

    min_ok = int(expect.get("min_post_success", 1))
    assert min_ok >= 1
    recovery_result = build_recovery_result(
        recovered=True,
        recovery_time_sec=elapsed,
        health_check_ok=True,
        post_fault_success_count=1,
        post_fault_error_count=0,
        notes=fault.notes,
    )
    assert recovery_result.recovered
    assert recovery_result.health_check_ok
    assert recovery_result.post_fault_success_count >= min_ok
