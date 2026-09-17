# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end HTTP API test for weight transfer.

This test starts a vllm-omni server with weight_transfer_config enabled,
then verifies the full four-phase lifecycle through HTTP endpoints.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from contextlib import ExitStack

import pytest
import requests

from tests.helpers.mark import hardware_test

MODEL = "tiny-random/Qwen-Image"
TOKENIZER_MODEL = "Qwen/Qwen2-1.5B-Instruct"
SERVER_PORT = 8123
SERVER_STARTUP_TIMEOUT = 180  # seconds


@pytest.fixture(scope="module")
def server():
    """Start vllm-omni server with weight transfer enabled."""
    import json
    config_json = json.dumps({"backend": "ipc"})

    cmd = [
        "python", "-m", "vllm_omni.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "bfloat16",
        "--max-model-len", "1058",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.5",
        "--enforce-eager",
        "--disable-log-stats",
        "--port", str(SERVER_PORT),
        "--weight-transfer-config", config_json,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://localhost:{SERVER_PORT}"

    # Wait for server to be ready
    print(f"\n[Fixture] Starting server on {base_url}...")
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"[Fixture] Server ready after {i}s")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        proc.wait()
        pytest.fail(f"Server failed to start within {SERVER_STARTUP_TIMEOUT}s")

    yield base_url

    # Cleanup
    print("\n[Fixture] Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_api_reachability(server: str):
    """Test if the weight transfer endpoints are reachable."""
    endpoints = [
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/update_weights",
        "/finish_weight_update",
    ]

    for endpoint in endpoints:
        url = f"{server}{endpoint}"
        resp = requests.options(url, timeout=5)
        # OPTIONS should succeed or at least not raise
        assert resp.status_code in (200, 405), f"{endpoint} not reachable"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_four_phase_protocol(server: str):
    """Test the complete four-phase weight transfer protocol."""

    # Phase 1: Initialize
    resp = requests.post(
        f"{server}/init_weight_transfer_engine",
        json={"init_info": {"backend": "ipc"}},
        timeout=30,
    )
    assert resp.status_code == 200, f"Init failed: {resp.text}"
    result = resp.json()
    assert "message" in result or "error" not in result

    # Phase 2: Start weight update session
    resp = requests.post(
        f"{server}/start_weight_update",
        timeout=30,
    )
    assert resp.status_code == 200, f"Start failed: {resp.text}"
    result = resp.json()
    assert "message" in result or "error" not in result

    # Phase 3: Update weights (with mock data)
    update_info = {
        "names": ["test.layer.weight"],
        "tensors": [[1.0, 2.0, 3.0, 4.0]],
    }
    resp = requests.post(
        f"{server}/update_weights",
        json={"update_info": update_info},
        timeout=30,
    )
    assert resp.status_code == 200, f"Update failed: {resp.text}"
    result = resp.json()
    assert "message" in result or "error" not in result

    # Phase 4: Finish weight update
    resp = requests.post(
        f"{server}/finish_weight_update",
        timeout=30,
    )
    assert resp.status_code == 200, f"Finish failed: {resp.text}"
    result = resp.json()
    assert "message" in result or "error" not in result


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_error_handling(server: str):
    """Test error handling for invalid requests."""

    # Test 1: Missing init_info
    resp = requests.post(
        f"{server}/init_weight_transfer_engine",
        json={},
        timeout=5,
    )
    assert resp.status_code == 400, f"Expected 400 for missing init_info, got {resp.status_code}"

    # Test 2: Missing update_info
    resp = requests.post(
        f"{server}/update_weights",
        json={},
        timeout=5,
    )
    assert resp.status_code == 400, f"Expected 400 for missing update_info, got {resp.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
