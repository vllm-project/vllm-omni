# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end HTTP API test for weight transfer.

Starts a vllm-omni server with ``--weight-transfer-config`` (the flag is
inherited from upstream vLLM ``EngineArgs``) and verifies the control plane:
the four-phase endpoints are reachable, the real IPC init handshake reaches
the stage worker, and invalid or missing payloads are rejected instead of
being silently swallowed.

The data plane (real weight payloads changing generation output) is covered
by ``test_weight_transfer_changes_output.py`` in-process, which can build
valid CUDA IPC payloads from the live model.
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest
import requests

from tests.helpers.mark import hardware_test

MODEL = "tiny-random/Qwen-Image"
SERVER_PORT = 8123
SERVER_STARTUP_TIMEOUT = 300  # seconds

# Same handshake payload IPCTrainerWeightTransferEngine.trainer_init ships.
IPC_INIT_INFO = {"packed": False}


@pytest.fixture(scope="module")
def server():
    """Start vllm-omni server with weight transfer enabled."""
    cmd = [
        "python",
        "-m",
        "vllm_omni.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1058",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.5",
        "--enforce-eager",
        "--disable-log-stats",
        "--port",
        str(SERVER_PORT),
        "--weight-transfer-config",
        json.dumps({"backend": "ipc"}),
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
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"Server exited early (code {proc.returncode}):\n{output[-4000:]}")
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
def test_init_handshake_reaches_engine(server: str):
    """The trainer-side IPC handshake payload reaches the stage worker."""
    resp = requests.post(
        f"{server}/init_weight_transfer_engine",
        json={"init_info": IPC_INIT_INFO},
        timeout=60,
    )
    assert resp.status_code == 200, f"Init handshake failed: {resp.status_code} {resp.text}"
    assert "error" not in resp.json()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_invalid_init_payload_rejected(server: str):
    """A hand-crafted payload with wrong fields must fail, not no-op.

    The IPC init info accepts ``packed`` only; the backend is chosen
    server-side via ``weight_transfer_config``. This guards the regression
    where any dict was accepted and errors were swallowed.
    """
    resp = requests.post(
        f"{server}/init_weight_transfer_engine",
        json={"init_info": {"backend": "ipc"}},
        timeout=60,
    )
    assert resp.status_code >= 400, f"Invalid init_info should be rejected, got {resp.status_code}: {resp.text}"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_missing_fields_rejected(server: str):
    """Missing init_info / update_info must return 400."""
    resp = requests.post(f"{server}/init_weight_transfer_engine", json={}, timeout=10)
    assert resp.status_code == 400, f"Expected 400 for missing init_info, got {resp.status_code}"

    resp = requests.post(f"{server}/update_weights", json={}, timeout=10)
    assert resp.status_code == 400, f"Expected 400 for missing update_info, got {resp.status_code}"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_update_before_start_rejected(server: str):
    """Protocol violations must surface as errors, not silent success.

    update_weights without a matching start_weight_update is rejected by the
    worker state machine; the orchestrator forwards that failure to HTTP.
    """
    resp = requests.post(
        f"{server}/update_weights",
        json={"update_info": {"names": ["some.weight"], "dtype_names": ["float32"], "shapes": [[1]]}},
        timeout=60,
    )
    assert resp.status_code >= 400, (
        f"update_weights before start_weight_update should fail, got {resp.status_code}: {resp.text}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
