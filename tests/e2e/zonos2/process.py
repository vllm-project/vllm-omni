# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Own only test subprocess groups; never kill other users' model workers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from tests.e2e.zonos2.runtime import ROOT, deployment, environment, model_path


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def check_gpu_cleanup(directory: Path):
    owned = {path.stem.split("-")[-1] for path in (directory / "trace").glob("trace-*.jsonl")}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    for _ in range(100):
        active = subprocess.check_output(
            ["nvidia-smi", "-i", visible, "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).splitlines()
        if not owned.intersection(active):
            return
        time.sleep(0.1)
    raise AssertionError(f"Test GPU processes were not released: {owned.intersection(active)}")


def run_offline(directory: Path, *, concurrent: bool):
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(environment(directory))
    mode = "--concurrent" if concurrent else "--offline"
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "tests.e2e.zonos2.worker", mode, str(directory)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=1800)
            assert status == 0, f"ZONOS2 worker failed ({status}); see {directory}/run.log"
        finally:
            terminate(process)
    check_gpu_cleanup(directory)


@contextmanager
def speech_server(directory: Path):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(environment(directory))
    command = [
        sys.executable,
        "-m",
        "tests.e2e.zonos2.worker",
        "--serve",
        model_path(),
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "zonos2-e2e",
        "--deploy-config",
        str(deployment(directory)),
        "--trust-remote-code",
        "--disable-log-stats",
    ]
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=900, trust_env=False) as client:
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    assert process.poll() is None, f"HTTP server exited; see {directory}/server.log"
                    try:
                        if client.get("/health", timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise AssertionError("ZONOS2 HTTP server did not become ready within 900s")
                yield client
        finally:
            terminate(process)
            check_gpu_cleanup(directory)
