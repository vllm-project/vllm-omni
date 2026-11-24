"""Direct launchers must reject collisions and clean up only their own servers."""

import fcntl
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1]
FAKE_PYTHON = """#!/usr/bin/env python3
import os, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
role = 'worker' if 'vllm_omni.entrypoints.cli.main' in sys.argv else 'policy'
Path(os.environ['COSMOS3_LOG_DIR'], role + '.started').write_text(str(os.getpid()))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])), Handler).serve_forever()
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def launch_env(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}")
    python = tmp_path / "python"
    python.write_text(FAKE_PYTHON)
    python.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("worker.log", "policy.log"):
        (logs / name).write_text("existing-instance\n")
    ports = {free_port(), free_port()}
    while len(ports) < 2:
        ports.add(free_port())
    worker, policy = ports
    return logs, {
        **os.environ,
        "VLLM_PYTHON": str(python),
        "COSMOS3_MODEL": str(model),
        "COSMOS3_LOG_DIR": str(logs),
        "COSMOS3_WORKER_PORT": str(worker),
        "COSMOS3_POLICY_PORT": str(policy),
        "COSMOS3_STARTUP_TIMEOUT": "10",
    }


@pytest.mark.parametrize("script", ["run_server.sh", "run_server_faster.sh"])
@pytest.mark.parametrize("role", ["WORKER", "POLICY"])
def test_port_collision_does_not_touch_existing_service(launch_env, script, role):
    logs, env = launch_env
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        env[f"COSMOS3_{role}_PORT"] = str(sock.getsockname()[1])
        result = subprocess.run(["bash", str(EXAMPLES / script)], env=env, capture_output=True, timeout=5)
    assert result.returncode == 2
    assert b"is in use" in result.stderr
    assert not list(logs.glob("*.started"))
    assert (logs / "worker.log").read_text() == "existing-instance\n"
    assert (logs / "policy.log").read_text() == "existing-instance\n"


def test_launch_lock_preserves_existing_logs(launch_env):
    logs, env = launch_env
    with (logs / "server.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(EXAMPLES / "run_server.sh")], env=env, capture_output=True, timeout=5)
    assert result.returncode == 2
    assert not list(logs.glob("*.started"))
    assert (logs / "worker.log").read_text() == "existing-instance\n"


@pytest.mark.parametrize("script", ["run_server.sh", "run_server_faster.sh"])
def test_direct_servers_are_cleaned_up_on_interrupt(launch_env, script):
    logs, env = launch_env
    launcher = subprocess.Popen(["bash", str(EXAMPLES / script)], env=env)
    try:
        deadline = time.monotonic() + 12
        while not (logs / "policy.started").exists():
            assert launcher.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        launcher.terminate()
        launcher.wait(timeout=10)
    assert launcher.returncode == 143
    assert not (logs / "server.pid").exists()
    for role in ("worker", "policy"):
        pid = int((logs / f"{role}.started").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_managed_launcher_rejects_external_worker(launch_env):
    logs, env = launch_env
    result = subprocess.run(
        ["bash", str(EXAMPLES / "run_server.sh"), "--vllm-base-url", "http://127.0.0.1:1"],
        env=env,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert b"manages its own worker" in result.stderr
    assert not list(logs.glob("*.started"))
