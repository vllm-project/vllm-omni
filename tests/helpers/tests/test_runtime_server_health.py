# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A server that accepts TCP but cannot serve must not count as ready.

Regression coverage for the tiny-model online flake where ``OmniServer``
declared readiness from a listening socket and the first request then died with
``openai.APIConnectionError: Server disconnected without sending a response``.
"""

import subprocess
import sys
import time

import pytest

from tests.helpers import runtime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# The harness opens one bare TCP connection before it probes ``/health``, so the
# stub must keep accepting: the first connection carries no request.
_STUB_ACCEPTS_BUT_UNHEALTHY = """
import socket

listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("{host}", {port}))
listener.listen()
listener.settimeout(0.2)
while True:
    try:
        conn, _ = listener.accept()
    except socket.timeout:
        continue
    conn.settimeout(0.2)
    try:
        conn.recv(4096)
        conn.sendall(
            b"HTTP/1.1 503 Service Unavailable\\r\\nContent-Length: 0\\r\\nConnection: close\\r\\n\\r\\n"
        )
    except OSError:
        pass
    finally:
        conn.close()
"""

_STUB_DIES_AFTER_READY = """
import socket
import sys
import time

listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("{host}", {port}))
listener.listen()
listener.settimeout(0.2)
deadline = time.time() + 4.0
while time.time() < deadline:
    try:
        conn, _ = listener.accept()
    except socket.timeout:
        continue
    conn.settimeout(0.2)
    try:
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\nConnection: close\\r\\n\\r\\n{{}}")
    except OSError:
        pass
    finally:
        conn.close()
sys.stderr.write("simulated engine failure: stage 0 raised\\n")
sys.stderr.flush()
time.sleep(0.2)
sys.exit(3)
"""


def _stand_in_for_server_subprocess(server, body: str, monkeypatch) -> None:
    """Run *body* as the server child, owning ``server.port``."""
    popen = subprocess.Popen
    source = body.format(host=server.host, port=server.port)

    def launch(*args, **kwargs):
        # Keep the harness's own Popen options (stderr pipe, cwd, env): the
        # diagnostic under test reads the child's captured stderr.
        return popen([sys.executable, "-c", source], **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)


def test_listening_but_unhealthy_server_is_not_ready(monkeypatch):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [], startup_timeout=6)
    _stand_in_for_server_subprocess(server, _STUB_ACCEPTS_BUT_UNHEALTHY, monkeypatch)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as failure:
        with server:
            pass
    assert "failed to start within" in str(failure.value)
    # Proves the probe read an HTTP status, not just a refused connection.
    assert "503" in str(failure.value)
    # The probe must give up on its own budget instead of hanging.
    assert time.monotonic() - started < 60


def test_death_during_test_reports_server_stderr(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [], startup_timeout=30)
    _stand_in_for_server_subprocess(server, _STUB_DIES_AFTER_READY, monkeypatch)

    with pytest.raises(AssertionError, match="test body failed"):
        with server:
            # Longer than the stub's 4s serving window, so the child is gone
            # before the test body fails.
            time.sleep(6.0)
            raise AssertionError("test body failed after the server died")

    output = capsys.readouterr().out
    assert "OmniServer exited during the test (code 3)" in output
    assert "simulated engine failure" in output
