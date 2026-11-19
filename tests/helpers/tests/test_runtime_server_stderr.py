# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request failures must retain stderr from a server that was already ready."""

import http.client
import io
import subprocess
import sys
import textwrap

import pytest

from tests.helpers import runtime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_request_disconnect_preserves_server_stderr(monkeypatch, capfd):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        code = textwrap.dedent(
            f"""\
            import socket
            import sys

            with socket.socket() as listener:
                listener.bind(({server.host!r}, {server.port}))
                listener.listen()
                while True:
                    connection, _ = listener.accept()
                    with connection:
                        # The startup TCP probe sends no request.
                        if connection.recv(4096):
                            print('injected server failure after readiness', file=sys.stderr, flush=True)
                            break
            raise SystemExit(7)
            """
        )
        pipe_kwargs = {key: kwargs[key] for key in ("stderr", "encoding", "errors", "text", "bufsize") if key in kwargs}
        return popen([sys.executable, "-c", code], **pipe_kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with server:
        connection = http.client.HTTPConnection(server.host, server.port, timeout=5)
        try:
            connection.request("POST", "/v1/images/generations")
            with pytest.raises(http.client.RemoteDisconnected):
                connection.getresponse()
        finally:
            connection.close()
        assert server.proc is not None
        assert server.proc.wait(timeout=5) == 7

    # Teardown replays the buffered diagnostics once instead of streaming each line
    # while the server is handling measured requests.
    assert "injected server failure after readiness" in capfd.readouterr().err


def test_ready_server_stderr_is_deferred_until_teardown(monkeypatch, capfd):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        code = textwrap.dedent(
            f"""\
            import socket
            import sys

            with socket.socket() as listener:
                listener.bind(({server.host!r}, {server.port}))
                listener.listen()
                print('deferred server diagnostic', file=sys.stderr, flush=True)
                while True:
                    connection, _ = listener.accept()
                    connection.close()
            """
        )
        pipe_kwargs = {key: kwargs[key] for key in ("stderr", "encoding", "errors", "text", "bufsize") if key in kwargs}
        return popen([sys.executable, "-c", code], **pipe_kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with server:
        assert "deferred server diagnostic" not in capfd.readouterr().err
    assert "deferred server diagnostic" in capfd.readouterr().err


def test_closed_stderr_capture_does_not_block_the_server(monkeypatch):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [], startup_timeout=5)
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        # More than a pipe buffer: stopping the drain thread would leave the
        # child stuck in write(), hiding its exit behind a startup timeout.
        code = "import sys; sys.stderr.write('injected log line\\n' * 10000); raise SystemExit(7)"
        pipe_kwargs = {key: kwargs[key] for key in ("stderr", "encoding", "errors", "text", "bufsize") if key in kwargs}
        return popen([sys.executable, "-c", code], **pipe_kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    closed_capture = io.StringIO()
    closed_capture.close()
    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(runtime.sys, "stderr", closed_capture)
        with pytest.raises(RuntimeError, match="exited with code 7"):
            with server:
                pass
