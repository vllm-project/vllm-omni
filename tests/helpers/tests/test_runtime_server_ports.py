# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A listening port alone must not make a different test's server ready."""

import socket
import subprocess
import sys

import pytest
from filelock import FileLock

from tests.helpers import runtime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_server_does_not_accept_another_process_listener(monkeypatch):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    popen = subprocess.Popen

    # Occupy the chosen port after allocation, while the requested subprocess
    # is still importing. The old TCP-only probe returns ready for this socket.
    with socket.socket() as foreign_listener:

        def launch(*args, **kwargs):
            foreign_listener.bind((server.host, server.port))
            foreign_listener.listen()
            # Forward OmniServer's stderr pipe kwargs so capture stays valid.
            pipe_kwargs = {
                key: kwargs[key] for key in ("stderr", "encoding", "errors", "text", "bufsize") if key in kwargs
            }
            return popen(
                [sys.executable, "-c", "import time; time.sleep(0.2); raise SystemExit(7)"],
                **pipe_kwargs,
            )

        monkeypatch.setattr(runtime.subprocess, "Popen", launch)
        with pytest.raises(RuntimeError, match="exited with code 7"):
            with server:
                pass
    # Failed __enter__ must also release the lease for a later server.
    retry = runtime.OmniServer("retry-model", [], port=server.port)
    try:
        retry._reserve_port()
    finally:
        retry.__exit__(None, None, None)


def test_auto_ports_are_reserved_before_subprocesses_bind(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    # Both constructors see the same free port before either server has bound.
    first = runtime.OmniServer("first-model", [])
    candidate = first.port
    monkeypatch.setattr(runtime, "get_open_port", lambda *args: candidate)
    second = runtime.OmniServer("second-model", [])
    first._reserve_port()
    with socket.socket() as occupied, socket.socket() as probe:
        occupied.bind((first.host, candidate))
        probe.bind((first.host, 0))
        alternative = probe.getsockname()[1]
    monkeypatch.setattr(runtime, "get_open_port", lambda *args: alternative)
    try:
        second._reserve_port()
        assert first.port == candidate
        assert second.port == alternative
        # Check OS-level lock exclusion, not only the Python object's state.
        lock_path = next(tmp_path.glob(f"*-{candidate}.lock"))
        contender = FileLock(lock_path)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\nfrom filelock import FileLock, Timeout\n"
                "try:\n FileLock(sys.argv[1]).acquire(timeout=0)\n"
                "except Timeout:\n sys.exit(23)\n",
                str(lock_path),
            ],
            timeout=10,
            check=False,
        )
        assert result.returncode == 23
        explicit = runtime.OmniServer("explicit-model", [], port=candidate)
        with pytest.raises(RuntimeError, match="already reserved"):
            with explicit:
                pass
    finally:
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)
    with contender.acquire(timeout=0):
        pass


@pytest.mark.parametrize("in_child", [False, True])
def test_server_accepts_its_own_listener(monkeypatch, in_child):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        code = (
            "import socket,time; s=socket.socket(); "
            f"s.bind(({server.host!r}, {server.port})); s.listen(); time.sleep(30)"
        )
        if in_child:
            code = f"import subprocess,sys; child=subprocess.Popen([sys.executable,'-c',{code!r}]); child.wait()"
        pipe_kwargs = {key: kwargs[key] for key in ("stderr", "encoding", "errors", "text", "bufsize") if key in kwargs}
        return popen([sys.executable, "-c", code], **pipe_kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with server:
        assert server.proc is not None
        assert server.proc.poll() is None
        with socket.create_connection((server.host, server.port), timeout=1):
            pass


def test_auto_port_retries_on_address_already_in_use(monkeypatch):
    """Child EADDRINUSE on an auto port must re-reserve and relaunch."""
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    monkeypatch.setattr(runtime, "SERVER_PORT_IN_USE_RETRIES", 3)
    server = runtime.OmniServer("fake-model", [])
    popen = subprocess.Popen
    launches: list[int] = []

    def launch(cmd, *args, **kwargs):
        port = int(cmd[cmd.index("--port") + 1])
        launches.append(port)
        pipe_kwargs = {key: kwargs[key] for key in ("stderr", "encoding", "errors", "text", "bufsize") if key in kwargs}
        if len(launches) == 1:
            code = (
                "import sys\nsys.stderr.write('OSError: [Errno 98] Address already in use\\n')\nraise SystemExit(1)\n"
            )
            return popen([sys.executable, "-c", code], **pipe_kwargs)
        code = (
            f"import socket, time\ns = socket.socket(); s.bind(({server.host!r}, {port})); s.listen(); time.sleep(30)\n"
        )
        return popen([sys.executable, "-c", code], **pipe_kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with server:
        assert len(launches) == 2
        assert server.proc is not None
        assert server.proc.poll() is None
        assert server.port == launches[1]
        with socket.create_connection((server.host, server.port), timeout=1):
            pass


def test_fixed_port_does_not_retry_on_address_already_in_use(monkeypatch):
    """An explicit port must surface EADDRINUSE instead of hopping ports."""
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        fixed_port = holder.getsockname()[1]
    server = runtime.OmniServer("fake-model", [], port=fixed_port)
    popen = subprocess.Popen
    launches = 0

    def launch(cmd, *args, **kwargs):
        nonlocal launches
        launches += 1
        pipe_kwargs = {key: kwargs[key] for key in ("stderr", "encoding", "errors", "text", "bufsize") if key in kwargs}
        code = "import sys\nsys.stderr.write('OSError: [Errno 98] Address already in use\\n')\nraise SystemExit(1)\n"
        return popen([sys.executable, "-c", code], **pipe_kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="Address already in use"):
        with server:
            pass
    assert launches == 1


def test_startup_timeout_does_not_retry_on_address_in_use_log(monkeypatch):
    """A still-running child must not retry even if stderr mentions EADDRINUSE."""
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    monkeypatch.setattr(runtime, "SERVER_PORT_IN_USE_RETRIES", 3)
    server = runtime.OmniServer("fake-model", [])
    calls = {"n": 0}

    class _AliveProc:
        def poll(self):
            return None

    def once(self):
        calls["n"] += 1
        self.proc = _AliveProc()
        self._server_error_lines.append("OSError: [Errno 98] Address already in use\n")
        raise RuntimeError(
            "Server failed to start within 4 seconds\n"
            "=== OmniServer stderr ===\n"
            "OSError: [Errno 98] Address already in use\n"
        )

    monkeypatch.setattr(runtime.OmniServer, "_start_server_once", once)
    monkeypatch.setattr(runtime.OmniServer, "_abandon_launch_attempt", lambda self: None)
    with pytest.raises(RuntimeError, match="failed to start within"):
        server._start_server()
    assert calls["n"] == 1
