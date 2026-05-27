"""Tests for the diffusion benchmark runner."""

import socket

import pytest

from tests.dfx.perf.scripts import run_diffusion_benchmark as runner


class _ExitedProcess:
    stdout = None

    def poll(self):
        return 42


def test_wait_for_server_ready_fails_fast_when_process_exits():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]

    with pytest.raises(RuntimeError, match="exited with code 42"):
        runner._wait_for_server_ready(
            "127.0.0.1",
            unused_port,
            _ExitedProcess(),
            timeout=30,
        )


def test_wait_for_server_ready_includes_process_output_on_exit():
    class ExitedWithOutput:
        def poll(self):
            return 1

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]

    with pytest.raises(RuntimeError, match="last line"):
        runner._wait_for_server_ready(
            "127.0.0.1",
            unused_port,
            ExitedWithOutput(),
            timeout=30,
            log_lines=["first line\n", "last line\n"],
        )
