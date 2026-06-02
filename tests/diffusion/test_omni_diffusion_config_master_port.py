# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniDiffusionConfig master_port resolution (issue #3794)."""

import socket

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _bind_available_port_below(max_port: int, port_inc: int) -> socket.socket:
    for _ in range(100):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
            if port >= max_port - port_inc:
                sock.close()
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as next_sock:
                next_sock.bind(("", port + port_inc))
            return sock
        except OSError:
            sock.close()
    raise RuntimeError(f"Failed to reserve an available port below {max_port - port_inc}")


class TestOmniDiffusionConfigMasterPort:
    def test_honors_master_port_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        port = _get_available_port()
        monkeypatch.setenv("MASTER_PORT", str(port))
        config = OmniDiffusionConfig(model="test")
        assert config.master_port == port

    def test_honors_explicit_master_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        port = _get_available_port()
        config = OmniDiffusionConfig(model="test", master_port=port)
        assert config.master_port == port

    def test_master_port_env_takes_precedence_over_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env_port = _get_available_port()
        kwarg_port = _get_available_port()
        monkeypatch.setenv("MASTER_PORT", str(env_port))
        config = OmniDiffusionConfig(model="test", master_port=kwarg_port)
        assert config.master_port == env_port

    def test_explicit_master_port_is_stable_without_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        port = _get_available_port()
        ports = {OmniDiffusionConfig(model="test", master_port=port).master_port for _ in range(5)}
        assert ports == {port}

    def test_default_master_port_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        config = OmniDiffusionConfig(model="test")
        assert config.master_port is not None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", config.master_port))

    def test_settle_port_steps_when_port_is_busy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        with _bind_available_port_below(max_port=60000, port_inc=37) as sock:
            busy_port = sock.getsockname()[1]
            config = OmniDiffusionConfig(model="test", master_port=busy_port)
            assert config.master_port == busy_port + 37
