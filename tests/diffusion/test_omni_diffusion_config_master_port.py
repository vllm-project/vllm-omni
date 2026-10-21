# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for OmniDiffusionConfig master_port resolution (issue #3794)."""

import socket

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Fixed ports for precedence/stability assertions. These are never bound by the
# tests (availability is frozen via _freeze_port_availability), so concurrent
# processes cannot steal them and force settle_port to step +37.
_ENV_PORT = 39001
_KWARG_PORT = 39002


def _freeze_port_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report every port as available so port assertions are deterministic.

    Ports from get_open_port() are not reserved, so a concurrent process can
    steal one between the call and the assertion; settle_port then steps +37
    by design (PR #3803) and the equality check fails. Freezing availability
    keeps the resolution-precedence logic under test without that race.
    """
    monkeypatch.setattr("vllm_omni.diffusion.data.is_port_available", lambda port: True)


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
        _freeze_port_availability(monkeypatch)
        monkeypatch.setenv("MASTER_PORT", str(_ENV_PORT))
        config = OmniDiffusionConfig(model="test")
        assert config.master_port == _ENV_PORT

    def test_honors_explicit_master_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _freeze_port_availability(monkeypatch)
        monkeypatch.delenv("MASTER_PORT", raising=False)
        config = OmniDiffusionConfig(model="test", master_port=_KWARG_PORT)
        assert config.master_port == _KWARG_PORT

    def test_master_port_env_takes_precedence_over_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _freeze_port_availability(monkeypatch)
        monkeypatch.setenv("MASTER_PORT", str(_ENV_PORT))
        config = OmniDiffusionConfig(model="test", master_port=_KWARG_PORT)
        assert config.master_port == _ENV_PORT

    def test_explicit_master_port_is_stable_without_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _freeze_port_availability(monkeypatch)
        monkeypatch.delenv("MASTER_PORT", raising=False)
        ports = {OmniDiffusionConfig(model="test", master_port=_KWARG_PORT).master_port for _ in range(5)}
        assert ports == {_KWARG_PORT}

    def test_default_master_port_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        # The default port comes from an unreserved ephemeral port, so a
        # concurrent process can transiently steal it; retry the whole
        # resolve-and-bind sequence before declaring failure.
        last_error: Exception | None = None
        for _ in range(10):
            try:
                config = OmniDiffusionConfig(model="test")
                assert config.master_port is not None
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("", config.master_port))
                return
            except OSError as exc:
                last_error = exc
        pytest.fail(f"default master_port was not bindable in 10 attempts: {last_error}")

    def test_settle_port_steps_when_port_is_busy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        with _bind_available_port_below(max_port=60000, port_inc=37) as sock:
            busy_port = sock.getsockname()[1]
            config = OmniDiffusionConfig(model="test", master_port=busy_port)
            assert config.master_port == busy_port + 37
