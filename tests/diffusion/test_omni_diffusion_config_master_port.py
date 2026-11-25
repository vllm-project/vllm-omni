# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for OmniDiffusionConfig master_port resolution (issue #3794)."""

import socket

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Distinct from each other and from a single +37 settle step.
_ENV_PORT = 41017
_KWARG_PORT = 42019
_EXPLICIT_PORT = 43013


def _stub_port_availability(monkeypatch: pytest.MonkeyPatch, *, unavailable: frozenset[int] = frozenset()) -> list[int]:
    """Control the check ``settle_port`` uses, instead of a live bind.

    ``settle_port`` calls ``is_port_available`` through the name imported in
    ``vllm_omni.diffusion.data``. A port returned by ``get_open_port()`` is not
    reserved, so under CI contention the real check can fail and the resolver
    steps forward by 37. That fallback is intended. Tests of which candidate
    was selected stub the check so they do not depend on the port staying free.

    The returned list is every port the stub was asked about, so a test fails
    if resolution stops consulting this check.
    """
    checked: list[int] = []

    def _is_available(port: int) -> bool:
        checked.append(port)
        return port not in unavailable

    monkeypatch.setattr("vllm_omni.diffusion.data.is_port_available", _is_available)
    return checked


class TestOmniDiffusionConfigMasterPort:
    def test_honors_master_port_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        checked = _stub_port_availability(monkeypatch)
        monkeypatch.setenv("MASTER_PORT", str(_ENV_PORT))
        config = OmniDiffusionConfig(model="test")
        assert config.master_port == _ENV_PORT
        assert checked == [_ENV_PORT]

    def test_honors_explicit_master_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        checked = _stub_port_availability(monkeypatch)
        config = OmniDiffusionConfig(model="test", master_port=_EXPLICIT_PORT)
        assert config.master_port == _EXPLICIT_PORT
        assert checked == [_EXPLICIT_PORT]

    def test_master_port_env_takes_precedence_over_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        checked = _stub_port_availability(monkeypatch)
        monkeypatch.setenv("MASTER_PORT", str(_ENV_PORT))
        config = OmniDiffusionConfig(model="test", master_port=_KWARG_PORT)
        assert config.master_port == _ENV_PORT
        assert checked == [_ENV_PORT]

    def test_explicit_master_port_is_stable_without_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        checked = _stub_port_availability(monkeypatch)
        ports = {OmniDiffusionConfig(model="test", master_port=_EXPLICIT_PORT).master_port for _ in range(5)}
        assert ports == {_EXPLICIT_PORT}
        assert checked == [_EXPLICIT_PORT] * 5

    def test_default_master_port_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        # The default port still comes from live get_open_port(), which does
        # not reserve it, so retry the resolve-and-bind. Bind loopback only;
        # CodeQL flags a wildcard bind.
        last_error: Exception | None = None
        for _ in range(10):
            try:
                config = OmniDiffusionConfig(model="test")
                assert config.master_port is not None
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", config.master_port))
                return
            except OSError as exc:
                last_error = exc
        pytest.fail(f"default master_port was not bindable in 10 attempts: {last_error}")

    def test_settle_port_steps_when_port_is_busy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        busy_port = _EXPLICIT_PORT
        checked = _stub_port_availability(monkeypatch, unavailable=frozenset({busy_port}))
        config = OmniDiffusionConfig(model="test", master_port=busy_port)
        assert config.master_port == busy_port + 37
        assert checked == [busy_port, busy_port + 37]
