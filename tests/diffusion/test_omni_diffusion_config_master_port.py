# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniDiffusionConfig master_port resolution (issue #3794)."""

import socket

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniDiffusionConfigMasterPort:
    def test_honors_master_port_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MASTER_PORT", "43210")
        config = OmniDiffusionConfig(model="test")
        assert config.master_port == 43210

    def test_honors_explicit_master_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        config = OmniDiffusionConfig(model="test", master_port=40000)
        assert config.master_port == 40000

    def test_master_port_env_takes_precedence_over_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MASTER_PORT", "43210")
        config = OmniDiffusionConfig(model="test", master_port=40000)
        assert config.master_port == 43210

    def test_explicit_master_port_is_stable_without_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        ports = {OmniDiffusionConfig(model="test", master_port=40123).master_port for _ in range(5)}
        assert ports == {40123}

    def test_default_master_port_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        config = OmniDiffusionConfig(model="test")
        assert config.master_port is not None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", config.master_port))

    def test_settle_port_steps_when_port_is_busy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            busy_port = sock.getsockname()[1]
            config = OmniDiffusionConfig(model="test", master_port=busy_port)
            assert config.master_port == busy_port + 37
