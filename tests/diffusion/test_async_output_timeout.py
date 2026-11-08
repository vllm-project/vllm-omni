# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the async-output wait bound (_async_output_timeout)."""

import logging

import pytest

from vllm_omni.diffusion.diffusion_engine import (
    _ASYNC_OUTPUT_TIMEOUT_DEFAULT,
    _ASYNC_OUTPUT_TIMEOUT_ENV,
    _async_output_timeout,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestAsyncOutputTimeoutIsConfigurable:
    def test_default_is_generous(self, monkeypatch):
        """The bound covers a copy queued behind a full denoise step, so the
        default must not be tight enough to abort a slow but healthy render.
        """
        monkeypatch.delenv(_ASYNC_OUTPUT_TIMEOUT_ENV, raising=False)
        assert _async_output_timeout() == _ASYNC_OUTPUT_TIMEOUT_DEFAULT == 600.0

    def test_env_var_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, "1800")
        assert _async_output_timeout() == 1800.0

    def test_env_var_accepts_a_float(self, monkeypatch):
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, "45.5")
        assert _async_output_timeout() == 45.5

    def test_value_is_not_frozen_at_import(self, monkeypatch):
        """Resolved per call rather than captured in a module constant, so the
        value never goes stale relative to os.environ and no module reload is
        needed to exercise it.
        """
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, "60")
        assert _async_output_timeout() == 60.0
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, "120")
        assert _async_output_timeout() == 120.0


class TestAsyncOutputTimeoutRejectsBadValues:
    """This runs on the request path, so a typo in the environment must degrade
    to the default rather than fail the generation.
    """

    @pytest.mark.parametrize("value", ["", "abc", "30s", "None"])
    def test_non_numeric_falls_back_to_the_default(self, monkeypatch, caplog, value):
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, value)
        with caplog.at_level(logging.WARNING):
            assert _async_output_timeout() == _ASYNC_OUTPUT_TIMEOUT_DEFAULT

    @pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
    def test_non_positive_falls_back_to_the_default(self, monkeypatch, caplog, value):
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, value)
        with caplog.at_level(logging.WARNING):
            assert _async_output_timeout() == _ASYNC_OUTPUT_TIMEOUT_DEFAULT
