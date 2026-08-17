# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the async-output wait bound (_async_output_timeout)."""

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

    def test_value_is_read_per_call(self, monkeypatch):
        """Read at call time rather than import time, so operators are not
        forced to restart the server to widen the bound.
        """
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, "60")
        assert _async_output_timeout() == 60.0
        monkeypatch.setenv(_ASYNC_OUTPUT_TIMEOUT_ENV, "120")
        assert _async_output_timeout() == 120.0
