# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM layer bootstrap for ``tests/model_executor`` to reduce duplicate CustomOp issues."""

from __future__ import annotations

import pytest

from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules


def pytest_configure(config: pytest.Config) -> None:
    bootstrap_vllm_layer_custom_op_modules()
