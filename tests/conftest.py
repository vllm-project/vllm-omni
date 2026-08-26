# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Root pytest entrypoint for the vLLM-Omni test suite.

- `tests/conftest.py` stays thin: plugin registration only.
- Importable utilities live under `tests/helpers/`.
- Helper unit tests live under `tests/helpers/tests/`.
- Fixtures live under `tests/helpers/fixtures/` and are loaded via `pytest_plugins`.
"""

from __future__ import annotations

# Before ``pytest_plugins`` and before any other test path imports vLLM, pin op
# registration order (see :func:`tests.model_executor.helpers.bootstrap_vllm_layer_custom_op_modules`).
# Subdir ``conftest`` hooks can run after other tests are collected/imported, which is
# too late and can trigger duplicate ``vllm::flashinfer_rotary_embedding`` (etc.) errors.
from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

bootstrap_vllm_layer_custom_op_modules()

pytest_plugins = (
    "tests.helpers.fixtures.config",
    "tests.helpers.fixtures.clean",
    "tests.helpers.fixtures.log",
    "tests.helpers.fixtures.media",
    "tests.helpers.fixtures.run_args",
    "tests.helpers.fixtures.runtime",
    "tests.helpers.fixtures.xdist",
)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Marker for Buildkite log folding before pytest summary lines.
    terminalreporter.write_line("--- Running Summary")
