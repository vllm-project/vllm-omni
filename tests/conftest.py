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

import os

# Before ``pytest_plugins`` and before any other test path imports vLLM, pin op
# registration order (see :func:`tests.model_executor.helpers.bootstrap_vllm_layer_custom_op_modules`).
# Subdir ``conftest`` hooks can run after other tests are collected/imported, which is
# too late and can trigger duplicate ``vllm::flashinfer_rotary_embedding`` (etc.) errors.
from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

bootstrap_vllm_layer_custom_op_modules()

pytest_plugins = (
    "tests.helpers.fixtures.config",
    "tests.helpers.fixtures.clean",
    "tests.helpers.fixtures.cpu_threads",
    "tests.helpers.fixtures.log",
    "tests.helpers.fixtures.media",
    "tests.helpers.fixtures.pytest_collection",
    "tests.helpers.fixtures.pytest_run_args",
    "tests.helpers.fixtures.runtime",
)


def pytest_configure(config):
    # Opt in only for the existing CPU shell's direct pytest child. Consume the
    # registration token so nested pytest sessions cannot create extra observers.
    if os.environ.get("VLLM_OMNI_NATIVE_PHASE_PROBE") == "1":
        # A timeout can prevent the terminal summary and artifact upload. Keep
        # a bounded receipt of the gate without logging arbitrary env values.
        receipt = (
            f"native-phase-probe: gate pid={os.getpid()} ppid={os.getppid()} "
            f"token_present={'NATIVE_PHASE_PROBE_PARENT_PID' in os.environ} "
            f"token_matches={os.environ.get('NATIVE_PHASE_PROBE_PARENT_PID') == str(os.getppid())} "
            f"output_configured={'NATIVE_PHASE_PROBE_OUTPUT_DIR' in os.environ}\n"
        )
        try:
            os.write(2, receipt.encode("ascii"))
        except OSError:
            pass  # Diagnostic output must not change the original test result.
    if (
        os.environ.get("VLLM_OMNI_NATIVE_PHASE_PROBE") == "1"
        and os.environ.get("NATIVE_PHASE_PROBE_PARENT_PID") == str(os.getppid())
        and "NATIVE_PHASE_PROBE_OUTPUT_DIR" in os.environ
    ):
        os.environ.pop("NATIVE_PHASE_PROBE_PARENT_PID", None)
        from tests.helpers import native_phase_probe

        config.pluginmanager.register(native_phase_probe, "native-phase-probe")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Marker for Buildkite log folding before pytest summary lines.
    terminalreporter.write_line("--- Running Summary")
