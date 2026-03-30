# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test-only worker extension for custom diffusion pipeline E2E tests."""

from __future__ import annotations

from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension


class vLLMOmniColocateWorkerExtensionForTest(CustomPipelineWorkerExtension):
    """Minimal worker extension used by tests.

    This intentionally stays lightweight: we only inherit the base custom
    pipeline extension and add one test function.
    """

    @staticmethod
    def test_extension_name() -> str:
        """Return a stable identifier for assertions in unit tests."""
        return "vllm-omni-colocate-worker-extension-for-test"
