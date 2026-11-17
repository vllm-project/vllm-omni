# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MRv2 runner path must not depend on vLLM's Model Runner V1.

vLLM removed Model Runner V1 in vllm-project/vllm#63373. Each check imports
real modules in a fresh interpreter, so modules loaded by earlier tests do not
leak into ``sys.modules``.
"""

import importlib.util
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VLLM_HAS_MRV1 = importlib.util.find_spec("vllm.v1.worker.gpu_model_runner") is not None


def _run_fresh(code: str) -> None:
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_mrv2_worker_setup_does_not_load_mrv1_runners() -> None:
    _run_fresh(
        """
        import sys

        from vllm_omni.worker.gpu_ar_worker import GPUARWorker
        from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker
        from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector
        from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        assert GPUARWorker.v2_model_runner_cls is OmniARModelRunner
        assert GPUGenerationWorker.v2_model_runner_cls is OmniGenerationModelRunner
        validate_worker_omni_connector(GPUARWorker, True, use_v2_model_runner=True)
        validate_worker_omni_connector(GPUGenerationWorker, True, use_v2_model_runner=True)

        mrv1_runners = (
            "vllm.v1.worker.gpu_model_runner",
            "vllm_omni.worker.gpu_ar_model_runner",
            "vllm_omni.worker.gpu_generation_model_runner",
        )
        loaded = [name for name in mrv1_runners if name in sys.modules]
        assert not loaded, loaded
        """
    )


@pytest.mark.skipif(not VLLM_HAS_MRV1, reason="needs a vLLM with Model Runner V1")
def test_workers_keep_mrv1_runner_classes() -> None:
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
    from vllm_omni.worker.gpu_ar_worker import GPUARWorker
    from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
    from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker
    from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector

    assert GPUARWorker.model_runner_cls is GPUARModelRunner
    assert GPUGenerationWorker.model_runner_cls is GPUGenerationModelRunner
    validate_worker_omni_connector(GPUARWorker, True)
    validate_worker_omni_connector(GPUGenerationWorker, True)


@pytest.mark.skipif(VLLM_HAS_MRV1, reason="needs a vLLM without Model Runner V1 (vllm-project/vllm#63373)")
@pytest.mark.parametrize(
    "module",
    [
        "vllm_omni.worker_v2.omni_ar_model_runner",
        "vllm_omni.worker_v2.omni_generation_model_runner",
        "vllm_omni.worker.gpu_ar_worker",
        "vllm_omni.worker.gpu_generation_worker",
        "vllm_omni.worker.omni_connector_validation",
        "vllm_omni.diffusion.diffusion_kv.model_runner_backend",
    ],
)
def test_mrv2_path_imports_without_mrv1(module: str) -> None:
    _run_fresh(f"import {module}\n")
