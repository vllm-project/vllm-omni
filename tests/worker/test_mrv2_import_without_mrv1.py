# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MRv2 runner path must import without vLLM's Model Runner V1 modules.

vLLM removed Model Runner V1 in vllm-project/vllm#63373. Each case runs in a
fresh interpreter that blocks the removed modules, so an MRv2-only stage keeps
loading once vLLM-Omni moves past the vLLM release that still ships them.
"""

import subprocess
import sys
import textwrap

import pytest
import vllm.v1.kv_cache_interface as kv_cache_interface

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Modules deleted by vllm-project/vllm#63373 that vLLM-Omni has imported.
REMOVED_MRV1_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu_input_batch",
    "vllm.v1.worker.block_table",
    "vllm.v1.worker.gpu_ubatch_wrapper",
    "vllm.v1.worker.kv_connector_model_runner_mixin",
    "vllm.v1.worker.ec_connector_model_runner_mixin",
    "vllm.v1.sample.sampler",
    "vllm.v1.sample.metadata",
    "vllm.v1.sample.rejection_sampler",
    "vllm.v1.sample.logits_processor",
    "vllm.v1.spec_decode.eagle",
    "vllm.v1.spec_decode.dflash",
    "vllm.v1.spec_decode.draft_model",
    "vllm.v1.spec_decode.extract_hidden_states",
    "vllm.v1.spec_decode.gemma4",
    "vllm.v1.spec_decode.metadata",
    "vllm.v1.spec_decode.ngram_proposer_gpu",
)

_PRELUDE = textwrap.dedent(
    """
    import builtins
    import sys

    BLOCKED = {blocked!r}
    _real_import = builtins.__import__

    def _is_blocked(name):
        return any(name == m or name.startswith(m + ".") for m in BLOCKED)

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Only vLLM-Omni's own imports are blocked: the vLLM release under test
        # still imports these modules internally and drops them with MRv1.
        # Checking the importer here (not in a finder) also catches modules
        # that vLLM already loaded into sys.modules.
        importer = (globals or {{}}).get("__name__") or ""
        if importer == "vllm_omni" or importer.startswith("vllm_omni."):
            if level:
                package = globals.get("__package__") or importer.rpartition(".")[0]
                base = package.rsplit(".", level - 1)[0] if level > 1 else package
                absolute = f"{{base}}.{{name}}" if name else base
            else:
                absolute = name
            targets = [absolute] + [f"{{absolute}}.{{item}}" for item in (fromlist or ()) if item != "*"]
            for target in targets:
                if _is_blocked(target):
                    raise ModuleNotFoundError(
                        f"No module named {{target!r}} (removed with MRv1; imported by {{importer}})"
                    )
        return _real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _guarded_import

    # vllm#63373 also drops VllmConfig.use_v2_model_runner, which
    # vllm_omni.patch overrides at package import.
    from vllm.config import VllmConfig

    if hasattr(VllmConfig, "use_v2_model_runner"):
        del VllmConfig.use_v2_model_runner
    """
).format(blocked=REMOVED_MRV1_MODULES)


def _run_blocked(body: str) -> None:
    code = _PRELUDE + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


@pytest.mark.parametrize(
    "module",
    [
        "vllm_omni.worker_v2.omni_ar_model_runner",
        "vllm_omni.worker_v2.omni_generation_model_runner",
        "vllm_omni.worker.gpu_ar_worker",
        "vllm_omni.worker.gpu_generation_worker",
        "vllm_omni.worker.omni_connector_validation",
        pytest.param(
            "vllm_omni.diffusion.diffusion_kv.model_runner_backend",
            marks=pytest.mark.skipif(
                not hasattr(kv_cache_interface, "get_block_table_width"),
                reason="this vLLM still keeps get_block_table_width in the MRv1 block_table",
            ),
        ),
    ],
)
def test_mrv2_path_imports_without_mrv1(module: str) -> None:
    _run_blocked(f"import {module}\n")


def test_workers_resolve_mrv2_runners_without_mrv1() -> None:
    _run_blocked(
        """
        from vllm_omni.worker.gpu_ar_worker import GPUARWorker
        from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker
        from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector
        from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        assert GPUARWorker.v2_model_runner_cls is OmniARModelRunner
        assert GPUGenerationWorker.v2_model_runner_cls is OmniGenerationModelRunner
        validate_worker_omni_connector(GPUARWorker, True, use_v2_model_runner=True)
        validate_worker_omni_connector(GPUGenerationWorker, True, use_v2_model_runner=True)
        assert "vllm.v1.worker.gpu_model_runner" not in sys.modules
        """
    )


def test_workers_keep_mrv1_runner_classes() -> None:
    """On a vLLM that still ships MRv1, ``model_runner_cls`` is the V1 runner."""
    pytest.importorskip("vllm.v1.worker.gpu_model_runner")
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
    from vllm_omni.worker.gpu_ar_worker import GPUARWorker
    from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
    from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker
    from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector

    assert GPUARWorker.model_runner_cls is GPUARModelRunner
    assert GPUGenerationWorker.model_runner_cls is GPUGenerationModelRunner
    validate_worker_omni_connector(GPUARWorker, True)
    validate_worker_omni_connector(GPUGenerationWorker, True)
