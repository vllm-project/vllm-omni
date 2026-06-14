# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner


def test_v2_model_runners_have_shutdown_for_vllm_020_worker_exit():
    for runner_cls in (OmniARModelRunner, OmniGenerationModelRunner, OmniGPUModelRunner):
        assert hasattr(runner_cls, "shutdown")


if __name__ == "__main__":
    test_v2_model_runners_have_shutdown_for_vllm_020_worker_exit()


def test_load_model_uses_class_patch_lock():
    import inspect

    source = inspect.getsource(OmniGPUModelRunner.load_model)
    assert "_model_state_patch_lock" in source
