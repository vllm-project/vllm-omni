# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner


def test_v2_model_runners_accept_vllm_020_is_profile_keyword():
    for runner_cls in (OmniARModelRunner, OmniGenerationModelRunner, OmniGPUModelRunner):
        assert "is_profile" in inspect.signature(runner_cls.execute_model).parameters


if __name__ == "__main__":
    test_v2_model_runners_accept_vllm_020_is_profile_keyword()
