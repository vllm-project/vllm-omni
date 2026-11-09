# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend generation runner with the shared V2 chunk lifecycle."""

from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniV2RunnerMixin
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner


class NPUGenerationModelRunnerV2(NPUOmniV2RunnerMixin, OmniGenerationModelRunner, NPUModelRunner):
    """Run Code2Wav with Ascend batches and synchronous, owning host outputs.

    Generation does not build attention metadata or sample tokens. The shared
    runner keeps exact codec shapes and recycles each completed chunk's slot;
    its non-CUDA output path finishes the host copy before publishing audio.
    """
