# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend generation runner with the shared V2 chunk lifecycle."""

from typing import Any

from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniV2RunnerMixin
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner


class NPUGenerationModelRunnerV2(NPUOmniV2RunnerMixin, OmniGenerationModelRunner, NPUModelRunner):
    """Run Code2Wav with Ascend batches and synchronous, owning host outputs.

    Generation does not build attention metadata or sample tokens. The shared
    runner keeps exact codec shapes and recycles each completed chunk's slot;
    its non-CUDA output path finishes the host copy before publishing audio.
    """

    def _forward_context(self, attn_metadata: Any, vllm_config: Any, **kwargs: Any) -> Any:
        if kwargs.get("slot_mapping") is not None:
            raise NotImplementedError(
                "NPU generation stages do not support slot_mapping; use this runner only for "
                "attention-free stages such as Code2Wav."
            )
        return super()._forward_context(attn_metadata, vllm_config, **kwargs)
