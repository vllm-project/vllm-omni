# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-Image startup compilation for configured production shapes."""

from typing import TYPE_CHECKING

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.forward_context import set_forward_context

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


class MingImageDiffusionEngine(DiffusionEngine):
    def _dummy_run(self) -> None:
        if self.od_config.additional_config.get("ming_image_compile_buckets"):
            self.collective_rpc("warmup_ming_image_compile_buckets")
        else:
            super()._dummy_run()


class MingImageCompileWorkerExtension:
    model_runner: "DiffusionModelRunner"

    def warmup_ming_image_compile_buckets(self) -> None:
        with set_forward_context(
            vllm_config=self.model_runner.vllm_config,
            omni_diffusion_config=self.model_runner.od_config,
            attn_metadata={},
        ):
            self.model_runner.pipeline.warmup_compile_buckets()
