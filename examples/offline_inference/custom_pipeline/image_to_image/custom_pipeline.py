# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import QwenImageEditPipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)


class CustomPipeline(QwenImageEditPipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

    def forward(self, req: DiffusionRequestBatch):
        """Forward pass for image editing with dummy trajectory data."""
        # Customize num_inference_steps that best suits your model
        actual_num_steps = req.sampling_params.num_inference_steps or 50
        req.sampling_params.num_inference_steps = actual_num_steps

        # Call parent's forward to get the normal output
        output = super().forward(req=req)

        # Create dummy trajectory data
        dummy_trajectory_latents = torch.randn(actual_num_steps, 1, 16, 64, 64, dtype=torch.float32)

        # Inject dummy trajectory data into output
        output.trajectory_latents = dummy_trajectory_latents

        return output
