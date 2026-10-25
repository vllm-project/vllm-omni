# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext

import torch
from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
from torch import nn

from vllm_omni.diffusion.offloader.sequential_backend import sequential_offload_component


class HyNomasRuntime(CPTCausalInferencePipeline):
    omni_cpu_offload = False

    def _component_context(self, module: nn.Module):
        if self.omni_cpu_offload:
            return sequential_offload_component(module)
        return nullcontext()

    def _encode_prompt(self, prompt, negative_prompt=None, data_type="video", encode_negative=None):
        with self._component_context(self.text_encoder):
            return super()._encode_prompt(prompt, negative_prompt, data_type, encode_negative)

    def _dit_denoise_residency(self):
        # Keep the DiT resident through camera projection, segment denoising, and KV writes.
        return self._component_context(self.transformer)

    def _decode_latents(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        with self._component_context(self.vae):
            return super()._decode_latents(latents, num_frames)
