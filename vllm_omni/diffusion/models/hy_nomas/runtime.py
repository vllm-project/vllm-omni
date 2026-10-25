# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager, nullcontext

import torch
from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
from hymm.pipeline.solvers.euler import EulerSolver
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm_omni.diffusion.offloader.sequential_backend import sequential_offload_component


class HyNomasRuntime(CPTCausalInferencePipeline):
    omni_cpu_offload = False

    @contextmanager
    def _component_context(self, module: nn.Module):
        residency = sequential_offload_component(module) if self.omni_cpu_offload else nullcontext()
        # Step execution enters each component separately; restore the reference
        # SDPA policy for text encoding and streaming decode as well as DiT.
        with (
            residency,
            sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ),
        ):
            yield

    def _encode_prompt(self, prompt, negative_prompt=None, data_type="video", encode_negative=None):
        with self._component_context(self.text_encoder):
            return super()._encode_prompt(prompt, negative_prompt, data_type, encode_negative)

    def _dit_denoise_residency(self):
        # Keep the DiT resident through camera projection, segment denoising, and KV writes.
        return self._component_context(self.transformer)

    def omni_forward_with_cache(self, **kwargs):
        with self._dit_denoise_residency(), torch.autocast(device_type="cuda", dtype=self.target_dtype):
            return self.transformer.forward_with_cache(**kwargs)

    def omni_scheduler(self, steps: int) -> EulerSolver:
        scheduler = EulerSolver(shift=5.0, reverse=True, solver="euler")
        scheduler.set_timesteps(steps, device=self.device)
        return scheduler

    def _decode_latents(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        with self._component_context(self.vae):
            return super()._decode_latents(latents, num_frames)
