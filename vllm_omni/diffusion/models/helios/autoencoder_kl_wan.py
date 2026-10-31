# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from diffusers.models.autoencoders.vae import DecoderOutput

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan


class HeliosAutoencoderKLWan(DistributedAutoencoderKLWan):
    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        decoded = super().tiled_decode(z, return_dict=False)[0]
        self.clear_cache()
        if self.is_distributed_enabled():
            # Every Helios rank appends each decoded chunk to its video history.
            decoded = self.distributed_executor._sync_final_result(decoded, z.ndim, z.device, self.dtype)
        return DecoderOutput(sample=decoded) if return_dict else (decoded,)
