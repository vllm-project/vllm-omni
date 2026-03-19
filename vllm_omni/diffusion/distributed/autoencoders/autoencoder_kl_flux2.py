# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import (
    DistributedAutoencoderKL_base,
)


class DistributedAutoencoderKLFlux2(DistributedAutoencoderKL_base, AutoencoderKLFlux2):
    pass
