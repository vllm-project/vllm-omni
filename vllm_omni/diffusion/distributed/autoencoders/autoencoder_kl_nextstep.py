# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import (
    DistributedAutoencoderKL_base,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    TileTask,
)
from vllm_omni.diffusion.models.nextstep_1_1.modeling_flux_vae import (
    AutoencoderKL as NextStepAutoencoderKL,
)


class DistributedAutoencoderKLNextStep(DistributedAutoencoderKL_base, NextStepAutoencoderKL):
    def tile_exec(self, task: TileTask) -> torch.Tensor:
        return self.decoder(task.tensor)
