# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# for now, it suffices to use vLLM's implementation directly
# as this is a user-facing variable, defined here to so that user can directly import LoRARequest from vllm_omni
import torch
from vllm.lora.request import LoRARequest


class TensorLoRARequest(LoRARequest):
    """Native in-process adapter payload for diffusion merged updates.

    ``lora_tensors`` contains PEFT-named A/B tensors in checkpoint (global)
    layout. ``peft_config`` is the corresponding adapter configuration. The
    inherited path is unused; IPC sessions transport the tensors separately.
    """

    lora_tensors: dict[str, torch.Tensor] | None = None
    peft_config: dict | None = None


__all__ = ["LoRARequest", "TensorLoRARequest"]
