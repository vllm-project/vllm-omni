# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA

from .base_linear import DiffusionBaseLinearLayerWithLoRA


class DiffusionReplicatedLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    ReplicatedLinearWithLoRA,
):
    """
    Diffusion ReplicatedLinear with LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA
    """

    pass
