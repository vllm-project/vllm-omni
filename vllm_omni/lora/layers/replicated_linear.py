# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA

from .base_linear import OmniBaseLinearLayerWithLoRA


class OmniReplicatedLinearWithLoRA(
    OmniBaseLinearLayerWithLoRA,
    ReplicatedLinearWithLoRA,
):
    """
    ReplicatedLinear with the shared Omni LoRA merge binding.
    """

    pass


DiffusionReplicatedLinearWithLoRA = OmniReplicatedLinearWithLoRA
