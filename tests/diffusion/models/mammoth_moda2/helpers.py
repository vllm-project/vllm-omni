# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch


@torch.no_grad()
def initialize_block_weights(block: torch.nn.Module) -> None:
    """Populate standalone blocks whose vLLM linears normally load a checkpoint.

    Callers seed torch before constructing the block for reproducible weights.
    """
    for name, parameter in block.named_parameters():
        if parameter.ndim > 1:
            torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        elif name.endswith("bias"):
            parameter.zero_()
        else:
            parameter.fill_(1.0)
