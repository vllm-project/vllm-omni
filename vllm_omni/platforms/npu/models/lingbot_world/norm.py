# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch



def projection_rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the flat TP1 projection before splitting into heads."""
    import torch_npu

    return torch_npu.npu_rms_norm(value, weight, epsilon=eps)[0]
