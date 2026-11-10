# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MAGI-2 modality adapter for the NPU RMSNorm primitive."""

from __future__ import annotations

import torch
import torch_npu

from ...common_layers import ModalityDispatcher
from ..native.normalization import MultiModalityRMSNorm as NativeMultiModalityRMSNorm


class MultiModalityRMSNorm(NativeMultiModalityRMSNorm):
    """Adapt zero-delta modality weights without adding checkpoint state."""

    def forward(
        self,
        tensor: torch.Tensor,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        if tensor.device.type != "npu" or self.num_patterns != 1 or tensor.shape[-1] != self.dim:
            return super().forward(tensor, modality_dispatcher)

        original_dtype = tensor.dtype
        compute = tensor.float()
        weights = (self.weight.view(self.num_modality, self.dim) + 1.0).to(compute.dtype)
        if self.num_modality == 1:
            result = torch_npu.npu_rms_norm(compute, gamma=weights[0], epsilon=self.eps)[0]
        else:
            if modality_dispatcher is None:
                raise ValueError("modality_dispatcher is required for multimodal RMSNorm")
            inputs = modality_dispatcher.dispatch(compute)
            outputs = [
                part if part.shape[0] == 0 else torch_npu.npu_rms_norm(part, gamma=weights[index], epsilon=self.eps)[0]
                for index, part in enumerate(inputs)
            ]
            result = modality_dispatcher.undispatch(*outputs)
        return result.to(self.out_dtype or original_dtype)


__all__ = ["MultiModalityRMSNorm"]
