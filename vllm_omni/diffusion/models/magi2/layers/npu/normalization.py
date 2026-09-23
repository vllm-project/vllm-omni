# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend RMSNorm implementation for MAGI-2."""

from __future__ import annotations

import os

import torch

from ..native.dispatcher import ModalityDispatcher
from ..native.normalization import MultiModalityRMSNorm as NativeMultiModalityRMSNorm

_USE_NPU_RMS_NORM = os.environ.get("MAGI2_USE_NPU_RMS_NORM", "1") != "0"


class MultiModalityRMSNorm(NativeMultiModalityRMSNorm):
    """Use ``torch_npu.npu_rms_norm`` when its MAGI-2 contract is supported."""

    def forward(
        self,
        tensor: torch.Tensor,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        if not _USE_NPU_RMS_NORM or tensor.device.type != "npu" or self.num_patterns != 1:
            return super().forward(tensor, modality_dispatcher)

        import torch_npu

        original_dtype = tensor.dtype
        compute = tensor.float()
        weights = (self.weight.view(self.num_modality, self.dim) + 1.0).to(compute.dtype)
        if self.num_modality == 1:
            result = torch_npu.npu_rms_norm(compute, weights[0], epsilon=self.eps)[0]
        else:
            if modality_dispatcher is None:
                raise ValueError("modality_dispatcher is required for multimodal RMSNorm")
            inputs = modality_dispatcher.dispatch(compute)
            outputs = [
                part
                if part.shape[0] == 0
                else torch_npu.npu_rms_norm(part, weights[index], epsilon=self.eps)[0]
                for index, part in enumerate(inputs)
            ]
            result = modality_dispatcher.undispatch(*outputs)
        return result.to(self.out_dtype or original_dtype)


__all__ = ["MultiModalityRMSNorm"]
