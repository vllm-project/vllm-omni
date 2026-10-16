# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU RMSNorm operator override for MAGI-2."""

from __future__ import annotations

import os

import torch

from ...common_layers import MultiModalityRMSNorm as NativeMultiModalityRMSNorm

_USE_NPU_RMS_NORM = os.environ.get("MAGI2_USE_NPU_RMS_NORM", "1") != "0"


class MultiModalityRMSNorm(NativeMultiModalityRMSNorm):
    """Use ``torch_npu.npu_rms_norm`` when its MAGI-2 contract is supported."""

    def _normalize(
        self,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        reduce_dims: int | tuple[int, ...],
    ) -> torch.Tensor:
        if (
            not _USE_NPU_RMS_NORM
            or tensor.device.type != "npu"
            or self.num_patterns != 1
            or tensor.shape[-1] != self.dim
        ):
            return super()._normalize(tensor, weight, reduce_dims)
        if self.num_modality > 1 and tensor.shape[0] == 0:
            return tensor.float()

        import torch_npu

        return torch_npu.npu_rms_norm(
            tensor.float(),
            gamma=weight.reshape(self.dim).to(torch.float32),
            epsilon=self.eps,
        )[0]


__all__ = ["MultiModalityRMSNorm"]
