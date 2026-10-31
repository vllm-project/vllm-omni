# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local Marlin scratch for Boogu's serial online-FP8 MLLM calls.

Reuse requires the same CUDA stream or caller-established GPU dependencies;
unordered cross-stream use is unsupported. Boogu's default model runner does not compile or capture the MLLM.
"""

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_MAX_BLOCKS_PER_SM,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear

_WORKSPACE_NAME = "_boogu_marlin_workspace"
logger = init_logger(__name__)


class _BooguMarlinFP8Kernel(MarlinFP8ScaledMMLinearKernel):
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        workspace = marlin_make_workspace_new(layer.weight.device, MARLIN_MAX_BLOCKS_PER_SM)
        layer.register_buffer(_WORKSPACE_NAME, workspace, persistent=False)

    def apply_weights(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=getattr(layer, self._block_scale_name(layer)),
            workspace=getattr(layer, _WORKSPACE_NAME),
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            input_dtype=self.marlin_input_dtype,
            bias=bias,
        )


def bind_boogu_marlin_workspaces(mllm: nn.Module) -> None:
    """Adapt Boogu's already-selected online per-block Marlin kernels before loading."""
    for layer in mllm.modules():
        method = getattr(layer, "quant_method", None)
        if type(method) is not Fp8PerBlockOnlineLinearMethod:
            continue
        selected = getattr(method, "fp8_linear", None)
        if type(selected) is not MarlinFP8ScaledMMLinearKernel:
            continue
        local = _BooguMarlinFP8Kernel(selected.config, selected.layer_param_names)
        local.marlin_input_dtype = selected.marlin_input_dtype
        method.fp8_linear = local
