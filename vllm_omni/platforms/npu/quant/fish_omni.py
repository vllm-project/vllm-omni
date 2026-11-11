# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend block-FP8 experts and linear methods for Drama 3."""

from types import MethodType

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config


def _resolve_experts_on_cpu(self, layer: torch.nn.Module, weight_name: str, scale_name: str) -> torch.Tensor:
    """Expand this method's expert tensor on the host, then return it on the device.

    Upstream expands in place on the device, so the FP8 expert stack and its
    bfloat16 expansion are resident together; on a 64 GB 910B that peak aborts
    the load. Moving both parameters to the host first releases the FP8 device
    storage before the expansion allocates, leaving only the expanded tensor on
    the device. The expansion itself stays with the upstream implementation,
    which keeps its block-scale shape validation and its bounded FP32 staging.
    """
    from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockFusedMoEMethod

    weight_param = getattr(layer, weight_name)
    scale_param = getattr(layer, scale_name)
    device = weight_param.device
    weight_param.data = weight_param.data.to("cpu")
    scale_param.data = scale_param.data.to("cpu")
    if device.type == "npu":
        torch.npu.empty_cache()
    resolved = AscendFp8BlockFusedMoEMethod._resolve_experts(self, layer, weight_name, scale_name)
    return resolved.to(device=device)


def _dequantize_npu_block_fp8_on_cpu(method: QuantizeMethodBase | None) -> None:
    """Scope the CPU expansion workaround to this Drama 3 quantization method.

    Ascend's adapter owns the block-FP8 scheme in ``quant_method``. Override
    only that instance; other models and the Atlas 950 MXFP8 path retain
    their upstream implementation.
    """
    from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockFusedMoEMethod

    scheme = getattr(method, "quant_method", None)
    if isinstance(scheme, AscendFp8BlockFusedMoEMethod):
        scheme._resolve_experts = MethodType(_resolve_experts_on_cpu, scheme)


def get_fish_omni_quant_method(
    config: Fp8Config, layer: torch.nn.Module, prefix: str, tid2eid: torch.Tensor | None
) -> QuantizeMethodBase | None:
    """Ascend's own methods: block-FP8 routed experts, bf16 Ascend linears.

    CUDA's ``Fp8MoEMethod`` selects no Ascend kernel (``Fp8MoeBackend.NONE``),
    and the generic ``UnquantizedLinearMethod`` would skip the FRACTAL_NZ
    weight conversion and the Ascend GEMM that ``AscendLinearBase`` installs
    for an unquantized checkpoint, so both sides need the vLLM-Ascend
    implementation.
    """
    # Delayed imports: vllm_ascend exists only on an Ascend host.
    if isinstance(layer, LinearBase):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        return AscendUnquantizedLinearMethod()
    from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config

    npu_config = AscendFp8Config(
        is_checkpoint_fp8_serialized=config.is_checkpoint_fp8_serialized,
        activation_scheme=config.activation_scheme,
        weight_block_size=config.weight_block_size,
    )
    method = npu_config.get_quant_method(layer, prefix, tid2eid=tid2eid)
    _dequantize_npu_block_fp8_on_cpu(method)
    return method
