# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MindIE-SD fusions for MiniMax H3 diffusion transformer blocks."""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNLayout
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention, MiniMaxH3MLP

_FP32_INFO = torch.finfo(torch.float32)
_QKV = "EagleSplitNormRopeConcat"
_SWIGLU = "EagleQuantBatchMatmulSwiglu"


def _has_eagle_kernel(kernels: Path, op: str, registration: object) -> bool:
    binaries = registration.get("binaryList") if isinstance(registration, dict) else None
    if not isinstance(binaries, list):
        return False
    for entry in binaries:
        try:
            if not (kernels / entry["binPath"]).is_file():
                continue
            descriptor = json.loads((kernels / entry["jsonPath"]).read_text(encoding="utf-8"))["supportInfo"]
            inputs = descriptor["inputs"]
            if descriptor["outputs"][0]["dtype"] != "bfloat16":
                continue
            if op == _QKV:
                if inputs[0]["dtype"] == "bfloat16":
                    return True
            elif (
                inputs[0]["dtype"] == inputs[1]["dtype"] == "int8"
                and inputs[1]["format"] == "ND"
                and inputs[2]["dtype"] == "float32"
            ):
                return True
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            continue
    return False


def _ascend_device_name() -> str | None:
    try:
        from vllm_ascend.utils import get_ascend_device_type

        return get_ascend_device_type().name
    except (ImportError, KeyError, RuntimeError):
        return None


def _npu_project_qkv(
    self: MiniMaxH3Attention, x: torch.Tensor, rope_table: torch.Tensor | None, vdn_window: VDNLayout | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = [self.num_heads, self.num_kv_heads, self.num_kv_heads]
    rows = x.shape[0]
    qkv_numel = rows * sum(heads) * self.head_dim
    eps = self.q_norm.variance_epsilon
    if (
        x.dtype != torch.bfloat16
        or rope_table is None
        or (self.vdn is not None and vdn_window is None)
        or self.head_dim != 128
        or self.rot_dim not in (32, 64, 96, 128)
        or eps != self.k_norm.variance_epsilon
        or not _FP32_INFO.tiny <= eps <= _FP32_INFO.max
        or not 0 < qkv_numel <= torch.iinfo(torch.int32).max
    ):
        return type(self)._project_qkv(self, x, rope_table, vdn_window)
    try:
        from mindiesd.fused_ops import eagle_split_norm_rope_concat
    except ImportError:
        return type(self)._project_qkv(self, x, rope_table, vdn_window)

    qkv, _ = self.qkv_proj(x)
    packed = qkv.view(rows, sum(heads), self.head_dim)
    q_raw, k_raw, v = packed.split(heads, dim=1)
    if qkv.device.type == "npu":
        # H3 packs half-width cos and sin side by side; Eagle takes contiguous halves.
        cos, sin = rope_table.reshape(rows, 2, self.rot_dim // 2).transpose(0, 1).contiguous().unbind(0)
        try:
            q, k, v = eagle_split_norm_rope_concat(
                packed.contiguous(),
                self.q_norm.weight.contiguous(),
                self.k_norm.weight.contiguous(),
                cos,
                sin,
                heads,
                norm_type=3,  # weighted RMSNorm
                rope_mode=1,  # HALF (NeoX)
                rotary_dim=self.rot_dim,
                epsilon=eps,
                output_bnsd=False,
            )
        except ImportError:
            # The facade can discover a missing native symbol after projection.
            pass
        else:
            return q, k, v, q_raw, k_raw
    q, k = fused_qk_norm_rope(q_raw, k_raw, self.q_norm.weight, self.k_norm.weight, rope_table, eps)
    return q, k, v, q_raw, k_raw


def forward_h3_mlp(self: MiniMaxH3MLP, x: torch.Tensor) -> torch.Tensor:
    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

    # LoRA can wrap fc1 after the platform installs this forward.
    if (
        x.device.type == "npu"
        and x.dtype == torch.bfloat16
        and type(self.fc1) is MergedColumnParallelLinear
        and isinstance(self.fc1.quant_method, NPUInt8LinearMethod)
    ):
        try:
            import torch_npu
            from mindiesd.fused_ops import eagle_quant_batch_matmul_swiglu

            quantized_x, scale = torch_npu.npu_dynamic_quant(x.reshape(-1, x.shape[-1]))
            hidden = eagle_quant_batch_matmul_swiglu(
                quantized_x, self.fc1.weight, self.fc1.weight_scale, pertoken_scale=scale, output_dtype=x.dtype
            )
        except ImportError:
            pass
        else:
            out, _ = self.fc2(hidden.reshape(*x.shape[:-1], -1))
            return out
    return type(self).forward(self, x)


def apply_minimax_h3_dit_fusions(model: nn.Module, early_opp_paths: frozenset[str]) -> None:
    """Bind eligible H3 blocks without changing classes or loading MindIE-SD on other platforms."""
    device_name = _ascend_device_name()
    if device_name not in ("A2", "A5") or not early_opp_paths:
        return
    sdk = find_spec("mindiesd")
    if sdk is None or sdk.origin is None:
        return
    vendor = (Path(sdk.origin).parent / "ops/vendors/aie_ascendc").resolve()
    active_paths = early_opp_paths.intersection(os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep))
    if not any(Path(path).resolve() == vendor for path in active_paths):
        return
    tbe = vendor / "op_impl/ai_core/tbe"
    kernels = tbe / "kernel"
    soc = "ascend910b" if device_name == "A2" else "ascend950"
    try:
        op_info = json.loads((tbe / "config" / soc / f"aic-{soc}-ops-info.json").read_text(encoding="utf-8"))
        index = json.loads((kernels / "config" / soc / "binary_info_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(index, dict) or not isinstance(op_info, dict):
        return
    use_qkv = _QKV in op_info and _has_eagle_kernel(kernels, _QKV, index.get(_QKV))
    use_swiglu = device_name == "A2" and _SWIGLU in op_info and _has_eagle_kernel(kernels, _SWIGLU, index.get(_SWIGLU))
    for module in model.modules():
        if use_qkv and isinstance(module, MiniMaxH3Attention):
            module._project_qkv = MethodType(_npu_project_qkv, module)  # type: ignore[method-assign]
        elif use_swiglu and isinstance(module, MiniMaxH3MLP):
            module.forward = MethodType(forward_h3_mlp, module)  # type: ignore[method-assign]
