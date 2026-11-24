# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU patches for the MiniMax H3 text encoder and diffusion transformer."""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_QKV = "EagleSplitNormRopeConcat"
_SWIGLU = "EagleQuantBatchMatmulSwiglu"

_ROPE_PATCHED = False
_SDPA_PATCHED = False
_SWIGLU_PATCHED = False


def _apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL text RoPE with fused BNSD rotary multiplication."""
    from vllm_omni.platforms.npu.layers.rotary_embedding import npu_rotary_mul_with_bsnd_fallback

    return (
        npu_rotary_mul_with_bsnd_fallback(q, cos, sin, unsqueeze_dim=1),
        npu_rotary_mul_with_bsnd_fallback(k, cos, sin, unsqueeze_dim=1),
    )


def apply_minimax_h3_qwen3vl_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text RoPE to the Ascend fused operator."""
    global _ROPE_PATCHED
    if _ROPE_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
    _ROPE_PATCHED = True
    logger.debug("Applied NPU fused RoPE patch for MiniMax H3 Qwen3-VL text encoder")


def _scaled_dot_product_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run causal SDPA with compressed K/V heads through NPU native GQA."""
    num_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    num_value_heads = value.shape[1]
    if num_key_value_heads != num_value_heads:
        raise ValueError(
            "GQA requires key and value to have the same number of heads, "
            f"got k_heads={num_key_value_heads} and v_heads={num_value_heads}."
        )
    if num_key_value_heads == 0:
        raise ValueError("GQA requires at least one KV head.")
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            "GQA requires query heads to be a multiple of KV heads, "
            f"got q_heads={num_heads} and kv_heads={num_key_value_heads}."
        )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=num_heads != num_key_value_heads,
    )


def apply_minimax_h3_qwen3vl_sdpa_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text attention to NPU native GQA."""
    global _SDPA_PATCHED
    if _SDPA_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._scaled_dot_product_attention = _scaled_dot_product_attention_npu
    _SDPA_PATCHED = True
    logger.debug("Applied NPU SDPA patch for MiniMax H3 Qwen3-VL text encoder")


def npu_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply fused SwiGLU to a packed gate/up projection tensor."""
    import torch_npu

    return torch_npu.npu_swiglu(gate_up, dim=-1)


def _forward_minimax_h3_qwen3vl_text_mlp_npu(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Run the Qwen3-VL MLP with one packed GEMM and fused SwiGLU."""
    gate_up = self.gate_up_proj(x)
    return self.down_proj(npu_swiglu_from_packed(gate_up))


def apply_minimax_h3_qwen3vl_swiglu_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text MLP to the Ascend fused SwiGLU path."""
    global _SWIGLU_PATCHED
    if _SWIGLU_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder.MiniMaxH3Qwen3VLTextMLP.forward = _forward_minimax_h3_qwen3vl_text_mlp_npu  # type: ignore[method-assign]
    _SWIGLU_PATCHED = True


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
            if op == _QKV and inputs[0]["dtype"] == "bfloat16":
                return True
            if (
                op == _SWIGLU
                and inputs[0]["dtype"] == inputs[1]["dtype"] == "int8"
                and inputs[1]["format"] == "ND"
                and inputs[2]["dtype"] == "float32"
            ):
                return True
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            continue
    return False


def _npu_project_qkv(
    self: Any, x: torch.Tensor, rope_table: torch.Tensor | None, vdn_window=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = [self.num_heads, self.num_kv_heads, self.num_kv_heads]
    rows = x.shape[0]
    qkv_numel = rows * sum(heads) * self.head_dim
    eps = self.q_norm.variance_epsilon
    if (
        x.dtype != torch.bfloat16
        or rope_table is None
        or (self.vdn is not None and vdn_window is None)
        or not 0 < qkv_numel <= torch.iinfo(torch.int32).max
    ):
        return type(self)._project_qkv(self, x, rope_table, vdn_window)

    qkv, _ = self.qkv_proj(x)
    packed = qkv.view(rows, sum(heads), self.head_dim)
    q_raw, k_raw, v = packed.split(heads, dim=1)
    try:
        from mindiesd.fused_ops import eagle_split_norm_rope_concat

        # Eagle takes contiguous cosine and sine halves of H3's packed RoPE table.
        cos, sin = rope_table.reshape(rows, 2, self.rot_dim // 2).transpose(0, 1).contiguous().unbind(0)
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
        from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

        q, k = fused_qk_norm_rope(q_raw, k_raw, self.q_norm.weight, self.k_norm.weight, rope_table, eps)
    return q, k, v, q_raw, k_raw


def forward_h3_mlp(self: Any, x: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

    # LoRA can wrap fc1 after the platform installs this forward.
    if (
        x.dtype != torch.bfloat16
        or type(self.fc1) is not MergedColumnParallelLinear
        or not isinstance(self.fc1.quant_method, NPUInt8LinearMethod)
    ):
        return type(self).forward(self, x)

    try:
        import torch_npu
        from mindiesd.fused_ops import eagle_quant_batch_matmul_swiglu

        quantized_x, scale = torch_npu.npu_dynamic_quant(x.reshape(-1, x.shape[-1]))
        hidden = eagle_quant_batch_matmul_swiglu(
            quantized_x, self.fc1.weight, self.fc1.weight_scale, pertoken_scale=scale, output_dtype=x.dtype
        )
    except ImportError:
        return type(self).forward(self, x)

    out, _ = self.fc2(hidden.reshape(*x.shape[:-1], -1))
    return out


def apply_minimax_h3_dit_fusions(model: torch.nn.Module, early_opp_paths: frozenset[str]) -> None:
    """Bind eligible H3 blocks without changing classes or loading MindIE-SD on other platforms."""
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention, MiniMaxH3MLP

    try:
        from vllm_ascend.utils import get_ascend_device_type

        soc = {"A2": "ascend910b", "A5": "ascend950"}.get(get_ascend_device_type().name)
    except (ImportError, KeyError, RuntimeError):
        return
    if soc is None or not early_opp_paths:
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
    try:
        op_info = json.loads((tbe / "config" / soc / f"aic-{soc}-ops-info.json").read_text(encoding="utf-8"))
        index = json.loads((kernels / "config" / soc / "binary_info_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(index, dict) or not isinstance(op_info, dict):
        return
    use_qkv = _QKV in op_info and _has_eagle_kernel(kernels, _QKV, index.get(_QKV))
    use_swiglu = soc == "ascend910b" and _SWIGLU in op_info and _has_eagle_kernel(kernels, _SWIGLU, index.get(_SWIGLU))
    for module in model.modules():
        if use_qkv and isinstance(module, MiniMaxH3Attention):
            module._project_qkv = MethodType(_npu_project_qkv, module)  # type: ignore[method-assign]
        elif use_swiglu and isinstance(module, MiniMaxH3MLP):
            module.forward = MethodType(forward_h3_mlp, module)  # type: ignore[method-assign]
