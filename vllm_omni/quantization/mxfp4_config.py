# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A4 MXFP4 (Microscaling FP4) online/offline quantization for diffusion transformers.

Architecture mirrors mxfp8_config.py:

  MXFPLinearMethodBase                    – platform-agnostic skeleton (imported from mxfp8_config)
    NPUMxfp4LinearMethod                  – NPU single-scale offline (W4A4 MXFP4)
      NPUMxfp4OnlineLinearMethod          – NPU single-scale online (BF16 → FP4)
    ROCmMxfp4LinearMethod                 – ROCm base class (AITER quant + shuffle; online-only)
      ROCmMxfp4OnlineLinearMethod         – ROCm online (BF16 → FP4 via AITER)
    NPUMxfp4DualScaleLinearMethod         – NPU dual-scale offline (W4A4 MXFP4 DualScale)
      NPUMxfp4DualScaleOnlineLinearMethod – NPU dual-scale online (BF16 → FP4)

Quantization configs:

  DiffusionMXFP4Config            – single-scale online/offline (quant_method="mxfp4")
  DiffusionMXFP4DualScaleMixedConfig – dual-scale + per-layer BF16 fallback (quant_method="mxfp4_dualscale")
      Offline: ignored_layers from config.json routes interleaved BF16 layers
      Online:  num_bf16_fallback_layers leading blocks stay in BF16 (default 5)

Key differences from MXFP8:

  1. Precision: float4_e2m1fn_x2 (FP4 packed, 2 values per element).
     npu_dynamic_mx_quant(x) without dst_type defaults to float4_e2m1fn_x2.

  2. Weight layout: stored as (N, K) — NOT pre-transposed.
     FP4 uses a packed format; transposing a packed tensor is not safely contiguous.
     Transpose is done inline in _quant_matmul via layer.weight.transpose(0, 1).

  3. GEMM signature: npu_quant_matmul requires explicit
     x1_dtype=float4_e2m1fn_x2 and x2_dtype=float4_e2m1fn_x2.

  Scale layout: (N, S/2, 2) — same reshape as MXFP8, also NOT pre-transposed;
  transposed inline in _quant_matmul.

Dual-scale (W4A4_MXFP4_DUALSCALE):

  Two-level quantization: fine scale (per-32 K) + coarse scale (per-512 K) + per-channel
  activation pre-scale (mul_scale from calibration). Uses npu_dynamic_dual_level_mx_quant
  and npu_dual_level_quant_matmul. Weight stored in NZ hardware format via npu_format_cast.

  Checkpoint tensor shapes for dual-scale:
    weight            : (N, K)          float8_e4m3fn     – FP4 packed
    weight_scale      : (N, K//32)      uint8    – fine scale (float8_e8m0fnu bits)
    weight_dual_scale : (N, K//512, 1)  float32  – coarse scale (extra dim avoids shape assert)
    mul_scale         : (K,)            float32  – per-input-channel activation pre-scale

Reference: MindIE-SD W4A4MXFP4QuantLinear / W4A4MXFP4DualQuantLinear (mindiesd/quantization/layer.py).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.mxfp8_config import (
    MXFPLinearMethodBase,
    _LazyWeightMixin,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionMXFP4Config(QuantizationConfig):
    """W4A4 MXFP4 quantization config for diffusion transformers.

    Supports both online (BF16 checkpoint → quantize at load time) and offline
    (pre-quantized MXFP4 checkpoint) modes, mirroring DiffusionMXFP8Config.

    MX (microscaling) format: groups of 32 K-dimension elements share one
    float8_e8m0fnu exponent scale. Weight and activation are float4_e2m1fn_x2.
    """

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = False,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP4Config:
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_mxfp4_serialized"], False)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_mxfp4_serialized=is_serialized,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if current_omni_platform.is_npu():
                if self.is_checkpoint_mxfp4_serialized:
                    return NPUMxfp4LinearMethod(self)
                return NPUMxfp4OnlineLinearMethod(self)
            if current_omni_platform.is_rocm():
                gcn_arch = torch.cuda.get_device_properties(torch.accelerator.current_device_index()).gcnArchName
                if "gfx950" not in gcn_arch:
                    raise NotImplementedError(f"MXFP4 on ROCm requires gfx950 (MI355X). Detected: {gcn_arch}")
                if self.is_checkpoint_mxfp4_serialized:
                    raise NotImplementedError("Pre-quantized MXFP4 checkpoints are not yet supported on ROCm.")
                return ROCmMxfp4OnlineLinearMethod(self)
            raise NotImplementedError(
                "DiffusionMXFP4Config (W4A4 MXFP4) is currently only supported "
                "on NPU (Ascend) and ROCm (AMD, gfx950) platforms."
            )
        return None


# ---------------------------------------------------------------------------
# NPU MXFP4 single-scale offline method (pre-quantized checkpoint)
# ---------------------------------------------------------------------------


class NPUMxfp4LinearMethod(MXFPLinearMethodBase):
    """NPU W4A4 MXFP4 offline linear method for pre-quantized checkpoints.

    Weight canonical layout after process_weights_after_loading:
      weight      : (N, K) in float4_e2m1fn_x2  — NOT pre-transposed (FP4 packed)
      weight_scale: (N, S/2, 2) in float8_e8m0fnu — NOT pre-transposed

    Both are transposed inline in _quant_matmul, unlike MXFP8 which pre-transposes.
    NPUMxfp4OnlineLinearMethod normalizes to the same layout so apply() is shared.
    """

    def __init__(self, quant_config: DiffusionMXFP4Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # BF16 placeholder; cast to float4_e2m1fn_x2 in process_weights.
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        # Scale stored as uint8 in safetensors (float8_e8m0fnu is same bit width).
        # Using uint8 avoids a lossy float32 round-trip when loading the checkpoint.
        num_groups = (input_size_per_partition + 31) // 32
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, num_groups, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        # NPU: cast to float4_e2m1fn_x2. Weight stays (N, K) — no pre-transpose.
        w = layer.weight
        if w.dtype != torch_npu.float4_e2m1fn_x2:
            w = torch_npu.npu_dtype_cast(w.npu(), torch_npu.float4_e2m1fn_x2)

        # Scale: checkpoint stores uint8 bytes that ARE float8_e8m0fnu bits.
        # Only convert if neither uint8 nor the target NPU dtype already.
        # (N, K_groups) → (N, K_groups/2, 2). Not pre-transposed; done inline in _quant_matmul.
        s = layer.weight_scale.data
        if s.dtype not in (torch.uint8, torch_npu.float8_e8m0fnu):
            s = s.to(torch_npu.float8_e8m0fnu)
        N, K_groups = s.shape
        if K_groups % 2 == 1:
            s = torch.cat([s, torch.zeros(N, 1, dtype=s.dtype, device=s.device)], dim=1)
        s = s.reshape(N, -1, 2).contiguous()

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        layer._already_called_process_weights_after_loading = True

    # --- NPU MXFP4 ops — shared with online path via inheritance ---

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        # No dst_type: npu_dynamic_mx_quant defaults to float4_e2m1fn_x2.
        return torch_npu.npu_dynamic_mx_quant(x)

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        import torch_npu

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        # FP4 differences vs FP8:
        #   weight (N,K) transposed inline → (K,N); scale (N,S/2,2) transposed inline → (S/2,N,2).
        #   x1_dtype / x2_dtype required — FP4 dtype not inferred from tensor dtype.
        return torch_npu.npu_quant_matmul(
            x_q,
            layer.weight.transpose(0, 1),  # (K, N) inline
            layer.weight_scale.transpose(0, 1),  # (S/2, N, 2) inline
            scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=ori_dtype,
            group_sizes=[1, 1, 32],
        )


# ---------------------------------------------------------------------------
# NPU MXFP4 single-scale online method (BF16 checkpoint → quantize at load time)
# ---------------------------------------------------------------------------


class NPUMxfp4OnlineLinearMethod(_LazyWeightMixin, NPUMxfp4LinearMethod):
    """NPU W4A4 MXFP4 online linear method.

    MRO: NPUMxfp4OnlineLinearMethod → _LazyWeightMixin → NPUMxfp4LinearMethod
         → MXFPLinearMethodBase → LinearMethodBase

      create_weights  : _LazyWeightMixin          (meta device + patched loader)
      process_weights : NPUMxfp4OnlineLinearMethod (BF16 → FP4 + normalize)
      apply / ops     : NPUMxfp4LinearMethod / MXFPLinearMethodBase (shared)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # NPU: quantize BF16/FP16 (N, K) → FP4. No dst_type → float4_e2m1fn_x2.
        weight_fp4, weight_scale_raw = torch_npu.npu_dynamic_mx_quant(layer.weight)

        # Weight stays (N, K) — no pre-transpose for FP4 packed format.
        # Scale: (N, S) → (N, S/2, 2). Not pre-transposed; done inline in _quant_matmul.
        weight_scale = weight_scale_raw.reshape(weight_scale_raw.shape[0], -1, 2).contiguous()

        replace_parameter(layer, "weight", weight_fp4)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer._already_called_process_weights_after_loading = True


# ---------------------------------------------------------------------------
# ROCm MXFP4 base method (AITER) — used as base class for the online subclass.
# Online-only for now; offline (pre-quantized checkpoint) not yet supported.
# ---------------------------------------------------------------------------


def _register_rocm_mxfp4_op() -> None:
    """Register the vllm_omni::rocm_mxfp4_gemm custom op for torch.compile.

    Wraps activation quantization + GEMM in a single opaque op so that
    torch.compile/inductor doesn't try to trace through AITER internals.
    """
    import aiter

    @torch.library.custom_op("vllm_omni::rocm_mxfp4_gemm", mutates_args=())
    def _rocm_mxfp4_gemm(
        a: torch.Tensor,
        w_quant: torch.Tensor,
        w_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        a_quant, a_scale = quant_func(a, shuffle=True)
        return aiter.gemm_a4w4(
            a_quant,
            w_quant,
            a_scale,
            w_scale,
            bpreshuffle=True,
            bias=bias,
        )

    @_rocm_mxfp4_gemm.register_fake
    def _rocm_mxfp4_gemm_fake(
        a: torch.Tensor,
        w_quant: torch.Tensor,
        w_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M, _ = a.shape
        N, _ = w_quant.shape
        return torch.empty(M, N, dtype=a.dtype, device=a.device)


class ROCmMxfp4LinearMethod(MXFPLinearMethodBase):
    """ROCm W4A4 MXFP4 linear method using AITER (gemm_a4w4).

    Weight buffers after process_weights_after_loading:
      weight_shuffle : FP4 quantized + shuffled via shuffle_weight(layout=(16,16))
      weight_scale   : per-group-of-32 scales from AITER per_1x32

    Forward path:
      _quantize_activation: pass-through (activation quant is inside the custom op)
      _quant_matmul: torch.ops.vllm_omni.rocm_mxfp4_gemm wraps activation quant
                     then gemm_a4w4 in one custom op
    """

    def __init__(self, quant_config: DiffusionMXFP4Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        if not hasattr(torch.ops.vllm_omni, "rocm_mxfp4_gemm"):
            _register_rocm_mxfp4_op()

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import aiter
        from aiter.ops.shuffle import shuffle_weight

        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_quant, weight_scale = quant_func(layer.weight.data, shuffle=True)
        weight_shuffled = shuffle_weight(weight_quant, layout=(16, 16))

        # Store quantized tensors as non-parameter buffers; delete original weight.
        layer.register_buffer("weight_shuffle", weight_shuffled, persistent=True)
        layer.register_buffer("weight_scale", weight_scale, persistent=True)

        # Remove the original weight parameter to free memory.
        if hasattr(layer, "weight") and isinstance(layer.weight, torch.nn.Parameter):
            delattr(layer, "weight")
            layer.register_parameter("weight", None)

        layer._already_called_process_weights_after_loading = True

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Activation quantization is inside the custom op called by _quant_matmul,
        # so pass through the raw activation here.
        return x, None

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.ops.vllm_omni.rocm_mxfp4_gemm(
            x_q,
            layer.weight_shuffle,
            layer.weight_scale,
            bias,
        )
        if output.dtype != ori_dtype:
            output = output.to(ori_dtype)
        return output


# ---------------------------------------------------------------------------
# ROCm MXFP4 online method (BF16 checkpoint → quantize at load time)
# ---------------------------------------------------------------------------


class ROCmMxfp4OnlineLinearMethod(_LazyWeightMixin, ROCmMxfp4LinearMethod):
    """ROCm W4A4 MXFP4 online linear method using AITER.

    MRO: ROCmMxfp4OnlineLinearMethod → _LazyWeightMixin → ROCmMxfp4LinearMethod
         → MXFPLinearMethodBase → LinearMethodBase

      create_weights  : _LazyWeightMixin          (meta device + patched loader)
      process_weights : ROCmMxfp4OnlineLinearMethod  (meta → materialize, then AITER quant + shuffle)
      apply / ops     : ROCmMxfp4LinearMethod / MXFPLinearMethodBase (shared)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Materialise from meta device if needed (same pattern as NPU online).
        if layer.weight is not None and layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # Delegate to the base class which does AITER quant + shuffle.
        ROCmMxfp4LinearMethod.process_weights_after_loading(self, layer)


# ---------------------------------------------------------------------------
# NPU MXFP4 dual-scale offline method (W4A4_MXFP4_DUALSCALE checkpoint)
# ---------------------------------------------------------------------------


class NPUMxfp4DualScaleLinearMethod(MXFPLinearMethodBase):
    """NPU W4A4 MXFP4 dual-scale offline method for pre-quantized checkpoints.

    Checkpoint tensors and their canonical post-load shapes:
      weight            : (N, K) float8_e4m3fn       – FP4 packed (2 values per byte)
      weight_scale      : (N, K//32) uint8  – fine scale (float8_e8m0fnu bits); reshaped to (N, K//64, 2)
      weight_dual_scale : (N, K//512, 1) float32 – coarse scale; transposed to (K//512, N)
      mul_scale         : (K,) float32      – per-input-channel activation pre-scale (from calibration)

    Forward pass:
      x_q, l0, l1 = npu_dynamic_dual_level_mx_quant(x, smooth_scale=mul_scale)
      out          = npu_dual_level_quant_matmul(x_q, weight, l0, weight_dual_scale, l1, weight_scale)

    Reference: MindIE-SD W4A4MXFP4DualQuantLinear.
    """

    def __init__(self, quant_config: Any) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # FP4 packed: 2 values per float8_e4m3fn byte → checkpoint stores as float8_e4m3fn; register as BF16
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        # Fine scale: one uint8 exponent (float8_e8m0fnu bit pattern) per group of 32 K elements.
        # input_dim=1: RowParallelLinear weight_loader shards along the K-group dimension (dim 1)
        # so each rank receives only its (K/TP)//32 groups; ColumnParallelLinear only uses
        # output_dim=0 for sharding and leaves dim 1 intact.
        num_groups_fine = (input_size_per_partition + 31) // 32
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, num_groups_fine, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        # Coarse scale: one float32 per group of 512 K elements.
        # Shape (N, K_coarse, 1) matches checkpoint layout exactly, avoiding the
        # shape-mismatch assert in linear.py:1344.
        # input_dim=1: same TP sharding rationale as weight_scale above.
        num_groups_coarse = (input_size_per_partition + 511) // 512
        layer.register_parameter(
            "weight_dual_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, num_groups_coarse, 1, dtype=torch.float32),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        # mul_scale is a float32 calibration tensor; register as float32
        # to avoid precision loss from an implicit BF16 cast during weight loading.
        # input_dim=0: RowParallelLinear shards the 1-D per-input-channel tensor along dim 0
        # so each rank receives only its K/TP channels; ColumnParallelLinear leaves it intact.
        layer.register_parameter(
            "mul_scale",
            ModelWeightParameter(
                data=torch.empty(input_size_per_partition, dtype=torch.float32),
                input_dim=0,
                output_dim=None,
                weight_loader=weight_loader,
            ),
        )
        setattr(layer.mul_scale, "ignore_warning", True)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        # float8_e4m3fn (FP4 packed) → float4_e2m1fn_x2 → NZ hardware format (format ID 29).
        # NZ layout is required by npu_dual_level_quant_matmul for FP4 weight matrices.
        w = torch_npu.npu_dtype_cast(layer.weight.data.npu(), torch_npu.float4_e2m1fn_x2)
        w = torch_npu.npu_format_cast(w.view(torch.int8), 29, customize_dtype=torch.int8)

        # Fine scale: (N, K//32) uint8 → cast to float8_e8m0fnu → (N, K//64, 2).
        # The cast reinterprets the stored uint8 bit-patterns as float8_e8m0fnu exponents,
        # which is required for npu_dual_level_quant_matmul to compute correct scale values.
        s = layer.weight_scale.data.npu()
        s = s.reshape(s.shape[0], -1, 2).contiguous()

        # Coarse scale: (N, K//512, 1) → squeeze → (N, K//512) → transpose → (K//512, N).
        ds = layer.weight_dual_scale.data.squeeze(-1).transpose(0, 1).npu().contiguous()

        ms = layer.mul_scale.to(torch.bfloat16).data.view(-1).npu().contiguous()

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        replace_parameter(layer, "weight_dual_scale", ds)
        replace_parameter(layer, "mul_scale", ms)
        layer._already_called_process_weights_after_loading = True

    def _apply_inner(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dual-scale inner loop: dtype guard → 3-tuple quantize (mul_scale as smooth) → matmul.

        Overrides the default single-scale _apply_inner from MXFPLinearMethodBase.
        apply() in the base class handles reshape/unreshape; this method is not
        responsible for that.
        """
        if ori_dtype not in (torch.bfloat16, torch.float16):
            x = x.to(torch.bfloat16)
        x_q, l0_scale, l1_scale = self._quantize_activation(x, layer.mul_scale)
        return self._quant_matmul(x_q, l0_scale, l1_scale, layer, bias, ori_dtype)

    def _quantize_activation(  # type: ignore[override]
        self,
        x: torch.Tensor,
        smooth_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch_npu

        return torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=smooth_scale)

    def _quant_matmul(  # type: ignore[override]
        self,
        x_q: torch.Tensor,
        l0_scale: torch.Tensor,
        l1_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        import torch_npu

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        # weight_scale is (N, K//64, 2) float8_e8m0fnu — operator expects output-major layout.
        # weight_dual_scale is (K//512, N) float32 — transposed to K-major in process_weights.
        return torch_npu.npu_dual_level_quant_matmul(
            x_q,
            layer.weight,
            l0_scale,
            layer.weight_dual_scale,
            l1_scale,
            layer.weight_scale,
            bias=bias,
            output_dtype=ori_dtype,
        )


# ---------------------------------------------------------------------------
# NPU MXFP4 dual-scale online method (BF16 checkpoint → quantize at load time)
# ---------------------------------------------------------------------------


class NPUMxfp4DualScaleOnlineLinearMethod(_LazyWeightMixin, NPUMxfp4DualScaleLinearMethod):
    """NPU W4A4 MXFP4 dual-scale online method: quantises BF16 weights at load time.

    MRO: NPUMxfp4DualScaleOnlineLinearMethod → _LazyWeightMixin
         → NPUMxfp4DualScaleLinearMethod → MXFPLinearMethodBase

      create_weights        : _LazyWeightMixin              (meta device + patched loader)
      process_weights       : NPUMxfp4DualScaleOnlineLinearMethod (BF16 → FP4 + dual scales)
      apply / _quant_matmul : NPUMxfp4DualScaleLinearMethod (shared with offline path)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # Quantize BF16 weight → FP4 + dual-level scales (no smooth pre-scale for online).
        # Returns: (weight_fp4, l0_scale[coarse per-512], l1_scale[fine per-32])
        weight_fp4, w_l0_scale, w_l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
            layer.weight.data.npu(), smooth_scale=None
        )

        # NZ hardware format for the FP4 weight (same as offline path).
        w = torch_npu.npu_format_cast(weight_fp4.view(torch.int8), 29, customize_dtype=torch.int8)

        # Fine scale (l1): (N, K//32) → (N, K//64, 2). Dtype from op output (float8_e8m0fnu).
        s = w_l1_scale.reshape(w_l1_scale.shape[0], -1, 2).contiguous()

        # Coarse scale (l0): (N, K_coarse) → (K_coarse, N). Dtype from op output.
        ds = w_l0_scale.reshape(w_l0_scale.shape[0], -1).transpose(0, 1).contiguous()

        # No calibration available: identity pre-scale (no smooth quantization effect).
        ms = torch.ones(layer.input_size_per_partition, dtype=torch.bfloat16, device="npu")

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        replace_parameter(layer, "weight_dual_scale", ds)
        replace_parameter(layer, "mul_scale", ms)
        layer._already_called_process_weights_after_loading = True


# ---------------------------------------------------------------------------
# Block-index helper (shared by DiffusionMXFP4DualScaleMixedConfig)
# ---------------------------------------------------------------------------

_BLOCK_IDX_RE = re.compile(r"^blocks\.(\d+)\.")


def _parse_block_idx(prefix: str) -> int | None:
    """Extract block index from prefix like 'blocks.5.attn1.to_q'."""
    m = _BLOCK_IDX_RE.match(prefix)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Config: MXFP4 DualScale + per-layer BF16 fallback
# ---------------------------------------------------------------------------


class DiffusionMXFP4DualScaleMixedConfig(QuantizationConfig):
    """W4A4 MXFP4 DualScale with per-layer BF16 fallback for diffusion transformers.

    Sensitive layers fall back to BF16 (original weights) while all other linear
    layers use W4A4 MXFP4 DualScale.  BF16 fallback layers may be interleaved
    anywhere in the transformer.

    Offline mode (is_checkpoint_serialized=True):
        Layers whose prefix appears in ignored_layers → UnquantizedLinearMethod (BF16)
        All other linear layers → NPUMxfp4DualScaleLinearMethod

        ignored_layers is injected into transformer/config.json by the merge script
        and contains the prefixes of all non-MXFP4 linear layers.

    Online mode (is_checkpoint_serialized=False):
        Layer routing applies two rules in priority order:
          1. ignored_layers (explicit per-layer BF16 override, user-supplied) → BF16
          2. Blocks 0 .. num_bf16_fallback_layers-1 (coarse leading-block rule) → BF16
          3. All other linear layers → NPUMxfp4DualScaleOnlineLinearMethod

        num_bf16_fallback_layers defaults to 5 when not specified.
        Set ignored_layers to pin arbitrary interleaved layers to BF16 without
        needing an offline checkpoint (useful for accuracy debugging).
        Layers outside "blocks.N.*" (condition_embedder etc.) always use online MXFP4
        unless they appear in ignored_layers.

    Config injected by merge_mxfp4_dualscale_checkpoint.py:
        {
            "quant_method": "mxfp4_dualscale",
            "is_checkpoint_serialized": true,
            "ignored_layers": ["blocks.0.attn1.to_q", ...]
        }
    """

    def __init__(
        self,
        is_checkpoint_serialized: bool = False,
        ignored_layers: list[str] | None = None,
        num_bf16_fallback_layers: int = 5,
    ) -> None:
        super().__init__()
        self.is_checkpoint_serialized = is_checkpoint_serialized
        self.ignored_layers = ignored_layers or []
        self.num_bf16_fallback_layers = num_bf16_fallback_layers

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4_dualscale"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP4DualScaleMixedConfig:
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_serialized"], False)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        num_bf16_fallback_layers = cls.get_from_keys_or(config, ["num_bf16_fallback_layers"], 5)
        return cls(
            is_checkpoint_serialized=is_serialized,
            ignored_layers=ignored_layers,
            num_bf16_fallback_layers=num_bf16_fallback_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None

        if self.is_checkpoint_serialized:
            # Offline: ignored_layers lists interleaved BF16 fallback layer prefixes.
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not current_omni_platform.is_npu():
                raise NotImplementedError(
                    "DiffusionMXFP4DualScaleMixedConfig is currently only supported on NPU (Ascend) platforms."
                )
            return NPUMxfp4DualScaleLinearMethod(self)

        # Online: explicit ignored_layers take priority (user-specified per-layer BF16 override),
        # then fall back to the coarse leading-block rule.
        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()
        block_idx = _parse_block_idx(prefix)
        if block_idx is not None and block_idx < self.num_bf16_fallback_layers:
            return UnquantizedLinearMethod()

        if not current_omni_platform.is_npu():
            raise NotImplementedError(
                "DiffusionMXFP4DualScaleMixedConfig is currently only supported on NPU (Ascend) platforms."
            )
        return NPUMxfp4DualScaleOnlineLinearMethod(self)
