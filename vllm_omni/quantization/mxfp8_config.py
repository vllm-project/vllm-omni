# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W8A8 MXFP8 (Microscaling FP8) online/offline quantization for diffusion transformers.

Architecture:

  MXFPLinearMethodBase            – platform-agnostic skeleton; defines apply() (reshape only),
                                     _apply_inner() (default single-scale dispatch), and two
                                     abstract ops (_quantize_activation, _quant_matmul).
    NPUMxfp8LinearMethod          – NPU offline: create_weights for pre-quantized checkpoint,
                                     process_weights normalization, and NPU MXFP8 ops.
      NPUMxfp8OnlineLinearMethod  – NPU online: _LazyWeightMixin for create_weights,
                                     overrides process_weights to quantize BF16 → FP8.

XPU reuses vLLM's own MXFP8 methods and adds only what diffusion needs on top.
Offline AutoRound MXFP8 checkpoints never reach this file: they are served by
vLLM's INC path (see inc_config.py).

  VllmMxfp8OnlineLinearMethod   – vLLM Mxfp8OnlineLinearMethod plus the (B, S, K) reshape
  VllmMxfp8OnlineMoEMethod      – vLLM Mxfp8OnlineMoEMethod, re-exported unchanged

Extending to a new platform (e.g. CUDA MXFP8 once the ops are available):

    class CUDAMxfp8LinearMethod(MXFPLinearMethodBase):
        def create_weights(self, ...): ...
        def process_weights_after_loading(self, layer): ...
        def _quantize_activation(self, x): ...   # CUDA FP8 quant op
        def _quant_matmul(self, ...): ...         # CUDA FP8 GEMM

    class CUDAMxfp8OnlineLinearMethod(_LazyWeightMixin, CUDAMxfp8LinearMethod):
        def process_weights_after_loading(self, layer): ...  # BF16 → FP8

Extending to a new MX precision with a different calling convention (e.g. dual-scale FP4):

    class NPUMxfp4DualScaleLinearMethod(MXFPLinearMethodBase):
        def _apply_inner(self, layer, x, bias, ori_dtype): ...  # override for 3-tuple path
        def _quantize_activation(self, x, smooth_scale): ...    # returns 3-tuple
        def _quant_matmul(self, x_q, l0, l1, layer, ...): ...  # dual-level GEMM
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.reload.meta import (
    CopyCounter as CopyNumelCounter,
)
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization._copy_missing_attrs import (
    copy_missing_attrs as _copy_missing_attrs,
)

if TYPE_CHECKING:
    from mindiesd.quantization import MXFP8BF16Fallback
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


def _mx_scale_alg() -> int:
    return int(os.environ.get("VLLM_OMNI_MXFP8_SCALE_ALG", "0"))


def _current_denoise_step() -> int | None:
    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    step = get_forward_context().denoise_step_idx
    return None if step is None else int(step)


def _build_mxfp8_bf16_fallback(quant_config: DiffusionMXFP8Config) -> MXFP8BF16Fallback | None:
    if not quant_config.prefix_bf16_steps:
        return None
    try:
        from mindiesd.quantization import MXFP8BF16Fallback, TimestepPolicyConfig
    except ImportError as exc:
        raise ImportError("MXFP8 BF16 prefix fallback requires MindIE-SD with MXFP8BF16Fallback support.") from exc

    policy = TimestepPolicyConfig()
    policy.register(
        range(quant_config.prefix_bf16_steps),
        "W16A16",
        target="mxfp8_linear",
    )
    # DLO is a vLLM-Omni storage lifecycle. MindIE-SD only needs to retain
    # the ordinary CPU master; DLO later streams it through bounded buffers.
    master_mode = "cpu" if quant_config.prefix_bf16_master_mode == "dlo" else quant_config.prefix_bf16_master_mode
    return MXFP8BF16Fallback(policy, master_mode=master_mode)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionMXFP8Config(QuantizationConfig):
    """W8A8 MXFP8 quantization config for diffusion transformers.

    Supports both online (BF16 checkpoint → quantize at load time) and offline
    (pre-quantized MXFP8 checkpoint) modes, mirroring DiffusionInt8Config.

    MX (microscaling) format: groups of 32 K-dimension elements share one
    float8_e8m0fnu exponent scale, matching MXFP8 as defined in the OCP MX spec.
    """

    def __init__(
        self,
        is_checkpoint_mxfp8_serialized: bool = False,
        ignored_layers: list[str] | None = None,
        prefix_bf16_steps: int = 0,
        prefix_bf16_master_mode: str = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(prefix_bf16_steps, bool) or not isinstance(prefix_bf16_steps, int):
            raise TypeError("prefix_bf16_steps must be an integer")
        if prefix_bf16_steps < 0:
            raise ValueError("prefix_bf16_steps must be non-negative")
        if prefix_bf16_master_mode not in ("cpu", "resident", "dlo"):
            raise ValueError(f"Unknown MXFP8 prefix master mode: {prefix_bf16_master_mode}")
        if is_checkpoint_mxfp8_serialized and prefix_bf16_steps:
            raise ValueError("BF16 prefix fallback requires online MXFP8 weights, not a serialized checkpoint")

        self.is_checkpoint_mxfp8_serialized = is_checkpoint_mxfp8_serialized
        self.ignored_layers = ignored_layers or []
        self.prefix_bf16_steps = prefix_bf16_steps
        self.prefix_bf16_master_mode = prefix_bf16_master_mode

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

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
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP8Config:
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_mxfp8_serialized"], False)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_mxfp8_serialized=is_serialized,
            ignored_layers=ignored_layers,
            prefix_bf16_steps=cls.get_from_keys_or(config, ["prefix_bf16_steps"], 0),
            prefix_bf16_master_mode=cls.get_from_keys_or(config, ["prefix_bf16_master_mode"], "cpu"),
        )

    _XPU_OFFLINE_UNSUPPORTED = (
        "Native MXFP8 offline mode is not supported on XPU. "
        "Use AutoRound MXFP8 checkpoints (quant_method='auto-round', data_type='mx_fp') instead, "
        "or use online MXFP8 mode without is_checkpoint_mxfp8_serialized."
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
                if self.is_checkpoint_mxfp8_serialized:
                    return NPUMxfp8LinearMethod(self)
                return NPUMxfp8OnlineLinearMethod(self)
            if current_omni_platform.is_xpu():
                if self.prefix_bf16_steps:
                    raise NotImplementedError("BF16 prefix fallback is only supported for online NPU MXFP8")
                if self.is_checkpoint_mxfp8_serialized:
                    raise NotImplementedError(self._XPU_OFFLINE_UNSUPPORTED)
                return VllmMxfp8OnlineLinearMethod()
            raise NotImplementedError(
                "DiffusionMXFP8Config (W8A8 MXFP8) is currently only supported "
                "on NPU (Ascend) and XPU (Intel) platforms."
            )
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if current_omni_platform.is_xpu():
                if self.is_checkpoint_mxfp8_serialized:
                    raise NotImplementedError(self._XPU_OFFLINE_UNSUPPORTED)
                return VllmMxfp8OnlineMoEMethod(layer=layer)
            # NPU has no MXFP8 MoE op yet; leave the experts in BF16 rather
            # than failing the whole model load.
            return None
        return None


# ---------------------------------------------------------------------------
# _LazyWeightMixin — shared by all online methods
# ---------------------------------------------------------------------------


class _LazyWeightMixin:
    """Weight registered on meta device, materialised just-in-time on first load chunk.

    Platform-agnostic; shared by all online MXFP methods.
    Imported by mxfp4_config.py to avoid duplication.
    """

    uses_meta_device: bool = True

    # This mixin knows when a layer's weight is final, so it can hand the layer
    # back to the host right there. Loaders that intend to offload the whole
    # model after loading opt in per layer via ``enable_offload_after_quant``.
    supports_offload_after_quant: bool = True
    _offload_after_quant: bool = False

    def enable_offload_after_quant(self) -> None:
        """Return each layer to host memory as soon as it has been quantized.

        Caps the load-time device footprint at one layer instead of the whole
        model. A quant method instance belongs to a single layer, so this is not
        a global switch.
        """
        self._offload_after_quant = True

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

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                weight = ModelWeightParameter(
                    data=torch.empty_like(layer.weight, device=layer._load_device),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                _copy_missing_attrs(layer.weight, weight)
                layer.register_parameter("weight", weight)
                del layer._load_device

            param = layer.weight
            counter = CopyNumelCounter()
            with counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)  # type: ignore[misc]
            layer._loaded_numel += counter.copied_numel

            if layer._loaded_numel == layer.weight.numel():
                self.process_weights_after_loading(layer)  # type: ignore[attr-defined]
                layer._already_called_process_weights_after_loading = True

                # This layer's weight is final, so nothing needs it on the
                # accelerator until inference.
                if self._offload_after_quant:
                    layer.to("cpu")
            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer._load_device = torch.get_default_device()
        layer.register_parameter("weight", weight)


# ---------------------------------------------------------------------------
# Abstract base — platform-agnostic apply skeleton
# ---------------------------------------------------------------------------


class MXFPLinearMethodBase(LinearMethodBase, ABC):
    """Platform-agnostic MXFP linear method base.

    Defines the apply() skeleton (flatten → _apply_inner → reshape) and three
    hooks that subclasses implement:

      _quantize_activation(x)                              → tuple  (arity depends on variant)
      _quant_matmul(x_q, x_scale, layer, bias, ori_dtype)  → Tensor (single-scale default)
      _apply_inner(layer, x, bias, ori_dtype)               → Tensor (override for dual-scale)

    Extension guide:
      Single-scale (MXFP8, MXFP4): implement _quantize_activation + _quant_matmul only.
      Dual-scale (MXFP4 DualScale): override _apply_inner for a different calling convention.
      Subclasses must NOT override apply() — reshape logic lives here exclusively.
    """

    @abstractmethod
    def _quantize_activation(self, x: torch.Tensor) -> tuple:
        """Quantize 2-D activation. Return arity must match what _apply_inner expects."""

    @abstractmethod
    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fused MXFP quantized GEMM. Weight and scale accessed from layer."""

    def _apply_inner(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Default single-scale inner loop: 2-tuple quantize → matmul.

        Override this (not apply()) when a different quantize/matmul convention
        is needed, e.g. dual-scale variants that return a 3-tuple from
        _quantize_activation and pass extra tensors to _quant_matmul.
        """
        x_q, x_scale = self._quantize_activation(x)
        return self._quant_matmul(x_q, x_scale, layer, bias, ori_dtype)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared apply skeleton: reshape → _apply_inner → unreshape.

        Do NOT override this method. Override _apply_inner() instead.
        """
        ori_shape = x.shape
        ori_dtype = x.dtype
        x = x.reshape(-1, ori_shape[-1])
        output = self._apply_inner(layer, x, bias, ori_dtype)
        return output.reshape(*ori_shape[:-1], -1)


# ---------------------------------------------------------------------------
# NPU MXFP8 offline method (pre-quantized checkpoint)
# ---------------------------------------------------------------------------


class NPUMxfp8LinearMethod(MXFPLinearMethodBase):
    """NPU W8A8 MXFP8 offline linear method for pre-quantized checkpoints.

    Weight canonical layout after process_weights_after_loading:
      weight      : (K, N)              float8_e4m3fn   – pre-transposed for GEMM
      weight_scale: (K_groups/2, N, 2)  float8_e8m0fnu  – reshaped + pre-transposed

    NPUMxfp8OnlineLinearMethod normalises to the same layout so apply() is shared.
    """

    def __init__(self, quant_config: DiffusionMXFP8Config) -> None:
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
        """Register weight and per-group MX scale for a pre-quantized checkpoint."""
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

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
                input_dim=None,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        """Cast checkpoint weight to FP8 and normalise to canonical GEMM layout."""
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        # Weight: BF16 → float8_e4m3fn via npu_dtype_cast, then transpose (N,K) → (K,N).
        w = layer.weight
        if w.dtype != torch_npu.float8_e4m3fn:
            w = torch_npu.npu_dtype_cast(w.npu(), torch_npu.float8_e4m3fn)
        w = w.transpose(0, 1).contiguous()

        # Scale: checkpoint stores uint8 bytes that ARE float8_e8m0fnu bits.
        # Only convert if neither uint8 nor the target NPU dtype already.
        # Pad K_groups to even so the (K_groups/2, N, 2) reshape is always valid.
        s = layer.weight_scale.data
        if s.dtype not in (torch.uint8, torch_npu.float8_e8m0fnu):
            s = s.to(torch_npu.float8_e8m0fnu)
        N, K_groups = s.shape
        if K_groups % 2 == 1:
            s = torch.cat([s, torch.zeros(N, 1, dtype=s.dtype, device=s.device)], dim=1)
            K_groups += 1
        s = s.reshape(N, K_groups // 2, 2).transpose(0, 1).contiguous()

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        layer._already_called_process_weights_after_loading = True

    # --- NPU MXFP8 ops — shared with online path via inheritance ---

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        return torch_npu.npu_dynamic_mx_quant(
            x,
            dst_type=torch_npu.float8_e4m3fn,
            scale_alg=_mx_scale_alg(),
        )

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        import torch_npu

        # NPU npu_quant_matmul requires bias in float32.
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        return torch_npu.npu_quant_matmul(
            x_q,
            layer.weight,  # (K, N) float8_e4m3fn
            layer.weight_scale,  # (S/2, N, 2) float8_e8m0fnu
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=ori_dtype,
            group_sizes=[1, 1, 32],
        )


# ---------------------------------------------------------------------------
# NPU MXFP8 online method (BF16 checkpoint → quantize at load time)
# ---------------------------------------------------------------------------


class NPUMxfp8OnlineLinearMethod(_LazyWeightMixin, NPUMxfp8LinearMethod):
    """NPU W8A8 MXFP8 online linear method.

    MRO: NPUMxfp8OnlineLinearMethod → _LazyWeightMixin → NPUMxfp8LinearMethod
         → MXFPLinearMethodBase → LinearMethodBase

      create_weights   : _LazyWeightMixin      (meta device + patched loader)
      process_weights  : NPUMxfp8OnlineLinearMethod  (BF16 → FP8 + normalize)
      apply / ops      : NPUMxfp8LinearMethod / MXFPLinearMethodBase  (shared)
    """

    def __init__(self, quant_config: DiffusionMXFP8Config) -> None:
        super().__init__(quant_config)
        self._bf16_fallback = _build_mxfp8_bf16_fallback(quant_config)

    def get_layerwise_prefix_weights(self, layer: Module):
        if not self.quant_config.prefix_bf16_steps or self.quant_config.prefix_bf16_master_mode != "dlo":
            return None
        master = self._bf16_fallback.get_master(layer)
        if master is None:
            raise RuntimeError("DLO BF16 prefix requested without the original master weight")
        return (
            self.quant_config.prefix_bf16_steps,
            {"_mxfp8_prefix_master": master},
            ("weight", "weight_scale"),
        )

    def _apply_inner(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._bf16_fallback is not None:
            step = _current_denoise_step()
            if self._bf16_fallback.use_bf16(step):
                master = self._bf16_fallback.get_master(layer)
                if getattr(layer, "_layerwise_prefix_managed", False) and (
                    master is None or master.device != x.device or master.numel() == 0
                ):
                    raise RuntimeError("DLO BF16 prefix master was not prefetched for this block")
                output = self._bf16_fallback.apply(layer, x, bias, ori_dtype, step)
                if output is not None:
                    return output
        return super()._apply_inner(layer, x, bias, ori_dtype)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        # Materialise weight if still on meta device (dummy-weight init path).
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

        if self._bf16_fallback is not None:
            self._bf16_fallback.save_master(layer)
            layer._mxfp8_prefix_master_mode = self.quant_config.prefix_bf16_master_mode

        # NPU: quantize BF16/FP16 (N, K) → FP8 (N, K) + MX scale (N, S).
        weight_fp8, weight_scale_raw = torch_npu.npu_dynamic_mx_quant(
            layer.weight,
            dst_type=torch_npu.float8_e4m3fn,
            scale_alg=_mx_scale_alg(),
        )

        # Normalize to canonical layout shared with offline path.
        weight_scale = weight_scale_raw.reshape(weight_scale_raw.shape[0], -1, 2).transpose(0, 1).contiguous()
        weight_fp8 = weight_fp8.transpose(0, 1).contiguous()

        replace_parameter(layer, "weight", weight_fp8)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer._already_called_process_weights_after_loading = True


# ---------------------------------------------------------------------------
# XPU method — vLLM's online MXFP8 method plus what diffusion needs on top
#
# Weight quantization, the e8m0 scale relayout and the oneDNN fp8_gemm call are
# all vLLM's (Mxfp8OnlineLinearMethod -> XPUMxFp8LinearKernel), and so is the
# layer-by-layer meta-device loading. Only the (B, S, K) activation reshape is
# added here.
# ---------------------------------------------------------------------------

try:
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineMoEMethod as VllmMxfp8OnlineMoEMethod,
    )
except ImportError as e:
    raise ImportError(
        "vLLM MXFP8 support is not available. "
        "MXFP8 quantization requires vLLM with Mxfp8OnlineLinearMethod support. "
        "Please upgrade vLLM or use a compatible build."
    ) from e


class VllmMxfp8OnlineLinearMethod(Mxfp8OnlineLinearMethod):
    """Online MXFP8 linear method: BF16 checkpoint, quantized at load time.

    Weight creation (meta device, materialized and quantized layer by layer),
    the BF16 -> MXFP8 quantization and the GEMM are all vLLM's.
    """

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flatten the diffusion (B, S, K) hidden states the MX GEMMs reject."""
        if x.dim() <= 2:
            return super().apply(layer, x, bias)
        ori_shape = x.shape
        output = super().apply(layer, x.reshape(-1, ori_shape[-1]), bias)
        return output.reshape(*ori_shape[:-1], -1)
