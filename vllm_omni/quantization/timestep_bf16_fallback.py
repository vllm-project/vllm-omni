# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic timestep-scheduled BF16 fallback for quantized diffusion linears.

Decouples four orthogonal concerns:
1. Step schedule (``TimestepBF16Policy`` reading ``ForwardContext.denoise_step_idx``).
2. Execution interception (``TimestepBF16FallbackLinearMethod`` wrapping any
   ``LinearMethodBase`` so BF16 fallback bypasses low-bit quantization, smooth
   scales, and low-rank paths without duplicating logic per quant format).
3. Weight attachment (``attach_bf16_fallback_weight`` + saved partition metadata
   so offline BF16 delta loaders can reuse the layer's exact TP ``weight_loader``).
4. Residency & streaming transport (delegated to ``PrefixBlockStreamer`` in
   ``stream`` mode or ``DistributedLayerwiseOffloadHook`` in ``dlo`` mode).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter

logger = init_logger(__name__)

MasterMode = Literal["stream", "dlo", "resident"]
VALID_MASTER_MODES: frozenset[str] = frozenset({"stream", "dlo", "resident"})
BF16_FALLBACK_WEIGHT_ATTR = "_bf16_fallback_weight"
BF16_FALLBACK_MANAGED_ATTR = "_bf16_fallback_managed_by"
_QUANT_REPLACEABLE_PARAM_NAMES = (
    "weight",
    "weight_scale",
    "weight_scale_inv",
    "qweight",
    "wscales",
    "wcscales",
    "proj_up",
    "input_scale",
)
_LOGGED_ACTIVE_MODES: set[tuple[str, int]] = set()


def current_denoise_step() -> int | None:
    """Return the active denoising step index from ``ForwardContext``, if any."""
    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    step = get_forward_context().denoise_step_idx
    return None if step is None else int(step)


@dataclass(frozen=True)
class TimestepBF16Policy:
    """Schedule and residency policy for timestep BF16 fallback."""

    steps: frozenset[int]
    master_mode: MasterMode = "stream"

    def __post_init__(self) -> None:
        if self.master_mode not in VALID_MASTER_MODES:
            raise ValueError(
                f"Invalid prefix_bf16_master_mode={self.master_mode!r}; expected one of {sorted(VALID_MASTER_MODES)}"
            )
        for step in self.steps:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"BF16 fallback step indices must be non-negative ints, got {step!r}")
        if self.master_mode == "dlo" and self.steps and self.prefix_steps == 0:
            raise ValueError("prefix_bf16_master_mode='dlo' requires a contiguous 0..K-1 step prefix")

    @classmethod
    def from_prefix(
        cls,
        prefix_bf16_steps: int,
        master_mode: str = "stream",
    ) -> TimestepBF16Policy:
        if isinstance(prefix_bf16_steps, bool) or not isinstance(prefix_bf16_steps, int):
            raise TypeError(f"prefix_bf16_steps must be an int, got {type(prefix_bf16_steps).__name__}")
        if prefix_bf16_steps < 0:
            raise ValueError(f"prefix_bf16_steps must be non-negative, got {prefix_bf16_steps}")
        if master_mode not in VALID_MASTER_MODES:
            raise ValueError(
                f"Invalid prefix_bf16_master_mode={master_mode!r}; expected one of {sorted(VALID_MASTER_MODES)}"
            )
        return cls(
            steps=frozenset(range(prefix_bf16_steps)),
            master_mode=master_mode,  # type: ignore[arg-type]
        )

    @classmethod
    def from_steps(
        cls,
        steps: Iterable[int],
        master_mode: str = "stream",
    ) -> TimestepBF16Policy:
        if master_mode not in VALID_MASTER_MODES:
            raise ValueError(
                f"Invalid prefix_bf16_master_mode={master_mode!r}; expected one of {sorted(VALID_MASTER_MODES)}"
            )
        return cls(
            steps=frozenset(steps),
            master_mode=master_mode,  # type: ignore[arg-type]
        )

    @property
    def enabled(self) -> bool:
        return bool(self.steps)

    @property
    def prefix_steps(self) -> int:
        """Return K if ``steps == {0, ..., K-1}``; otherwise 0."""
        count = len(self.steps)
        if count > 0 and self.steps == frozenset(range(count)):
            return count
        return 0

    def is_active(self, step_idx: int | None) -> bool:
        return step_idx is not None and step_idx in self.steps

    def is_current_step_active(self) -> bool:
        return self.is_active(current_denoise_step())


def attach_bf16_fallback_weight(layer: nn.Module, weight: torch.Tensor) -> None:
    """Attach a materialized 2D ``(N, K)`` BF16 weight onto a wrapped linear layer."""
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError(f"BF16 fallback weight must be a non-empty 2D matrix, got shape={tuple(weight.shape)}")
    weight = weight.detach()
    if weight.dtype != torch.bfloat16:
        weight = weight.to(dtype=torch.bfloat16)
    if not weight.is_contiguous():
        weight = weight.contiguous()
    setattr(layer, BF16_FALLBACK_WEIGHT_ATTR, weight)


def create_cpu_bf16_load_parameter(layer: nn.Module) -> nn.Parameter:
    """Create a temporary CPU BF16 parameter matching ``layer``'s TP partition."""
    shape = getattr(layer, "_bf16_fallback_weight_shape", None)
    if shape is None:
        raise RuntimeError(f"Layer {type(layer).__name__} lacks saved BF16 fallback partition shape")
    weight_loader = getattr(layer, "_bf16_fallback_weight_loader", None)
    extra_attrs: dict[str, Any] = getattr(layer, "_bf16_fallback_extra_weight_attrs", {})
    data = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    try:
        param: nn.Parameter = ModelWeightParameter(
            data=data,
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
    except AssertionError:
        param = nn.Parameter(data, requires_grad=False)
        param.input_dim = 1  # type: ignore[attr-defined]
        param.output_dim = 0  # type: ignore[attr-defined]
        if weight_loader is not None:
            param.weight_loader = weight_loader  # type: ignore[attr-defined]
    for key, value in extra_attrs.items():
        if not hasattr(param, key):
            setattr(param, key, value)
    return param


class TimestepBF16FallbackLinearMethod(LinearMethodBase):
    """Universal ``LinearMethodBase`` decorator for timestep BF16 fallback.

    Wraps any underlying quantized ``LinearMethodBase`` without modifying its
    quantized storage or GEMM path. During active fallback steps, executes pure
    BF16 ``F.linear`` using the layer's materialized ``(N, K)`` BF16 weight;
    during all other steps, forwards straight to ``self.inner.apply()``.
    """

    def __init__(
        self,
        inner: LinearMethodBase,
        policy: TimestepBF16Policy,
        *,
        quant_config: Any | None = None,
    ) -> None:
        if isinstance(inner, TimestepBF16FallbackLinearMethod):
            raise ValueError("Cannot nest TimestepBF16FallbackLinearMethod wrappers")
        self.inner = inner
        self.policy = policy
        self.quant_config = quant_config if quant_config is not None else getattr(inner, "quant_config", None)

    @property
    def uses_meta_device(self) -> bool:
        return bool(getattr(self.inner, "uses_meta_device", False))

    @property
    def supports_pre_processed_weights(self) -> bool:
        return bool(getattr(self.inner, "supports_pre_processed_weights", False))

    @property
    def supports_offload_after_quant(self) -> bool:
        return bool(getattr(self.inner, "supports_offload_after_quant", False))

    def enable_offload_after_quant(self) -> None:
        enable_fn = getattr(self.inner, "enable_offload_after_quant", None)
        if callable(enable_fn):
            enable_fn()

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        layer._bf16_fallback_weight_shape = (sum(output_partition_sizes), input_size_per_partition)
        layer._bf16_fallback_weight_loader = extra_weight_attrs.get("weight_loader")
        layer._bf16_fallback_extra_weight_attrs = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        setattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
        setattr(layer, BF16_FALLBACK_MANAGED_ATTR, None)
        self.inner.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.inner.process_weights_after_loading(layer)

    def get_layerwise_prefix_weights(
        self,
        layer: nn.Module,
    ) -> tuple[int, dict[str, torch.Tensor], tuple[str, ...]] | None:
        """Expose BF16 master to ``DistributedLayerwiseOffloadHook`` in DLO mode."""
        if not self.policy.enabled or self.policy.master_mode != "dlo":
            return None
        master = getattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
        if not isinstance(master, torch.Tensor):
            raise RuntimeError("DLO BF16 prefix requested without a materialized CPU BF16 fallback weight")
        replaced = tuple(
            name for name in _QUANT_REPLACEABLE_PARAM_NAMES if name in layer._parameters or name in layer._buffers
        )
        if not replaced:
            raise RuntimeError("DLO BF16 prefix could not find quantized weight parameters to replace on layer")
        return (
            self.policy.prefix_steps,
            {BF16_FALLBACK_WEIGHT_ATTR: master},
            replaced,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        step = current_denoise_step()
        if self.policy.is_active(step):
            assert step is not None
            weight = getattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
            if not isinstance(weight, torch.Tensor) or weight.numel() == 0:
                raise RuntimeError(
                    f"Timestep BF16 fallback active at step={step} (mode={self.policy.master_mode}) "
                    "but layer has no materialized BF16 weight"
                )
            managed_by = getattr(layer, BF16_FALLBACK_MANAGED_ATTR, None)
            if managed_by is None and getattr(layer, "_layerwise_prefix_managed", False):
                managed_by = "dlo"
            if managed_by in ("stream", "dlo", "resident") and weight.device != x.device:
                raise RuntimeError(
                    f"Timestep BF16 fallback weight (device={weight.device}) is not on activation "
                    f"device={x.device} at step={step} (mode={managed_by}); refusing implicit copy"
                )
            if weight.device != x.device:
                weight = weight.to(device=x.device)
            ori_dtype = x.dtype
            compute_dtype = weight.dtype
            x_in = x if x.dtype == compute_dtype else x.to(dtype=compute_dtype)
            bias_in = None
            if bias is not None:
                bias_in = (
                    bias
                    if (bias.device == x_in.device and bias.dtype == compute_dtype)
                    else bias.to(device=x_in.device, dtype=compute_dtype)
                )
            key = (self.policy.master_mode, step)
            if key not in _LOGGED_ACTIVE_MODES:
                _LOGGED_ACTIVE_MODES.add(key)
                logger.info(
                    "Timestep BF16 fallback active: executing BF16 F.linear at denoise_step=%d (mode=%s, weight=%s)",
                    step,
                    self.policy.master_mode,
                    tuple(weight.shape),
                )
            out = F.linear(x_in, weight, bias_in)
            return out if out.dtype == ori_dtype else out.to(dtype=ori_dtype)
        return self.inner.apply(layer, x, bias)


def unwrap_linear_method(method: Any) -> Any:
    """Return the underlying ``LinearMethodBase`` when wrapped by BF16 fallback."""
    if isinstance(method, TimestepBF16FallbackLinearMethod):
        return method.inner
    return method


def iter_bf16_fallback_layers(
    module: nn.Module,
) -> Iterable[tuple[str, nn.Module, TimestepBF16FallbackLinearMethod]]:
    """Yield ``(name, layer, wrapper)`` for all BF16-fallback-wrapped linears."""
    for name, child in module.named_modules():
        method = getattr(child, "quant_method", None)
        if isinstance(method, TimestepBF16FallbackLinearMethod):
            yield name, child, method


__all__ = [
    "BF16_FALLBACK_MANAGED_ATTR",
    "BF16_FALLBACK_WEIGHT_ATTR",
    "MasterMode",
    "TimestepBF16FallbackLinearMethod",
    "TimestepBF16Policy",
    "VALID_MASTER_MODES",
    "attach_bf16_fallback_weight",
    "create_cpu_bf16_load_parameter",
    "current_denoise_step",
    "iter_bf16_fallback_layers",
    "unwrap_linear_method",
]
