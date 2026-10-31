# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt native H3 decoder linears to the existing online NPU INT8 methods.

Install after loading the floating checkpoint and before residency snapshots.
The native Attention/FFN forwards and boundary projections remain unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8OnlineLinearMethod

logger = init_logger(__name__)
_COMPUTE_DTYPES = (torch.float16, torch.bfloat16)
_INPUT_DTYPES = (*_COMPUTE_DTYPES, torch.float32)
_LEAVES = (("attn", "to_qkv"), ("attn", "to_out"), ("ff", "w1"), ("ff", "w2"))


def resolve_h3_video_vae_quant_config(config: QuantizationConfig | None) -> QuantizationConfig | None:
    """Opt in only through an explicit, exact ``video_vae`` component key.

    Global and default configs retain their existing scope. Layer exclusions
    use ignored_layers rather than nested video_vae component routes.
    """
    from vllm_omni.quantization.component_config import ComponentQuantizationConfig

    if not isinstance(config, ComponentQuantizationConfig):
        return None
    components = config.component_configs
    if any(key.startswith("video_vae.") for key in components):
        raise ValueError("H3 VAE INT8 requires the exact video_vae component key; use ignored_layers for exclusions")
    return components.get("video_vae")


def _require_npu_device(device: torch.device) -> None:
    if device.type != "npu" or not current_omni_platform.is_npu():
        raise ValueError("H3 Video VAE online INT8 is supported only on NPU")


def validate_h3_video_vae_quant_config(
    config: QuantizationConfig | None,
    *,
    device: torch.device,
    encode_only: bool = False,
    decoder_source: str | Mapping[str, str] | None = None,
) -> None:
    """Validate before loading a checkpoint or resolving a replacement decoder."""
    if config is None:
        return
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    _require_npu_device(device)
    if not isinstance(config, DiffusionInt8Config):
        raise ValueError("H3 Video VAE supports only an explicit online int8 quantization config")
    if config.is_checkpoint_int8_serialized or config.activation_scheme != "dynamic":
        raise ValueError("H3 Video VAE INT8 requires a floating checkpoint and dynamic online quantization")
    if not isinstance(config.ignored_layers, list) or not all(isinstance(path, str) for path in config.ignored_layers):
        raise ValueError("H3 Video VAE ignored_layers must be a list of layer names")
    if encode_only:
        raise ValueError("An encode-only H3 Video VAE has no decoder linears to quantize")
    if decoder_source is not None:
        raise ValueError("H3 Video VAE online INT8 cannot be combined with a replacement video decoder")


def _check_uncompiled(module: nn.Module, path: str) -> None:
    if (
        hasattr(module, "_orig_mod")
        or getattr(module, "_compiled_call_impl", None) is not None
        or getattr(module, "_compiled_forward", None) is not None
        or getattr(module, "_compile_forward_enabled", False)
    ):
        raise ValueError(f"{path}: install INT8 before compiling the decoder")


def _check_linear(source: nn.Module, path: str) -> None:
    from vllm_omni.quantization.int8_config import NPU_QUANT_MATMUL_MAX_OUT_FEATURES

    if type(source) is not nn.Linear or "forward" in source.__dict__:
        raise ValueError(f"{path}: online INT8 supports ordinary nn.Linear without custom forward behavior")
    _check_uncompiled(source, path)
    hook_names = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    if any(getattr(source, name, None) for name in hook_names):
        raise ValueError(f"{path}: replacing a Linear with registered hooks is unsupported")
    if set(source._parameters) != {"weight", "bias"} or source._buffers or source._modules:
        raise ValueError(f"{path}: custom Linear parameters, buffers, or submodules are unsupported")
    if source.in_features <= 0 or not 0 < source.out_features <= NPU_QUANT_MATMUL_MAX_OUT_FEATURES:
        raise ValueError(
            f"{path}: NPU INT8 requires K > 0 and 0 < N <= {NPU_QUANT_MATMUL_MAX_OUT_FEATURES}; "
            "exclude this layer with ignored_layers"
        )
    tensors = (("weight", source.weight, (source.out_features, source.in_features)),)
    if source.bias is not None:
        tensors += (("bias", source.bias, (source.out_features,)),)
    for name, tensor, shape in tensors:
        if tensor.is_meta or tensor.dtype not in _INPUT_DTYPES or tuple(tensor.shape) != shape:
            raise ValueError(f"{path}.{name}: expected a materialized floating tensor of shape {shape}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{path}.{name}: non-finite values cannot be quantized")


class H3VAEInt8Linear(nn.Module):
    """Tensor-returning native Linear interface backed by the NPU INT8 method.

    ``weight`` changes from [N,K] floating to [K,N] INT8 during construction.
    This module does not implement a serialized quantized checkpoint format.
    """

    weight: nn.Parameter
    weight_scale: nn.Parameter
    bias: nn.Parameter | None
    _int8_method: NPUInt8OnlineLinearMethod

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        source: nn.Linear,
        quant_config: DiffusionInt8Config,
        *,
        quant_device: torch.device,
        storage_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> H3VAEInt8Linear:
        validate_h3_video_vae_quant_config(quant_config, device=quant_device)
        if compute_dtype not in _COMPUTE_DTYPES:
            raise ValueError("H3 Video VAE INT8 compute_dtype must be FP16 or BF16")
        if storage_device.type not in ("cpu", quant_device.type):
            raise ValueError("H3 Video VAE INT8 weights must reside on CPU or NPU")
        _check_linear(source, "linear")
        from vllm_omni.quantization.int8_config import NPUInt8OnlineLinearMethod

        result = cls()
        result.in_features = source.in_features
        result.out_features = source.out_features
        result.compute_dtype = compute_dtype
        weight = source.weight.detach().to(device=quant_device, dtype=compute_dtype).contiguous()
        if not bool(torch.isfinite(weight).all().item()):
            raise ValueError("H3 Video VAE weight overflows its FP16/BF16 quantization input dtype")
        result.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        result.register_parameter("weight_scale", None)
        bias = source.bias.detach().to(dtype=torch.float32) if source.bias is not None else None
        result.register_parameter("bias", nn.Parameter(bias, requires_grad=False) if bias is not None else None)
        # A public quant_method would let the diffusion loader mistake this
        # already-eager-loaded module for a LazyWeightMixin streaming layer.
        result._int8_method = NPUInt8OnlineLinearMethod(quant_config)
        result._int8_method.process_weights_after_loading(result)
        if (
            result.weight.dtype != torch.int8
            or tuple(result.weight.shape) != (result.in_features, result.out_features)
            or result.weight_scale.dtype != torch.float32
            or tuple(result.weight_scale.shape) != (result.out_features,)
            or not getattr(result, "_already_called_process_weights_after_loading", False)
        ):
            raise RuntimeError("H3 Video VAE INT8 post-load method returned an unexpected weight/scale layout")
        if not bool(torch.isfinite(result.weight_scale).all().item()):
            raise ValueError("H3 Video VAE INT8 produced non-finite weight scales")
        result.to(device=storage_device)
        return result.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_npu_device(x.device)
        if x.device != self.weight.device:
            raise ValueError("H3 Video VAE INT8 input and weights must be on the same NPU")
        if x.dtype not in _INPUT_DTYPES or x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("H3 Video VAE INT8 expects floating input with the original Linear input width")
        if self.weight.dtype != torch.int8 or self.weight_scale.dtype != torch.float32:
            raise RuntimeError("Do not change H3 Video VAE INT8 weight or scale dtypes after installation")
        if self.bias is not None and self.bias.dtype != torch.float32:
            raise RuntimeError("H3 Video VAE INT8 bias must remain FP32; use device-only module moves")
        output_dtype = torch.get_autocast_dtype("npu") if torch.is_autocast_enabled("npu") else x.dtype
        compute_dtype = output_dtype if output_dtype in _COMPUTE_DTYPES else self.compute_dtype
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.out_features), dtype=output_dtype)
        with torch.autocast(device_type="npu", enabled=False):
            y = self._int8_method.apply(self, x.to(dtype=compute_dtype), bias=None)
            y = y.to(dtype=output_dtype)
            if self.bias is not None:
                y = y + self.bias.to(dtype=output_dtype)
        return y


@dataclass(frozen=True)
class VAEInt8LayerReport:
    path: str
    in_features: int
    out_features: int
    quantized: bool
    device: torch.device
    weight_dtype: torch.dtype
    scale_dtype: torch.dtype | None
    bias_dtype: torch.dtype | None
    weight_bytes: int
    scale_bytes: int
    bias_bytes: int


@dataclass(frozen=True)
class VAEInt8InstallReport:
    block_count: int
    compute_dtype: torch.dtype
    layers: tuple[VAEInt8LayerReport, ...]
    elapsed_seconds: float

    @property
    def candidate_count(self) -> int:
        return len(self.layers)

    @property
    def quantized_count(self) -> int:
        return sum(layer.quantized for layer in self.layers)

    @property
    def ignored_count(self) -> int:
        return self.candidate_count - self.quantized_count


@dataclass(frozen=True)
class _LinearPlan:
    parent: nn.Module
    leaf: str
    path: str
    source: nn.Linear
    ignored: bool


def _check_aliases(decoder: nn.Module, plans: list[_LinearPlan]) -> None:
    selected = [plan.source for plan in plans if not plan.ignored]
    module_counts = Counter(id(module) for _, module in decoder.named_modules(remove_duplicate=False))
    if any(module_counts[id(module)] != 1 for module in selected):
        raise ValueError("H3 Video VAE INT8 does not support shared Linear modules")
    selected_parameters = {id(parameter) for module in selected for parameter in module.parameters()}
    storages: dict[tuple[torch.device, int, int], list[int]] = {}
    for _, parameter in decoder.named_parameters(remove_duplicate=False):
        if parameter.is_meta or parameter.numel() == 0:
            continue
        storage = parameter.untyped_storage()
        key = (parameter.device, storage.data_ptr(), storage.nbytes())
        storages.setdefault(key, []).append(id(parameter))
    if any(len(ids) > 1 and any(item in selected_parameters for item in ids) for ids in storages.values()):
        raise ValueError("H3 Video VAE INT8 does not support tied or shared weight/bias storage")


def _layer_report(path: str, module: nn.Linear | H3VAEInt8Linear, *, quantized: bool) -> VAEInt8LayerReport:
    scale = getattr(module, "weight_scale", None)
    bias = module.bias
    return VAEInt8LayerReport(
        path=path,
        in_features=module.in_features,
        out_features=module.out_features,
        quantized=quantized,
        device=module.weight.device,
        weight_dtype=module.weight.dtype,
        scale_dtype=scale.dtype if scale is not None else None,
        bias_dtype=bias.dtype if bias is not None else None,
        weight_bytes=module.weight.numel() * module.weight.element_size(),
        scale_bytes=scale.numel() * scale.element_size() if scale is not None else 0,
        bias_bytes=bias.numel() * bias.element_size() if bias is not None else 0,
    )


@torch.no_grad()
def install_h3_video_vae_int8(
    decoder: nn.Module,
    quant_config: DiffusionInt8Config,
    *,
    quant_device: torch.device,
    storage_device: torch.device,
    compute_dtype: torch.dtype,
) -> VAEInt8InstallReport:
    """Replace only four block linears, before creating any residency stager.

    Static validation completes before replacing modules. An operator error
    during conversion aborts startup; no runtime floating-weight fallback is kept.
    """
    validate_h3_video_vae_quant_config(quant_config, device=quant_device)
    if compute_dtype not in _COMPUTE_DTYPES:
        raise ValueError("H3 Video VAE INT8 compute_dtype must be FP16 or BF16")
    _check_uncompiled(decoder, "video_vae.decoder")
    identity = (tuple(quant_config.ignored_layers), compute_dtype, str(quant_device), str(storage_device))
    installed = getattr(decoder, "_omni_h3_vae_int8_config", None)
    if installed is not None:
        if installed != identity:
            raise ValueError("H3 Video VAE INT8 is already installed with a different configuration")
        return decoder._omni_h3_vae_int8_report
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("H3 Video VAE INT8 requires a decoder.transformer_blocks ModuleList")
    start = perf_counter()
    plans: list[_LinearPlan] = []
    for index, block in enumerate(blocks):
        for parent_name, leaf in _LEAVES:
            parent = getattr(block, parent_name, None)
            source = getattr(parent, leaf, None)
            path = f"video_vae.decoder.transformer_blocks.{index}.{parent_name}.{leaf}"
            if not isinstance(parent, nn.Module) or not isinstance(source, nn.Linear):
                raise ValueError(f"{path}: expected a native Linear submodule")
            for owner in (block, parent):
                _check_uncompiled(owner, path)
            ignored = is_layer_skipped(
                prefix=path,
                ignored_layers=quant_config.ignored_layers,
                fused_mapping=getattr(quant_config, "packed_modules_mapping", {}),
            )
            if not ignored:
                _check_linear(source, path)
            plans.append(_LinearPlan(parent, leaf, path, source, ignored))
    if all(plan.ignored for plan in plans):
        raise ValueError("H3 Video VAE INT8 was enabled but ignored_layers excludes every candidate")
    _check_aliases(decoder, plans)
    reports: list[VAEInt8LayerReport] = []
    for plan in plans:
        if plan.ignored:
            reports.append(_layer_report(plan.path, plan.source, quantized=False))
            continue
        converted = H3VAEInt8Linear.from_linear(
            plan.source,
            quant_config,
            quant_device=quant_device,
            storage_device=storage_device,
            compute_dtype=compute_dtype,
        )
        setattr(plan.parent, plan.leaf, converted)
        reports.append(_layer_report(plan.path, converted, quantized=True))
    report = VAEInt8InstallReport(len(blocks), compute_dtype, tuple(reports), perf_counter() - start)
    decoder._omni_h3_vae_int8_config = identity
    decoder._omni_h3_vae_int8_report = report
    logger.info(
        "Installed H3 Video VAE INT8: blocks=%d, quantized=%d, ignored=%d, compute_dtype=%s, "
        "quant_device=%s, storage_device=%s, elapsed=%.3fs",
        report.block_count,
        report.quantized_count,
        report.ignored_count,
        compute_dtype,
        quant_device,
        storage_device,
        report.elapsed_seconds,
    )
    return report
