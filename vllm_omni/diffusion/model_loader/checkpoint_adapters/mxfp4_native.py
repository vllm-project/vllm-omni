# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bind native Wan MXFP4 storage before allocation and stream its tensor sources."""

from __future__ import annotations

from collections.abc import Generator
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config
from vllm_omni.quantization.tools.mxfp4_native import EXPERTS, NativeMXFP4Expert, _ignored_names, validate_native_root

if TYPE_CHECKING:
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader


def prepare_native_mxfp4(
    config: QuantizationConfig | None, original_model_path: str | None, component: str
) -> QuantizationConfig | None:
    if not isinstance(config, DiffusionMXFP4Config) or config.native_checkpoint_path is None:
        return config
    if original_model_path is None or component not in EXPERTS.values():
        raise ValueError("Native MXFP4 requires a local BF16 Wan2.2 T2V model and an exact expert component")
    original_root, quant_root = Path(original_model_path), Path(config.native_checkpoint_path)
    validate_native_root(original_root, quant_root)
    source = next(name for name, target in EXPERTS.items() if target == component)
    if len(list((quant_root / source).glob("*.safetensors"))) != 1:
        raise ValueError("Native MXFP4 runtime requires one safetensors per expert")
    expert = NativeMXFP4Expert(quant_root / source, original_root / component, config.require_smooth_scale)
    # Runtime ignore cannot turn packed codes into real BF16 weights. Native
    # FLOAT descriptors own the storage policy, independently for each expert.
    for runtime_layer in _ignored_names(expert.quant_layers):
        if is_layer_skipped(runtime_layer, config.ignored_layers, {"to_qkv": ["to_q", "to_k", "to_v"]}):
            raise ValueError(f"Native MXFP4 ignored layer has packed weights, not BF16: {runtime_layer}")
    result = copy(config)
    result.ignored_layers = expert.ignored
    result.require_smooth_scale = bool(expert.smooth)
    result.native_checkpoint = expert
    return result


def get_native_mxfp4_weights(
    model: nn.Module, source: DiffusersPipelineLoader.ComponentSource
) -> Generator[tuple[str, torch.Tensor], None, None] | None:
    """Return a prepared native iterator, or None for other checkpoint formats."""
    component = source.prefix.rstrip(".")
    module = getattr(model, component, None)
    expert = getattr(module, "_native_mxfp4_checkpoint", None)
    if expert is None:
        return None
    if not isinstance(expert, NativeMXFP4Expert):
        raise TypeError("Native MXFP4 requires a validated expert checkpoint")
    if component not in EXPERTS.values() or source.subfolder != component:
        raise ValueError("Native MXFP4 source must select an exact Wan expert")
    return ((source.prefix + name, tensor) for name, tensor in expert.weights())
