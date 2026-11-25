# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt the native LTX-2.3 Gemma text model for merged LoRA binding."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config.lora import LoRAConfig
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_omni.lora.layers import OmniReplicatedLinearWithLoRA


class LTX23TextLinear(ReplicatedLinear):
    """Keep the Hugging Face Tensor-to-Tensor projection contract."""

    # vLLM's CPU post-load hook otherwise packs the projection and removes
    # ``weight``. The HF forward and merged-weight bank require the dense
    # Parameter to remain present through loading and later reloads.
    _cpu_skip_gemm_dispatch = True

    @classmethod
    def from_linear(cls, source: nn.Linear) -> LTX23TextLinear:
        # ReplicatedLinear creates its parameters in the constructor. Meta
        # construction avoids allocating a second dense weight before the
        # original Parameter objects are installed below.
        with torch.device("meta"):
            result = cls(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
                return_bias=False,
                disable_tp=True,
                params_dtype=source.weight.dtype,
            )
        result.weight = source.weight
        result.bias = source.bias
        result.in_features = source.in_features
        result.out_features = source.out_features
        result.train(source.training)
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # HF Gemma consumes a tensor, whereas the stock vLLM linear can return
        # (output, bias). This also retains the native dense CPU path.
        return F.linear(input, self.weight, self.bias)

    def _create_omni_lora_layer(
        self,
        *,
        max_loras: int,
        lora_config: LoRAConfig,
        packed_modules_list: list[str],
        model_config=None,
    ) -> OmniReplicatedLinearWithLoRA | None:
        if packed_modules_list:
            return None
        instance = OmniReplicatedLinearWithLoRA(self)
        instance.create_lora_weights(max_loras, lora_config, model_config)
        return instance


def ltx23_text_lora_aliases(full_name: str) -> tuple[str, ...]:
    """Accept native HF names while retaining the full manager binding key."""
    prefix = "text_encoder."
    return (full_name[len(prefix) :],) if full_name.startswith(prefix) else ()


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, storage.data_ptr())


def prepare_ltx23_text_encoder_lora(pipeline: nn.Module) -> None:
    """Atomically convert only plain linears in model.language_model.

    The caller owns the LTX version, recipe, and merged-mode gate. This helper
    has no configuration side effects and may be called again after success.
    """
    text_encoder = getattr(pipeline, "text_encoder", None)
    model = getattr(text_encoder, "model", None)
    language_model = getattr(model, "language_model", None)
    if (
        not isinstance(text_encoder, nn.Module)
        or not isinstance(model, nn.Module)
        or not isinstance(language_model, nn.Module)
    ):
        raise ValueError("LTX-2.3 text_encoder.model.language_model must be a torch.nn.Module")

    all_modules = list(pipeline.named_modules(remove_duplicate=False))
    module_counts = Counter(id(module) for _, module in all_modules)
    subtree = list(language_model.named_modules(remove_duplicate=False))
    candidates: list[tuple[nn.Module, str, nn.Linear]] = []
    adapted_count = 0
    for name, module in subtree:
        if name and isinstance(module, nn.Linear):
            if type(module) is not nn.Linear:
                raise ValueError(f"LTX text LoRA rejects custom Linear subclass {name!r}")
            if any(
                (
                    module._forward_hooks,
                    module._forward_pre_hooks,
                    module._backward_hooks,
                    module._backward_pre_hooks,
                )
            ) or hasattr(module, "parametrizations"):
                raise ValueError(f"LTX text LoRA rejects hooks or parametrization on {name!r}")
            if module_counts[id(module)] != 1:
                raise ValueError(f"LTX text LoRA rejects shared module {name!r}")
            for param in (module.weight, module.bias):
                if param is None:
                    continue
                if param.is_meta or param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(f"LTX text LoRA requires materialized fp16/bf16/fp32 parameters: {name!r}")
            if module.bias is not None and (
                module.bias.device != module.weight.device or module.bias.dtype != module.weight.dtype
            ):
                raise ValueError(f"LTX text LoRA rejects mixed device or dtype parameters: {name!r}")
            parent_name, _, leaf = name.rpartition(".")
            parent = language_model.get_submodule(parent_name) if parent_name else language_model
            candidates.append((parent, leaf, module))
        elif isinstance(module, LTX23TextLinear):
            adapted_count += 1

    if not candidates and not adapted_count:
        raise ValueError("LTX text LoRA found no plain Linear projections in language_model")

    if getattr(pipeline, "_ltx23_text_encoder_lora_prepared", False):
        if candidates:
            raise ValueError("LTX text LoRA was already prepared but contains new plain Linear projections")
        return

    # Count every reference, including aliases outside the selected subtree.
    # Storage identity catches distinct Parameter objects sharing dense memory.
    all_tensors = list(pipeline.named_parameters(remove_duplicate=False))
    all_tensors.extend(pipeline.named_buffers(remove_duplicate=False))
    storage_counts = Counter(_storage_key(tensor) for _, tensor in all_tensors if not tensor.is_meta)
    parameter_counts = Counter(id(tensor) for _, tensor in all_tensors)
    for _, _, source in candidates:
        for param in (source.weight, source.bias):
            if param is not None and (parameter_counts[id(param)] != 1 or storage_counts[_storage_key(param)] != 1):
                raise ValueError("LTX text LoRA rejects shared Parameter or storage alias")

    # Build every replacement before publishing one. A late constructor error
    # leaves the original module graph intact.
    replacements = [(parent, leaf, LTX23TextLinear.from_linear(source)) for parent, leaf, source in candidates]
    for parent, leaf, replacement in replacements:
        setattr(parent, leaf, replacement)
    pipeline._ltx23_text_encoder_lora_prepared = True
