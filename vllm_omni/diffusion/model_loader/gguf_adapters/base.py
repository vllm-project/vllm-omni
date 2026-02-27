# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import gguf
import numpy as np
import torch


@dataclass
class MappedTensor:
    name: str
    tensor: Any
    tensor_type: Any
    row_slice: slice | None = None
    swap_scale_shift: bool = False


class GGUFAdapter(ABC):
    """Base class for model-specific GGUF adapters."""

    _include_qkv_virtuals: bool = False
    _include_add_kv_proj_virtuals: bool = False
    _include_to_out_virtuals: bool = False
    _include_w13_virtuals: bool = False
    _shard_tokens: tuple[str, ...] = ()
    _prefer_exact_qweight: bool = True

    def __init__(self, gguf_file: str, model: torch.nn.Module, source, od_config) -> None:
        self.gguf_file = gguf_file
        self.model = model
        self.source = source
        self.od_config = od_config

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        return False

    @abstractmethod
    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError

    def _get_target_module(self) -> torch.nn.Module:
        prefix = getattr(self.source, "prefix", "")
        return self.model.get_submodule(prefix.rstrip(".")) if prefix else self.model

    def _build_allowed_names(self) -> set[str]:
        target = self._get_target_module()
        allowed = {name for name, _ in target.named_parameters()}
        allowed.update(name for name, _ in target.named_buffers())
        for name in list(allowed):
            if name.endswith(".qweight"):
                allowed.add(name.replace(".qweight", ".weight"))
            elif name.endswith(".qweight_type"):
                allowed.add(name.replace(".qweight_type", ".weight"))

        virtual_names = set()
        for name in allowed:
            if self._include_qkv_virtuals and ".to_qkv." in name:
                virtual_names.add(name.replace(".to_qkv.", ".to_q."))
                virtual_names.add(name.replace(".to_qkv.", ".to_k."))
                virtual_names.add(name.replace(".to_qkv.", ".to_v."))
            if self._include_add_kv_proj_virtuals and ".add_kv_proj." in name:
                virtual_names.add(name.replace(".add_kv_proj.", ".add_q_proj."))
                virtual_names.add(name.replace(".add_kv_proj.", ".add_k_proj."))
                virtual_names.add(name.replace(".add_kv_proj.", ".add_v_proj."))
            if self._include_w13_virtuals and ".w13." in name:
                virtual_names.add(name.replace(".w13.", ".w1."))
                virtual_names.add(name.replace(".w13.", ".w3."))
            if self._include_to_out_virtuals and ".to_out." in name:
                virtual_names.add(name.replace(".to_out.", ".to_out.0."))
        allowed.update(virtual_names)
        return allowed

    def _build_param_names(self) -> set[str]:
        target = self._get_target_module()
        return {name for name, _ in target.named_parameters()}

    def _resolve_linear_qweight(self, name: str, param_names: set[str]) -> str | None:
        if not name.endswith(".weight"):
            return None
        if self._prefer_exact_qweight:
            candidate = name.replace(".weight", ".qweight")
            if candidate in param_names:
                return candidate
        if ".to_out.0." in name:
            alt_name = name.replace(".to_out.0.", ".to_out.")
            candidate = alt_name.replace(".weight", ".qweight")
            if candidate in param_names:
                return candidate
            name = alt_name
        for shard_token in self._shard_tokens:
            if shard_token in name:
                return name.replace(".weight", ".qweight")
        candidate = name.replace(".weight", ".qweight")
        if candidate in param_names:
            return candidate
        return None

    def _build_gguf_name_map(self) -> dict[str, str]:
        def resolve_model_type() -> str:
            cfg = self.od_config.tf_model_config
            model_type = None
            if cfg is not None:
                model_type = cfg.get("model_type")
            if model_type:
                return model_type
            model_class = self.od_config.model_class_name or ""
            if model_class.startswith("QwenImage"):
                return "qwen_image"
            if model_class.startswith("Flux2"):
                return "flux"
            raise ValueError("Cannot infer gguf model_type for diffusion model.")

        def resolve_arch(model_type: str):
            for key, value in gguf.MODEL_ARCH_NAMES.items():
                if value == model_type:
                    return key
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        def resolve_num_layers(target_module: torch.nn.Module) -> int:
            if hasattr(target_module, "transformer_blocks"):
                return len(getattr(target_module, "transformer_blocks"))
            if hasattr(target_module, "double_blocks"):
                return len(getattr(target_module, "double_blocks"))
            cfg = self.od_config.tf_model_config
            if cfg is not None:
                for key in ("num_hidden_layers", "num_layers", "n_layers"):
                    value = cfg.get(key)
                    if isinstance(value, int) and value > 0:
                        return value
            raise ValueError("Cannot infer gguf num_layers for diffusion model.")

        def get_target_module(root: torch.nn.Module, prefix: str) -> torch.nn.Module:
            if not prefix:
                return root
            prefix = prefix.rstrip(".")
            if hasattr(root, "get_submodule"):
                return root.get_submodule(prefix)
            current = root
            for part in prefix.split("."):
                current = getattr(current, part)
            return current

        def split_name(name: str) -> tuple[str, str]:
            if name.endswith("_weight"):
                return name[:-7], "weight"
            if "." in name:
                base, suffix = name.rsplit(".", 1)
                return base, suffix
            return name, ""

        reader = gguf.GGUFReader(self.gguf_file)
        gguf_tensor_names = {tensor.name for tensor in reader.tensors}

        model_type = resolve_model_type()
        arch = resolve_arch(model_type)
        target_module = get_target_module(self.model, self.source.prefix)
        num_layers = resolve_num_layers(target_module)
        name_map = gguf.get_tensor_name_map(arch, num_layers)

        gguf_to_model_map: dict[str, str] = {}
        for name, _ in target_module.named_parameters():
            base_name, suffix = split_name(name)
            gguf_base = name_map.get_name(base_name)
            if gguf_base is None:
                continue
            candidates = []
            if suffix:
                candidates.append(f"{gguf_base}.{suffix}")
                if suffix == "weight":
                    candidates.append(f"{gguf_base}.scale")
            else:
                candidates.append(gguf_base)
            gguf_name = next((c for c in candidates if c in gguf_tensor_names), None)
            if gguf_name is None:
                continue
            gguf_to_model_map[gguf_name] = name

        for name, _ in target_module.named_buffers():
            base_name, suffix = split_name(name)
            gguf_base = name_map.get_name(base_name)
            if gguf_base is None:
                continue
            candidates = []
            if suffix:
                candidates.append(f"{gguf_base}.{suffix}")
                if suffix == "weight":
                    candidates.append(f"{gguf_base}.scale")
            else:
                candidates.append(gguf_base)
            gguf_name = next((c for c in candidates if c in gguf_tensor_names), None)
            if gguf_name is None:
                continue
            gguf_to_model_map[gguf_name] = name

        if not gguf_to_model_map:
            raise RuntimeError(f"No GGUF tensors were mapped for model_class_name={self.od_config.model_class_name!r}.")
        return gguf_to_model_map


# FIXME(Isotr0py): Sync implemnentation with upstream vLLM?
def gguf_quant_weights_iterator(gguf_file: str) -> Generator[tuple[str, torch.Tensor]]:
    """
    Iterate over the quant weights in the model gguf files and convert
    them to torch tensors.
    Be careful of the order of yielding weight types and weights data,
    we have to yield all weight types first before yielding any weights.
    Otherwise it would cause issue when loading weights with for packed
    layer with different quant types.
    """

    reader = gguf.GGUFReader(gguf_file)

    for tensor in reader.tensors:
        weight_type = tensor.tensor_type
        name = tensor.name

        if weight_type.name not in ("F32", "F16"):
            weight_type_name = name.replace("weight", "qweight_type")
            weight_type = torch.tensor(weight_type)
            yield weight_type_name, weight_type

    for tensor in reader.tensors:
        weight = tensor.data
        weight_type = tensor.tensor_type
        name = tensor.name
        if weight_type.name not in ("F32", "F16"):
            name = name.replace("weight", "qweight")
        if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
            # BF16 is currently the only "quantization" type that isn't
            # actually quantized but is read as a raw byte tensor.
            # Reinterpret as `torch.bfloat16` tensor.
            weight = weight.view(np.uint16)
            if reader.byte_order == "S":
                # GGUF endianness != system endianness
                weight = weight.byteswap()
            param = torch.tensor(weight).view(torch.bfloat16)
        else:
            param = torch.tensor(weight)
        yield name, param
