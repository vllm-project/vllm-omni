# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastH3's serialized NVFP4 projections, using the existing parallel linears.

Each exported projection keeps its own global scale. Q/K/V are evaluated
separately; merging their packed bytes into a single GEMM would change scales.
Only scale layout and TP slices are changed during loading, never quantization.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import regex as re
import torch
from safetensors import safe_open
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .fasth3 import _resolve_native_target
from .vae_nvfp4 import _packed_nvfp4_linear

if TYPE_CHECKING:
    from vllm_omni.quantization.fp8_config import SerializedPerChannelFp8Config


def fasth3_fp8_config(root: Path) -> "SerializedPerChannelFp8Config":
    """Resolve native projection names from serialized FP8 tensor headers."""
    from vllm_omni.quantization.fp8_config import SerializedPerChannelFp8Config

    index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    targets: set[str] = set()
    parts: dict[str, set[str]] = {}
    for filename in sorted(set(index.values())):
        with safe_open(root / filename, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if not key.endswith(".weight") or reader.get_slice(key).get_dtype() != "F8_E4M3":
                    continue
                if key.removesuffix(".weight") + ".weight_scale" not in index:
                    raise ValueError(f"FastH3 FP8 weight has no per-channel scale: {key}")
                target = _resolve_native_target(key.removesuffix(".weight"))
                if target is None:
                    raise ValueError(f"Unsupported FastH3 FP8 projection: {key}")
                targets.add(target[0])
                parts.setdefault(target[0], set()).add(target[1])
    for target, layouts in parts.items():
        if target.endswith(".qkv_proj") and layouts != {"q", "k", "v"}:
            raise ValueError(f"FastH3 FP8 QKV must contain all three projections: {target}")
    if not targets:
        raise ValueError("FastH3 FP8 checkpoint contains no serialized FP8 weights")
    return SerializedPerChannelFp8Config("(?:" + "|".join(re.escape(name) for name in sorted(targets)) + r")\.weight")


@dataclass(frozen=True)
class NVFP4Projection:
    source: str
    rows: int
    columns: int
    tensors: Mapping[str, str]
    swap_halves: bool = False
    reciprocal_scale: bool = False


def _linear_scales(scales: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    return scales.reshape(rows // 128, columns // 64, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, columns // 16)


def _swizzled_scales(scales: torch.Tensor) -> torch.Tensor:
    rows, groups = scales.shape
    return scales.reshape(rows // 128, 4, 32, groups // 4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(rows, groups)


class _Projection(nn.Module):
    def __init__(self, spec: NVFP4Projection, rows: int, columns: int, rank: int) -> None:
        super().__init__()
        if rows % 128 or columns % 64:
            raise ValueError(f"FastH3 NVFP4 TP shards require N divisible by 128 and K by 64, got {rows}x{columns}")
        self.spec = spec
        self.rows, self.columns, self.rank = rows, columns, rank
        shapes = {"weight": (rows, columns // 2), "scales": (rows, columns // 16)}
        for field in spec.tensors:
            dtype = torch.uint8 if field in shapes else torch.float32
            parameter = nn.Parameter(torch.empty(shapes.get(field, (1,)), dtype=dtype), requires_grad=False)
            parameter.weight_loader = self.weight_loader
            parameter.missing_param_init = "error"
            parameter.nvfp4_field = field
            self.register_parameter(field, parameter)
        if "input_scale" not in spec.tensors:
            self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32))

    def weight_loader(self, parameter: nn.Parameter, value: torch.Tensor, shard_id: str | int | None = None) -> None:
        del shard_id
        field = parameter.nvfp4_field
        spec = self.spec
        if field in ("weight", "scales"):
            if value.dtype != torch.uint8:
                raise ValueError(f"{spec.source}: packed NVFP4 {field} must be UINT8")
            expected = (spec.rows, spec.columns // (2 if field == "weight" else 16))
            if tuple(value.shape) != expected:
                raise ValueError(f"{spec.source}: {field} shape {tuple(value.shape)}, expected {expected}")
            if field == "scales":
                value = _linear_scales(value, spec.rows, spec.columns)
            if spec.swap_halves:
                first, second = value.chunk(2, dim=0)
                count = self.rows // 2
                value = torch.cat(
                    (second.narrow(0, self.rank * count, count), first.narrow(0, self.rank * count, count))
                )
            elif self.rows != spec.rows:
                value = value.narrow(0, self.rank * self.rows, self.rows)
            if self.columns != spec.columns:
                count = self.columns // (2 if field == "weight" else 16)
                value = value.narrow(1, self.rank * count, count)
            if field == "scales":
                value = _swizzled_scales(value)
        else:
            if value.numel() != 1 or not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError(f"{spec.source}: {field} must be a finite positive scalar")
            value = value.float().reshape(1)
            if field == "alpha" and spec.reciprocal_scale:
                value = value.reciprocal()
        parameter.data.copy_(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _packed_nvfp4_linear(x, self.weight, self.scales, self.alpha / self.input_scale, self.input_scale)


class _NVFP4Method(LinearMethodBase):
    def __init__(self, projections: list[NVFP4Projection], rank: int) -> None:
        self.projections = projections
        self.rank = rank

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype, extra_weight_attrs
        sizes = output_partition_sizes if len(self.projections) > 1 else [sum(output_partition_sizes)]
        layer.packed = nn.ModuleList(
            _Projection(spec, rows, input_size_per_partition, self.rank)
            for spec, rows in zip(self.projections, sizes, strict=True)
        )
        # There is no post-load conversion or device work for packed exports.
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        parts = [projection(x) for projection in layer.packed]
        output = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return output if bias is None else output + bias


class FastH3NVFP4Config(QuantizationConfig):
    """Model-local checkpoint adapter; does not change the global quant registry."""

    def __init__(self, projections: dict[str, list[NVFP4Projection]], *, rank: int) -> None:
        super().__init__()
        self.projections = projections
        self.rank = rank
        self.weight_map = {
            source: f"{target}.packed.{index}.{field}"
            for target, specs in projections.items()
            for index, spec in enumerate(specs)
            for field, source in spec.tensors.items()
        }

    def get_name(self) -> str:
        return "fasth3_nvfp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "FastH3NVFP4Config":
        raise ValueError("FastH3 NVFP4 is selected from the component's serialized weights")

    def get_quant_method(self, layer: nn.Module, prefix: str) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        specs = self.projections.get(prefix.removeprefix("text_encoder."))
        return _NVFP4Method(specs, self.rank) if specs else UnquantizedLinearMethod()

    @classmethod
    def from_dit(cls, path: Path, *, rank: int) -> "FastH3NVFP4Config":
        fields = {
            "_nvfp4_weight": "weight",
            "_nvfp4_weight_scale": "scales",
            "_nvfp4_alpha": "alpha",
            "_weight_global_sf": "weight_global_scale",
            "_nvfp4_input_global_sf": "input_scale",
        }
        groups: dict[str, dict[str, str]] = {}
        projections: dict[str, list[NVFP4Projection]] = {}
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                source, sep, field = key.partition("::")
                if not sep or field not in fields:
                    raise ValueError(f"Unsupported FastH3 NVFP4 tensor: {key}")
                groups.setdefault(source, {})[fields[field]] = key
            # Q/K/V must keep their execution order, independently of file order.
            order = {"q": 0, "k": 1, "v": 2}
            for source, tensors in groups.items():
                if not {"weight", "scales", "alpha", "weight_global_scale"} <= tensors.keys():
                    raise ValueError(f"Incomplete FastH3 NVFP4 projection: {source}")
                name = source.replace(".ff.fc_in", ".ff.net.0.proj").replace(".ff.fc_out", ".ff.net.2")
                if name.endswith(".attn.to_out"):
                    name += ".0"
                target = _resolve_native_target(name)
                if target is None:
                    raise ValueError(f"Unsupported FastH3 NVFP4 projection: {source}")
                rows, packed_columns = reader.get_slice(tensors["weight"]).get_shape()
                spec = NVFP4Projection(source, rows, packed_columns * 2, tensors, target[1] == "swap_halves")
                projections.setdefault(target[0], []).append(spec)
            for target_name, specs in projections.items():
                if target_name.endswith(".qkv_proj"):
                    specs.sort(key=lambda spec: order[spec.source[-1]])
                    if [spec.source[-1] for spec in specs] != ["q", "k", "v"]:
                        raise ValueError("FastH3 NVFP4 QKV must contain all three projections")
        return cls(projections, rank=rank)

    @classmethod
    def from_encoder(cls, root: Path, *, rank: int) -> "FastH3NVFP4Config":
        config = json.loads((root / "config.json").read_text())["quantization_config"]
        for field, expected in (
            ("quant_method", "nvfp4"),
            ("activation_scheme", "dynamic"),
            ("fmt", "e2m1"),
            ("group_size", 16),
            ("scale_fmt", "e4m3"),
            ("scale_layout", "128x4"),
        ):
            if config.get(field) != expected:
                raise ValueError(f"Unsupported FastH3 text encoder {field}: {config.get(field)!r}")
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        projections: dict[str, list[NVFP4Projection]] = {}
        for filename in sorted(set(index.values())):
            with safe_open(root / filename, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if not key.endswith(".weight_packed"):
                        continue
                    source = key.removesuffix(".weight_packed")
                    target = source.replace("model.language_model.", "text_model.")
                    for part in ("q", "k", "v"):
                        target = target.replace(f".self_attn.{part}_proj", ".self_attn.qkv_proj")
                    for part in ("gate", "up"):
                        target = target.replace(f".mlp.{part}_proj", ".mlp.gate_up_proj")
                    tensors = {
                        "weight": key,
                        "scales": source + ".weight_scale",
                        "alpha": source + ".weight_global_scale",
                    }
                    if not all(name in index for name in tensors.values()):
                        raise ValueError(f"Incomplete FastH3 encoder NVFP4 projection: {source}")
                    rows, packed_columns = reader.get_slice(key).get_shape()
                    spec = NVFP4Projection(source, rows, packed_columns * 2, tensors, reciprocal_scale=True)
                    projections.setdefault(target, []).append(spec)
        order = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "gate_proj": 0, "up_proj": 1}
        for target, specs in projections.items():
            if len(specs) > 1:
                specs.sort(key=lambda spec: order[spec.source.rsplit(".", 1)[-1]])
            expected = 3 if target.endswith("qkv_proj") else 2 if target.endswith("gate_up_proj") else 1
            if len(specs) != expected:
                raise ValueError(f"Incomplete FastH3 encoder NVFP4 projections for {target}")
        return cls(projections, rank=rank)
