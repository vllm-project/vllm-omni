# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read-only validation of msModelSlim mindie_format_saver Wan MXFP4 exports.

Independent of the inference stack. Packed bytes are never unpacked or requantized.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open

SOURCE_COMMIT = "2e58ef003eb6166143f2f857d76295ae3a0b7a91"
C7_REFERENCE_COMMIT = "ff3963d777cda951f40c6b2f376f5722895245f4"
SOURCE_CONTRACT = "msmodelslim-mindie-w4a4-mxfp4-uint8-v1"
QUANT_TYPE = "W4A4_MXFP4"
EXPERTS = {"high_noise_model": "transformer", "low_noise_model": "transformer_2"}
FLOAT_DTYPES = {"BF16", "F16", "F32"}

# Wan's native checkpoint names, not substring aliases for arbitrary model keys.
ROOT_NAMES = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "patch_embedding": "patch_embedding",
}
BLOCK_NAMES = {
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    "norm1": "norm1",
    "norm2": "norm3",
    "norm3": "norm2",
    **{
        f"{src}.{part}": f"{dst}.{target}"
        for src, dst in (("self_attn", "attn1"), ("cross_attn", "attn2"))
        for part, target in (
            ("q", "to_q"),
            ("k", "to_k"),
            ("v", "to_v"),
            ("o", "to_out.0"),
            ("norm_q", "norm_q"),
            ("norm_k", "norm_k"),
        )
    },
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 重复键: {key}")
        result[key] = value
    return result


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"必须为 JSON object: {path}")
    return value


def _rename(key: str) -> str:
    for attr in ("weight", "weight_scale", "bias"):
        if key.endswith(".linear." + attr):
            key = key.removesuffix(".linear." + attr) + "." + attr
            break
    if key.endswith(".div.mul_scale"):
        key = key.removesuffix(".div.mul_scale") + ".mul_scale"
    parts = key.split(".", 2)
    if len(parts) == 3 and parts[0] == "blocks" and parts[1].isascii() and parts[1].isdigit():
        prefix, rest, names = f"blocks.{parts[1]}.", parts[2], BLOCK_NAMES
    else:
        prefix, rest, names = "", key, ROOT_NAMES
    for source, target in names.items():
        if rest == source or rest.startswith(source + "."):
            return prefix + target + rest[len(source) :]
    raise ValueError(f"不支持的原生 Wan tensor/配方字段: {key}")


@dataclass(frozen=True)
class TensorInfo:
    file: pathlib.Path
    shape: tuple[int, ...]
    dtype: str


class TensorArchive:
    """Read headers first and materialize one tensor at a time (no full BF16 load)."""

    def __init__(self, directory: pathlib.Path, pattern: str, *, strict_metadata: bool = False):
        self.tensors: dict[str, TensorInfo] = {}
        files = sorted(directory.glob(pattern))
        if not files:
            raise ValueError(f"缺少 safetensors: {directory}/{pattern}")
        if set(files) != set(directory.glob("*.safetensors")):
            raise ValueError(f"存在额外/不支持的 safetensors（旋转或其它格式）: {directory}")
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as archive:
                if strict_metadata and (archive.metadata() or {}) not in ({}, {"format": "pt"}):
                    raise ValueError(f"不支持的 safetensors header 元数据: {path}")
                for name in archive.keys():
                    if name in self.tensors:
                        raise ValueError(f"分片重复 tensor: {name}")
                    view = archive.get_slice(name)
                    self.tensors[name] = TensorInfo(path, tuple(view.get_shape()), view.get_dtype())
        indices = list(directory.glob("*.safetensors.index.json"))
        if len(indices) > 1:
            raise ValueError(f"多个 safetensors index: {directory}")
        if indices:
            index = _read_json(indices[0]).get("weight_map")
            expected = {name: info.file.name for name, info in self.tensors.items()}
            if index != expected:
                raise ValueError(f"safetensors index 与实际分片 tensor 不一致: {indices[0]}")

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.tensors[name].file), framework="pt", device="cpu") as archive:
            return archive.get_tensor(name)


def _qkv_group(layer: str) -> str | None:
    prefix, separator, projection = layer.rpartition(".attn1.to_")
    return prefix + ".attn1" if separator and projection in ("q", "k", "v") else None


def _ignored_names(names: set[str]) -> list[str]:
    result = set()
    for name in names:
        if group := _qkv_group(name):
            name = group + ".to_qkv"
        name = name.replace(".ffn.net.0.proj", ".ffn.net_0.proj").replace(".ffn.net.2", ".ffn.net_2")
        result.add(name.removesuffix(".0") if name.endswith(".to_out.0") else name)
    return sorted(result)


def _reject_sidecars(directory: pathlib.Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file() and any(token in path.name.lower() for token in ("quarot", "rotation", "rotate", "hadamard")):
            raise ValueError(f"不支持旋转配方/sidecar: {path}")


class NativeMXFP4Expert:
    def __init__(self, quant_dir: pathlib.Path, original_dir: pathlib.Path, require_smooth: bool):
        if quant_dir.is_dir() and any(path.is_dir() for path in quant_dir.iterdir()):
            raise ValueError(f"不支持专家内嵌套/rank 子目录: {quant_dir}")
        self.original = TensorArchive(original_dir, "diffusion_pytorch_model*.safetensors")
        self.quant = TensorArchive(quant_dir, "quant_model_weight*.safetensors", strict_metadata=True)
        self.config = _read_json(original_dir / "config.json")
        if self.config.get("_class_name") != "WanTransformer3DModel" or self.config.get("quantization_config"):
            raise ValueError(f"原模型必须为未量化的 WanTransformer3DModel: {original_dir}")
        descriptions = sorted(quant_dir.glob("quant_model_description*.json"))
        if len(descriptions) != 1:
            raise ValueError(f"需要唯一 quant_model_description JSON: {quant_dir}")
        description_path = descriptions[0]
        description = _read_json(description_path)
        self.source_metadata = {}
        for name in ("model_quant_type", "version", "group_size"):
            if name in description:
                self.source_metadata[name] = description.pop(name)
        if self.source_metadata.get("model_quant_type") != QUANT_TYPE:
            raise ValueError(f"不支持/缺少 model_quant_type: {self.source_metadata}")
        if "version" in self.source_metadata and not isinstance(self.source_metadata["version"], str):
            raise ValueError("version 必须是字符串；不推测版本兼容性")
        if "group_size" in self.source_metadata and (
            type(self.source_metadata["group_size"]) is not int or self.source_metadata["group_size"] != 32
        ):
            raise ValueError("单级 MXFP4 仅支持 group_size=32")
        for name, label in description.items():
            if label not in (QUANT_TYPE, "FLOAT"):
                raise ValueError(f"不支持的 quant label/元数据: {name}={label!r}")
        undescribed = self.quant.tensors.keys() - description.keys()
        if undescribed:
            raise ValueError(f"tensor 缺量化描述: {sorted(undescribed)}")
        self.description = description
        self.mapping: dict[str, str] = {}
        self.quant_layers: set[str] = set()
        for name, label in description.items():
            target = _rename(name)
            if target in self.mapping:
                raise ValueError(f"重命名冲突: {name} 与 {self.mapping[target]} -> {target}")
            self.mapping[target] = name
            if label == QUANT_TYPE:
                if not name.endswith((".weight", ".weight_scale")):
                    raise ValueError(f"单级量化标签只能标注 weight/weight_scale: {name}")
                if name not in self.quant.tensors:
                    raise ValueError(f"缺少量化 tensor，禁止 BF16 补权重: {name}")
                if name.endswith(".weight"):
                    self.quant_layers.add(target.removesuffix(".weight"))
        if not self.quant_layers:
            raise ValueError("专家没有 W4A4_MXFP4 权重")
        # Root embeddings/head are floating in Wan. Both runtime and the CLI
        # reject their quantized form; numeric unpacking must not hide a lost scale.
        quantized_suffixes = {
            "attn1.to_qkv",
            "attn1.to_out",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out",
            "ffn.net_0.proj",
            "ffn.net_2",
        }
        for runtime_layer in _ignored_names(self.quant_layers):
            parts = runtime_layer.split(".", 2)
            if len(parts) != 3 or parts[0] != "blocks" or not parts[1].isdigit() or parts[2] not in quantized_suffixes:
                raise ValueError(
                    f"Native MXFP4 quantized layer is not supported by the Wan Linear method: {runtime_layer}"
                )
        self.smooth: dict[str, torch.Tensor] = {}
        self.reused_float: list[str] = []
        self._validate_layers(require_smooth)
        self._validate_mapping()
        self._validate_qkv()
        self.ignored = _ignored_names(
            {
                name.removesuffix(".weight")
                for name in self.original.tensors
                if name.endswith(".weight") and name.removesuffix(".weight") not in self.quant_layers
            }
        )
        self.description_hash = hashlib.sha256(description_path.read_bytes()).hexdigest()

    def _validate_layers(self, require_smooth: bool) -> None:
        for layer in sorted(self.quant_layers):
            target = layer + ".weight"
            name = self.mapping[target]
            source_prefix = name.removesuffix(".weight")
            original = self.original.tensors.get(target)
            if original is None or original.dtype != "BF16" or len(original.shape) != 2:
                raise ValueError(f"量化层缺少对应 BF16 二维原模型 shape: {name}")
            n, k = original.shape
            if n <= 0 or k <= 0 or k % 32:
                raise ValueError(f"需要完整 group32 的二维权重: {name}, {original.shape}")
            packed = self.quant.tensors[name]
            if packed.dtype != "U8" or packed.shape != (n, k // 2):
                raise ValueError(f"weight packing 必须为 uint8[{n},{k // 2}]: {name}, {packed}")
            scale_name = source_prefix + ".weight_scale"
            if self.description.get(scale_name) != QUANT_TYPE or scale_name not in self.quant.tensors:
                raise ValueError(f"缺少或标签不匹配的 weight_scale: {scale_name}")
            scale_info = self.quant.tensors[scale_name]
            if scale_info.dtype != "U8" or scale_info.shape != (n, k // 32):
                raise ValueError(f"weight_scale 必须为 uint8[{n},{k // 32}]: {scale_name}")
            if bool((self.quant.get(scale_name) == 255).any()):
                raise ValueError(f"weight_scale 含 E8M0 NaN (255): {scale_name}")
            bias_target = layer + ".bias"
            if bias_target in self.original.tensors:
                bias_name = source_prefix + ".bias"
                if self.description.get(bias_name) != "FLOAT" or bias_name not in self.quant.tensors:
                    raise ValueError(f"量化层缺导出 bias，禁止从 BF16 补齐: {bias_name}")
            wrapped = source_prefix.endswith(".linear")
            smooth_name = source_prefix.removesuffix(".linear") + ".div.mul_scale"
            if wrapped or require_smooth:
                if not wrapped or self.description.get(smooth_name) != "FLOAT":
                    raise ValueError(f"声明 Smooth 必须有 .linear 与 .div.mul_scale: {name}")
                if smooth_name not in self.quant.tensors:
                    raise ValueError(f"声明 Smooth 却缺少 mul_scale: {smooth_name}")
                info = self.quant.tensors[smooth_name]
                if info.dtype not in FLOAT_DTYPES or info.shape != (k,):
                    raise ValueError(f"mul_scale 必须为浮点 [{k}]: {smooth_name}")
                value = self.quant.get(smooth_name).float()
                runtime_scale = value.to(torch.bfloat16)
                if not bool(torch.isfinite(runtime_scale).all()) or not bool((runtime_scale > 0).all()):
                    raise ValueError(f"mul_scale 必须有限且正: {smooth_name}")
                self.smooth[layer] = value
        if self.smooth and len(self.smooth) != len(self.quant_layers):
            raise ValueError("不支持专家内部分 Smooth：当前 require_smooth_scale 是专家级合同")

    def _validate_mapping(self) -> None:
        for target, name in self.mapping.items():
            label = self.description[name]
            parent, _, attr = target.rpartition(".")
            if attr == "weight_scale":
                if parent not in self.quant_layers or label != QUANT_TYPE:
                    raise ValueError(f"孤立/非法 weight_scale: {name}")
                continue
            if attr == "mul_scale":
                if parent not in self.smooth or not name.endswith(".div.mul_scale"):
                    raise ValueError(f"孤立/非法 Smooth mul_scale: {name}")
                continue
            original = self.original.tensors.get(target)
            if original is None:
                raise ValueError(f"导出 tensor 无原模型映射（不支持配方）: {name} -> {target}")
            if label == QUANT_TYPE:
                continue
            if ".linear." in name and not (parent in self.quant_layers and attr == "bias"):
                raise ValueError(f"不支持 FLOAT Smooth wrapper: {name}")
            if name not in self.quant.tensors:
                if original.dtype != "BF16":
                    raise ValueError(f"明确 FLOAT 层只能回用原 BF16: {name}")
                self.reused_float.append(target)
                continue
            info = self.quant.tensors[name]
            if info.dtype not in FLOAT_DTYPES or info.shape != original.shape:
                raise ValueError(f"FLOAT tensor dtype/shape 不匹配: {name}")
            if attr == "weight" and len(info.shape) == 2 and info.dtype != "BF16":
                raise ValueError(f"未量化 Linear 必须是 BF16: {name}")
            if attr == "bias" and not bool(torch.isfinite(self.quant.get(name)).all()):
                raise ValueError(f"bias 必须为有限浮点: {name}")
        for target, original in self.original.tensors.items():
            if original.dtype not in FLOAT_DTYPES:
                raise ValueError(f"原模型含非浮点 scaffold: {target}")
            if target.endswith(".weight") and len(original.shape) == 2 and target not in self.mapping:
                raise ValueError(f"原模型 Linear 缺量化描述，不能推断为 BF16: {target}")

    def _validate_qkv(self) -> None:
        groups = {
            group
            for name in self.original.tensors
            if name.endswith(".weight") and (group := _qkv_group(name.removesuffix(".weight")))
        }
        for group in sorted(groups):
            layers = [group + ".to_" + part for part in "qkv"]
            if any(layer + ".weight" not in self.mapping for layer in layers):
                raise ValueError(f"融合 QKV 缺少完整 Q/K/V 描述: {group}")
            if len({layer in self.quant_layers for layer in layers}) != 1:
                raise ValueError(f"融合 QKV 不允许部分 BF16/量化: {group}")
            shapes = {self.original.tensors[layer + ".weight"].shape for layer in layers}
            biases = {layer + ".bias" in self.original.tensors for layer in layers}
            if len(shapes) != 1 or len(biases) != 1:
                raise ValueError(f"融合 QKV shape/bias 不一致: {group}")
            if layers[0] in self.smooth and not all(
                torch.equal(self.smooth[layers[0]], self.smooth[layer]) for layer in layers[1:]
            ):
                raise ValueError(f"融合 QKV mul_scale 必须完全一致: {group}")

    def weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Stream export tensors and only the permitted original FLOAT scaffold."""
        for target in sorted(self.original.tensors.keys() | self.mapping.keys()):
            source = self.mapping.get(target)
            if source is not None and source in self.quant.tensors:
                yield target, self.quant.get(source)
            else:
                yield target, self.original.get(target)


def validate_native_root(original_root: pathlib.Path, quant_root: pathlib.Path) -> None:
    """Only the native two-expert Wan T2V saver layout is supported."""
    model_index = _read_json(original_root / "model_index.json")
    if model_index.get("_class_name") != "WanPipeline" or any(name not in model_index for name in EXPERTS.values()):
        raise ValueError("Native MXFP4 requires a BF16 Diffusers WanPipeline with two experts")
    _reject_sidecars(quant_root)
    if not quant_root.is_dir():
        raise ValueError(f"Native MXFP4 directory does not exist: {quant_root}")
    directories = {path.name for path in quant_root.iterdir() if path.is_dir()}
    if (
        directories != set(EXPERTS)
        or list(quant_root.glob("*.safetensors"))
        or list(quant_root.glob("quant_model_description*.json"))
    ):
        raise ValueError("Native MXFP4 requires high_noise_model/low_noise_model without rank directories")
