#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Import a single-scale msModelSlim MindIE Wan2.2 T2V export on CPU.

See README_mxfp4_single_checkpoint.md for the deliberately narrow source contract.
This tool unpacks E2M1 codes; it never dequantizes, calibrates or requantizes weights.
Only torch and safetensors are needed, without importing the inference package.
"""

from __future__ import annotations

import argparse

# File-based loading keeps this optional CLI usable without vLLM installed.
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

_native_spec = importlib.util.spec_from_file_location(
    "_omni_mxfp4_native", pathlib.Path(__file__).with_name("mxfp4_native.py")
)
assert _native_spec is not None and _native_spec.loader is not None
_native = importlib.util.module_from_spec(_native_spec)
sys.modules[_native_spec.name] = _native
_native_spec.loader.exec_module(_native)
SOURCE_COMMIT = _native.SOURCE_COMMIT
C7_REFERENCE_COMMIT = _native.C7_REFERENCE_COMMIT
SOURCE_CONTRACT = _native.SOURCE_CONTRACT
QUANT_TYPE = _native.QUANT_TYPE
EXPERTS = _native.EXPERTS
validate_native_root = _native.validate_native_root

if TYPE_CHECKING:
    from vllm_omni.quantization.tools.mxfp4_native import NativeMXFP4Expert
else:
    NativeMXFP4Expert = _native.NativeMXFP4Expert


def _write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    # E2M1 sign-magnitude, low nibble first along K. All values (including -0)
    # are exactly representable in BF16, the current Omni loader's placeholder.
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    return table[codes].reshape(packed.shape[0], packed.shape[1] * 2).to(torch.bfloat16)


class ExpertConversion(NativeMXFP4Expert):
    def write(self, directory: pathlib.Path, max_shard_bytes: int) -> dict:
        directory.mkdir()
        tensors: dict[str, torch.Tensor] = {}
        weight_map = {}
        total_size = 0
        shard_size = 0
        shard_number = 0

        def flush():
            nonlocal shard_number, shard_size
            if not tensors:
                return
            shard_number += 1
            name = f"diffusion_pytorch_model-{shard_number:05d}.safetensors"
            save_file(tensors, str(directory / name), metadata={"format": "pt"})
            weight_map.update({key: name for key in tensors})
            tensors.clear()
            shard_size = 0

        for target, value in self.weights():
            source = self.mapping.get(target)
            if source is not None and target.endswith(".weight") and self.description[source] == QUANT_TYPE:
                value = _unpack_fp4(value)
            elif target.endswith(".mul_scale"):
                value = value.float()
            size = value.numel() * value.element_size()
            if shard_size + size > max_shard_bytes:
                flush()
            tensors[target] = value.contiguous()
            total_size += size
            shard_size += size
        flush()
        _write_json(
            directory / "diffusion_pytorch_model.safetensors.index.json",
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
        )
        self.config["quantization_config"] = {
            "quant_method": "mxfp4",
            "is_checkpoint_mxfp4_serialized": True,
            "ignored_layers": self.ignored,
            "require_smooth_scale": bool(self.smooth),
        }
        _write_json(directory / "config.json", self.config)
        _write_json(
            directory / "quant_model_description.json",
            {target: self.description[source] for target, source in self.mapping.items()},
        )
        return {
            "source_metadata": self.source_metadata,
            "description_sha256": self.description_hash,
            "quantized_layers": sorted(self.quant_layers),
            "ignored_layers": self.ignored,
            "smooth": "required" if self.smooth else "absent; does not validate Smooth",
            "original_bf16_for_explicit_float": sorted(self.reused_float),
            "source_files": sorted({str(info.file) for info in self.quant.tensors.values()}),
            "output_weight_dtype": "BF16 numeric E2M1, no block scale applied",
            "output_scale_dtype": "uint8 E8M0 bytes unchanged",
            "total_size": total_size,
        }


def repack(
    model_type: str,
    original_model_path: pathlib.Path,
    quant_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    require_smooth_scale: bool = False,
    max_shard_bytes: int = 4 * 1024**3,
) -> dict:
    if model_type != "Wan2.2-T2V-A14B":
        raise ValueError("当前只支持 Wan2.2-T2V-A14B 双专家")
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes 必须是正整数")
    original_model_path, quant_path, output_path = (
        pathlib.Path(path).resolve() for path in (original_model_path, quant_path, output_path)
    )
    if output_path.exists():
        raise ValueError(f"输出目录已存在，拒绝覆盖: {output_path}")
    for source in (original_model_path, quant_path):
        if not source.is_dir() or output_path.is_relative_to(source) or source.is_relative_to(output_path):
            raise ValueError(f"输入/输出目录必须独立且输入存在: {source}, {output_path}")
    validate_native_root(original_model_path, quant_path)
    experts = {
        destination: ExpertConversion(quant_path / source, original_model_path / destination, require_smooth_scale)
        for source, destination in EXPERTS.items()
    }
    report: dict[str, Any] = {
        "source_contract": SOURCE_CONTRACT,
        "source_reference_commit": SOURCE_COMMIT,
        "additional_source_reference_commit": C7_REFERENCE_COMMIT,
        "producer_version": "unknown; descriptor version is not a verified installed msModelSlim version",
        "algorithm_provenance": "not verified: C7=7.25 and enable_search=false require generation evidence",
        "bf16_replacement_policy": "explicit FLOAT only; no replacement of unsupported quantization",
        "experts": {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A failed second expert must not leave an apparently complete first expert.
    with tempfile.TemporaryDirectory(prefix=".mxfp4-import-", dir=output_path.parent) as temporary:
        staging = pathlib.Path(temporary) / "model"

        def ignore_experts(directory, names):
            return set(names) & set(EXPERTS.values()) if pathlib.Path(directory) == original_model_path else set()

        shutil.copytree(original_model_path, staging, ignore=ignore_experts)
        for destination, expert in experts.items():
            report["experts"][destination] = expert.write(staging / destination, max_shard_bytes)
        _write_json(staging / "mxfp4_conversion_report.json", report)
        if output_path.exists():
            raise ValueError(f"输出目录在转换期间被创建，拒绝覆盖: {output_path}")
        staging.rename(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="严格导入单级 MXFP4 Wan2.2 T2V 双专家（仅 CPU 格式转换）。")
    parser.add_argument("--model-type", default="Wan2.2-T2V-A14B", choices=["Wan2.2-T2V-A14B"])
    parser.add_argument("--original-model", required=True, type=pathlib.Path, help="原始 BF16 Diffusers 模型")
    parser.add_argument("--quant-path", required=True, type=pathlib.Path, help="msModelSlim 两专家导出根目录")
    parser.add_argument("--output-path", required=True, type=pathlib.Path, help="尚不存在的目标目录")
    parser.add_argument("--require-smooth-scale", action="store_true", help="要求两专家全部量化层具有真实 Smooth")
    parser.add_argument("--max-shard-size-mb", type=int, default=4096, help="输出分片 MiB；单 tensor 不拆分")
    args = parser.parse_args()
    try:
        report = repack(
            args.model_type,
            args.original_model,
            args.quant_path,
            args.output_path,
            require_smooth_scale=args.require_smooth_scale,
            max_shard_bytes=args.max_shard_size_mb * 1024**2,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"转换失败: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
