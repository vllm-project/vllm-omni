# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_LORA_DOWN_SUFFIX = ".lora_down.weight"
_LORA_UP_SUFFIX = ".lora_up.weight"
_ALPHA_SUFFIX = ".alpha"


def _module_name(key: str, suffix: str) -> str:
    if not key.endswith(suffix):
        raise ValueError(f"Unexpected SenseNova LoRA tensor: {key}")
    return key[: -len(suffix)]


def _read_scalar_alpha(module_name: str, tensor: torch.Tensor) -> int:
    if tensor.numel() != 1:
        raise ValueError(f"LoRA alpha for {module_name} must be a scalar, got shape {tuple(tensor.shape)}")
    alpha = tensor.item()
    if not isinstance(alpha, int | float) or int(alpha) != alpha or alpha <= 0:
        raise ValueError(f"LoRA alpha for {module_name} must be a positive integer, got {alpha!r}")
    return int(alpha)


def _convert_tensors(
    source_tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int, int, list[str]]:
    down: dict[str, torch.Tensor] = {}
    up: dict[str, torch.Tensor] = {}
    alphas: dict[str, int] = {}

    for key, tensor in source_tensors.items():
        if key.endswith(_LORA_DOWN_SUFFIX):
            down[_module_name(key, _LORA_DOWN_SUFFIX)] = tensor
        elif key.endswith(_LORA_UP_SUFFIX):
            up[_module_name(key, _LORA_UP_SUFFIX)] = tensor
        elif key.endswith(_ALPHA_SUFFIX):
            module_name = _module_name(key, _ALPHA_SUFFIX)
            alphas[module_name] = _read_scalar_alpha(module_name, tensor)
        else:
            raise ValueError(f"Unexpected SenseNova LoRA tensor: {key}")

    modules = sorted(set(down) | set(up) | set(alphas))
    if not modules:
        raise ValueError("No LoRA tensors found in the SenseNova checkpoint")

    ranks: set[int] = set()
    alpha_values: set[int] = set()
    converted: dict[str, torch.Tensor] = {}
    for module_name in modules:
        if module_name not in down:
            raise ValueError(f"{module_name} is missing lora_down weight")
        if module_name not in up:
            raise ValueError(f"{module_name} is missing lora_up weight")
        if module_name not in alphas:
            raise ValueError(f"{module_name} is missing alpha")

        lora_a = down[module_name]
        lora_b = up[module_name]
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"{module_name} LoRA weights must be matrices, got {tuple(lora_a.shape)} and {tuple(lora_b.shape)}"
            )
        rank = lora_a.shape[0]
        if rank <= 0:
            raise ValueError(f"{module_name} LoRA weights must have a positive rank, got {rank}")
        if lora_b.shape[1] != rank:
            raise ValueError(f"{module_name} rank mismatch: lora_down rank {rank}, lora_up rank {lora_b.shape[1]}")

        ranks.add(rank)
        alpha_values.add(alphas[module_name])
        prefix = f"base_model.model.{module_name}"
        converted[f"{prefix}.lora_A.weight"] = lora_a
        converted[f"{prefix}.lora_B.weight"] = lora_b

    if len(ranks) != 1:
        raise ValueError(f"PEFT adapter_config.json requires a single rank, found {sorted(ranks)}")
    if len(alpha_values) != 1:
        raise ValueError(f"PEFT adapter_config.json requires a single alpha, found {sorted(alpha_values)}")

    return converted, ranks.pop(), alpha_values.pop(), modules


def convert_sensenova_lora(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    weights_path = output_dir / "adapter_model.safetensors"
    config_path = output_dir / "adapter_config.json"

    if not source_path.is_file():
        raise FileNotFoundError(f"SenseNova LoRA checkpoint not found: {source_path}")
    if not overwrite and (weights_path.exists() or config_path.exists()):
        raise FileExistsError(f"PEFT adapter already exists in {output_dir}; pass overwrite=True to replace it")

    source_tensors = load_file(str(source_path), device="cpu")
    converted, rank, alpha, target_modules = _convert_tensors(source_tensors)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(weights_path), metadata={"format": "pt"})
    config = {
        "bias": "none",
        "lora_alpha": alpha,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": target_modules,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an official SenseNova-U1 single-file LoRA checkpoint to PEFT format."
    )
    parser.add_argument("source", type=Path, help="Path to the official .safetensors checkpoint")
    parser.add_argument("output_dir", type=Path, help="Directory for adapter_config.json and adapter weights")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing converted adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_sensenova_lora(args.source, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
