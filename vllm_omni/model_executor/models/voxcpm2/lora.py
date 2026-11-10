# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Merge a native VoxCPM2 LoRA checkpoint at startup for all requests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn


def merge_voxcpm2_lora(
    adapter_path: str,
    *,
    base_lm: nn.Module,
    residual_lm: nn.Module,
    tts: nn.Module,
) -> int:
    """Validate and merge one local native safetensors adapter.

    Call once after base weight loading and before fusion or graph capture.
    Validate all checkpoint keys and tensors before changing weights.
    """
    path = Path(adapter_path)
    with (path / "lora_config.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    config = metadata.get("lora_config") if isinstance(metadata, dict) else None
    if not isinstance(config, dict):
        raise ValueError("VoxCPM2 lora_config.json must contain a lora_config object")
    rank, alpha = config.get("r"), config.get("alpha")
    if type(rank) is not int or rank <= 0:
        raise ValueError("VoxCPM2 LoRA r must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError("VoxCPM2 LoRA alpha must be finite")

    targets: dict[str, nn.Linear] = {}
    groups = (
        ("lm", {"base_lm": base_lm, "residual_lm": residual_lm}, "target_modules_lm"),
        ("dit", {"feat_decoder.estimator": tts.feat_decoder.estimator}, "target_modules_dit"),
        ("proj", {"": tts}, "target_proj_modules"),
    )
    for group, roots, target_key in groups:
        enabled = config.get(f"enable_{group}", False)
        if type(enabled) is not bool:
            raise ValueError(f"VoxCPM2 LoRA enable_{group} must be a boolean")
        if not enabled:
            continue
        defaults = (
            ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]
            if group == "proj"
            else ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        names = config.get(target_key, defaults)
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError(f"VoxCPM2 LoRA {target_key} must be a non-empty list of names")
        for prefix, root in roots.items():
            matched = set()
            for name, module in root.named_modules():
                selector = name if group == "proj" else name.rsplit(".", 1)[-1]
                if selector in names and isinstance(module, nn.Linear):
                    targets[f"{prefix}.{name}" if prefix else name] = module
                    matched.add(selector)
            if missing := set(names) - matched:
                raise ValueError(f"VoxCPM2 LoRA targets not found in {prefix or 'tts'}: {sorted(missing)}")
    if not targets:
        raise ValueError("VoxCPM2 LoRA config enables no target layers")

    state = load_file(str(path / "lora_weights.safetensors"), device="cpu")
    expected = {f"{name}.lora_{suffix}" for name in targets for suffix in ("A", "B")}
    if set(state) != expected:
        raise ValueError(
            "VoxCPM2 LoRA checkpoint keys do not match the configured targets: "
            f"missing={sorted(expected - set(state))}, unexpected={sorted(set(state) - expected)}"
        )
    for name, module in targets.items():
        if not module.weight.is_floating_point() or module.weight.is_meta:
            raise ValueError(f"VoxCPM2 LoRA requires materialized floating-point weights: {name}")
        for suffix, shape in (("A", (rank, module.in_features)), ("B", (module.out_features, rank))):
            key = f"{name}.lora_{suffix}"
            tensor = state[key]
            if tensor.shape != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid VoxCPM2 LoRA tensor {key}: expected finite floating-point shape {shape}")

    with torch.no_grad():
        for name, module in targets.items():
            # Merge one layer at a time in fp32, preserving the base Parameter.
            a = state[f"{name}.lora_A"].to(device=module.weight.device, dtype=torch.float32)
            b = state[f"{name}.lora_B"].to(device=module.weight.device, dtype=torch.float32)
            merged = torch.addmm(module.weight.float(), b, a, alpha=alpha / rank)
            module.weight.copy_(merged)
    return len(targets)
