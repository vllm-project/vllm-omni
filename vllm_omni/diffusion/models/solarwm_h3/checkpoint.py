# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Checkpoint contract adapted from SolarWM, Copyright (c) SolarWM contributors.
# Source: Junchao-cs/SolarWM@ce1da4e7705391eda8eeda6016c0fd3f614b975e
"""Strict Stage2 EMA loading and fixed native H3 LoRA sidecars."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

LORA_RANK = 384
LORA_ALPHA = 384
TARGET_DIMS = {
    "attn.to_q": (5376, 7168),
    "attn.to_k": (5376, 7168),
    "attn.to_v": (5376, 7168),
    "attn.to_out.0": (7168, 5376),
    "ff.net.0.proj": (5376, 28672),
    "ff.net.2": (14336, 5376),
}
BLOCK_COUNTS = (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2))


@dataclass(frozen=True)
class Stage2Adapter:
    tensors: Mapping[str, torch.Tensor]
    manifest: Mapping[str, Any]
    step: int
    rank: int = LORA_RANK
    alpha: int = LORA_ALPHA


def canonical_lora_key(key: str) -> str:
    """Normalize FSDP/checkpoint/PEFT wrappers without guessing layer names."""
    parts = [part for part in key.split(".") if part not in {"_fsdp_wrapped_module", "_checkpoint_wrapped_module"}]
    key = (
        ".".join(parts)
        .replace(".lora_A.default.weight", ".lora_A.weight")
        .replace(".lora_B.default.weight", ".lora_B.weight")
    )
    if key.startswith("base_model.model."):
        key = key[len("base_model.model.") :]
    return key


def _expected_targets() -> dict[str, tuple[int, int]]:
    return {
        f"{prefix}.{index}.{suffix}": dims
        for prefix, count in BLOCK_COUNTS
        for index in range(count)
        for suffix, dims in TARGET_DIMS.items()
    }


def _validate_tensors(values: Mapping[str, Any], *, rank: int) -> dict[str, torch.Tensor]:
    canonical: dict[str, torch.Tensor] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("SolarWM-H3 EMA tensor keys must be strings")
        key = canonical_lora_key(key)
        if key in canonical:
            raise ValueError(f"duplicate canonical SolarWM-H3 adapter tensor: {key}")
        canonical[key] = value
    expected = {
        f"{target}.lora_{side}.weight": ((rank, dims[0]) if side == "A" else (dims[1], rank))
        for target, dims in _expected_targets().items()
        for side in ("A", "B")
    }
    if canonical.keys() != expected.keys():
        missing, extra = sorted(expected.keys() - canonical.keys()), sorted(canonical.keys() - expected.keys())
        raise ValueError(f"SolarWM-H3 adapter tensor set differs: missing={missing[:3]}, unexpected={extra[:3]}")
    for key, shape in expected.items():
        value = canonical[key]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype not in (torch.float32, torch.bfloat16)
        ):
            raise ValueError(f"SolarWM-H3 adapter tensor descriptor differs: {key}, expected={shape}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite SolarWM-H3 adapter tensor: {key}")
    return canonical


def _validate_manifest(root: Path, manifest: Mapping[str, Any], complete: Mapping[str, Any]) -> int:
    if manifest.get("schema") != "solarwm.checkpoint.v2" or complete.get("schema") != "solarwm.checkpoint-complete.v2":
        raise ValueError("unsupported SolarWM checkpoint manifest/COMPLETE schemas")
    if complete.get("manifest_digest") != hashlib.blake2s((root / "checkpoint-manifest.json").read_bytes()).hexdigest():
        raise ValueError("SolarWM checkpoint manifest digest differs from COMPLETE")
    step = manifest.get("step")
    if type(step) is not int or step < 0 or complete.get("step") != step:
        raise ValueError("SolarWM checkpoint step differs between manifest and COMPLETE")
    contract = manifest.get("contract", {})
    required = {
        "family": "minimax_h3",
        "stage": "stage2",
        "camera_translation_transform": "logd4",
        "parameterization": "peft-lora-r384-alpha384",
        "causal_mode": "self_gradient_forcing",
        "objective": "flow_matching",
        "objective_variant": "data_ward_velocity",
    }
    if not isinstance(contract, Mapping) or any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("SolarWM-H3 requires a Stage2 SGF data-ward-flow LoRA-384/logd4 checkpoint")
    extras = contract.get("extras", {})
    required_extra = {
        "chunk_latents": 5,
        "window_chunks": 6,
        "target_latents": 47,
        "rollout_latents": 50,
        "student_rope_mode": "sliding_local",
    }
    if not isinstance(extras, Mapping) or any(extras.get(key) != value for key, value in required_extra.items()):
        raise ValueError("SolarWM-H3 checkpoint causal geometry or RoPE mode differs")
    profile = extras.get("encoder_profile", {})
    if not isinstance(profile, Mapping) or tuple(profile.get(k) for k in ("pixel_frames", "height", "width")) != (
        158,
        768,
        1344,
    ):
        raise ValueError("SolarWM-H3 checkpoint encoder geometry differs")
    files = manifest.get("files", [])
    records = {entry.get("path"): entry for entry in files if isinstance(entry, Mapping)}
    ema = records.get("ema.pt")
    if (
        len(records) != len(files)
        or ema is None
        or type(ema.get("size")) is not int
        or ema["size"] != (root / "ema.pt").stat().st_size
    ):
        raise ValueError("SolarWM-H3 manifest lacks the intact EMA component")
    return step


def load_stage2_adapter(root: str | Path, *, expected_step: int | None = None) -> Stage2Adapter:
    """Load the published EMA shadow, without optimizer/training state."""
    root = Path(root)
    if not root.is_dir() or any(
        not (root / name).is_file() for name in ("COMPLETE.json", "checkpoint-manifest.json", "ema.pt")
    ):
        raise ValueError(f"incomplete SolarWM-H3 Stage2 checkpoint: {root}")
    manifest = json.loads((root / "checkpoint-manifest.json").read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    if not isinstance(manifest, Mapping) or not isinstance(complete, Mapping):
        raise ValueError("SolarWM-H3 manifest and COMPLETE must be JSON objects")
    step = _validate_manifest(root, manifest, complete)
    if expected_step is not None and step != expected_step:
        raise ValueError(f"SolarWM-H3 checkpoint step={step}, expected={expected_step}")
    payload = torch.load(root / "ema.pt", map_location="cpu", weights_only=True, mmap=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "solarwm.minimax-h3-ema.v1"
        or not isinstance(payload.get("shadow"), Mapping)
    ):
        raise ValueError("SolarWM-H3 checkpoint has no compatible EMA shadow")
    return Stage2Adapter(_validate_tensors(payload["shadow"], rank=LORA_RANK), manifest, step, LORA_RANK, LORA_ALPHA)


class _StaticLoRASidecar(nn.Module):
    """Fixed BF16 LoRA factors; buffers move together with their owning linear."""

    def __init__(self, factors: list[tuple[torch.Tensor, torch.Tensor]], scale: float, *, device: torch.device):
        super().__init__()
        self.count, self.scale = len(factors), scale
        for index, (A, B) in enumerate(factors):
            self.register_buffer(f"A{index}", A.detach().to(device=device, dtype=torch.bfloat16).contiguous())
            self.register_buffer(f"B{index}", B.detach().to(device=device, dtype=torch.bfloat16).contiguous())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        parts = []
        for index in range(self.count):
            A, B = getattr(self, f"A{index}"), getattr(self, f"B{index}")
            parts.append(F.linear(F.linear(value.to(A.dtype), A), B) * self.scale)
        return torch.cat(parts, -1) if len(parts) > 1 else parts[0]


def _sidecar_forward(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
    base = output[0] if isinstance(output, tuple) else output
    delta = module.solarwm_lora(inputs[0]).to(base.dtype)
    result = base + delta
    return (result, *output[1:]) if isinstance(output, tuple) else result


def install_static_adapter(transformer: nn.Module, adapter: Stage2Adapter) -> None:
    """Attach the fixed Stage2 EMA without modifying any pretrained weights.

    Initial support requires unquantized, tensor-parallel-size-one projections.
    Ulysses sequence parallelism is compatible: sidecars operate on local rows.
    Native H3 QKV output is Q|K|V; native SwiGLU output is gate|up, whereas the
    Diffusers source is up|gate. All descriptors are checked before mutation.
    """
    values = _validate_tensors(adapter.tensors, rank=adapter.rank)
    modules = dict(transformer.named_modules())
    pending: dict[str, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    for target in _expected_targets():
        native = target.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.").replace(
            "transformer_blocks.", "blocks."
        )
        A, B = (values[f"{target}.lora_{side}.weight"] for side in ("A", "B"))
        if ".attn.to_" in native and native.rsplit(".", 1)[-1] in ("to_q", "to_k", "to_v"):
            native, slot = native.rsplit(".", 1)
            native += ".qkv_proj"
        elif native.endswith(".attn.to_out.0"):
            native, slot = native.removesuffix(".to_out.0") + ".out_proj", "plain"
        elif native.endswith(".ff.net.0.proj"):
            native, slot = native.removesuffix(".ff.net.0.proj") + ".mlp.fc1", "plain"
            up, gate = B.chunk(2, 0)
            B = torch.cat((gate, up), 0)
        elif native.endswith(".ff.net.2"):
            native, slot = native.removesuffix(".ff.net.2") + ".mlp.fc2", "plain"
        else:
            raise ValueError(f"unsupported SolarWM-H3 native target: {target}")
        pending.setdefault(native, []).append((slot, A, B))
    installations = []
    for name, factors in pending.items():
        module = modules.get(name)
        weight = getattr(module, "weight", None)
        if module is None or not isinstance(weight, torch.Tensor) or hasattr(module, "solarwm_lora"):
            raise ValueError(f"SolarWM-H3 native projection missing or already adapted: {name}")
        factors.sort(key=lambda item: {"to_q": 0, "to_k": 1, "to_v": 2, "plain": 0}[item[0]])
        input_dims = {A.shape[1] for _, A, _ in factors}
        expected_shape = (sum(B.shape[0] for _, _, B in factors), factors[0][1].shape[1])
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != expected_shape or len(input_dims) != 1:
            raise ValueError(f"SolarWM-H3 live adapter requires BF16 unsharded native projections: {name}")
        if name.endswith("qkv_proj") and [item[0] for item in factors] != ["to_q", "to_k", "to_v"]:
            raise ValueError(f"incomplete SolarWM-H3 native QKV target: {name}")
        installations.append((module, [(A, B) for _, A, B in factors], weight.device))
    for module, factors, device in installations:
        module.add_module("solarwm_lora", _StaticLoRASidecar(factors, adapter.alpha / adapter.rank, device=device))
        module.register_forward_hook(_sidecar_forward)
