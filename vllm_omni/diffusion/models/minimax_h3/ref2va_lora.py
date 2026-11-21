# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Immutable eight-step Ref2VA adapter, applied after the base linear operation.

The Larry v4 artifact uses contiguous Q/K/V and rank-16 AdaLN deltas in addition
to its rank-64 attention/MLP deltas. It must not be interpreted as the native
FlashGen or Diffusers Turbo formats, or merged into quantized base weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

FILENAME = "minimax_h3_turbo_v4_step600_ema.safetensors"
SHA256 = "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3"


def expected_targets() -> dict[str, tuple[int, int, int]]:
    targets = {}
    for prefix, count in (("blocks", 50), ("token_refiner.blocks", 2)):
        for index in range(count):
            for suffix, dims in {
                "attn.qkv_proj": (5376, 21504, 64),
                "attn.out_proj": (7168, 5376, 64),
                "mlp.fc1": (5376, 28672, 64),
                "mlp.fc2": (14336, 5376, 64),
            }.items():
                targets[f"{prefix}.{index}.{suffix}"] = dims
    for index in range(50):
        targets[f"blocks.{index}.adaln_proj.linear"] = (2688, 96768, 16)
    targets["final_layer.adaln_proj.linear"] = (2688, 10752, 16)
    return targets


def resolve_adapter_path(config: OmniDiffusionConfig) -> Path | None:
    """Claim only the known immutable startup adapter through existing CLI flags."""
    path = getattr(config, "lora_path", None)
    if isinstance(path, (list, tuple)):
        if len(path) != 1:
            return None
        path = path[0]
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / FILENAME
    if candidate.name != FILENAME:
        return None
    if getattr(config, "lora_backend", None) != "distill":
        raise ValueError("the eight-step Ref2VA adapter requires --lora-backend distill")
    if float(getattr(config, "lora_scale", 1.0)) != 1.0:
        raise ValueError("the eight-step Ref2VA adapter requires --lora-scale 1")
    if not candidate.is_file():
        raise ValueError(f"Ref2VA LoRA file does not exist: {candidate}")
    return candidate


def load_checked(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError("Ref2VA v4-step600 EMA LoRA SHA256 does not match the supported artifact")
    state = load_file(str(path), device="cpu")
    expected = expected_targets()
    if set(state) != {f"{name}.lora_{side}.weight" for name in expected for side in "AB"}:
        raise ValueError("Ref2VA LoRA must contain exactly the 259 expected A/B pairs")
    pairs = {}
    for name, (inputs, outputs, rank) in expected.items():
        a, b = (state[f"{name}.lora_{side}.weight"] for side in "AB")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError(f"Ref2VA LoRA requires BF16 tensors: {name}")
        if tuple(a.shape) != (rank, inputs) or tuple(b.shape) != (outputs, rank):
            raise ValueError(f"Ref2VA LoRA tensor dimensions do not match {name}")
        pairs[name] = (a, b)
    return pairs


class _ForwardDelta(torch.nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)

    def forward(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(x.to(torch.bfloat16), self.a), self.b)
        if update.shape != output.shape:
            raise ValueError("Ref2VA LoRA output dimensions do not match the base projection")
        return output + update.to(output.dtype)


def _add_delta(
    module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor | None]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("Ref2VA LoRA expects a vLLM linear (output, bias) tuple")
    result, bias = output
    return module.ref2va_delta(args[0], result), bias


def install_adapter(model: torch.nn.Module, path: Path) -> None:
    """Install once; keep base quantization and adapter BF16 arithmetic separate."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("the eight-step Ref2VA adapter requires DiT TP=1; Ulysses SP is supported")
    pairs = load_checked(path)
    modules = []
    for name, (inputs, outputs, _) in expected_targets().items():
        module = model.get_submodule(name)
        if getattr(module, "input_size", None) != inputs or getattr(module, "output_size", None) != outputs:
            raise ValueError(f"Ref2VA LoRA target dimensions do not match {name}")
        if getattr(module, "skip_bias_add", False) or hasattr(module, "ref2va_delta"):
            raise ValueError(f"Ref2VA LoRA target cannot be bound: {name}")
        modules.append((name, module))
    # Validate every target before changing any module.
    for name, module in modules:
        module.add_module("ref2va_delta", _ForwardDelta(*pairs[name]))
        module.register_forward_hook(_add_delta)
    model.adaln_cache.clear()

    def prepare(module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        # Called after the model's normal offload hook. Registered delta buffers
        # also follow subsequent module device moves.
        x = kwargs["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("Ref2VA DiT expects the x tensor as a keyword argument")
        device = x.device
        for _, target in modules:
            if target.ref2va_delta.a.device != device:
                target.ref2va_delta.to(device)

    model.register_forward_pre_hook(prepare, with_kwargs=True)
