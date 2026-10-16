# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flux3 raw LoRA checkpoint loading."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import safetensors
import safetensors.torch
import torch
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest

_PEFT_CONFIG_FILE = "adapter_config.json"
_SAFETENSORS_SUFFIX = ".safetensors"


def infer_peft_helper_from_file(safetensor_path: str | Path) -> PEFTHelper:
    """Infer PEFT metadata from a bare Flux3 LoRA safetensors checkpoint."""
    header_r: int | None = None
    header_alpha: float | None = None
    header_target_modules: list[str] | None = None

    with safetensors.safe_open(str(safetensor_path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        for key in ("rank", "r"):
            if key in metadata:
                try:
                    header_r = int(metadata[key])
                    break
                except (TypeError, ValueError):
                    pass
        for key in ("alpha", "lora_alpha"):
            if key in metadata:
                try:
                    header_alpha = float(metadata[key])
                    break
                except (TypeError, ValueError):
                    pass
        if "target_modules" in metadata:
            try:
                modules = json.loads(metadata["target_modules"])
                if isinstance(modules, (list, tuple)) and modules:
                    header_target_modules = sorted({module.rsplit(".", 1)[-1] for module in modules})
            except (AttributeError, json.JSONDecodeError, TypeError):
                pass

        need_r = header_r is None
        need_alpha = header_alpha is None
        need_modules = header_target_modules is None
        tensor_ranks: list[int] = []
        tensor_leaf_modules: list[str] = []
        alpha_from_tensor: float | None = None
        if need_r or need_alpha or need_modules:
            for key in checkpoint.keys():
                if "lora_alpha" in key.lower():
                    if need_alpha:
                        tensor = checkpoint.get_tensor(key)
                        if tensor.numel() == 1:
                            alpha_from_tensor = float(tensor.item())
                    continue
                if not (need_r or need_modules) or not key.endswith(".lora_A.weight"):
                    continue
                tensor = checkpoint.get_tensor(key)
                if need_r:
                    tensor_ranks.append(tensor.shape[0])
                if need_modules:
                    tensor_leaf_modules.append(key[: -len(".lora_A.weight")].rsplit(".", 1)[-1])

    if header_r is not None:
        rank = header_r
    elif tensor_ranks:
        rank = Counter(tensor_ranks).most_common(1)[0][0]
    else:
        raise ValueError(
            f"Cannot determine LoRA rank: no rank in header and no lora_A tensors found in {safetensor_path}"
        )
    lora_alpha = (
        header_alpha
        if header_alpha is not None
        else alpha_from_tensor
        if alpha_from_tensor is not None
        else float(rank)
    )
    target_modules = header_target_modules if header_target_modules is not None else sorted(set(tensor_leaf_modules))
    return PEFTHelper.from_dict(
        {
            "r": rank,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
        }
    )


def load_flux3_raw_lora(
    *,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    weights_mapper: Any | None,
) -> tuple[LoRAModel, PEFTHelper] | None:
    """Load a Flux3 raw safetensors adapter, or decline a PEFT directory."""
    path = Path(lora_path)
    if path.is_dir():
        if (path / _PEFT_CONFIG_FILE).is_file():
            return None
        safetensors_files = sorted(path.glob(f"*{_SAFETENSORS_SUFFIX}"))
    elif path.suffix == _SAFETENSORS_SUFFIX:
        safetensors_files = [path]
    else:
        raise ValueError(
            "Flux3 raw LoRA adapters must be a .safetensors file or a directory containing .safetensors files; "
            f"got {str(path)!r}"
        )

    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {str(path)!r}")

    tensors: dict[str, torch.Tensor] = {}
    for safetensors_file in safetensors_files:
        # Sorting makes overlapping keys deterministic: later files win.
        tensors.update(safetensors.torch.load_file(str(safetensors_file), device="cpu"))

    peft_helper = infer_peft_helper_from_file(str(safetensors_files[0]))
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        model_vocab_size=None,
        weights_mapper=weights_mapper,
    )
    return lora_model, peft_helper
