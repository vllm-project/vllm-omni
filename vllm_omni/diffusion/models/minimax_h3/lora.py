# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import torch
from safetensors import safe_open
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.lora.plan import (
    DiffusionLoRAApplyPlan,
    DiffusionLoRALoadPlan,
)

_MINIMAX_H3_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "qkv_proj",
    "out_proj",
    "fc1",
    "fc2",
)

MINIMAX_H3_LORA_APPLY_PLAN = DiffusionLoRAApplyPlan(
    component_names=("transformer",),
    target_modules=_MINIMAX_H3_LORA_TARGETS,
    packed_modules_mapping={"qkv_proj": ("to_q", "to_k", "to_v")},
)

_TURBO_LORA_CONFIG = {
    # Legacy v0.1 has no alpha metadata and its official inference path uses 8.
    # Newer releases carry their training alpha in safetensors metadata.
    "lora_alpha": 8,
    "target_modules": list(_MINIMAX_H3_LORA_TARGETS),
}

_TURBO_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "token_refiner.refiner_blocks.": "token_refiner.blocks.",
        "transformer_blocks.": "blocks.",
        ".attn.to_out.0.": ".attn.out_proj.",
        ".ff.net.0.proj.": ".mlp.fc1.",
        ".ff.net.2.": ".mlp.fc2.",
        ".lora_A.default.": ".lora_A.",
        ".lora_B.default.": ".lora_B.",
    }
)


def _convert_turbo_lora_state_dict(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Restore Diffusers SwiGLU LoRA output rows to native H3 order."""

    converted = dict(tensors)
    for name, tensor in tensors.items():
        if ".ff.net.0.proj.lora_B." not in name:
            continue
        if tensor.shape[0] % 2:
            raise ValueError(
                f"MiniMax-H3 Turbo FFN lora_B output dimension must be even, got {tuple(tensor.shape)} for {name}."
            )
        value, gate = tensor.chunk(2, dim=0)
        converted[name] = torch.cat((gate, value), dim=0).contiguous()
    return converted


def _turbo_lora_config(adapter_path: str) -> dict[str, object]:
    with safe_open(adapter_path, framework="pt", device="cpu") as checkpoint:
        raw_alpha = (checkpoint.metadata() or {}).get("alpha")
    if raw_alpha is None:
        return dict(_TURBO_LORA_CONFIG)

    try:
        alpha_value = float(raw_alpha)
    except ValueError as error:
        raise ValueError(f"MiniMax-H3 Turbo LoRA alpha must be numeric, got {raw_alpha!r}.") from error
    if not math.isfinite(alpha_value) or alpha_value <= 0 or not alpha_value.is_integer():
        raise ValueError(f"MiniMax-H3 Turbo LoRA alpha must be a positive integer, got {raw_alpha!r}.")
    return {**_TURBO_LORA_CONFIG, "lora_alpha": int(alpha_value)}


def minimax_h3_lora_load_plan(
    adapter_path: str,
    tensor_keys: tuple[str, ...],
) -> DiffusionLoRALoadPlan | None:
    """Describe the supported lightx2v MiniMax-H3 Turbo checkpoint."""

    if any(
        ("transformer_blocks." in key or "token_refiner.refiner_blocks." in key)
        and (".lora_A.default." in key or ".lora_B.default." in key)
        for key in tensor_keys
    ):
        return DiffusionLoRALoadPlan(
            peft_config=_turbo_lora_config(adapter_path),
            state_dict_converter=_convert_turbo_lora_state_dict,
            weights_mapper=_TURBO_WEIGHTS_MAPPER,
        )
    return None
