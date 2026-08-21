# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import torch
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_wan_lora_to_diffusers,
)

from vllm_omni.diffusion.lora.plan import (
    AdditiveBiasUpdate,
    ConvertedLoRAState,
    DiffusionLoRAApplyPlan,
    DiffusionLoRALoadPlan,
)
from vllm_omni.diffusion.lora.utils import fold_diffusers_lora_alpha

_WAN_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "add_k_proj",
    "add_v_proj",
    "proj",
    "net_2",
    "time_proj",
    "proj_out",
    "linear_1",
    "linear_2",
)

WAN_LORA_APPLY_PLAN = DiffusionLoRAApplyPlan(
    component_names=("transformer", "transformer_2"),
    target_modules=_WAN_LORA_TARGETS,
    packed_modules_mapping={"to_qkv": ("to_q", "to_k", "to_v")},
)


def _wan_lora_component(adapter_path: str, has_transformer_2: bool) -> str:
    if not has_transformer_2:
        return "transformer"

    filename = Path(adapter_path).name.lower()
    if "high_noise" in filename:
        return "transformer"
    if "low_noise" in filename:
        return "transformer_2"
    raise ValueError(
        "Wan2.2 uses separate high- and low-noise transformers. The LoRA "
        "filename must contain 'high_noise' or 'low_noise' so it can be "
        "assigned without relying on adapter argument order."
    )


def convert_wan_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    component_name: str,
) -> ConvertedLoRAState:
    """Normalize one published Wan LoRA for the selected transformer."""

    if any(key.endswith(".alpha") for key in state_dict) and any(
        key.endswith((".lora_A.weight", ".lora_B.weight")) for key in state_dict
    ):
        state_dict = fold_diffusers_lora_alpha(state_dict)

    # Diffusers silently drops unsupported ``.diff`` tensors such as norm
    # deltas because published adapters normally store them as zero-valued
    # placeholders. Preserve that compatibility, but never discard a real
    # dense update. ``head.head.diff`` is the one dense delta that the upstream
    # converter explicitly represents as a LoRA A/B pair.
    unsupported_dense_deltas = {
        key: value
        for key, value in state_dict.items()
        if key.endswith(".diff") and not key.removeprefix("diffusion_model.").endswith("head.head.diff")
    }
    nonzero_dense_deltas = [key for key, value in unsupported_dense_deltas.items() if torch.count_nonzero(value).item()]
    if nonzero_dense_deltas:
        raise ValueError(f"This Wan adapter contains unsupported non-zero dense deltas: {nonzero_dense_deltas[:3]}")
    if unsupported_dense_deltas:
        state_dict = {key: value for key, value in state_dict.items() if key not in unsupported_dense_deltas}

    if any(key.startswith("diffusion_model.") for key in state_dict):
        state_dict = _convert_non_diffusers_wan_lora_to_diffusers(dict(state_dict))

    unsupported = [key for key in state_dict if key.endswith((".diff", ".diff_b"))]
    if unsupported:
        raise ValueError(f"This Wan adapter contains unsupported dense deltas: {unsupported[:3]}")

    converted: dict[str, torch.Tensor] = {}
    auxiliary_updates: list[AdditiveBiasUpdate] = []
    for key, value in state_dict.items():
        key = key.replace(".ffn.net.0.", ".ffn.net_0.")
        key = key.replace(".ffn.net.2.", ".ffn.net_2.")
        key = key.replace(".to_out.0.", ".to_out.")
        if key.startswith("transformer."):
            key = f"{component_name}.{key.removeprefix('transformer.')}"
        elif not key.startswith(f"{component_name}."):
            key = f"{component_name}.{key}"
        if key.endswith(".lora_B.bias"):
            auxiliary_updates.append(
                AdditiveBiasUpdate(
                    module_name=key.removesuffix(".lora_B.bias"),
                    tensor=value,
                )
            )
        else:
            converted[key] = value
    return ConvertedLoRAState(
        lora_tensors=converted,
        auxiliary_updates=tuple(auxiliary_updates),
    )


def wan_lora_load_plan(
    adapter_path: str,
    tensor_keys: tuple[str, ...],
    *,
    has_transformer_2: bool,
) -> DiffusionLoRALoadPlan | None:
    if not any(key.endswith((".lora_A.weight", ".lora_down.weight")) for key in tensor_keys):
        return None

    component_name = _wan_lora_component(adapter_path, has_transformer_2)

    def convert(state_dict: dict[str, torch.Tensor]) -> ConvertedLoRAState:
        return convert_wan_lora_state_dict(state_dict, component_name=component_name)

    return DiffusionLoRALoadPlan(
        peft_config={
            "lora_alpha": None,
            "target_modules": list(_WAN_LORA_TARGETS),
        },
        state_dict_converter=convert,
    )


class WanLoRAPlanMixin:
    def get_lora_apply_plan(self) -> DiffusionLoRAApplyPlan:
        return WAN_LORA_APPLY_PLAN

    def get_lora_load_plan(
        self,
        adapter_path: str,
        tensor_keys: tuple[str, ...],
    ) -> DiffusionLoRALoadPlan | None:
        return wan_lora_load_plan(
            adapter_path,
            tensor_keys,
            has_transformer_2=bool(getattr(self, "has_transformer_2", False)),
        )
