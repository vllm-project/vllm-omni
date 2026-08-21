# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.lora.plan import AdditiveBiasUpdate, ConvertedLoRAState
from vllm_omni.diffusion.models.wan2_2.lora import (
    WAN_LORA_APPLY_PLAN,
    convert_wan_lora_state_dict,
    wan_lora_load_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _published_wan_t2v_state_dict() -> dict[str, torch.Tensor]:
    return {
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.ones(2, 3),
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.ffn.0.lora_down.weight": torch.ones(2, 3),
        "diffusion_model.blocks.0.ffn.0.lora_up.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.ffn.2.lora_down.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.ffn.2.lora_up.weight": torch.ones(3, 2),
    }


@pytest.mark.parametrize(
    ("filename", "component_name"),
    [
        ("wan_high_noise_lora.safetensors", "transformer"),
        ("wan_low_noise_lora.safetensors", "transformer_2"),
    ],
)
def test_wan22_lora_plan_routes_by_noise_range(filename: str, component_name: str) -> None:
    state_dict = _published_wan_t2v_state_dict()
    plan = wan_lora_load_plan(filename, tuple(state_dict), has_transformer_2=True)

    assert plan is not None
    assert plan.state_dict_converter is not None
    converted = plan.state_dict_converter(state_dict)
    assert isinstance(converted, ConvertedLoRAState)
    assert set(converted.lora_tensors) == {
        f"{component_name}.blocks.0.attn1.to_q.lora_A.weight",
        f"{component_name}.blocks.0.attn1.to_q.lora_B.weight",
        f"{component_name}.blocks.0.ffn.net_0.proj.lora_A.weight",
        f"{component_name}.blocks.0.ffn.net_0.proj.lora_B.weight",
        f"{component_name}.blocks.0.ffn.net_2.lora_A.weight",
        f"{component_name}.blocks.0.ffn.net_2.lora_B.weight",
    }


def test_wan_single_transformer_does_not_require_noise_range_name() -> None:
    state_dict = _published_wan_t2v_state_dict()
    plan = wan_lora_load_plan("wan21.safetensors", tuple(state_dict), has_transformer_2=False)

    assert plan is not None
    assert plan.state_dict_converter is not None
    converted = plan.state_dict_converter(state_dict)
    assert isinstance(converted, ConvertedLoRAState)
    assert all(key.startswith("transformer.") for key in converted.lora_tensors)


def test_wan_diffusers_alpha_is_folded_into_lora_b() -> None:
    lora_a = torch.ones(2, 3)
    lora_b = torch.ones(4, 2)
    converted = convert_wan_lora_state_dict(
        {
            "transformer.blocks.0.attn1.to_q.lora_A.weight": lora_a,
            "transformer.blocks.0.attn1.to_q.lora_B.weight": lora_b,
            "transformer.blocks.0.attn1.to_q.alpha": torch.tensor(1.0),
        },
        component_name="transformer",
    )

    assert set(converted.lora_tensors) == {
        "transformer.blocks.0.attn1.to_q.lora_A.weight",
        "transformer.blocks.0.attn1.to_q.lora_B.weight",
    }
    torch.testing.assert_close(
        converted.lora_tensors["transformer.blocks.0.attn1.to_q.lora_B.weight"],
        lora_b * 0.5,
    )


def test_wan22_ambiguous_adapter_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="high_noise.*low_noise"):
        wan_lora_load_plan(
            "adapter.safetensors",
            ("diffusion_model.blocks.0.self_attn.q.lora_down.weight",),
            has_transformer_2=True,
        )


def test_wan21_bias_deltas_are_normalized_for_shared_backend() -> None:
    state_dict = _published_wan_t2v_state_dict()
    state_dict["diffusion_model.blocks.0.self_attn.q.diff_b"] = torch.ones(4)
    state_dict["diffusion_model.blocks.0.self_attn.norm_q.diff"] = torch.zeros(4)

    converted = convert_wan_lora_state_dict(state_dict, component_name="transformer")

    assert not any(key.endswith((".diff", ".diff_b", ".bias")) for key in converted.lora_tensors)
    assert len(converted.auxiliary_updates) == 1
    update = converted.auxiliary_updates[0]
    assert isinstance(update, AdditiveBiasUpdate)
    assert update.module_name == "transformer.blocks.0.attn1.to_q"
    assert update.tensor is state_dict["diffusion_model.blocks.0.self_attn.q.diff_b"]


def test_wan_nonzero_unsupported_dense_delta_is_rejected() -> None:
    state_dict = _published_wan_t2v_state_dict()
    state_dict["diffusion_model.blocks.0.self_attn.norm_q.diff"] = torch.ones(4)

    with pytest.raises(ValueError, match="unsupported non-zero dense deltas.*norm_q.diff"):
        convert_wan_lora_state_dict(state_dict, component_name="transformer")


def test_wan_lora_plan_describes_both_transformers() -> None:
    assert WAN_LORA_APPLY_PLAN.component_names == ("transformer", "transformer_2")
    assert WAN_LORA_APPLY_PLAN.packed_modules_mapping == {"to_qkv": ("to_q", "to_k", "to_v")}
