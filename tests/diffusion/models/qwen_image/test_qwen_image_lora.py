# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.lora import (
    QWEN_IMAGE_LORA_APPLY_PLAN,
    convert_qwen_image_lora_state_dict,
    qwen_image_lora_load_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_qwen_image_published_lora_conversion_preserves_alpha_scale() -> None:
    lora_a = torch.ones(2, 3)
    lora_b = torch.ones(4, 2)
    converted = convert_qwen_image_lora_state_dict(
        {
            "transformer_blocks.0.attn.to_out.0.lora_down.weight": lora_a,
            "transformer_blocks.0.attn.to_out.0.lora_up.weight": lora_b,
            "transformer_blocks.0.attn.to_out.0.alpha": torch.tensor(1.0),
        }
    )

    assert set(converted) == {
        "transformer.transformer_blocks.0.attn.to_out.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_out.lora_B.weight",
    }
    assert torch.equal(
        converted["transformer.transformer_blocks.0.attn.to_out.lora_A.weight"],
        lora_a * 0.5,
    )
    assert torch.equal(
        converted["transformer.transformer_blocks.0.attn.to_out.lora_B.weight"],
        lora_b,
    )


def test_qwen_image_diffusers_lora_alpha_does_not_repeat_conversion() -> None:
    lora_a = torch.ones(2, 3)
    lora_b = torch.ones(4, 2)
    converted = convert_qwen_image_lora_state_dict(
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": lora_a,
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": lora_b,
            "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
        }
    )

    assert set(converted) == {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
    }
    torch.testing.assert_close(
        converted["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"],
        lora_b * 0.5,
    )


def test_qwen_image_unprefixed_diffusers_keys_receive_component_prefix() -> None:
    converted = convert_qwen_image_lora_state_dict(
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
            "transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
        }
    )

    assert set(converted) == {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
    }
    torch.testing.assert_close(
        converted["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"],
        torch.full((4, 2), 0.5),
    )


def test_qwen_image_lora_plan_describes_packed_projections() -> None:
    plan = qwen_image_lora_load_plan(
        "lightning.safetensors",
        ("transformer_blocks.0.attn.to_q.lora_down.weight",),
    )

    assert plan is not None
    assert plan.peft_config["lora_alpha"] is None
    assert QWEN_IMAGE_LORA_APPLY_PLAN.component_names == ("transformer",)
    assert QWEN_IMAGE_LORA_APPLY_PLAN.packed_modules_mapping == {
        "to_qkv": ("to_q", "to_k", "to_v"),
        "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    }


def test_qwen_image_unknown_single_file_is_not_claimed() -> None:
    assert qwen_image_lora_load_plan("unknown.safetensors", ("weight",)) is None
