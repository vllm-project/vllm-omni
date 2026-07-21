# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-Omni Thinker fused-MoE LoRA setup."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch import nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.model_manager import LoRAModelManager
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    _ensure_thinker_architecture,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_EXPERTS = 2
_HIDDEN_SIZE = 4
_INTERMEDIATE_SIZE = 4
_LORA_RANK = 1
_EXPERTS_MODULE = "language_model.model.layers.0.mlp.experts"
_PEFT_TARGET_PARAMETERS = [
    "thinker.model.layers.0.mlp.experts.gate_up_proj",
    "thinker.model.layers.0.mlp.experts.down_proj",
]


class _TinyExperts(nn.Module):
    """Expert parameters with the same 3-D layout as Qwen3-Omni."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(_NUM_EXPERTS, 2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE))
        self.down_proj = nn.Parameter(torch.zeros(_NUM_EXPERTS, _HIDDEN_SIZE, _INTERMEDIATE_SIZE))


class _TinyQwen3OmniMoe(nn.Module):
    """Minimal module tree needed for PEFT to save expert LoRA weights."""

    def __init__(self, model_path: Path):
        super().__init__()
        self.thinker = nn.ModuleDict(
            {
                "model": nn.ModuleDict(
                    {"layers": nn.ModuleList([nn.ModuleDict({"mlp": nn.ModuleDict({"experts": _TinyExperts()})})])}
                )
            }
        )
        self.name_or_path = str(model_path)
        self.config = {"model_type": "qwen3_omni_moe"}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _CpuFusedMoE3DWithLoRA(FusedMoE3DWithLoRA):
    """CPU harness for the real 3-D MoE LoRA weight methods."""

    def __init__(self):
        nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(
            hidden_dim=_HIDDEN_SIZE,
            num_local_experts=_NUM_EXPERTS,
            num_experts=_NUM_EXPERTS,
            intermediate_size_per_partition=_INTERMEDIATE_SIZE,
            moe_parallel_config=SimpleNamespace(
                tp_size=1,
                tp_rank=0,
                ep_rank=0,
                use_ep=False,
            ),
        )
        self.device = torch.device("cpu")
        self.tp_size = 1
        self.tp_rank = 0
        self._w13_slices = 1


def _save_peft_adapter(tmp_path: Path) -> Path:
    base_model_dir = tmp_path / "base_model"
    base_model_dir.mkdir()
    (base_model_dir / "config.json").write_text('{"model_type": "qwen3_omni_moe"}', encoding="utf-8")

    peft_model = get_peft_model(
        _TinyQwen3OmniMoe(base_model_dir),
        PeftLoraConfig(
            r=_LORA_RANK,
            lora_alpha=_LORA_RANK,
            target_modules=[],
            target_parameters=_PEFT_TARGET_PARAMETERS,
        ),
    )
    with torch.no_grad():
        for value, parameter in enumerate(
            (parameter for parameter in peft_model.parameters() if parameter.requires_grad),
            start=1,
        ):
            parameter.fill_(value)

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    return adapter_dir


def test_peft_expert_lora_loads_into_thinker(tmp_path: Path):
    adapter_dir = _save_peft_adapter(tmp_path)
    peft_helper = PEFTHelper.from_local_dir(str(adapter_dir), max_position_embeddings=128)
    lora_model = LoRAModel.from_local_checkpoint(
        str(adapter_dir),
        {"experts"},
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        dtype=torch.float32,
        weights_mapper=Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper,
    )

    thinker_config = PretrainedConfig()
    _ensure_thinker_architecture(thinker_config)
    fused_moe = _CpuFusedMoE3DWithLoRA()
    fused_moe.create_lora_weights(
        max_loras=1,
        lora_config=LoRAConfig(
            max_lora_rank=_LORA_RANK,
            max_loras=1,
            lora_dtype=torch.float32,
        ),
        model_config=thinker_config,
    )

    manager = object.__new__(LoRAModelManager)
    manager.is_pooling_model = False
    manager._is_3d_moe_model = Qwen3OmniMoeThinkerForConditionalGeneration.is_3d_moe_weight
    manager._stack_moe_lora_weights(lora_model, fused_moe, _EXPERTS_MODULE)

    expert_lora = lora_model.get_lora(_EXPERTS_MODULE)
    assert expert_lora is not None
    fused_moe.set_lora(0, expert_lora.lora_a, expert_lora.lora_b)

    assert fused_moe.adapter_enabled.tolist() == [1, 0]
    for weights in (
        fused_moe.w13_lora_a_stacked[0],
        fused_moe.w13_lora_b_stacked[0],
        fused_moe.w2_lora_a_stacked[0],
        fused_moe.w2_lora_b_stacked[0],
    ):
        assert torch.count_nonzero(weights[0]) == weights[0].numel()


def test_existing_thinker_architecture_is_preserved():
    config = PretrainedConfig(architectures=["CustomThinkerArchitecture"])

    _ensure_thinker_architecture(config)

    assert config.architectures == ["CustomThinkerArchitecture"]
