# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora import lora_model
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.qwen_image.lora import QwenImageLoRAMixin
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _Pipeline(torch.nn.Module, QwenImageLoRAMixin):
    def __init__(self, wrapped: bool):
        super().__init__()
        self.transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
        torch.nn.Module.__init__(self.transformer)
        output = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(output)
        if wrapped:
            wrapper = torch.nn.Module()
            wrapper.add_module("base_layer", output)
            self.transformer.add_module("to_out", wrapper)
        else:
            self.transformer.add_module("to_out", output)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("target", ["attn.to_out.0", "attn.to_out", "to_out.0", "to_out"])
def test_qwen_lora_output_name_and_scaling(tmp_path, monkeypatch, wrapped, target):
    monkeypatch.setattr(lora_model, "PIN_MEMORY", False)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "target_modules": [target], "peft_type": "LORA"})
    )
    projection = target if target.startswith("attn.") else f"attn.{target}"
    prefix = f"transformer.transformer_blocks.0.{projection}"
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    save_file({prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b}, tmp_path / "adapter_model.safetensors")
    manager = DiffusionLoRAManager.__new__(DiffusionLoRAManager)
    manager.pipeline = _Pipeline(wrapped)
    manager.dtype = torch.float32
    manager._expected_lora_modules = {"to_out"}
    model, helper = manager._load_adapter(LoRARequest("output", 1, str(tmp_path)))
    assert helper.target_modules == [target.removesuffix(".0")]
    assert set(model.loras) == {"transformer.transformer_blocks.0.attn.to_out"}
    weights = manager._get_lora_weights(model, "transformer.transformer_blocks.0.attn.to_out")
    assert weights is not None and weights.scaling == 1
    torch.testing.assert_close(weights.lora_a, a, rtol=0, atol=0)
    torch.testing.assert_close(weights.lora_b, b * 2, rtol=0, atol=0)


def test_qwen_lora_still_rejects_unknown_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_model, "PIN_MEMORY", False)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2, "target_modules": ["unknown"], "peft_type": "LORA"})
    )
    save_file(
        {"transformer.unknown.lora_A.weight": torch.ones(2, 3), "transformer.unknown.lora_B.weight": torch.ones(4, 2)},
        tmp_path / "adapter_model.safetensors",
    )
    with pytest.raises(ValueError, match="expected target modules"):
        _Pipeline(False)._load_diffusion_lora_adapter(
            lora_request=LoRARequest("unknown", 2, str(tmp_path)), lora_path=str(tmp_path), dtype=torch.float32
        )
