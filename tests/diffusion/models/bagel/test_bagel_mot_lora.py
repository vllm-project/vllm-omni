# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from safetensors.torch import save_file
from vllm.config import VllmConfig, set_current_vllm_config

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers.mot.mot_qkv_parallel_linear import MoTQKVParallelLinear
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}))],
)
@pytest.mark.parametrize("text_count", [0, 4, 8])
@pytest.mark.parametrize("return_bias", [False, True])
def test_mot_qkv_lora_preserves_expert_routing(tmp_path, monkeypatch, device, text_count, return_bias):
    import vllm.model_executor.parameter as parameter

    import vllm_omni.diffusion.layers.mot.mot_qkv_parallel_linear as mot

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    if device == "cpu":
        monkeypatch.setattr(mot.current_platform, "is_cuda", lambda: False)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    with set_current_vllm_config(VllmConfig()), torch.device(device), torch.inference_mode():
        layer = MoTQKVParallelLinear(
            256,
            128,
            2,
            1,
            bias=True,
            vae_bias=True,
            params_dtype=dtype,
            disable_tp=True,
            return_bias=return_bias,
        )
        layer.weight.fill_(0.01)
        layer.gen_exp.weight.fill_(0.02)
        layer.bias.fill_(0.03)
        layer.gen_exp.bias.fill_(0.04)
        pipeline = torch.nn.Module()
        pipeline._lora_components = ["bagel"]
        pipeline.bagel = torch.nn.Module()
        pipeline.bagel.qkv_proj = layer
        # BAGEL exposes the same backbone under more than one component.
        pipeline.transformer = pipeline.bagel
        manager = DiffusionLoRAManager(pipeline=pipeline, device=torch.device(device), dtype=dtype)
        a = torch.full((2, 256), 0.01, dtype=dtype)
        b = torch.linspace(0.01, 0.03, 512 * 2, dtype=torch.float32).reshape(512, 2).to(dtype)
        save_file(
            {
                "base_model.model.bagel.qkv_proj.lora_A.weight": a.cpu(),
                "base_model.model.bagel.qkv_proj.lora_B.weight": b.cpu(),
            },
            str(tmp_path / "adapter_model.safetensors"),
        )
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 2,
                    "lora_alpha": 2,
                    "target_modules": ["bagel.qkv_proj"],
                }
            )
        )
        request = LoRARequest(lora_name="mot", lora_int_id=1, lora_path=str(tmp_path))
        x = torch.arange(8 * 256, dtype=torch.float32).reshape(8, 256).to(dtype) / 2048
        token_order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        text_indices = token_order[:text_count]
        vae_indices = token_order[text_count:]
        baseline = layer(x, text_indices, vae_indices)
        baseline = baseline[0] if return_bias else baseline
        for scale in (1.0, 2.0):
            manager.set_active_adapter(request, lora_scale=scale)
            result = pipeline.bagel.qkv_proj(x, text_indices, vae_indices)
            output = result[0] if return_bias else result
            expected = baseline.clone()
            expected[text_indices] += (x[text_indices] @ a.T) @ (b * scale).T
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(output[vae_indices], baseline[vae_indices], rtol=0, atol=0)
            # The no-routing understanding path must still use the text expert.
            plain = pipeline.bagel.qkv_proj(x)
            plain = plain[0] if return_bias else plain
            reference = torch.nn.functional.linear(x, layer.weight, layer.bias) + (x @ a.T) @ (b * scale).T
            torch.testing.assert_close(plain, reference)
        manager.set_active_adapter(None)
        restored = pipeline.bagel.qkv_proj(x, text_indices, vae_indices)
        restored = restored[0] if return_bias else restored
        torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
