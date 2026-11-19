# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

MLP = "transformer_blocks.0.img_mlp"


def _pipeline(monkeypatch, tp_size=1, tp_rank=0):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr("vllm_omni.diffusion.lora.loader._resolve_lora_compute_device", lambda: None)
    pipe = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipe)
    block = nn.Module()
    block.img_mlp = nn.Module()
    with set_current_vllm_config(VllmConfig()):
        for name in ("gate_layer", "proj"):
            layer = linear.ColumnParallelLinear(4, 6, bias=False, params_dtype=torch.float32)
            nn.init.zeros_(layer.weight)
            setattr(block.img_mlp, name, layer)
    pipe.transformer = nn.Module()
    pipe.transformer.transformer_blocks = nn.ModuleList([block])
    return pipe


@pytest.mark.parametrize(
    ("tp_size", "tp_rank", "prefix", "alpha"),
    [
        (1, 0, "diffusion_model", None),
        (2, 0, "diffusion_model", None),
        (2, 1, "diffusion_model", 4.0),
        (1, 0, "transformer", None),
    ],
)
def test_fused_gate_up_matches_explicit_split(tmp_path, monkeypatch, tp_size, tp_rank, prefix, alpha):
    pipe = _pipeline(monkeypatch, tp_size, tp_rank)
    a = torch.arange(8, dtype=torch.float32).view(2, 4)
    b = torch.arange(24, dtype=torch.float32).view(12, 2)
    source = f"{prefix}.{MLP}.gate_up"
    fused = {f"{source}.lora_A.weight": a, f"{source}.lora_B.weight": b}
    scale = 1.0
    if alpha is not None:
        fused[f"{source}.alpha"] = torch.tensor(alpha)
        scale = alpha / a.shape[0]
    path = tmp_path / "fused.safetensors"
    save_file(fused, path)
    pipe.load_lora_weights(str(path), "fused")
    mlp = pipe.transformer.transformer_blocks[0].img_mlp

    # Split global rows into branches before selecting this rank's rows.
    split = {}
    for name, rows in zip(("gate_layer", "proj"), b.chunk(2), strict=True):
        key = f"transformer.{MLP}.{name}"
        split[f"{key}.lora_A.weight"] = a
        split[f"{key}.lora_B.weight"] = rows * scale
        expected = (rows @ a).chunk(tp_size)[tp_rank] * scale
        torch.testing.assert_close(getattr(mlp, name).weight, expected, rtol=0, atol=0)

    fused_weights = [p.detach().clone() for p in mlp.parameters()]
    pipe.unload_lora_weights("fused")
    assert all(torch.count_nonzero(p) == 0 for p in mlp.parameters())
    pipe.load_lora_weights(split, "split")
    for weight, expected in zip(mlp.parameters(), fused_weights, strict=True):
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("error", ["missing-b", "wrong-rows", "mixed-layout"])
def test_invalid_gate_up_rejected_before_fusion(monkeypatch, error):
    pipe = _pipeline(monkeypatch)
    key = f"transformer.{MLP}"
    a = torch.ones(2, 4)
    state = {f"{key}.gate_up.lora_A.weight": a}
    if error != "missing-b":
        state[f"{key}.gate_up.lora_B.weight"] = torch.ones(10 if error == "wrong-rows" else 12, 2)
    if error == "mixed-layout":
        state[f"{key}.proj.lora_A.weight"] = a
        state[f"{key}.proj.lora_B.weight"] = torch.ones(6, 2)
    with pytest.raises(ValueError, match="gate_up|Conflicting"):
        pipe.load_lora_weights(state, "invalid")
    assert not pipe.lora_loaded
    assert all(torch.count_nonzero(p) == 0 for p in pipe.transformer.parameters())


def test_unknown_gate_up_preserves_matched_projection(monkeypatch):
    pipe = _pipeline(monkeypatch)
    a = torch.ones(2, 4)
    b = torch.ones(6, 2)
    unknown = "transformer.transformer_blocks.99.img_mlp.gate_up"
    state = {
        f"transformer.{MLP}.proj.lora_A.weight": a,
        f"transformer.{MLP}.proj.lora_B.weight": b,
        f"{unknown}.lora_A.weight": a,
        f"{unknown}.lora_B.weight": torch.ones(12, 2),
    }

    pipe.load_lora_weights(state, "mixed")
    mlp = pipe.transformer.transformer_blocks[0].img_mlp
    torch.testing.assert_close(mlp.proj.weight, b @ a)
    assert torch.count_nonzero(mlp.gate_layer.weight) == 0
    pipe.unload_lora_weights("mixed")
    assert torch.count_nonzero(mlp.proj.weight) == 0
