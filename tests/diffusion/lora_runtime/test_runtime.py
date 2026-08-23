# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.lora_runtime import (
    DiffusionLoRABindingPlan,
    DiffusionLoRADeployment,
    DiffusionLoRARuntime,
    DiffusionLoRASelection,
    DiffusionLoRASupport,
    LoadedDiffusionLoRA,
    LowRankUpdate,
    create_low_rank_executor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _StaticLoader:
    def __init__(self, updates: dict[str, LowRankUpdate]) -> None:
        self.updates = updates

    def load(self, deployment: DiffusionLoRADeployment, artifact_path: Path) -> LoadedDiffusionLoRA:
        assert artifact_path.is_file()
        return LoadedDiffusionLoRA(deployment.name, (self.updates[deployment.name],))


class _TinyPipeline(nn.Module):
    def __init__(self, updates: dict[str, LowRankUpdate]) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.proj = nn.Linear(2, 3, bias=False)
        self.transformer.proj.weight.data.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2))
        self.diffusion_lora_support = DiffusionLoRASupport(
            loader_factory=lambda pipeline: _StaticLoader(updates),
            binding_plan=DiffusionLoRABindingPlan(
                component_names=("transformer",),
                target_modules=("proj",),
            ),
            executor_factory=create_low_rank_executor,
            supports_composition=True,
        )


def test_default_executor_composes_two_adapters_without_duplicate_weight_bank(tmp_path):
    artifact = tmp_path / "adapter.safetensors"
    artifact.touch()
    updates = {
        "a": LowRankUpdate(
            "transformer",
            "proj",
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0], [4.0], [5.0]]),
        ),
        "b": LowRankUpdate(
            "transformer",
            "proj",
            torch.tensor([[2.0, -1.0]]),
            torch.tensor([[1.0], [2.0], [3.0]]),
            intrinsic_scale=0.5,
        ),
    }
    pipeline = _TinyPipeline(updates)
    base_weight = pipeline.transformer.proj.weight.detach().clone()
    runtime = DiffusionLoRARuntime(
        pipeline,
        [
            DiffusionLoRADeployment("a", str(artifact)),
            DiffusionLoRADeployment("b", str(artifact)),
        ],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    runtime.activate([DiffusionLoRASelection("a", 0.5), DiffusionLoRASelection("b", 2.0)])

    x = torch.tensor([[1.0, 2.0]])
    expected = x @ base_weight.T
    expected += 0.5 * ((x @ updates["a"].lora_a.T) @ updates["a"].lora_b.T)
    expected += (2.0 * 0.5) * ((x @ updates["b"].lora_a.T) @ updates["b"].lora_b.T)
    torch.testing.assert_close(pipeline.transformer.proj(x), expected)

    layer = pipeline.transformer.proj
    assert len(layer.banks) == 1
    assert layer.banks[0].lora_a.shape[0] == 2
    assert layer.banks[0].lora_b.shape[1] == 2

    with pytest.raises(ValueError, match="Unknown diffusion LoRA"):
        runtime.activate([DiffusionLoRASelection("not-deployed")])


def test_model_must_explicitly_declare_support(tmp_path):
    artifact = tmp_path / "adapter.safetensors"
    artifact.touch()
    with pytest.raises(ValueError, match="does not declare"):
        DiffusionLoRARuntime(
            nn.Module(),
            [DiffusionLoRADeployment("a", str(artifact))],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
