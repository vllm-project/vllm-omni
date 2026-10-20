# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.lora_weights import PackedLoRALayerWeights
from vllm.model_executor.layers.linear import LinearBase, MergedColumnParallelLinear, QKVParallelLinear

from vllm_omni.diffusion.models.minimax_h3.lora import load_minimax_h3_native_adapter
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def transformer():
    model = torch.nn.Module()
    for scope in ("blocks", "token_refiner.blocks"):
        parent = model
        for part in scope.split("."):
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        block = torch.nn.Module()
        parent.add_module("0", block)
        block.attn = torch.nn.Module()
        block.mlp = torch.nn.Module()
        for parent, name, cls, input_size, sizes in (
            (block.attn, "qkv_proj", QKVParallelLinear, 4, (4, 2, 2)),
            (block.attn, "out_proj", LinearBase, 4, (4,)),
            (block.mlp, "fc1", MergedColumnParallelLinear, 4, (6, 6)),
            (block.mlp, "fc2", LinearBase, 6, (4,)),
        ):
            layer = object.__new__(cls)
            torch.nn.Module.__init__(layer)
            layer.input_size, layer.output_size = input_size, sum(sizes)
            layer.output_sizes = list(sizes)
            parent.add_module(name, layer)
    return model


def load(path, transformer, **kwargs):
    return load_minimax_h3_native_adapter(
        transformer=transformer,
        lora_request=LoRARequest("native_adapter", 42, str(path)),
        lora_path=path,
        dtype=torch.float32,
        **kwargs,
    )


@pytest.mark.parametrize("rank", [4, 16, 32])
@pytest.mark.parametrize("alpha", [None, 0.0, 2.0])
@pytest.mark.parametrize("spelling", ["lora_A", "lora_down"])
def test_projection_deltas_match_comfy_scaling(tmp_path, transformer, rank, alpha, spelling):
    rng = torch.Generator().manual_seed(42)
    tensors = {}
    for name, module in transformer.named_modules():
        if not isinstance(module, LinearBase):
            continue
        prefix = f"diffusion_model.{name}"
        tensors[f"{prefix}.{spelling}.weight"] = torch.randn(rank, module.input_size, generator=rng)
        b_name = "lora_B" if spelling == "lora_A" else "lora_up"
        tensors[f"{prefix}.{b_name}.weight"] = torch.randn(module.output_size, rank, generator=rng)
        if alpha is not None:
            tensors[f"{prefix}.alpha"] = torch.tensor(alpha)
    path = tmp_path / "any-name.safetensors"
    save_file(tensors, path)
    model, helper = load(path, transformer)
    assert model.rank == helper.r == rank
    for name, weights in model.loras.items():
        input_size = transformer.get_submodule(name).input_size
        x = torch.randn(2, input_size, generator=rng)
        prefix = f"diffusion_model.{name}"
        expected = x @ tensors[f"{prefix}.{spelling}.weight"].T @ tensors[f"{prefix}.{b_name}.weight"].T
        expected *= 1.0 if alpha is None else alpha / rank
        if isinstance(weights, PackedLoRALayerWeights):
            actual = torch.cat(
                [x @ a.T @ b.T * scale for a, b, scale in zip(weights.lora_a, weights.lora_b, weights.scaling)],
                dim=-1,
            )
        else:
            actual = x @ weights.lora_a.T @ weights.lora_b.T * weights.scaling
        torch.testing.assert_close(actual, expected)


def test_partial_coverage_and_per_layer_rank_alpha(tmp_path, transformer):
    path = tmp_path / "partial.safetensors"
    save_file(
        {
            "blocks.0.attn.out_proj.lora_A.weight": torch.ones(2, 4),
            "blocks.0.attn.out_proj.lora_B.weight": torch.ones(4, 2),
            "blocks.0.attn.out_proj.alpha": torch.tensor(8.0),
            "blocks.0.mlp.fc2.lora_A.weight": torch.ones(3, 6),
            "blocks.0.mlp.fc2.lora_B.weight": torch.ones(4, 3),
        },
        path,
    )
    model, helper = load(path, transformer)
    assert len(model.loras) == 2 and helper.r == 3
    assert model.loras["blocks.0.attn.out_proj"].scaling == 4.0
    assert model.loras["blocks.0.mlp.fc2"].scaling == 1.0


@pytest.mark.parametrize("fault", ["missing", "unknown", "shape", "duplicate", "alpha", "nonfinite"])
def test_invalid_adapter_rejected(tmp_path, transformer, fault):
    prefix = "diffusion_model.blocks.0.attn.out_proj"
    tensors = {f"{prefix}.lora_A.weight": torch.ones(2, 4), f"{prefix}.lora_B.weight": torch.ones(4, 2)}
    if fault == "missing":
        del tensors[f"{prefix}.lora_A.weight"]
    elif fault == "unknown":
        tensors["diffusion_model.unknown.lora_A.weight"] = torch.ones(2, 4)
    elif fault == "shape":
        tensors[f"{prefix}.lora_B.weight"] = torch.ones(5, 2)
    elif fault == "duplicate":
        tensors[f"{prefix}.lora_down.weight"] = torch.ones(2, 4)
    elif fault == "alpha":
        tensors[f"{prefix}.alpha"] = torch.ones(2)
    else:
        tensors[f"{prefix}.alpha"] = torch.tensor(float("nan"))
    path = tmp_path / "invalid.safetensors"
    save_file(tensors, path)
    with pytest.raises(ValueError, match="H3 LoRA"):
        load(path, transformer)


def test_wrapped_layer_and_directory_selection(tmp_path, transformer):
    layer = transformer.blocks._modules["0"].attn.out_proj
    wrapper = object.__new__(BaseLayerWithLoRA)
    torch.nn.Module.__init__(wrapper)
    wrapper.base_layer = layer
    transformer.blocks._modules["0"].attn.out_proj = wrapper
    tensors = {
        "blocks.0.attn.out_proj.lora_A.weight": torch.ones(2, 4),
        "blocks.0.attn.out_proj.lora_B.weight": torch.ones(4, 2),
    }
    save_file(tensors, tmp_path / "first.safetensors")
    assert load(tmp_path, transformer) is not None
    save_file(tensors, tmp_path / "second.safetensors")
    with pytest.raises(ValueError, match="select one"):
        load(tmp_path, transformer)
    (tmp_path / "adapter_config.json").write_text("{}")
    assert load(tmp_path, transformer) is None


def test_unrelated_adapter_is_not_claimed(tmp_path, transformer):
    path = tmp_path / "other.safetensors"
    save_file({"x": torch.ones(1)}, path)
    assert load(path, transformer) is None


def test_unsupported_offload_rejected(tmp_path, transformer):
    path = tmp_path / "native_adapter.safetensors"
    save_file(
        {
            "blocks.0.attn.out_proj.lora_A.weight": torch.ones(2, 4),
            "blocks.0.attn.out_proj.lora_B.weight": torch.ones(4, 2),
        },
        path,
    )
    with pytest.raises(ValueError, match="model-level CPU offload"):
        load(path, transformer, unsupported_offload_mode="model-level CPU offload")


def test_separate_qkv_names_rejected(tmp_path, transformer):
    path = tmp_path / "separate.safetensors"
    save_file(
        {
            "blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
            "blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        },
        path,
    )
    with pytest.raises(ValueError, match="target does not exist"):
        load(path, transformer)


def test_pipeline_loads_native_adapter_without_distilled_schedule(tmp_path, transformer):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    path = tmp_path / "realism.safetensors"
    save_file(
        {
            "blocks.0.attn.out_proj.lora_A.weight": torch.ones(32, 4),
            "blocks.0.attn.out_proj.lora_B.weight": torch.ones(4, 32),
        },
        path,
    )
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline.transformer = transformer
    pipeline._turbo_lora_specs = {}
    pipeline._native_distilled_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    model, _ = pipeline._load_diffusion_lora_adapter(
        lora_request=LoRARequest("native_adapter", 42, str(path)),
        lora_path=path,
        dtype=torch.float32,
    )
    assert model.rank == 32
    assert pipeline._turbo_lora_specs == {}
    assert pipeline._native_distilled_lora_adapter_ids == set()
    assert pipeline._lora_sigma_schedules == {}


def test_loaded_packed_adapters_compose_without_double_scaling(tmp_path, transformer):
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

    generator = torch.Generator().manual_seed(42)
    models = []
    expected = torch.zeros(8, 4)
    for index, (rank, alpha, strength) in enumerate(((2, 4.0, 0.75), (3, 1.5, -0.5))):
        a = torch.randn(rank, 4, generator=generator)
        b = torch.randn(8, rank, generator=generator)
        path = tmp_path / f"adapter-{index}.safetensors"
        save_file(
            {
                "blocks.0.attn.qkv_proj.lora_A.weight": a,
                "blocks.0.attn.qkv_proj.lora_B.weight": b,
                "blocks.0.attn.qkv_proj.alpha": torch.tensor(alpha),
            },
            path,
        )
        model, _ = load(path, transformer)
        for weights in model.loras.values():
            weights.optimize()
        models.append((model, strength))
        expected += strength * alpha / rank * (b @ a)
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=3,
    )
    composed, _ = manager._compose_lora_models(100, models)
    weights = composed.loras["blocks.0.attn.qkv_proj"]
    actual = torch.cat([b @ a for a, b in zip(weights.lora_a, weights.lora_b)], dim=0)
    torch.testing.assert_close(actual, expected)
