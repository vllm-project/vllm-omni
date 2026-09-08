# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the fixed LTX refinement-phase adapter."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.ltx2 import ltx2_phase_adapter
from vllm_omni.diffusion.models.ltx2.ltx2_adapter_parser import (
    AdapterTarget,
    parse_ltx_adapter,
)
from vllm_omni.diffusion.models.ltx2.ltx2_components import LTX23_TWO_STAGE_COMPONENT_PROFILE
from vllm_omni.diffusion.models.ltx2.ltx2_phase_adapter import LTXPhaseAdapterRuntime
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX2_TWO_STAGE_RECIPE,
    LTX_DISTILLED_ADAPTER_SLOT,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _target(
    lora_a,
    lora_b,
    *,
    source_module="proj",
    module="proj",
    target_slice=None,
):
    return AdapterTarget(
        source_module=source_module,
        module=module,
        slice=target_slice,
        a_key="a",
        b_key="b",
        a_shape=tuple(lora_a.shape),
        b_shape=tuple(lora_b.shape),
    )


def _wrap(
    layer,
    lora_a,
    lora_b,
    *,
    source_module="proj",
    module="proj",
    target_slice=None,
    layer_fused=False,
):
    target = _target(
        lora_a,
        lora_b,
        source_module=source_module,
        module=module,
        target_slice=target_slice,
    )
    wrapper = ltx2_phase_adapter._PhaseAdapterLinear(
        layer,
        [target],
        adapter_dtype=lora_a.dtype,
        layer_fused=layer_fused,
    )
    wrapper.load_target(target, lora_a, lora_b)
    wrapper.set_enabled(True)
    return wrapper


class _LoRAAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_qkv = torch.nn.Identity()


class _LoRATransformer(torch.nn.Module):
    stacked_params_mapping = ((".attn1.to_qkv", ".attn1.to_q", "q"),)

    def __init__(self) -> None:
        super().__init__()
        self.proj_in = torch.nn.Linear(4, 6, bias=False)
        block = torch.nn.Module()
        block.attn1 = _LoRAAttention()
        self.transformer_blocks = torch.nn.ModuleList([block])


def test_ltx_adapter_parser_maps_official_and_packed_module_names(tmp_path):
    adapter_path = tmp_path / "distilled.safetensors"
    save_file(
        {
            "diffusion_model.text_embedding_projection.aggregate_embed.lora_A.weight": torch.randn(2, 4),
            "diffusion_model.patchify_proj.lora_A.weight": torch.randn(2, 4),
            "diffusion_model.patchify_proj.lora_B.weight": torch.randn(6, 2),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(2, 4),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(4, 2),
        },
        adapter_path,
    )

    manifest = parse_ltx_adapter(_LoRATransformer(), str(adapter_path))
    assert {(target.source_module, target.module, target.slice) for target in manifest.targets} == {
        ("proj_in", "proj_in", None),
        ("transformer_blocks.0.attn1.to_q", "transformer_blocks.0.attn1.to_qkv", "q"),
    }


def test_ltx_adapter_parser_rejects_unpaired_b_tensor(tmp_path):
    adapter_path = tmp_path / "distilled.safetensors"
    save_file(
        {"diffusion_model.patchify_proj.lora_B.weight": torch.randn(6, 2)},
        adapter_path,
    )

    with pytest.raises(ValueError, match="Missing paired LoRA A tensors"):
        parse_ltx_adapter(_LoRATransformer(), str(adapter_path))


def _pipeline(*, dtype, quantization_config):
    return SimpleNamespace(
        pipeline_recipe=LTX2_TWO_STAGE_RECIPE,
        component_profile=LTX23_TWO_STAGE_COMPONENT_PROFILE,
        transformer=object(),
        od_config=SimpleNamespace(
            model="unused",
            lora_path=None,
            dtype=dtype,
            quantization_config=quantization_config,
        ),
    )


@pytest.mark.parametrize(
    ("dtype", "quantization_config", "layer_fused"),
    [
        (torch.float32, object(), False),
        (torch.bfloat16, None, True),
    ],
)
def test_ltx_phase_adapter_selects_execution_from_quantization(
    monkeypatch,
    dtype,
    quantization_config,
    layer_fused,
):
    pipeline = _pipeline(dtype=dtype, quantization_config=quantization_config)
    manifest = object()
    captured: dict[str, Any] = {}

    def resolve_artifact(*_args, **kwargs):
        assert kwargs == {
            "model_revision": None,
            "artifact_revision": None,
        }
        return "adapter"

    def parse(transformer, path):
        assert transformer is pipeline.transformer
        assert path == "adapter"
        return manifest

    class Runtime:
        def __init__(self, transformer, parsed_manifest, **kwargs):
            captured.update(transformer=transformer, manifest=parsed_manifest, **kwargs)

    monkeypatch.setattr(ltx2_phase_adapter, "resolve_ltx_artifact", resolve_artifact)
    monkeypatch.setattr(ltx2_phase_adapter, "parse_ltx_adapter", parse)
    monkeypatch.setattr(ltx2_phase_adapter, "LTXPhaseAdapterRuntime", Runtime)

    runtime = ltx2_phase_adapter.build_ltx_phase_adapter(pipeline)

    assert isinstance(runtime, Runtime)
    assert captured == {
        "transformer": pipeline.transformer,
        "manifest": manifest,
        "dtype": dtype,
        "layer_fused": layer_fused,
    }


def test_ltx_phase_adapter_rejects_non_bfloat16_layer_fused(monkeypatch):
    pipeline = _pipeline(dtype=torch.float16, quantization_config=None)
    monkeypatch.setattr(ltx2_phase_adapter, "resolve_ltx_artifact", lambda *_args, **_kwargs: "adapter")

    with pytest.raises(ValueError, match="bfloat16"):
        ltx2_phase_adapter.build_ltx_phase_adapter(pipeline)


def test_ltx_phase_adapter_keeps_one_transformer_and_switches_a_fixed_slot(tmp_path):
    class TinyTransformer(torch.nn.Module):
        stacked_params_mapping = ()

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)

    adapter_path = tmp_path / "distilled.safetensors"
    save_file(
        {
            "diffusion_model.proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "diffusion_model.proj.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        adapter_path,
    )
    transformer = TinyTransformer()
    runtime = LTXPhaseAdapterRuntime(
        transformer,
        parse_ltx_adapter(transformer, str(adapter_path)),
        dtype=torch.float32,
    )

    assert runtime.base_parameter_name("proj.weight") == "proj.base_layer.weight"
    assert {name for name, _ in transformer.named_parameters()} == {"proj.base_layer.weight"}
    with torch.no_grad():
        transformer.proj.base_layer.weight.copy_(torch.eye(2))
    runtime.finalize()

    x = torch.tensor([[1.0, 1.0]])
    runtime.activate(None)
    torch.testing.assert_close(transformer.proj(x), torch.tensor([[1.0, 1.0]]))
    runtime.activate(LTX_DISTILLED_ADAPTER_SLOT)
    torch.testing.assert_close(transformer.proj(x), torch.tensor([[10.0, 13.0]]))


def test_ltx_layer_fused_adapter_matches_official_bf16_weight_fusion():
    layer = torch.nn.Linear(4, 3, bias=True, dtype=torch.bfloat16)
    base_weight = torch.tensor(
        [
            [0.5, -0.25, 0.125, 0.75],
            [-0.375, 0.625, 0.25, -0.5],
            [0.875, -0.125, -0.75, 0.375],
        ],
        dtype=torch.bfloat16,
    )
    lora_a = torch.tensor(
        [
            [0.1015625, -0.203125, 0.3046875, -0.40625],
            [0.5078125, -0.609375, 0.7109375, -0.8125],
        ],
        dtype=torch.bfloat16,
    )
    lora_b = torch.tensor(
        [[0.9140625, -0.1171875], [-0.21875, 0.3203125], [0.421875, -0.5234375]],
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        layer.weight.copy_(base_weight)
        layer.bias.copy_(torch.tensor([0.125, -0.25, 0.5], dtype=torch.bfloat16))
    wrapper = _wrap(layer, lora_a, lora_b, layer_fused=True)

    expected_weight = torch.matmul(lora_b, lora_a)
    expected_weight.add_(base_weight)
    x = torch.tensor([[0.6015625, -0.703125, 0.8046875, -0.90625]], dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, expected_weight, layer.bias)

    assert torch.equal(wrapper(x), expected)
    assert torch.equal(layer.weight, base_weight)


def test_ltx_layer_fused_row_parallel_uses_one_collective(monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class FakeRow(torch.nn.Module):
        input_size = 4
        output_size = 2
        input_size_per_partition = 2
        input_is_parallel = True
        tp_rank = 0
        tp_size = 2
        reduce_results = True
        skip_bias_add = False
        return_bias = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
            self.bias = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
            self.quant_method = UnquantizedLinearMethod()

    monkeypatch.setattr(ltx2_phase_adapter, "RowParallelLinear", FakeRow)
    reductions = []

    def all_reduce(value):
        reductions.append(value.clone())
        return value + 1

    monkeypatch.setattr(ltx2_phase_adapter, "tensor_model_parallel_all_reduce", all_reduce)
    layer = FakeRow()
    lora_a = torch.tensor([[0.5, -0.25, 0.75, -0.125]], dtype=torch.bfloat16)
    lora_b = torch.tensor([[0.25], [-0.5]], dtype=torch.bfloat16)
    wrapper = _wrap(layer, lora_a, lora_b, layer_fused=True)

    expected_weight = torch.matmul(lora_b, lora_a[:, :2])
    expected_weight.add_(layer.weight)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    expected_local = torch.nn.functional.linear(x, expected_weight, layer.bias)

    actual = wrapper(x)
    assert len(reductions) == 1
    assert torch.equal(reductions[0], expected_local)
    assert torch.equal(actual, expected_local + 1)


def test_ltx_layer_fused_column_parallel_uses_local_weight_shard(monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class FakeColumn(torch.nn.Module):
        input_size = 3
        output_size = 4
        output_size_per_partition = 2
        tp_rank = 1
        tp_size = 2
        gather_output = False
        skip_bias_add = False
        return_bias = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([[0.5, -0.25, 0.125], [-0.375, 0.625, 0.25]], dtype=torch.bfloat16)
            )
            self.bias = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
            self.quant_method = UnquantizedLinearMethod()

    monkeypatch.setattr(ltx2_phase_adapter, "ColumnParallelLinear", FakeColumn)
    layer = FakeColumn()
    base_weight = layer.weight.detach().clone()
    lora_a = torch.tensor([[0.5, -0.25, 0.75]], dtype=torch.bfloat16)
    lora_b = torch.tensor([[0.25], [-0.5], [0.75], [-1.0]], dtype=torch.bfloat16)
    wrapper = _wrap(layer, lora_a, lora_b, layer_fused=True)

    expected_weight = torch.matmul(lora_b[2:], lora_a)
    expected_weight.add_(base_weight)
    x = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, expected_weight, layer.bias)

    assert torch.equal(wrapper(x), expected)
    assert torch.equal(layer.weight, base_weight)


def test_ltx_layer_fused_packed_qkv_updates_only_the_target_slice(monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class FakeColumn(torch.nn.Module):
        pass

    class FakeQKV(FakeColumn):
        input_size = 4
        head_size = 2
        v_head_size = 2
        total_num_heads = 2
        total_num_kv_heads = 2
        num_heads = 1
        num_kv_heads = 1
        num_kv_head_replicas = 1
        tp_rank = 1
        tp_size = 2
        gather_output = False
        skip_bias_add = True
        return_bias = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16) / 16
            )
            self.bias = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).to(torch.bfloat16) / 8)
            self.quant_method = UnquantizedLinearMethod()

    monkeypatch.setattr(ltx2_phase_adapter, "ColumnParallelLinear", FakeColumn)
    monkeypatch.setattr(ltx2_phase_adapter, "QKVParallelLinear", FakeQKV)
    layer = FakeQKV()
    base_weight = layer.weight.detach().clone()
    lora_a = torch.tensor([[0.5, -0.25, 0.75, -0.125]], dtype=torch.bfloat16)
    lora_b = torch.tensor([[0.25], [-0.5], [0.75], [-1.0]], dtype=torch.bfloat16)
    wrapper = _wrap(
        layer,
        lora_a,
        lora_b,
        source_module="to_k",
        module="to_qkv",
        target_slice="k",
        layer_fused=True,
    )

    expected_weight = base_weight.clone()
    expected_k = torch.matmul(lora_b[2:], lora_a)
    expected_k.add_(base_weight[2:4])
    expected_weight[2:4].copy_(expected_k)
    x = torch.tensor([[1.0, -0.5, 0.25, -0.125]], dtype=torch.bfloat16)
    expected_output = torch.nn.functional.linear(x, expected_weight)

    actual_output, returned_bias = wrapper(x)
    assert torch.equal(actual_output, expected_output)
    assert returned_bias is layer.bias
    assert torch.equal(layer.weight, base_weight)


@pytest.mark.parametrize(
    ("layer_factory", "target_slice", "expected_a", "expected_b", "expected_output"),
    [
        (
            lambda: type(
                "FakeColumn",
                (torch.nn.Module,),
                {
                    "input_size": 4,
                    "output_size": 4,
                    "output_size_per_partition": 2,
                    "tp_rank": 1,
                    "gather_output": False,
                    "forward": lambda self, x: x.new_zeros((*x.shape[:-1], 2)),
                },
            )(),
            None,
            torch.tensor([[1.0, 10.0, 100.0, 1000.0]]),
            torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            torch.tensor([[12963.0, 17284.0]]),
        ),
        (
            lambda: type(
                "FakeRow",
                (torch.nn.Module,),
                {
                    "input_size": 4,
                    "output_size": 2,
                    "input_size_per_partition": 2,
                    "tp_rank": 1,
                    "reduce_results": False,
                    "forward": lambda self, x: x.new_zeros((*x.shape[:-1], 2)),
                },
            )(),
            None,
            torch.tensor([[1.0, 10.0, 100.0, 1000.0]]),
            torch.tensor([[2.0], [3.0]]),
            torch.tensor([[8600.0, 12900.0]]),
        ),
        (
            lambda: type(
                "FakeQKV",
                (torch.nn.Module,),
                {
                    "input_size": 4,
                    "head_size": 2,
                    "v_head_size": 2,
                    "total_num_heads": 2,
                    "total_num_kv_heads": 2,
                    "num_heads": 1,
                    "num_kv_heads": 1,
                    "num_kv_head_replicas": 1,
                    "tp_rank": 1,
                    "forward": lambda self, x: x.new_zeros((*x.shape[:-1], 6)),
                },
            )(),
            "k",
            torch.tensor([[1.0, 10.0, 100.0, 1000.0]]),
            torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            torch.tensor([[0.0, 0.0, 12963.0, 17284.0, 0.0, 0.0]]),
        ),
    ],
)
def test_ltx_phase_adapter_uses_local_tp_lora_slices(
    monkeypatch,
    layer_factory,
    target_slice,
    expected_a,
    expected_b,
    expected_output,
):
    layer = layer_factory()
    layer.register_buffer("device_anchor", torch.empty(0))
    layer_type = type(layer)
    if target_slice == "k":
        monkeypatch.setattr(ltx2_phase_adapter, "QKVParallelLinear", layer_type)
    elif "Column" in layer_type.__name__:
        monkeypatch.setattr(ltx2_phase_adapter, "ColumnParallelLinear", layer_type)
    else:
        monkeypatch.setattr(ltx2_phase_adapter, "RowParallelLinear", layer_type)

    wrapper = _wrap(layer, expected_a, expected_b, target_slice=target_slice)
    torch.testing.assert_close(wrapper(torch.tensor([[1.0, 2.0, 3.0, 4.0]])), expected_output)
    expected_local_a_width = getattr(layer, "input_size_per_partition", layer.input_size)
    assert wrapper.pieces[0].lora_a.shape[-1] == expected_local_a_width
