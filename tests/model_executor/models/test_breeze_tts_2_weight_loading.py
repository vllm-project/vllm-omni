# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage through Breeze, Qwen3 and the real packed linear loaders.

Only model construction and the unrelated depth/reference components are
reduced. No backbone loader, mapper or parameter weight loader is mocked.
"""

from collections.abc import Iterator

import pytest
import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from vllm.model_executor.models.qwen3 import Qwen3Model

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze import BreezeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PROJECTIONS = {
    "self_attn.qkv_proj.weight": ("q_proj", "k_proj", "v_proj"),
    "mlp.gate_up_proj.weight": ("gate_proj", "up_proj"),
}


class _EmptyReferenceEncoder(nn.Module):
    def load_weights(self, config: VllmConfig) -> set[str]:
        # This test exercises backbone checkpoint loading, not codec downloads.
        return set()


@pytest.fixture(scope="module")
def cpu_distributed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    rendezvous = tmp_path_factory.mktemp("breeze-distributed") / "rendezvous"
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=rendezvous.as_uri(),
                backend="gloo",
            )
            initialize_model_parallel(backend="gloo")
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.fixture
def talker(cpu_distributed: None) -> Iterator[BreezeForConditionalGeneration]:
    config = VllmConfig()
    with set_current_vllm_config(config), torch.device("cpu"):
        model = BreezeForConditionalGeneration.__new__(BreezeForConditionalGeneration)
        nn.Module.__init__(model)
        model.vllm_config = config
        model.model = Qwen3Model.__new__(Qwen3Model)
        nn.Module.__init__(model.model)
        layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            # Unequal Q/K/V sizes exercise GQA shard offsets as well as names.
            layer.self_attn.qkv_proj = QKVParallelLinear(
                8, 2, 4, 2, bias=False, params_dtype=torch.float32, disable_tp=True
            )
            layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.gate_up_proj = MergedColumnParallelLinear(
                8, [12, 12], bias=False, params_dtype=torch.float32, disable_tp=True
            )
            layer.mlp.down_proj = nn.Linear(12, 8, bias=False)
            layers.append(layer)
        model.model.layers = layers
        model.model.norm = nn.LayerNorm(8, bias=False)
        model.text_encoder = nn.Linear(8, 8, bias=False)
        model.text_encoder_proj = nn.Linear(8, 8, bias=False)
        model.lm_head = nn.Linear(8, 8, bias=False)
        model.depth_decoder = nn.Module()
        model.reference_encoder = _EmptyReferenceEncoder()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))
        yield model


def _checkpoint(
    model: BreezeForConditionalGeneration, layout: str = "split"
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    weights: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        value = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape) + index * 1000
        expected[name] = value
        source = name.replace("model.", "backbone_model.", 1) if name.startswith("model.") else name
        for packed, parts in _PROJECTIONS.items():
            if source.endswith(packed) and (layout == "split" or (layout == "mixed" and ".1." in source)):
                prefix = source.rsplit(".", 2)[0] + "."
                sizes = (8, 4, 4) if len(parts) == 3 else (12, 12)
                for part, shard in zip(parts, value.split(sizes), strict=True):
                    weights[prefix + part + ".weight"] = shard.clone()
                break
        else:
            weights[source] = value.clone()
    return weights, expected


@pytest.mark.parametrize("layout", ["split", "fused", "mixed"])
def test_complete_backbone_weights_preserve_values(talker: BreezeForConditionalGeneration, layout: str) -> None:
    weights, expected = _checkpoint(talker, layout)
    loaded = talker.load_weights(iter(weights.items()))
    assert loaded == set(expected)
    for name, parameter in talker.named_parameters():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("projection", ["q", "k", "v", "gate", "up"])
def test_missing_split_projection_is_rejected(
    talker: BreezeForConditionalGeneration, layer: int, projection: str
) -> None:
    weights, _ = _checkpoint(talker)
    module = "self_attn" if projection in ("q", "k", "v") else "mlp"
    source = f"backbone_model.layers.{layer}.{module}.{projection}_proj.weight"
    del weights[source]
    with pytest.raises(ValueError) as error:
        talker.load_weights(iter(weights.items()))
    assert source.removeprefix("backbone_model.") in str(error.value)


@pytest.mark.parametrize("packed", list(_PROJECTIONS))
def test_missing_projection_group_is_rejected(talker: BreezeForConditionalGeneration, packed: str) -> None:
    weights, _ = _checkpoint(talker)
    module = packed.split(".")[0]
    prefix = f"backbone_model.layers.1.{module}."
    for part in _PROJECTIONS[packed]:
        del weights[prefix + part + ".weight"]
    with pytest.raises(ValueError):
        talker.load_weights(iter(weights.items()))


def test_missing_non_projection_parameter_is_rejected(talker: BreezeForConditionalGeneration) -> None:
    weights, _ = _checkpoint(talker)
    del weights["backbone_model.norm.weight"]
    with pytest.raises(ValueError, match="Uninitialized Breeze parameters"):
        talker.load_weights(iter(weights.items()))


@pytest.mark.parametrize("layout", ["split", "fused"])
def test_projection_shape_check_is_preserved(talker: BreezeForConditionalGeneration, layout: str) -> None:
    weights, _ = _checkpoint(talker, layout)
    part = "q" if layout == "split" else "qkv"
    source = f"backbone_model.layers.1.self_attn.{part}_proj.weight"
    weights[source] = weights[source][:, :-1]
    with pytest.raises((AssertionError, RuntimeError)):
        talker.load_weights(iter(weights.items()))


def test_unexpected_backbone_parameter_is_rejected(talker: BreezeForConditionalGeneration) -> None:
    weights, _ = _checkpoint(talker)
    weights["backbone_model.layers.1.self_attn.unexpected.weight"] = torch.zeros(8, 8)
    with pytest.raises(ValueError, match="unexpected"):
        talker.load_weights(iter(weights.items()))
