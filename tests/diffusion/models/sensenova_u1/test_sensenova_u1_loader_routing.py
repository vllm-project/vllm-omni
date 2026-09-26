# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for SenseNova-U1's checkpoint loader."""

import os

import pytest
import torch
import torch.nn as nn
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import ConvDecoder, SenseNovaU1Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
PREFIX = "language_model.model.layers.0"


@pytest.fixture(autouse=True)
def tp_group():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


class TinyPipeline(SenseNovaU1Pipeline):
    def __init__(self, pixel_head: bool = False):
        nn.Module.__init__(self)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = QKVParallelLinear(8, 2, 4, 2, bias=False)
        layer.self_attn.qkv_proj_mot_gen = QKVParallelLinear(8, 2, 4, 2, bias=False)
        layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = MergedColumnParallelLinear(8, [8, 8], bias=False)
        layer.mlp_mot_gen = nn.Module()
        layer.mlp_mot_gen.gate_up_proj = MergedColumnParallelLinear(8, [8, 8], bias=False)
        self.language_model.model.layers = nn.ModuleList([layer])
        self.vision_model = nn.Linear(8, 8, bias=False)
        head = ConvDecoder(8, 8) if pixel_head else nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
        self.fm_modules = nn.ModuleDict({"fm_head": head})


def test_load_weights_places_distinct_qkv_shards_in_both_towers():
    pipe = TinyPipeline()
    weights = [
        (f"{PREFIX}.self_attn.{projection}_proj{tower}.weight", torch.full((rows, 8), value))
        for tower, offset in (("", 0), ("_mot_gen", 3))
        for projection, rows, value in (("q", 8, 1 + offset), ("k", 4, 2 + offset), ("v", 4, 3 + offset))
    ]
    loaded = pipe.load_weights(weights)

    assert loaded == {f"{PREFIX}.self_attn.qkv_proj{tower}.weight" for tower in ("", "_mot_gen")}
    layer = pipe.language_model.model.layers[0]
    for tower, offset in ((layer.self_attn.qkv_proj, 0), (layer.self_attn.qkv_proj_mot_gen, 3)):
        for shard, value in zip(tower.weight.split((8, 4, 4)), (1 + offset, 2 + offset, 3 + offset), strict=True):
            torch.testing.assert_close(shard, torch.full_like(shard, value))


def test_load_weights_places_gate_up_and_direct_parameters():
    pipe = TinyPipeline()
    weights = [
        (f"{PREFIX}.{mlp}.{projection}_proj.weight", torch.full((8, 8), value))
        for mlp, offset in (("mlp", 0), ("mlp_mot_gen", 2))
        for projection, value in (("gate", 1 + offset), ("up", 2 + offset))
    ]
    weights += [
        (f"{PREFIX}.self_attn.o_proj.weight", torch.full((8, 8), 7)),
        ("vision_model.weight", torch.full((8, 8), 8)),
    ]
    loaded = pipe.load_weights(weights)

    assert loaded == {
        f"{PREFIX}.mlp.gate_up_proj.weight",
        f"{PREFIX}.mlp_mot_gen.gate_up_proj.weight",
        f"{PREFIX}.self_attn.o_proj.weight",
        "vision_model.weight",
    }
    layer = pipe.language_model.model.layers[0]
    for mlp, values in ((layer.mlp, (1, 2)), (layer.mlp_mot_gen, (3, 4))):
        for shard, value in zip(mlp.gate_up_proj.weight.split((8, 8)), values, strict=True):
            torch.testing.assert_close(shard, torch.full_like(shard, value))
    torch.testing.assert_close(layer.self_attn.o_proj.weight, torch.full_like(layer.self_attn.o_proj.weight, 7))
    torch.testing.assert_close(pipe.vision_model.weight, torch.full_like(pipe.vision_model.weight, 8))


@pytest.mark.parametrize("pixel_head,parameter", [(False, "0.weight"), (True, "conv1.weight")])
def test_fm_head_checkpoint_keys_match_the_selected_branch(pixel_head, parameter):
    pipe = TinyPipeline(pixel_head)
    name = f"fm_modules.fm_head.{parameter}"
    tensor = torch.full_like(dict(pipe.named_parameters())[name], 9)
    loaded = pipe.load_weights([(name, tensor)])

    assert loaded == {name}
    torch.testing.assert_close(dict(pipe.named_parameters())[name], tensor)
    other_branch = "conv1.weight" if not pixel_head else "0.weight"
    assert f"fm_modules.fm_head.{other_branch}" not in dict(pipe.named_parameters())


@pytest.mark.parametrize("pixel_head,other_branch", [(False, "conv1.weight"), (True, "0.weight")])
def test_mismatched_fm_head_checkpoint_fails_explicitly(pixel_head, other_branch):
    pipe = TinyPipeline(pixel_head)
    with pytest.raises(ValueError, match="FM head checkpoint parameter"):
        pipe.load_weights([(f"fm_modules.fm_head.{other_branch}", torch.ones(1))])
