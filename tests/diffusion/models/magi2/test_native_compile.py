# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.magi2.test_native_packing import (
    _longer_text_tensors,
    _sampler_tensors,
    _tiny_model,
    _tiny_sampler,
)
from vllm_omni.diffusion.models.magi2.preview_data_proxy import Magi2PackedLayout
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_layer_regions_bracket_the_eager_kernels() -> None:
    moe_layer, dense_layer = _tiny_model().block.layers

    assert moe_layer.region_methods == ("_attention_input", "_moe_input", "_moe_output")
    assert dense_layer.region_methods == ("_attention_input", "_dense_output")


def test_compile_regions_fullgraph_matches_eager_bitwise() -> None:
    model = _tiny_model()
    sampler = _tiny_sampler(model)
    layout = Magi2PackedLayout()
    steps = [
        sampler.prepare_model_input(**_sampler_tensors(seed), t=t, cfg_config=CFGConfig(), layout=layout)
        for seed, t in enumerate((torch.tensor([900.0]), torch.tensor([450.0])))
    ]
    expected = [sampler.forward(step) for step in steps]

    torch._dynamo.reset()
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    model.compile_regions(fullgraph=True, backend="eager", dynamic=True)
    first = sampler.forward(steps[0])
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs_before
    with torch._dynamo.config.patch(error_on_recompile=True):
        second = sampler.forward(steps[1])

    for (video, audio), (video_ref, audio_ref) in zip((first, second), expected, strict=True):
        assert torch.equal(video, video_ref)
        assert torch.equal(audio, audio_ref)
    # pre adapter, post adapter, three MoE-layer regions, two dense-layer regions
    assert graphs == 7


def test_pipeline_setup_compile_forwards_config_dynamic() -> None:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.magi2.pipeline_magi2 import Magi2Pipeline

    calls: list[dict[str, object]] = []

    class _Transformer(torch.nn.Module):
        def compile_regions(self, **compile_kwargs: object) -> None:
            calls.append(compile_kwargs)

    pipeline = object.__new__(Magi2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(
        model="sand-ai/MAGI-2-preview",
        model_class_name="Magi2Pipeline",
        diffusion_compile_dynamic=False,
    )
    pipeline.transformer = _Transformer()

    pipeline.setup_compile()

    assert calls == [{"fullgraph": True, "dynamic": False, "options": {"emulate_precision_casts": True}}]


def test_compiled_graphs_are_shared_across_layers_of_one_kind() -> None:
    model = _tiny_model(moe_layers=2)
    sampler = _tiny_sampler(model)
    step = sampler.prepare_model_input(**_sampler_tensors(0), t=torch.tensor([900.0]), cfg_config=CFGConfig())
    expected = sampler.forward(step)

    torch._dynamo.reset()
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    model.compile_regions(fullgraph=True, backend="eager", dynamic=True)
    actual = sampler.forward(step)

    assert torch.equal(actual[0], expected[0])
    # Two MoE layers and two dense layers still compile one graph per region
    # kind: pre adapter, post adapter, three MoE regions, two dense regions.
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs_before == 7


def test_compiled_regions_survive_new_layouts_without_recompiling() -> None:
    model = _tiny_model()
    sampler = _tiny_sampler(model)
    t = torch.tensor([900.0])
    first_request = sampler.prepare_model_input(**_sampler_tensors(0), t=t, cfg_config=CFGConfig())
    second_request = sampler.prepare_model_input(**_sampler_tensors(1), t=t, cfg_config=CFGConfig())
    longer_request = sampler.prepare_model_input(**_longer_text_tensors(2), t=t, cfg_config=CFGConfig())
    expected = [sampler.forward(request) for request in (first_request, second_request, longer_request)]
    for request in (first_request, second_request, longer_request):
        request.layout = Magi2PackedLayout()

    torch._dynamo.reset()
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    model.compile_regions(fullgraph=True, backend="eager", dynamic=True)
    actual = [sampler.forward(first_request)]
    with torch._dynamo.config.patch(error_on_recompile=True):
        actual.append(sampler.forward(second_request))
        actual.append(sampler.forward(longer_request))

    for (video, audio), (video_ref, audio_ref) in zip(actual, expected, strict=True):
        assert torch.equal(video, video_ref)
        assert torch.equal(audio, audio_ref)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs_before == 7
