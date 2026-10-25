# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import load_transformer_weights
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def make_transformer():
    transformer = nn.Module()
    transformer.base = nn.Linear(2, 2, bias=False)
    transformer.action_modules = nn.ModuleList([nn.Linear(2, 2, bias=False)])
    return transformer


@pytest.mark.parametrize("camera_name", ["action_modules.0.weight", "double_blocks.0.action_module.weight"])
def test_complete_dit_checkpoint_with_reference_aliases(tmp_path, camera_name):
    model = make_transformer()
    checkpoint = tmp_path / "model.safetensors"
    save_file({"base.weight": model.base.weight, camera_name: model.action_modules[0].weight}, str(checkpoint))
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.data.zero_()
    assert load_transformer_weights(model, load_file(str(checkpoint)).items()) == set(expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_missing_camera_weights_are_rejected(tmp_path):
    model = make_transformer()
    checkpoint = tmp_path / "model.safetensors"
    save_file({"base.weight": model.base.weight}, str(checkpoint))
    with pytest.raises(ValueError, match="action_modules.0.weight"):
        load_transformer_weights(model, load_file(str(checkpoint)).items())


def test_standard_loader_accounts_for_all_components():
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = make_transformer()
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.vae = nn.Linear(2, 2)
    weights = [(f"transformer.{name}", tensor.clone()) for name, tensor in pipeline.transformer.state_dict().items()]
    loaded = pipeline.load_weights(weights)
    assert loaded == set(dict(pipeline.named_parameters()))
    with pytest.raises(ValueError, match="Unexpected Hy-Nomas component"):
        pipeline.load_weights(iter((("weight", torch.zeros(1)),)))
    with pytest.raises(ValueError, match="missing"):
        pipeline.load_weights(iter(()))


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "shape"])
def test_malformed_dit_weight_stream_is_rejected(kind):
    transformer = make_transformer()
    weights = list(transformer.state_dict().items())
    if kind == "duplicate":
        weights.append(("double_blocks.0.action_module.weight", weights[1][1]))
    elif kind == "unknown":
        weights.append(("extra.weight", torch.zeros(2, 2)))
    else:
        weights[0] = (weights[0][0], torch.zeros(3, 3))
    with pytest.raises((ValueError, AssertionError)):
        load_transformer_weights(transformer, weights)
