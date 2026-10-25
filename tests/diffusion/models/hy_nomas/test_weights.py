# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline
from vllm_omni.diffusion.models.hy_nomas.weights import validate_dit_checkpoint

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
    validate_dit_checkpoint(model, checkpoint)


def test_missing_camera_weights_are_rejected(tmp_path):
    model = make_transformer()
    checkpoint = tmp_path / "model.safetensors"
    save_file({"base.weight": model.base.weight}, str(checkpoint))
    with pytest.raises(ValueError, match="action_modules.0.weight"):
        validate_dit_checkpoint(model, checkpoint)


def test_self_loaded_pipeline_does_not_claim_every_parameter_was_streamed():
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    assert pipeline.load_weights(iter(())) is None
    with pytest.raises(ValueError, match="reference component loaders"):
        pipeline.load_weights(iter((("weight", torch.zeros(1)),)))
