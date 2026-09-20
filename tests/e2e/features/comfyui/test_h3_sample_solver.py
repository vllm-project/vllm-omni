# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
from comfyui_vllm_omni.nodes import VLLMOmniMiniMaxH3Params
from comfyui_vllm_omni.utils.models import _minimaxh3_params_builder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _build_fields(sample_solver=None):
    model_params = {
        "type": "minimax_h3",
        "audio_flow_shift": 3.0,
        "flow_shift": 12.0,
    }
    if sample_solver is not None:
        model_params["sample_solver"] = sample_solver
    return _minimaxh3_params_builder(
        model_params,
        extra_params={"task": "t2va"},
        width=1344,
        height=768,
    )


def test_h3_params_node_exposes_auto_as_default_sample_solver():
    options, config = VLLMOmniMiniMaxH3Params.INPUT_TYPES()["required"]["sample_solver"]

    assert options == ["auto", "euler", "res_multistep"]
    assert config["default"] == "auto"


@pytest.mark.parametrize("sample_solver", [None, "auto"], ids=["legacy", "auto"])
def test_h3_auto_and_legacy_params_preserve_server_default(sample_solver):
    fields = _build_fields(sample_solver)
    extra_params = json.loads(fields["extra_params"])

    assert fields["flow_shift"] == 12.0
    assert extra_params == {
        "task": "t2va",
        "audio_flow_shift": 3.0,
        "aspect_ratio": "16:9",
    }


@pytest.mark.parametrize("sample_solver", ["euler", "res_multistep"])
def test_h3_explicit_sample_solver_is_forwarded_in_extra_params(sample_solver):
    fields = _build_fields(sample_solver)

    assert json.loads(fields["extra_params"])["sample_solver"] == sample_solver


def test_h3_rejects_unknown_sample_solver():
    with pytest.raises(ValueError, match="Unsupported MiniMax-H3 sample_solver"):
        _build_fields("not-a-solver")
