# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-scoped umT5 residency through the model-level offload backend."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.kyutai_aura import pipeline_kyutai_aura as module
from vllm_omni.diffusion.models.kyutai_aura.pipeline_kyutai_aura import KyutaiAuraPipeline
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.config import OffloadStrategy
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def offload_case(mocker):
    events = []
    state = {"loaded": False, "fail": False}
    stager = mocker.Mock()

    def load():
        state["loaded"] = True
        events.append("load")

    def offload():
        state["loaded"] = False
        events.append("offload")

    stager.load.side_effect = load
    stager.offload.side_effect = offload
    factory = mocker.patch.object(module, "PinnedModuleStager", return_value=stager)

    class Encoder(torch.nn.Module):
        def forward(self, ids, mask):
            assert state["loaded"]
            events.append("encode")
            if state["fail"]:
                raise RuntimeError("text encoding failed")
            return SimpleNamespace(last_hidden_state=ids.float().unsqueeze(-1).repeat(1, 1, 2))

    pipeline = KyutaiAuraPipeline.__new__(KyutaiAuraPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.cfg = SimpleNamespace(text_len=4)
    pipeline.text_encoder = Encoder()
    pipeline._text_encoder_stager = None

    def tokenize(prompts, **kwargs):
        ids = [1, 2, 0, 0] if prompts[0] == "first prompt" else [3, 4, 5, 0]
        input_ids = torch.tensor([ids])
        return SimpleNamespace(input_ids=input_ids, attention_mask=input_ids.ne(0).long())

    pipeline.tokenizer = tokenize
    config = OffloadConfig(
        strategy=OffloadStrategy.MODEL_LEVEL,
        components=frozenset({"text_encoder"}),
    )
    backend = ModelLevelOffloadBackend(config, pipeline.device)
    return SimpleNamespace(
        pipeline=pipeline, backend=backend, factory=factory, stager=stager, events=events, state=state
    )


def test_each_prompt_reloads_and_releases_encoder_but_keeps_context(offload_case):
    case = offload_case
    case.backend.enable(case.pipeline)
    case.backend.enable(case.pipeline)
    case.factory.assert_called_once_with(
        case.pipeline.text_encoder,
        case.pipeline.device,
        pin_memory=True,
        retain_device_storage=False,
    )

    first = case.pipeline.encode_prompt("first prompt")
    assert not case.state["loaded"]
    second = case.pipeline.encode_prompt("different prompt")
    assert not case.state["loaded"]
    assert case.events == ["load", "encode", "offload"] * 2
    # Returned embeddings remain usable after offload and a different request.
    assert torch.equal(first, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    assert torch.equal(second, torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]))


def test_text_encoding_failure_releases_encoder(offload_case):
    case = offload_case
    case.backend.enable(case.pipeline)
    case.state["fail"] = True
    with pytest.raises(RuntimeError, match="text encoding failed"):
        case.pipeline.encode_prompt("first prompt")
    assert case.events == ["load", "encode", "offload"]
    assert not case.state["loaded"]


def test_disable_restores_resident_execution(offload_case):
    case = offload_case
    case.backend.enable(case.pipeline)
    case.backend.disable()
    assert case.pipeline._text_encoder_stager is None
    assert case.state["loaded"]
    case.events.clear()
    case.pipeline.encode_prompt("first prompt")
    assert case.events == ["encode"]
    case.backend.disable()


@pytest.mark.parametrize("components", [None, frozenset({"dit"}), frozenset({"dit", "text_encoder"})])
def test_unsupported_component_selection_is_rejected_before_staging(offload_case, components):
    case = offload_case
    case.backend.config.components = components
    with pytest.raises(ValueError, match='components=\\["text_encoder"\\] only'):
        case.backend.enable(case.pipeline)
    case.factory.assert_not_called()
    assert case.pipeline._text_encoder_stager is None


def test_hsdp_is_rejected_before_staging(offload_case):
    case = offload_case
    case.backend.config.use_hsdp = True
    with pytest.raises(ValueError, match="does not support HSDP"):
        case.backend.enable(case.pipeline)
    case.factory.assert_not_called()


@pytest.mark.parametrize("offload", [False, True])
def test_text_encoder_loads_on_host_when_selected(mocker, offload):
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.wan2_2 import pipeline_wan2_2_s2v

    encoder = mocker.Mock()
    encoder.to.return_value = encoder
    encoder.eval.return_value = encoder
    encoder.requires_grad_.return_value = encoder
    mocker.patch.object(AutoTokenizer, "from_pretrained", return_value=object())
    mocker.patch.object(pipeline_wan2_2_s2v, "_load_wan_t5_as_umt5", return_value=encoder)
    mocker.patch.object(module, "resolve_offload", return_value=SimpleNamespace(offloads=lambda component: offload))
    # Construction is meta-only, but avoid building the real XXL module in this unit test.
    mocker.patch("transformers.UMT5EncoderModel", return_value=encoder)
    pipeline = KyutaiAuraPipeline.__new__(KyutaiAuraPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cuda:0")
    pipeline.od_config = SimpleNamespace()
    loaded, _ = pipeline._load_text_encoder(Path("/unused"))
    assert loaded is encoder
    if offload:
        encoder.to.assert_not_called()
    else:
        encoder.to.assert_called_once_with(pipeline.device)
