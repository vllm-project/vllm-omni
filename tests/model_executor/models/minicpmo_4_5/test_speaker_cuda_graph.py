# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import ModelConfig, VllmConfig

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import CampplusTorch
from vllm_omni.model_executor.models.minicpmo_4_5.speaker_cuda_graph import SpeakerCudaGraph

pytestmark = [pytest.mark.core_model]


class _Speaker(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


def _config():
    config = VllmConfig()
    config.model_config = ModelConfig.__new__(ModelConfig)
    config.model_config.multimodal_config = None
    return config


@pytest.mark.cpu
def test_cpu_fallback_does_not_mutate_shared_model_or_config():
    speaker = CampplusTorch(_Speaker(), "cpu")
    config = _config()
    wrapper = SpeakerCudaGraph(speaker, config)
    feat = torch.randn(17, 80, dtype=torch.bfloat16)
    assert torch.equal(wrapper(feat), speaker(feat))
    assert config.model_config.multimodal_config is None
    assert wrapper.graph.vllm_config.model_config.multimodal_config is not None
    assert not wrapper.graph.graphs
    assert wrapper.speaker is speaker


@pytest.mark.cpu
def test_facade_rejects_non_torch_speaker():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import MiniCPMO45Token2wav

    facade = MiniCPMO45Token2wav.__new__(MiniCPMO45Token2wav)
    facade._core = SimpleNamespace(_spk_model=object())
    with pytest.raises(ValueError, match="CUDA Torch speaker"):
        facade.enable_speaker_cuda_graph(_config())


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_replay_ownership_shape_capacity_and_precision_fallback():
    speaker = CampplusTorch(_Speaker(), "cuda:0")
    wrapper = SpeakerCudaGraph(speaker, _config(), max_graphs=1)
    feat = torch.randn(17, 80, device="cuda")
    wrapper(feat)
    retained = wrapper(feat)
    expected = speaker(feat).clone()
    feat.add_(3)
    assert torch.equal(wrapper(feat), speaker(feat))
    assert torch.equal(retained, expected)
    assert len(wrapper.graph.graphs) == 1
    other = torch.randn(19, 80, device="cuda")
    assert torch.equal(wrapper(other), speaker(other))
    assert len(wrapper.graph.graphs) == 1
    hits = wrapper.graph.get_cumulative_stats()["graph_hits"]
    original = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = not original
        assert torch.equal(wrapper(feat), speaker(feat))
        assert wrapper.graph.get_cumulative_stats()["graph_hits"] == hits
    finally:
        torch.backends.cudnn.allow_tf32 = original
    assert torch.equal(wrapper(feat), speaker(feat))
    assert wrapper.graph.get_cumulative_stats()["graph_hits"] == hits + 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(wrapper(feat), speaker(feat))
    assert wrapper.graph.get_cumulative_stats()["graph_hits"] == hits + 1
