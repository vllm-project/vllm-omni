# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Constructor placement contract, following Qwen-Image initialization."""

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.helios.pipeline_helios as helios
from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LoadingModule(torch.nn.Module):
    """Observe placement without allocating CUDA or downloading heavy weights."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.placements = []

    def to(self, device):
        self.placements.append(torch.device(device))
        return self


@pytest.mark.parametrize("cpu_offload", [True, False], ids=["model_offload", "no_offload"])
def test_helios_constructor_places_encoder_and_vae_before_dit(monkeypatch, tmp_path, cpu_offload):
    """Real config policy must choose CPU under model-level offload, as Qwen does.

    Only heavyweight loaders/device allocation are isolated. This is a unit test
    of constructor placement, not evidence of model loading or offloader runtime.
    """
    config = OmniDiffusionConfig(model=str(tmp_path), enable_cpu_offload=cpu_offload)
    encoder = _LoadingModule()
    vae = _LoadingModule(SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8))
    text_config = SimpleNamespace(tie_word_embeddings=False)
    monkeypatch.setattr(helios, "get_local_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(helios.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(helios.AutoConfig, "from_pretrained", lambda *args, **kwargs: text_config)
    monkeypatch.setattr(helios.UMT5EncoderModel, "from_pretrained", lambda *args, **kwargs: encoder)
    monkeypatch.setattr(helios.AutoencoderKLWan, "from_pretrained", lambda *args, **kwargs: vae)

    expected = torch.device("cpu") if cpu_offload else torch.device("cuda:0")

    def create_transformer(*args, **kwargs):
        # The ordering matters: residency must be decided before DiT allocation.
        assert encoder.placements == [expected]
        assert vae.placements == [expected]
        return _LoadingModule()

    monkeypatch.setattr(helios, "create_transformer_from_config", create_transformer)
    pipeline = helios.HeliosPipeline(od_config=config)
    assert pipeline.device == torch.device("cuda:0")
    assert text_config.tie_word_embeddings is True
    assert pipeline.weights_sources[0].subfolder == "transformer"
