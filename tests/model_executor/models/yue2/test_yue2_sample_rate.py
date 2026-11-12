# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2 talker emits a VAE-config-derived sample rate."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.yue2.yue2 import Yue2ForCausalLM, YuE2VAEConfig
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_talker() -> Yue2ForCausalLM:
    model = object.__new__(Yue2ForCausalLM)
    nn.Module.__init__(model)
    model._states = {}
    model._audio_queue = []
    model._vae = None
    model._vae_device = None
    return model


def test_output_sample_rate_follows_loaded_vae_config() -> None:
    model = _bare_talker()
    model._vae = SimpleNamespace(config=SimpleNamespace(sample_rate=UNUSUAL_RATE))

    assert model._output_sample_rate() == UNUSUAL_RATE


def test_output_sample_rate_falls_back_to_config_default_before_load() -> None:
    # Error clips can ship before any successful decode; the pre-load value
    # is the VAE config class default, not a talker-side literal.
    model = _bare_talker()

    assert model._output_sample_rate() == int(YuE2VAEConfig().sample_rate) == 48000


def test_make_omni_output_emits_per_request_rate() -> None:
    model = _bare_talker()
    model._vae = SimpleNamespace(config=SimpleNamespace(sample_rate=UNUSUAL_RATE))
    model._audio_queue = [
        ("r1", torch.zeros(2, 5), False, False),
        ("r2", torch.zeros(2, 5), False, False),
    ]

    output = model.make_omni_output(torch.zeros(1))

    multimodal_outputs = output.multimodal_outputs
    assert len(multimodal_outputs["model_outputs"]) == 2
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 2
    assert all(t.dim() == 0 and t.dtype == torch.int32 for t in srs)
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE


def test_make_omni_output_zero_audio_step_emits_empty_rate_list() -> None:
    # The sparse payload rides every step, including ones that decoded
    # nothing; the rate list must stay parallel to model_outputs (empty)
    # rather than raise on the zero count.
    model = _bare_talker()
    model._vae = SimpleNamespace(config=SimpleNamespace(sample_rate=UNUSUAL_RATE))

    output = model.make_omni_output(torch.zeros(1))

    multimodal_outputs = output.multimodal_outputs
    assert multimodal_outputs["model_outputs"] == []
    assert multimodal_outputs[SAMPLE_RATE_KEY] == []


def test_ship_audio_appends_rate_to_current_step_payload() -> None:
    model = _bare_talker()
    model._vae = SimpleNamespace(config=SimpleNamespace(sample_rate=UNUSUAL_RATE))
    model._last_mm = {
        "model_outputs": [],
        SAMPLE_RATE_KEY: [],
        "meta": {"req_id": [], "truncated": [], "error": []},
    }

    model._ship_audio("r1", torch.zeros(2, 3), False)

    mm = model._last_mm
    assert len(mm["model_outputs"]) == 1
    srs = mm[SAMPLE_RATE_KEY]
    assert len(srs) == 1 and srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert int(srs[0]) == UNUSUAL_RATE
