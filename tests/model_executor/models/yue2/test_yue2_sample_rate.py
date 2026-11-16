# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2 talker emits a VAE-config-derived sample rate."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.yue2.yue2 import Yue2ForCausalLM
from vllm_omni.sample_rate import SAMPLE_RATE_KEY

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
