# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax Music 3 acoustic stage emits the canonical sample-rate payload shape."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.minimax_music3.acoustic import (
    MiniMaxMusic3AcousticForConditionalGeneration,
)
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_model() -> MiniMaxMusic3AcousticForConditionalGeneration:
    model = object.__new__(MiniMaxMusic3AcousticForConditionalGeneration)
    nn.Module.__init__(model)
    # The vocoder's output rate; the unusual value proves the emission reads
    # this attribute instead of the module constant at each site.
    model._output_sample_rate = UNUSUAL_RATE
    model._loaded = True
    return model


def test_silent_step_emits_one_rate_per_request() -> None:
    model = _bare_model()

    # A scheduled request whose span has not arrived yet decodes to silence
    # but must still carry its rate entry.
    output = model.forward(input_ids=torch.zeros(1), seq_token_counts=[4])

    multimodal_outputs = output.multimodal_outputs
    assert len(multimodal_outputs["model_outputs"]) == 1
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE
