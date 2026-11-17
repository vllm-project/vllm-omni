# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audex code2wav and xcodec stages emit the canonical sample-rate payload shape."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.audex.audex_code2wav import AudexCode2Wav
from vllm_omni.model_executor.models.audex.audex_xcodec import AudexXCodec1
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def test_code2wav_emits_config_derived_rate() -> None:
    model = object.__new__(AudexCode2Wav)
    nn.Module.__init__(model)
    # Unusual rate proves the emission follows the stage config's rate
    # (``config.sample_rate``) instead of a per-site literal.
    model._sample_rate = UNUSUAL_RATE

    # A scheduled request with no chunk payload yet decodes to empty audio
    # but keeps its rate entry.
    output = model.forward(runtime_additional_information=[{}])

    multimodal_outputs = output.multimodal_outputs
    assert len(multimodal_outputs["model_outputs"]) == 1
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE


def test_xcodec_emits_config_derived_rate() -> None:
    model = object.__new__(AudexXCodec1)
    nn.Module.__init__(model)
    # Unusual rate proves the emission follows the stage config's rate
    # (``config.sampling_rate``) instead of a per-site literal.
    model._sample_rate = UNUSUAL_RATE

    output = model.forward(runtime_additional_information=[{}])

    multimodal_outputs = output.multimodal_outputs
    assert len(multimodal_outputs["model_outputs"]) == 1
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE
