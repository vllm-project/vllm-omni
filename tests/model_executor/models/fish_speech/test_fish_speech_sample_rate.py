# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Speech DAC decoder emits the canonical sample-rate payload shape."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.fish_speech.fish_speech_dac_decoder import FishSpeechDACDecoder
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_decoder(monkeypatch: pytest.MonkeyPatch) -> FishSpeechDACDecoder:
    model = object.__new__(FishSpeechDACDecoder)
    nn.Module.__init__(model)
    model._num_codebooks = 8
    # An unusual rate proves the emission reads the decoder's output-rate
    # attribute (single-sourced from the DAC architecture constant) and not
    # a per-site literal.
    model._output_sample_rate = UNUSUAL_RATE
    model._codec = SimpleNamespace()
    monkeypatch.setattr(FishSpeechDACDecoder, "_ensure_codec_loaded", lambda self: None)
    return model


def test_empty_input_emits_canonical_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_decoder(monkeypatch)

    output = model.forward(input_ids=None)

    multimodal_outputs = output.multimodal_outputs
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE


def test_malformed_input_emits_one_rate_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # Codes not divisible by the codebook count decode nothing; the empty
    # result must still carry one rate entry per scheduled request.
    model = _bare_decoder(monkeypatch)

    output = model.forward(input_ids=torch.zeros(3, dtype=torch.long))

    multimodal_outputs = output.multimodal_outputs
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE
