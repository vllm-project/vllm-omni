# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GLM-TTS DiT stage emits the canonical sample-rate payload shape."""

import threading

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.glm_tts.glm_tts_dit_wrapper import GLMTTSDiTForGeneration
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_model() -> GLMTTSDiTForGeneration:
    model = object.__new__(GLMTTSDiTForGeneration)
    nn.Module.__init__(model)
    # The flow config's rate is what the waveform leaves at (vocoder output
    # is resampled back to it); the unusual value proves the emission reads
    # it instead of a literal.
    model.sample_rate = UNUSUAL_RATE
    model.max_num_seqs = 1
    model._stream_forward_count = 0
    model._stream_last_seen = {}
    model._stream_state_by_req = {}
    model._stream_cache_lock = threading.Lock()
    return model


def test_empty_tokens_emits_canonical_rate() -> None:
    model = _bare_model()

    # No speech tokens: every scheduled request gets empty audio but keeps
    # its rate entry.
    output = model.forward(input_ids=torch.zeros(0, dtype=torch.long), positions=None)

    multimodal_outputs = output.multimodal_outputs
    assert len(multimodal_outputs["audio"]) == 1
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE
