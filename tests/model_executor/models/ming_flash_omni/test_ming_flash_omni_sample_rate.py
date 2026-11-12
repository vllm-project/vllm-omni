# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming flash-omni talker emits the canonical sample-rate payload shape."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
    MingFlashOmniTalkerForConditionalGeneration,
)
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_talker(waveform: torch.Tensor) -> MingFlashOmniTalkerForConditionalGeneration:
    model = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_vae = SimpleNamespace(config=SimpleNamespace(sample_rate=UNUSUAL_RATE))
    model.audio_generator = SimpleNamespace(
        decode_to_waveform=lambda latents, stream_decode: waveform,
        trim_trailing_silence=lambda wav: wav,
    )
    return model


def test_batched_waveform_gets_one_rate_per_request() -> None:
    model = _bare_talker(torch.zeros(2, 10))
    output = model._decode_to_output([torch.zeros(1, 4, 8)], stream_decode=False)

    multimodal_outputs = output.multimodal_outputs
    assert multimodal_outputs["audio"].shape == (2, 10)
    # The emission used to be a single shared 0-d tensor; it must now be the
    # canonical per-request list matching the waveform's batch rows.
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 2
    assert all(t.dim() == 0 and t.dtype == torch.int32 for t in srs)
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE


def test_single_waveform_gets_single_rate() -> None:
    model = _bare_talker(torch.zeros(10))
    output = model._decode_to_output([torch.zeros(1, 4, 8)], stream_decode=True)

    srs = output.multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert get_sampling_rate(output.multimodal_outputs) == UNUSUAL_RATE
