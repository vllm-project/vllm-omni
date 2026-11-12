# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice AR decoder emits the canonical sample-rate payload shape."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.omnivoice.omnivoice import OmniVoiceModel
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# An unusual rate proves the emission reads the model config instead of a literal.
UNUSUAL_RATE = 22050


class _StubDecoder(nn.Module):
    """Stands in for the weight-loaded OmniVoiceDecoder ([B, 8, T] -> [B, samples])."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        return torch.zeros(audio_codes.shape[0], 16)


def _bare_model() -> OmniVoiceModel:
    model = object.__new__(OmniVoiceModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(sample_rate=UNUSUAL_RATE, llm_hidden_size=8)
    model.decoder = _StubDecoder()
    return model


def test_decoder_output_carries_per_request_rate_list() -> None:
    model = _bare_model()
    output = model._forward_decoder(
        input_ids=torch.zeros(2, 8, dtype=torch.long),
        kwargs={"runtime_additional_information": [{"audio_tokens": torch.zeros(2, 8, 5, dtype=torch.long)}]},
    )

    multimodal_outputs = output.multimodal_outputs
    assert multimodal_outputs["audio"].shape == (2, 16)
    # The emission used to be a plain int; it must now be the canonical
    # per-request list of 0-d int32 tensors, one per batch row.
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 2
    assert all(t.dim() == 0 and t.dtype == torch.int32 for t in srs)
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE


def test_decoder_output_unbatched_tokens_get_single_rate() -> None:
    model = _bare_model()
    output = model._forward_decoder(
        input_ids=torch.zeros(1, 8, dtype=torch.long),
        kwargs={"runtime_additional_information": [{"audio_tokens": torch.zeros(8, 5, dtype=torch.long)}]},
    )

    srs = output.multimodal_outputs[SAMPLE_RATE_KEY]
    # 2-D tokens are unsqueezed to one request, not broadcast across the batch.
    assert isinstance(srs, list) and len(srs) == 1
    assert get_sampling_rate(output.multimodal_outputs) == UNUSUAL_RATE
