# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex code2wav emits the canonical sample-rate payload shape."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.personaplex.personaplex_code2wav import PersonaPlexCode2Wav
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_model() -> PersonaPlexCode2Wav:
    model = object.__new__(PersonaPlexCode2Wav)
    nn.Module.__init__(model)
    # Unusual rate proves the emission follows the Mimi-config-derived
    # attribute instead of the 24000 fallback literal.
    model._output_sample_rate = UNUSUAL_RATE
    model._num_codebooks = 8
    model.mimi = SimpleNamespace()
    return model


def test_empty_input_emits_canonical_rate() -> None:
    model = _bare_model()

    output = model.forward(input_ids=None)

    multimodal_outputs = output.multimodal_outputs
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE


def test_batched_spans_get_one_rate_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_model()
    monkeypatch.setattr(
        PersonaPlexCode2Wav,
        "_decode_input_id_spans",
        lambda self, ids, seq_token_counts, request_ids, runtime_info: [torch.zeros(3), None],
    )

    output = model.forward(input_ids=torch.zeros(8, dtype=torch.long))

    multimodal_outputs = output.multimodal_outputs
    assert len(multimodal_outputs["model_outputs"]) == 2
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 2
    assert all(t.dim() == 0 and t.dtype == torch.int32 for t in srs)
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE
