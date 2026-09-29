# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_make_omni_output_keeps_token_rows_for_mixed_prefill_and_live_rows():
    # One live session carries one cached audio row; a newly admitted session
    # prefills 150 rows in the same step. The runner indexes the hidden states
    # with token-space logits indices, so the last prefill row must survive.
    hidden = torch.randn(151, 16)
    live_codes = torch.zeros(1, 8, dtype=torch.long)
    out = PersonaPlexTalkerForConditionalGeneration.make_omni_output(
        None,
        hidden,
        model_intermediate_buffer=[{"codes": {"audio": live_codes}}, {}],
    )

    assert isinstance(out, OmniOutput)
    assert out.text_hidden_states.shape[0] == 151
    logits_indices = torch.tensor([0, 150])
    assert out.text_hidden_states[logits_indices].shape == (2, 16)
    assert torch.equal(out.multimodal_outputs["codes"]["audio"], live_codes)
