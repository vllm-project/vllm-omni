# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexConfig,
)
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


def test_personaplex_talker_rejects_v2_model_runner():
    mock_vllm_config = MagicMock()
    mock_model_config = MagicMock()
    mock_model_config.hf_config = PersonaPlexConfig()
    mock_model_config.use_v2_model_runner = True
    mock_vllm_config.model_config = mock_model_config

    with pytest.raises(
        ValueError,
        match="PersonaPlex Stage 0 \\(Talker\\) requires model_runner: v1",
    ):
        PersonaPlexTalkerForConditionalGeneration(vllm_config=mock_vllm_config)
