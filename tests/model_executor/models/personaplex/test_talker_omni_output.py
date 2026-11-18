# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace
from unittest.mock import patch

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


def test_personaplex_talker_creates_owned_v2_state():
    model = SimpleNamespace()
    with patch("vllm_omni.model_executor.models.personaplex.model_state.PersonaPlexModelState") as state:
        result = PersonaPlexTalkerForConditionalGeneration.create_mrv2_model_state(model, "config", None, "cpu")
    state.assert_called_once_with("config", model, None, "cpu")
    assert result is state.return_value
    assert model.mm_outputs_fresh_per_step is True


def test_personaplex_tensor_forward_keeps_full_graph_eligible():
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    model = PersonaPlexTalkerForConditionalGeneration.__new__(PersonaPlexTalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.have_multimodal_outputs = True
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner._configure_cudagraph_output_contract()
    assert runner._model_has_structured_output is False
    assert runner._exclude_full_graph is False


@pytest.mark.parametrize(
    ("use_v2", "mode", "async_chunk", "supported"),
    [
        (True, "turn", False, False),
        (True, "turn", True, False),
        (True, "duplex", False, False),
        (True, "duplex", True, True),
        (False, "turn", False, True),
        (False, "duplex", True, True),
    ],
)
def test_personaplex_runner_mode_guard(use_v2, mode, async_chunk, supported):
    from vllm_omni.model_executor.models.personaplex.configuration_personaplex import PersonaPlexConfig

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            use_v2_model_runner=use_v2,
            session_mode=mode,
            async_chunk=async_chunk,
            hf_config=PersonaPlexConfig(),
            dtype=torch.bfloat16,
        )
    )
    # Unsupported modes must fail before constructing or allocating the backbone.
    with patch(
        "vllm_omni.model_executor.models.personaplex.personaplex_talker.HeliumModel",
        side_effect=RuntimeError("backbone reached"),
    ) as backbone:
        error = RuntimeError if supported else ValueError
        message = "backbone reached" if supported else "requires session_mode='duplex'"
        with pytest.raises(error, match=message):
            PersonaPlexTalkerForConditionalGeneration(vllm_config=config)
        assert backbone.called is supported
