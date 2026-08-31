# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from vllm_omni.model_executor.models.minimax_h3.text_encoder import MiniMaxH3TextEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model_without_weights() -> MiniMaxH3TextEncoder:
    model = MiniMaxH3TextEncoder.__new__(MiniMaxH3TextEncoder)
    model.config = SimpleNamespace(
        vision_start_token_id=1,
        vision_end_token_id=2,
        image_token_id=3,
        video_token_id=4,
    )
    model._token_tags = None
    return model


def test_embed_failure_clears_token_tags():
    model = _model_without_weights()
    model._token_tags = torch.tensor([0])

    with (
        patch.object(Qwen3VLForConditionalGeneration, "embed_input_ids", side_effect=RuntimeError("failed")),
        pytest.raises(RuntimeError, match="failed"),
    ):
        model.embed_input_ids(torch.tensor([5, 3]))

    assert model._token_tags is None


def test_invalid_output_length_consumes_token_tags():
    model = _model_without_weights()
    model._token_tags = torch.tensor([1, 0])

    with pytest.raises(RuntimeError, match="token tags exceed"):
        model.make_omni_output(torch.zeros(1, 8))

    assert model._token_tags is None


def test_make_omni_output_uses_shared_stage_payload_fields():
    model = _model_without_weights()
    model._token_tags = torch.tensor([1, 0])
    hidden = torch.zeros(2, 5120)

    output = model.make_omni_output(hidden)

    assert output.multimodal_outputs is not None
    assert torch.equal(output.multimodal_outputs["hidden_states"]["output"], hidden)
    assert torch.equal(output.multimodal_outputs["meta"]["token_role_ids"], torch.tensor([[1], [0]]))
    assert "encoder_hidden_states" not in output.multimodal_outputs
    assert "token_tags" not in output.multimodal_outputs
