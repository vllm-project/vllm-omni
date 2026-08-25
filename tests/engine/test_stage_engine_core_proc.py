from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.serialization import serialize_model_intermediate_buffer
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
        model_intermediate_buffer=serialize_model_intermediate_buffer(
            {"hidden_states": {"tts": torch.arange(4, dtype=torch.float32)}}
        ),
    )
    scheduler_request = SimpleNamespace()

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}
    assert torch.equal(
        result.model_intermediate_buffer["hidden_states"]["tts"],
        torch.arange(4, dtype=torch.float32),
    )
