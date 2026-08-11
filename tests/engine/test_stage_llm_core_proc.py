from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage.stage_llm_core_proc import StageLLMCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageLLMCoreProc.__new__(StageLLMCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
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
