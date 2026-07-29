from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
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


def test_reset_prefix_cache_resets_omni_state_after_scheduler_success():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.model_executor = MagicMock()

    with patch.object(EngineCoreProc, "reset_prefix_cache", return_value=True) as upstream_reset:
        assert engine.reset_prefix_cache(True, True) is True

    upstream_reset.assert_called_once_with(
        reset_running_requests=True,
        reset_connector=True,
    )
    engine.model_executor.collective_rpc.assert_called_once_with("reset_omni_prefix_cache")


def test_reset_prefix_cache_keeps_omni_state_when_scheduler_rejects_reset():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.model_executor = MagicMock()

    with patch.object(EngineCoreProc, "reset_prefix_cache", return_value=False):
        assert engine.reset_prefix_cache() is False

    engine.model_executor.collective_rpc.assert_not_called()
