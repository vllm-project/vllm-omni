# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import msgspec
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_preprocess_add_request_preserves_omni_fields(monkeypatch):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
        model_intermediate_buffer={"runtime": "state"},
        streaming_prompt_continuous=True,
    )
    scheduler_request = SimpleNamespace()

    monkeypatch.setattr(
        "vllm.v1.engine.core.EngineCoreProc.preprocess_add_request",
        lambda self, request: (scheduler_request, 3),
    )
    result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}
    assert result.model_intermediate_buffer == {"runtime": "state"}
    assert result.streaming_prompt_continuous is True


@pytest.mark.parametrize("wire_roundtrip", [False, True])
def test_scheduler_native_utilities_dispatch_through_engine_core(wire_roundtrip):
    calls: list[tuple[object, ...]] = []

    class Scheduler:
        def get_streaming_prompt_metrics(self, request_id):
            calls.append(("metrics", request_id))
            return {"omni_context_tokens": 17}

        def append_streaming_prompt_unit(
            self,
            request_id,
            token_ids,
            model_intermediate_buffer,
            *,
            operation_id,
            operation_fingerprint,
            sampling_params=None,
        ):
            calls.append(
                (
                    "append",
                    request_id,
                    token_ids,
                    model_intermediate_buffer,
                    operation_id,
                    operation_fingerprint,
                    sampling_params,
                )
            )
            return {"deduplicated": False}

    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = Scheduler()

    assert engine.get_streaming_prompt_metrics("request-1") == {"omni_context_tokens": 17}
    params = SamplingParams(temperature=0.8, max_tokens=7)
    args = ["request-1", [3, 5], {"runtime": "state"}, "operation-1", b"fingerprint", params]
    if wire_roundtrip:
        args = msgspec.msgpack.decode(msgspec.msgpack.encode(args))
    converted = EngineCoreProc._convert_msgspec_args(engine.append_streaming_prompt_unit, args)
    assert engine.append_streaming_prompt_unit(*converted) == {"deduplicated": False}

    assert calls == [
        ("metrics", "request-1"),
        (
            "append",
            "request-1",
            [3, 5],
            {"runtime": "state"},
            "operation-1",
            b"fingerprint",
            params,
        ),
    ]
