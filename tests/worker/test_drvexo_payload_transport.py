# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage routing metadata must not erase the caller's speech payload."""

from dataclasses import dataclass, field

import pytest

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.entrypoints.openai.drvexo_speech import speech_prompt
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class RequestState:
    additional_information_cpu: dict = field(default_factory=dict)


@dataclass
class ScheduledRequest:
    req_id: str
    model_intermediate_buffer: dict
    additional_information: object
    prompt_embeds: object = None


@dataclass
class ScheduledBatch:
    scheduled_new_reqs: list[ScheduledRequest]


@pytest.mark.parametrize("language", ["Auto", "Chinese", "English"])
def test_decode_preserves_fixed_speech_payload_alongside_final_stage_metadata(language):
    runner = object.__new__(OmniGPUModelRunner)
    runner.requests = {"speech": RequestState()}
    runner.model_intermediate_buffer = {}
    buffer = speech_prompt("Read this sentence.", language)["model_intermediate_buffer"]
    batch = ScheduledBatch(
        [ScheduledRequest("speech", buffer, serialize_additional_information({"omni_final_stage_id": 2}))]
    )
    runner._decode_and_store_request_payloads(batch)
    actual = runner.model_intermediate_buffer["speech"]
    assert actual["drvexo_fixed_reply"] is True
    assert actual["text"] == ["Read this sentence."]
    assert actual["language"] == [language]
    assert actual["omni_final_stage_id"] == 2
    assert runner.requests["speech"].additional_information_cpu == actual


def test_request_transport_snapshot_drops_previous_payload_keys():
    runner = object.__new__(OmniGPUModelRunner)
    runner.requests = {"request": RequestState()}
    runner.model_intermediate_buffer = {"request": {"old": True, "config": {"old": 1, "shared": 0}}}
    batch = ScheduledBatch(
        [
            ScheduledRequest(
                "request",
                {"new": True, "config": {"shared": 2}},
                serialize_additional_information({"omni_final_stage_id": 2}),
            )
        ]
    )
    runner._decode_and_store_request_payloads(batch)
    assert runner.model_intermediate_buffer["request"] == {
        "new": True,
        "config": {"shared": 2},
        "omni_final_stage_id": 2,
    }
    runner._decode_and_store_request_payloads(
        ScheduledBatch(
            [
                ScheduledRequest(
                    "request",
                    {},
                    serialize_additional_information({"only": "legacy"}),
                )
            ]
        )
    )
    assert runner.model_intermediate_buffer["request"] == {"only": "legacy"}


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_admission_preserves_preseeded_buffer(empty):
    runner = object.__new__(OmniGPUModelRunner)
    runner.requests = {"request": RequestState()}
    previous = {"seeded": True, "meta": {"num_processed_tokens": 3}}
    runner.model_intermediate_buffer = {"request": previous}
    runner._decode_and_store_request_payloads(ScheduledBatch([ScheduledRequest("request", {}, empty)]))
    assert runner.model_intermediate_buffer["request"] == previous


def test_bad_legacy_payload_does_not_fail_other_admissions():
    runner = object.__new__(OmniGPUModelRunner)
    runner.requests = {key: RequestState() for key in ("bad", "valid", "current")}
    runner.model_intermediate_buffer = {"bad": {"seeded": True}}
    runner._decode_and_store_request_payloads(
        ScheduledBatch(
            [
                ScheduledRequest("bad", {}, object()),
                ScheduledRequest("valid", {}, serialize_additional_information({"fresh": True})),
                ScheduledRequest("current", {"text": ["keep this"]}, object()),
            ]
        )
    )
    assert runner.model_intermediate_buffer["bad"] == {"seeded": True}
    assert runner.model_intermediate_buffer["valid"] == {"fresh": True}
    assert runner.model_intermediate_buffer["current"] == {"text": ["keep this"]}


@pytest.mark.parametrize("finished", [False, True], ids=["preempted-resume", "finished-id-reuse"])
def test_scheduler_payload_lifetime_distinguishes_resume_from_id_reuse(monkeypatch, finished):
    from unittest.mock import Mock

    from vllm import SamplingParams
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_omni.core.sched.output import OmniNewRequestData
    from vllm_omni.worker import gpu_model_runner as module

    runner = object.__new__(OmniGPUModelRunner)
    previous = {"text": ["previous caller"], "meta": {"num_processed_tokens": 7}}
    runner.model_intermediate_buffer = {"request": previous}
    runner.requests = {"request": CachedRequestState("request", [1], [], None, None, ([],), 0, [])}
    runner.num_prompt_logprobs = {}
    runner.speculative_config = None
    runner.use_async_spec_decode = False
    runner.use_async_scheduling = False
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.encoder_cache = {}
    runner.late_interaction_runner = Mock()
    runner._may_reorder_batch = Mock()
    runner._downstream_payload_cache = {"request": "old downstream"}
    runner._talker_mtp_generators = {"request": "old generator"}
    batch = Mock()
    batch.req_ids = ["request"]
    batch.req_id_to_index = {"request": 0}

    def remove_request(identifier):
        batch.req_id_to_index.pop(identifier, None)
        if identifier in batch.req_ids:
            batch.req_ids.remove(identifier)

    batch.remove_request.side_effect = remove_request
    runner.input_batch = batch
    monkeypatch.setattr(module, "get_pp_group", lambda: Mock(is_last_rank=True))
    new = OmniNewRequestData("request", [2], [], SamplingParams(temperature=0), None, ([],), 0, None)
    cached = CachedRequestData(["request"], {"request"}, [], {}, [([],)], [0], [0])
    output = SchedulerOutput(
        scheduled_new_reqs=[new] if finished else [],
        scheduled_cached_reqs=CachedRequestData.make_empty() if finished else cached,
        num_scheduled_tokens={"request": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids={"request"} if finished else set(),
        free_encoder_mm_hashes=[],
    )
    runner._update_states(output)
    if finished:
        assert runner.requests["request"].prompt_token_ids == [2]
        assert "request" not in runner.model_intermediate_buffer
        assert "request" not in runner._downstream_payload_cache
        assert "request" not in runner._talker_mtp_generators
    else:
        assert runner.model_intermediate_buffer["request"] is previous
        assert runner.requests["request"].prompt_token_ids == [1]
