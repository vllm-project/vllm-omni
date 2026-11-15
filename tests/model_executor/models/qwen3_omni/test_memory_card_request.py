# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Count and decode private Thinker requests without a model forward."""

import json
from dataclasses import replace

import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm_omni.outputs import OmniRequestOutput

from tests.model_executor.models.qwen3_omni.test_duplex_context_view_processor import (
    qwen_processor as _processor_fixture,
)
from tests.model_executor.models.qwen3_omni.test_memory_card_extraction import context as _context_fixture
from tests.model_executor.models.qwen3_omni.test_memory_card_extraction import result, upsert
from vllm_omni.model_executor.models.qwen3_omni.duplex.memory import build_memory_messages

context = _context_fixture
qwen_processor = _processor_fixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def build(context, processor, **limits):
    from vllm_omni.model_executor.models.qwen3_omni.duplex.memory_request import build_memory_request

    return build_memory_request(
        context,
        processor=processor,
        **dict({"input_token_limit": 2048, "output_token_limit": 512, "max_model_len": 4096}, **limits),
    )


def bound_request(context, processor):
    request = build(context, processor)
    assert request is not None
    assert request.context_work is None
    return replace(request, context_work=context.work)


def completion(text, *, finish_reason="stop", index=0):
    return CompletionOutput(index, text, [1], None, None, finish_reason=finish_reason)


def output(text, *, omni=False, **kwargs):
    if omni:
        return OmniRequestOutput(request_id="private", outputs=[completion(text)], stage_id=0, **kwargs)
    return RequestOutput("private", None, None, None, [completion(text)], kwargs.get("finished", True))


def decode(request, response):
    from vllm_omni.model_executor.models.qwen3_omni.duplex.memory_request import decode_memory_result

    return decode_memory_result(request=request, output=response)


def test_complete_template_ids_are_the_submitted_prompt(context, qwen_processor):
    request = build(context, qwen_processor)
    assert request is not None
    messages = build_memory_messages(context)
    rendered = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected = qwen_processor.tokenizer.encode(rendered, add_special_tokens=False)
    assert request.prompt == {"prompt_token_ids": expected}
    assert rendered.endswith("<|im_start|>assistant\n")
    assert 0 < len(expected) <= 2048
    assert len(expected) > len(qwen_processor.tokenizer.encode(messages[1]["content"], add_special_tokens=False))
    assert request.stage_id == 0 and request.model_source is context
    assert request.context_work is None
    sampling = request.sampling_params
    assert isinstance(sampling, SamplingParams)
    assert (sampling.n, sampling.temperature, sampling.max_tokens) == (1, 0, 512)
    assert sampling.detokenize and sampling.output_kind is RequestOutputKind.FINAL_ONLY
    assert sampling.stop is None or sampling.stop == []
    assert request.retry_cooldown_s is None


def test_exact_template_and_deployment_boundaries(context, qwen_processor):
    first = build(context, qwen_processor)
    count = len(first.prompt["prompt_token_ids"])
    assert build(context, qwen_processor, input_token_limit=count, max_model_len=count + 512) is not None
    assert build(context, qwen_processor, input_token_limit=count - 1) is None
    assert build(context, qwen_processor, max_model_len=count + 511) is None
    assert build(context, qwen_processor, max_model_len=count + 17, output_token_limit=17) is not None
    assert build(context, qwen_processor, input_token_limit=0) is None
    assert build(context, qwen_processor, output_token_limit=0) is None


@pytest.mark.parametrize(
    "limits",
    [
        {"input_token_limit": True},
        {"input_token_limit": -1},
        {"input_token_limit": 2049},
        {"input_token_limit": 1.0},
        {"output_token_limit": False},
        {"output_token_limit": -1},
        {"output_token_limit": 513},
        {"max_model_len": True},
        {"max_model_len": 0},
    ],
)
def test_invalid_allowance_is_rejected_before_processor_use(context, limits):
    with pytest.raises(ValueError, match="invalid_memory_request_limits"):
        build(context, object(), **limits)


def test_oversized_capture_is_deferred_without_clipping_or_progress(context, qwen_processor):
    excerpt = replace(context.evidence[0], text="杭州" * 3500, char_end=7040)
    larger = replace(context, evidence=(excerpt,))
    assert build(larger, qwen_processor) is None
    assert larger.evidence[0].text == "杭州" * 3500
    assert larger.work is context.work
    assert build(context, qwen_processor) is not None


@pytest.mark.parametrize("delimiter", ["<|audio_pad|>", "<|image_pad|>", "<|video_pad|>", "<|im_end|>", "<|im_start|>"])
def test_quoted_evidence_cannot_add_template_or_media_tokens(context, qwen_processor, delimiter):
    text = "Quoted " + delimiter + " literal."
    captured = replace(context, evidence=(replace(context.evidence[0], text=text, char_end=40 + len(text)),))
    messages = build_memory_messages(captured)
    assert json.loads(messages[1]["content"])["evidence"][0]["text"] == text
    assert delimiter not in messages[1]["content"]
    request = build(captured, qwen_processor)
    ids = request.prompt["prompt_token_ids"]
    special_id = qwen_processor.tokenizer.convert_tokens_to_ids(delimiter)
    # Only the two closed messages and the generation header may add controls.
    expected_count = {"<|im_end|>": 2, "<|im_start|>": 3}.get(delimiter, 0)
    assert ids.count(special_id) == expected_count


@pytest.mark.parametrize("omni", [False, True])
def test_final_typed_text_result_decodes_only_against_bound_capture(context, qwen_processor, omni):
    request = bound_request(context, qwen_processor)
    parsed = decode(request, output(json.dumps(result(upsert())), omni=omni))
    assert parsed is not None and len(parsed.operations) == 1
    assert parsed.operations[0].text == "The user lives in Hangzhou."
    assert decode(request, output(json.dumps(result()), omni=omni)).operations == ()
    assert decode(replace(request, context_work=None), output(json.dumps(result()))) is None
    assert decode(replace(request, context_work=replace(context.work, epoch=999)), output(json.dumps(result()))) is None
    assert decode(replace(request, model_source=None), output(json.dumps(result()))) is None
    assert decode(replace(request, stage_id=1), output(json.dumps(result()))) is None


@pytest.mark.parametrize(
    "case", ["unfinished", "length", "abort", "missing_reason", "two", "none", "index", "schema", "untyped"]
)
def test_partial_failed_or_ambiguous_completion_is_rejected(context, qwen_processor, case):
    request = bound_request(context, qwen_processor)
    response = output(json.dumps(result()))
    if case == "unfinished":
        response.finished = False
    elif case in {"length", "abort", "missing_reason"}:
        response.outputs[0].finish_reason = None if case == "missing_reason" else case
    elif case == "two":
        response.outputs *= 2
    elif case == "none":
        response.outputs = []
    elif case == "index":
        response.outputs[0].index = 1
    elif case == "schema":
        response.outputs[0].text = '{"operations":[]}'
    else:
        response = {"finished": True, "outputs": [{"text": json.dumps(result())}]}
    assert decode(request, response) is None


@pytest.mark.parametrize("fields", [{"stage_id": 1}, {"error": "failed"}, {"final_output_type": "audio"}])
def test_omni_stage_errors_and_nontext_output_are_rejected(context, qwen_processor, fields):
    response = output(json.dumps(result()), omni=True)
    for name, value in fields.items():
        setattr(response, name, value)
    assert decode(bound_request(context, qwen_processor), response) is None


def test_qwen_live_memory_hooks_remain_disabled():
    from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    assert plugin.supports_context_auxiliary_requests is False
    assert plugin.context_auxiliary_policy(runtime_config=None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("metered", [False, True])
async def test_request_round_trip_uses_only_private_stage_zero_and_owner_admission(qwen_processor, mocker, metered):
    from tests.engine.duplex.test_session_auxiliary import auxiliary_submissions, wait_until
    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex.commands import Heartbeat
    from vllm_omni.engine.duplex.contracts import duplex_is_auxiliary_request
    from vllm_omni.engine.duplex.events import SessionCreated, SessionHeartbeatAck, SessionUpdated
    from vllm_omni.engine.duplex.memory_types import DuplexContextAuxiliaryPolicy
    from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryRequest
    from vllm_omni.engine.duplex.session.auxiliary_budget import DuplexAuxiliaryBudgetLimits

    h = await open_harness(auto_response=False, stage_count=3)
    try:
        prepared: list[DuplexAuxiliaryRequest] = []
        decoded: list[DuplexAuxiliaryRequest] = []
        plugin = h.runner.plugin
        mocker.patch.object(plugin, "supports_context_auxiliary_requests", True)
        compute = DuplexAuxiliaryBudgetLimits(idle_s=0, cooldown_s=0) if metered else None
        mocker.patch.object(
            plugin, "context_auxiliary_policy", return_value=DuplexContextAuxiliaryPolicy(compute_budget=compute)
        )

        async def prepare(*, context, runtime_config):
            if prepared:
                return None
            request = build(context, qwen_processor)
            prepared.append(request)
            return request

        def consume(*, request, output):
            # The actual runtime has cleaned its stage resource before decode.
            assert ([output.request_id], True) in h.port.cleanups
            assert not any(duplex_is_auxiliary_request(rid) for _, rid in h.session.request_resources)
            decoded.append(request)
            return decode(request, output)

        mocker.patch.object(plugin, "prepare_context_auxiliary_request", side_effect=prepare)
        mocker.patch.object(plugin, "decode_context_auxiliary_result", side_effect=consume)
        h.session.register_history_item("old", {"role": "user", "content": "I live in Hangzhou."})
        before = h.session.history_snapshot()
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        submission = auxiliary_submissions(h)[0]
        assert submission.context.stage_id == submission.context.final_stage_id == 0
        assert submission.prompt == prepared[0].prompt
        assert submission.context.sampling_params[0] is prepared[0].sampling_params
        response = output(json.dumps(result(upsert())), omni=True)
        response.request_id = submission.context.request_id
        assert h.deliver(response, stage_id=0, segment_finished=True)
        await wait_until(lambda: len(h.session.context_ledger.card_candidates()) == 1)
        await h.settle()
        assert len(decoded) == 1
        assert decoded[0].context_work == prepared[0].model_source.work
        assert h.session.history_snapshot() == before
        assert all(isinstance(event, (SessionCreated, SessionHeartbeatAck, SessionUpdated)) for event in h.events), [
            (type(event).__module__, type(event).__name__) for event in h.events
        ]
        assert len(auxiliary_submissions(h)) == 1
        assert all(s.context.stage_id == 0 for s in h.port.submissions)
        if metered:
            assert h.session.context_compute_budget(compute).usage(now=h.manager._clock()) == (
                len(prepared[0].prompt["prompt_token_ids"]),
                512,
            )
            assert not h.manager._auxiliary_request_permits
    finally:
        await close_harness(h)
