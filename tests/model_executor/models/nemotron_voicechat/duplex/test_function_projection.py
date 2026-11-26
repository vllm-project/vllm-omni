# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks for scheduled function rows; no model or CUDA execution."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_voicechat.duplex.data_plane import NemotronVoiceChatDataPlaneSession
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration as Thinker,
)
from vllm_omni.utils.mm_outputs import build_mm_cpu
from vllm_omni.worker.output.payload_build import build_omni_mm_payload
from vllm_omni.worker.sparse_audio import resolve_sparse_mm_routing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def thinker_shell(*, forced=None):
    thinker = Thinker.__new__(Thinker)
    torch.nn.Module.__init__(thinker)
    thinker._use_function_head, thinker._dtype = True, torch.float32
    thinker.function_head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        thinker.function_head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    thinker._sessions = {"one": {}, "other": {}}
    if forced is not None:
        thinker._sessions["one"].update(forced_function_token=forced, forced_function_tokens=[forced])
    return thinker


@pytest.mark.parametrize("valid_rows", [1, 3])
@pytest.mark.parametrize("forced", [None, 99])
def test_function_projection_ignores_trailing_padding(valid_rows, forced):
    thinker = thinker_shell(forced=forced)
    hidden = torch.zeros(valid_rows + 1, 2)
    hidden[valid_rows - 1] = torch.tensor([1.0, 0.0])
    hidden[-1] = torch.tensor([0.0, 1.0])
    envelope = thinker.make_omni_output(
        hidden, model_intermediate_buffer=[{"request_id": "one"}], request_token_spans=[(0, valid_rows)]
    )
    # The same real linear head chooses 0 on the scheduled row and 1 on
    # padding. The projection must never turn the padding row into a call.
    expected = 0 if forced is None else forced
    assert int(envelope.multimodal_outputs["nvc_function_token"]) == expected
    output = thinker.postprocess(hidden[:valid_rows], request_id="one", multimodal_outputs=envelope.multimodal_outputs)
    assert int(output["nvc_prev_function_token"]) == expected


@pytest.mark.parametrize("spans", [None, [], [(0, 3)], [(1, 1)], [(True, 1)], ["invalid"]])
@pytest.mark.parametrize("forced", [None, 99])
def test_missing_or_invalid_span_uses_request_local_fallback(spans, forced):
    thinker = thinker_shell(forced=forced)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kwargs = {"model_intermediate_buffer": [{"request_id": "one"}]}
    if spans is not None:
        kwargs["request_token_spans"] = spans
    envelope = thinker.make_omni_output(hidden, **kwargs)
    assert "nvc_function_token" not in envelope.multimodal_outputs
    output = thinker.postprocess(hidden[:1], request_id="one", multimodal_outputs=envelope.multimodal_outputs)
    assert int(output["nvc_prev_function_token"]) == (0 if forced is None else forced)


def test_multi_request_projection_keeps_each_local_row():
    thinker = thinker_shell()
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    envelope = thinker.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "one"}, {"request_id": "other"}],
        request_token_spans=[(0, 1), (1, 2)],
    )
    tokens = envelope.multimodal_outputs["nvc_function_token"]
    assert [int(token) for token in tokens] == [0, 1]
    for index, request_id in enumerate(["one", "other"]):
        output = thinker.postprocess(
            hidden[index : index + 1], request_id=request_id, multimodal_outputs=envelope.multimodal_outputs
        )
        assert int(output["nvc_prev_function_token"]) == index


def test_intermediate_prefill_does_not_publish_or_consume_a_forced_token():
    thinker = thinker_shell(forced=99)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    info = {
        "request_id": "one",
        "_omni_is_prefill": True,
        "_omni_num_computed_tokens": 0,
        "_omni_prompt_len": 4,
    }
    envelope = thinker.make_omni_output(hidden, model_intermediate_buffer=[info], request_token_spans=[(0, 2)])
    assert "nvc_function_token" not in envelope.multimodal_outputs
    assert thinker._sessions["one"]["forced_function_token"] == 99

    final_info = dict(info, _omni_num_computed_tokens=2)
    final = thinker.make_omni_output(hidden, model_intermediate_buffer=[final_info], request_token_spans=[(0, 2)])
    assert int(final.multimodal_outputs["nvc_function_token"]) == 99
    assert thinker._sessions["one"]["forced_function_token"] == 99


def test_mixed_prefill_keeps_the_other_requests_token():
    thinker = thinker_shell()
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    envelope = thinker.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {
                "request_id": "one",
                "_omni_is_prefill": True,
                "_omni_num_computed_tokens": 0,
                "_omni_prompt_len": 8,
            },
            {"request_id": "other"},
        ],
        request_token_spans=[(0, 1), (1, 2)],
    )
    tokens = envelope.multimodal_outputs["nvc_function_token"]
    assert tokens[0].numel() == 0
    assert int(tokens[1]) == 1


def test_single_row_legacy_projection_is_unambiguous():
    thinker = thinker_shell()
    envelope = thinker.make_omni_output(torch.tensor([[1.0, 0.0]]), model_intermediate_buffer=[{"request_id": "one"}])
    assert int(envelope.multimodal_outputs["nvc_function_token"]) == 0


def request_payload(envelope, request_ids, spans, index, *, combined=False):
    start, end = spans[index]
    mm_cpu = build_mm_cpu(envelope.multimodal_outputs)
    _, sparse_index, sparse_audio = resolve_sparse_mm_routing(
        engine_output_type="audio",
        req_ids_output_copy=list(request_ids),
        downstream_req_ids=list(request_ids),
        multimodal_outputs=envelope.multimodal_outputs,
    )
    assert sparse_audio is False  # This Thinker never declares sparse audio.
    return build_omni_mm_payload(
        combined_multimodal_outputs={key: {rid: value for rid in request_ids} for key, value in mm_cpu.items()}
        if combined
        else None,
        mm_cpu=mm_cpu,
        rid=request_ids[index],
        idx=index,
        start=start,
        end=end,
        audio_sparse_output=sparse_audio,
        sparse_mm_index=sparse_index,
        hidden_seq_len=len(envelope.text_hidden_states),
        scheduled_seq_len=spans[-1][1],
    )


def client_events(projector, request_id, payload):
    output = SimpleNamespace(
        stage_id=0,
        request_output=SimpleNamespace(request_id=request_id, outputs=[SimpleNamespace(multimodal_output=payload)]),
    )
    return list(projector.project_output(output))


@pytest.mark.parametrize("first_rows", [1, 3])
@pytest.mark.parametrize("request_ids", [("one", "other"), ("other", "one")])
@pytest.mark.parametrize("combined", [False, True])
def test_batched_function_boundaries_reach_each_data_plane(first_rows, request_ids, combined):
    thinker = thinker_shell()
    spans = [(0, first_rows), (first_rows, first_rows + 1)]
    projectors = {request_id: NemotronVoiceChatDataPlaneSession(lambda *_: None) for request_id in request_ids}
    calls: dict[str, list[dict[str, object]]] = {request_id: [] for request_id in request_ids}
    for request_id, projector in projectors.items():
        projector.configure_runtime(
            {
                "nvc_text_bos_id": 10,
                "nvc_text_eos_id": 11,
                "nvc_text_pad_id": 12,
                "nvc_function_sotc_id": 0,
                "nvc_function_eotc_id": 1,
                "nvc_tokenizer_ref": "test",
            },
            tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: ""),
        )
        projector._decode = lambda ids, name=request_id: (
            '[{"name":"' + name + '","arguments":{"value":37}}]' if ids == [2] else ""
        )
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    # Staggered calls expose cross-request starts, bodies and ends.
    for frame, decisions in enumerate(((0, 2), (2, 0), (1, 2), (2, 1))):
        hidden = torch.zeros(first_rows + 2, 2)
        hidden[first_rows - 1], hidden[first_rows] = vectors[list(decisions)]
        hidden[-1] = vectors[1]  # Padding must never close either call.
        envelope = thinker.make_omni_output(
            hidden, model_intermediate_buffer=[{"request_id": rid} for rid in request_ids], request_token_spans=spans
        )
        for index, request_id in enumerate(request_ids):
            payload = request_payload(envelope, request_ids, spans, index, combined=combined)
            update = thinker.postprocess(
                hidden[slice(*spans[index])], request_id=request_id, multimodal_outputs=envelope.multimodal_outputs
            )
            assert int(update["nvc_prev_function_token"]) == decisions[index]
            assert "nvc_function_token" in thinker.omni_client_multimodal_output_keys
            assert int(payload["nvc_function_token"]) == decisions[index]
            calls[request_id].extend(
                event
                for event in client_events(projectors[request_id], request_id, payload)
                if event.get("function_call") is True
            )
        assert [len(calls[rid]) for rid in request_ids] == ([0, 0] if frame < 2 else [1, frame - 2])
    for request_id in request_ids:
        assert calls[request_id][0]["name"] == request_id
        assert calls[request_id][0]["arguments"] == '{"value":37}'
    assert calls[request_ids[0]][0]["call_id"] != calls[request_ids[1]][0]["call_id"]


@pytest.mark.parametrize("forced", [None, 99])
def test_intermediate_prefill_does_not_publish_or_consume_function_token(forced):
    thinker = thinker_shell(forced=forced)
    info = {"request_id": "one", "_omni_is_prefill": True, "_omni_num_computed_tokens": 0, "_omni_prompt_len": 4}
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    spans = [(0, 2)]
    projector = NemotronVoiceChatDataPlaneSession(lambda *_: None)
    projector.configure_runtime(
        {
            "nvc_text_bos_id": 10,
            "nvc_text_eos_id": 11,
            "nvc_text_pad_id": 12,
            "nvc_function_sotc_id": 0,
            "nvc_function_eotc_id": 1 if forced is None else forced,
            "nvc_tokenizer_ref": "test",
        },
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: ""),
    )
    projector._decode = lambda ids: '[{"name":"one","arguments":{}}]' if ids == [2] else ""
    for function_id in (0, 2):
        assert client_events(projector, "one", {"nvc_function_token": torch.tensor([function_id])}) == []
    for computed, expected in ((0, None), (2, 1 if forced is None else forced)):
        info["_omni_num_computed_tokens"] = computed
        envelope = thinker.make_omni_output(hidden, model_intermediate_buffer=[info], request_token_spans=spans)
        payload = request_payload(envelope, ["one"], spans, 0)
        token = payload.get("nvc_function_token")
        if expected is None:
            assert token is None or token.numel() == 0
            assert client_events(projector, "one", payload) == []
        else:
            assert int(token) == expected
            events = client_events(projector, "one", payload)
            assert len(events) == 1 and events[0].get("function_call") is True
        update = thinker.postprocess(hidden[:2], multimodal_outputs=envelope.multimodal_outputs, **info)
        if expected is None:
            assert update == {}
            if forced is not None:
                assert thinker._sessions["one"]["forced_function_tokens"] == [forced]
        else:
            assert int(update["nvc_prev_function_token"]) == expected
            assert thinker._sessions["one"].get("forced_function_tokens", []) == []
    info["_omni_is_prefill"] = False
    envelope = thinker.make_omni_output(hidden[:1], model_intermediate_buffer=[info], request_token_spans=[(0, 1)])
    payload = request_payload(envelope, ["one"], [(0, 1)], 0)
    assert int(payload["nvc_function_token"]) == 0  # The forced token is not emitted again.


@pytest.mark.parametrize("request_ids", [("one", "other"), ("other", "one")])
def test_mixed_prefill_decode_keeps_empty_slots_and_forced_ownership(request_ids):
    thinker = thinker_shell(forced=99)
    thinker._sessions["other"].update(forced_function_token=98, forced_function_tokens=[98])
    infos, spans, hidden_parts = [], [], []
    cursor = 0
    for request_id in request_ids:
        rows = 2 if request_id == "one" else 1
        infos.append(
            {
                "request_id": request_id,
                "_omni_is_prefill": request_id == "one",
                "_omni_num_computed_tokens": 0,
                "_omni_prompt_len": 4,
            }
        )
        spans.append((cursor, cursor + rows))
        cursor += rows
        hidden_parts.append(torch.ones(rows, 2))
    hidden = torch.cat([*hidden_parts, torch.zeros(1, 2)])
    envelope = thinker.make_omni_output(hidden, model_intermediate_buffer=infos, request_token_spans=spans)
    for index, info in enumerate(infos):
        request_id = info["request_id"]
        payload = request_payload(envelope, request_ids, spans, index)
        update = thinker.postprocess(
            hidden[slice(*spans[index])], multimodal_outputs=envelope.multimodal_outputs, **info
        )
        if request_id == "one":
            assert payload["nvc_function_token"].numel() == 0
            assert update == {}
            assert thinker._sessions["one"]["forced_function_tokens"] == [99]
        else:
            assert int(payload["nvc_function_token"]) == int(update["nvc_prev_function_token"]) == 98
            assert thinker._sessions["other"]["forced_function_tokens"] == []
