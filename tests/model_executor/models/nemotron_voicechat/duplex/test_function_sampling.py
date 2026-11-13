# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU sampling-state checks with scripted heads, without checkpoint inference.

The expected text policy follows NVIDIA-NeMo/Speech@097dfe9's
_apply_fc_state_machine and the subsequent active-call mask in infer_one_step.
"""

import pytest
import torch

from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration as Thinker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PAD, TEXT, SOTC, EOTC = 12, 17, 20, 21


class TextHead(torch.nn.Linear):
    def compute_logits(self, hidden):
        return self(hidden)


def sampler_shell(*, use_function_head=True):
    model = Thinker.__new__(Thinker)
    torch.nn.Module.__init__(model)
    model._use_function_head, model._dtype = use_function_head, torch.float32
    model.function_head = torch.nn.Identity()
    model.language_model = TextHead(32, 32)
    model.embed_tokens = torch.nn.Embedding(32, 32)
    with torch.no_grad():
        model.language_model.weight.zero_()
        model.language_model.bias.zero_()
        model.language_model.bias[TEXT] = 1
    model._w_text = model._w_audio = model._w_func = 1.0
    model._sessions = {
        name: {
            "nvc_text_pad_id": PAD,
            "nvc_function_sotc_id": SOTC,
            "nvc_function_eotc_id": EOTC,
            "function_call_active": False,
            "sampling_frame_eligible": True,
            "function_response_generation": 0,
            "forced_function_tokens": [],
        }
        for name in ("one", "two")
    }
    model._duplex_previous_text_tokens = {}
    model._duplex_sampling_rows = {}
    model._nvc_function_sampling_tokens = None
    return model


def sample_frame(model, functions, requests=("one",)):
    hidden = torch.nn.functional.one_hot(torch.tensor(functions), num_classes=32).to(torch.float32)
    envelope = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": name} for name in requests],
        request_token_spans=[(index, index + 1) for index in range(len(requests))],
    )
    logits = model.compute_logits(hidden)
    rows = tuple(
        DuplexSamplingRow(index, name, session_id=None, seq=None, payload=None, max_tokens=None)
        for index, name in enumerate(requests)
    )
    model.prepare_duplex_sampling(logits, None, rows)
    sampled = model.sample(logits, None)
    assert sampled is not None
    actual_functions = []
    for index, name in enumerate(requests):
        update = model.postprocess(
            hidden[index : index + 1], request_id=name, multimodal_outputs=envelope.multimodal_outputs
        )
        if model._use_function_head:
            actual_functions.append(int(update["nvc_prev_function_token"]))
        else:
            assert update == {}
    return sampled.sampled_token_ids.reshape(-1).tolist(), actual_functions


@pytest.mark.parametrize("raw_function,already_active", [(SOTC, False), (7, True)])
def test_active_function_call_pads_text_and_next_frame_token(raw_function, already_active):
    model = sampler_shell()
    model._sessions["one"]["function_call_active"] = already_active
    text, functions = sample_frame(model, [raw_function])
    assert text == [PAD]
    assert functions == [raw_function]
    assert model._duplex_previous_text_tokens["one"] == PAD
    assert model._sessions["one"]["function_call_active"] is True


def test_function_end_releases_text_and_restart_keeps_it_padded():
    model = sampler_shell()
    for raw, expected_text, active in (
        (SOTC, PAD, True),
        (7, PAD, True),
        (SOTC, PAD, True),
        (EOTC, TEXT, False),
        (7, TEXT, False),
    ):
        text, functions = sample_frame(model, [raw])
        assert text == [expected_text]
        assert functions == [raw]
        assert model._duplex_previous_text_tokens["one"] == expected_text
        assert model._sessions["one"]["function_call_active"] is active


@pytest.mark.parametrize("already_active", [False, True])
def test_forced_feedback_does_not_apply_raw_function_boundaries(already_active):
    model = sampler_shell()
    session = model._sessions["one"]
    session["function_call_active"] = already_active
    runtime = {
        "nvc_function_response_generation": 1,
        "nvc_function_response_batches": [{"generation": 1, "token_ids": [30, 31]}],
    }
    for raw, forced in ((EOTC, 30), (SOTC, 31)):
        model._sync_forced_function_response(session, runtime)
        text, functions = sample_frame(model, [raw])
        assert text == [PAD]
        assert functions == [forced]
        assert session["function_call_active"] is already_active
        assert model._duplex_previous_text_tokens["one"] == PAD
    assert session["forced_function_tokens"] == []
    model._sync_forced_function_response(session, runtime)
    text, functions = sample_frame(model, [EOTC])
    assert text == [TEXT]
    assert functions == [EOTC]
    assert session["function_call_active"] is False


def test_function_state_follows_request_identity_when_sampling_rows_move():
    model = sampler_shell()
    text, functions = sample_frame(model, [SOTC, 7], ("one", "two"))
    assert text == [PAD, TEXT]
    assert functions == [SOTC, 7]
    text, functions = sample_frame(model, [SOTC, EOTC], ("two", "one"))
    assert text == [PAD, TEXT]
    assert functions == [SOTC, EOTC]
    assert model._sessions["two"]["function_call_active"] is True
    assert model._sessions["one"]["function_call_active"] is False


@pytest.mark.parametrize("failure", ["missing_request", "duplicate_request", "missing_decisions"])
def test_invalid_sampling_batch_changes_no_request_policy(failure):
    model = sampler_shell()
    hidden = torch.nn.functional.one_hot(torch.tensor([SOTC, 7]), num_classes=32).to(torch.float32)
    logits = model.compute_logits(hidden)
    model._duplex_sampling_rows = {0: "one", 1: "two"}
    if failure == "missing_request":
        model._duplex_sampling_rows[1] = "unknown"
    elif failure == "duplicate_request":
        model._duplex_sampling_rows[1] = "one"
    else:
        model._nvc_function_sampling_tokens = None
    with pytest.raises(RuntimeError):
        model.sample(logits, None)
    assert model._sessions["one"]["function_call_active"] is False
    assert model._sessions["two"]["function_call_active"] is False
    assert model._duplex_previous_text_tokens == {}
    assert model._duplex_sampling_rows == {}
    assert model._nvc_function_sampling_tokens is None


def test_missing_sampling_rows_discard_the_forward_function_decisions():
    model = sampler_shell()
    hidden = torch.nn.functional.one_hot(torch.tensor([SOTC]), num_classes=32).to(torch.float32)
    logits = model.compute_logits(hidden)
    assert model.sample(logits, None) is None
    assert model._nvc_function_sampling_tokens is None
    assert model._sessions["one"]["function_call_active"] is False
    text, _ = sample_frame(model, [7])
    assert text == [TEXT]


@pytest.mark.parametrize("stale_forced_token", [None, 30])
def test_disabled_function_head_keeps_normal_text_sampling(stale_forced_token):
    model = sampler_shell(use_function_head=False)
    if stale_forced_token is not None:
        model._sessions["one"]["forced_function_token"] = stale_forced_token
    text, functions = sample_frame(model, [SOTC])
    assert text == [TEXT]
    assert functions == []  # postprocess has no function channel in this mode


@pytest.mark.parametrize("use_function_head", [False, True])
def test_duplex_open_requires_function_boundaries_only_for_an_enabled_head(monkeypatch, use_function_head):
    model = sampler_shell(use_function_head=use_function_head)
    model._sessions.clear()
    monkeypatch.setattr(model, "_duplex_stable_frame", lambda *_: torch.zeros(1, 32))
    runtime: dict[str, object] = {"nvc_text_pad_id": PAD, "nvc_prompt_token_ids": [0]}
    duplex = {"source_input_seq": 1, "runtime_config": runtime}
    if use_function_head:
        with pytest.raises(ValueError, match="function boundary"):
            model._preprocess_duplex(request_id="one", input_ids=torch.tensor([PAD]), info={}, duplex=duplex)
        assert model._sessions == {}
        runtime.update(nvc_function_sotc_id=SOTC, nvc_function_eotc_id=EOTC)
    model._preprocess_duplex(request_id="one", input_ids=torch.tensor([PAD]), info={}, duplex=duplex)
    assert model._sessions["one"]["function_call_active"] is False
    assert model._sessions["one"]["sampling_frame_eligible"] is False
    model._preprocess_duplex(
        request_id="one", input_ids=torch.tensor([PAD]), info={"_omni_num_computed_tokens": 1}, duplex=duplex
    )
    assert model._sessions["one"]["sampling_frame_eligible"] is True


@pytest.mark.parametrize("active,raw", [(False, SOTC), (True, EOTC)])
def test_intermediate_prefill_sampling_does_not_advance_function_state(active, raw):
    model = sampler_shell()
    session = model._sessions["one"]
    session["function_call_active"] = active
    session["sampling_frame_eligible"] = False
    previous = PAD if active else TEXT
    model._duplex_previous_text_tokens["one"] = previous
    hidden = torch.nn.functional.one_hot(torch.tensor([raw]), num_classes=32).to(torch.float32)
    logits = model.compute_logits(hidden)
    model.prepare_duplex_sampling(
        logits, None, (DuplexSamplingRow(0, "one", session_id=None, seq=None, payload=None, max_tokens=None),)
    )
    assert model.sample(logits, None) is not None
    assert session["function_call_active"] is active
    assert model._duplex_previous_text_tokens["one"] == previous
    assert model._nvc_function_sampling_tokens is None
    assert model._duplex_sampling_rows == {}


@pytest.mark.parametrize("scheduled_rows", [1, 3])
def test_padded_forward_and_selected_sampling_rows_make_the_same_function_decision(scheduled_rows):
    model = sampler_shell()
    model.function_head = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        model.function_head.weight.zero_()
        model.function_head.weight[SOTC, 0] = 1
        model.function_head.weight[EOTC, 1] = 1
    hidden = torch.zeros(scheduled_rows + 1, 32)
    hidden[scheduled_rows - 1, 0] = 1
    hidden[-1, 1] = 1
    envelope = model.make_omni_output(
        hidden, model_intermediate_buffer=[{"request_id": "one"}], request_token_spans=[(0, scheduled_rows)]
    )
    logits = model.compute_logits(hidden[scheduled_rows - 1 : scheduled_rows])
    model.prepare_duplex_sampling(
        logits, None, (DuplexSamplingRow(0, "one", session_id=None, seq=None, payload=None, max_tokens=None),)
    )
    sampled = model.sample(logits, None)
    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[PAD]]
    update = model.postprocess(
        hidden[:scheduled_rows], request_id="one", multimodal_outputs=envelope.multimodal_outputs
    )
    assert int(update["nvc_prev_function_token"]) == SOTC
    assert model._sessions["one"]["function_call_active"] is True


def test_finished_request_discards_its_function_policy_and_previous_text():
    model = sampler_shell()
    model._sessions["one"]["function_call_active"] = True
    model._duplex_previous_text_tokens["one"] = PAD
    model.on_requests_finished({"one"})
    assert "one" not in model._sessions
    assert "one" not in model._duplex_previous_text_tokens
    assert model._sessions["two"]["function_call_active"] is False
