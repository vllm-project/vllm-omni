# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU sampling-state checks with scripted heads, without checkpoint inference.

The expected text policy follows NVIDIA-NeMo/Speech@097dfe9's
_apply_fc_state_machine and the subsequent active-call mask in infer_one_step.
"""

from types import SimpleNamespace

import numpy as np
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


def _mrv2_sampler(model, monkeypatch):
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    from vllm_omni.model_executor.models.nemotron_voicechat import nemotron_voicechat_thinker
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    # Only the Triton counts kernel is replaced: request-row sampling goes
    # through the real native GPUModelRunner.sample and registered sampler.
    def cpu_counts(counts, seq_lens, cu_num_logits, idx_mapping, prefill_len):
        partial = seq_lens < prefill_len[idx_mapping]
        counts.masked_fill_(partial, 0)
        return counts, (cu_num_logits[1:] - cu_num_logits[:-1] - counts).masked_fill(partial, 0)

    monkeypatch.setattr(nemotron_voicechat_thinker, "get_num_sampled_and_rejected", cpu_counts, raising=False)
    state = object.__new__(OmniModelState)
    state.model = model
    states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.tensor([2, 2, 2])))
    fallback_calls = []

    def stock(logits, batch):
        fallback_calls.append(tuple(batch.req_ids))
        counts, rejected = cpu_counts(
            torch.ones(batch.num_reqs, dtype=torch.int32),
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            states.prefill_len.gpu,
        )
        return SamplerOutput(logits.argmax(-1).unsqueeze(-1), None, None, counts, rejected)

    class StockSampler:
        req_states = states

        def __call__(self, logits, batch):
            return stock(logits, batch)

    stock_sampler = StockSampler()
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(session_mode="duplex"))
    custom = state.custom_sampler(stock_sampler)
    runner = object.__new__(OmniARModelRunner)
    runner.model_state, runner.model = state, model
    runner.sampler = custom[0] if custom else stock_sampler
    runner.rejection_sampler = None
    runner.batch_sharder = runner.pp_handler = None
    return runner, fallback_calls


def _mrv2_frame(model, runner, functions, requests=("one",), *, partial=False):
    hidden = torch.nn.functional.one_hot(torch.tensor(functions), num_classes=32).float()
    infos = [{"request_id": name} for name in requests]
    # V2 runs postprocess before sampling. A forced response must still be
    # visible to both this frame's wire projection and its policy decision.
    for i, info in enumerate(infos):
        model.postprocess(hidden[i : i + 1], **info)
    wire = model.make_omni_output(
        hidden,
        model_intermediate_buffer=infos,
        request_token_spans=[(i, i + 1) for i in range(len(requests))],
    )
    batch = SimpleNamespace(
        req_ids=list(requests),
        num_reqs=len(requests),
        num_draft_tokens=0,
        logits_indices=torch.arange(len(requests)),
        idx_mapping=torch.tensor(list(reversed(range(len(requests))))),
        idx_mapping_np=np.array(list(reversed(range(len(requests))))),
        seq_lens=torch.full((len(requests),), 1 if partial else 2, dtype=torch.int32),
        cu_num_logits=torch.arange(len(requests) + 1, dtype=torch.int32),
    )
    output, counts, rejected = runner.sample(hidden, batch, None)
    assert output.num_sampled is counts and output.num_rejected is rejected
    return output, wire


@pytest.mark.parametrize("use_function_head", [False, True])
def test_mrv2_duplex_sample_reaches_next_frame_fusion(monkeypatch, use_function_head):
    model = sampler_shell(use_function_head=use_function_head)
    model._sessions.clear()
    monkeypatch.setattr(model, "_duplex_stable_frame", lambda *_: torch.zeros(1, 32))
    runtime = {
        "nvc_text_pad_id": PAD,
        "nvc_prompt_token_ids": [0],
        "nvc_function_sotc_id": SOTC,
        "nvc_function_eotc_id": EOTC,
    }
    duplex = {"source_input_seq": 1, "runtime_config": runtime}
    model._preprocess_duplex(request_id="one", input_ids=torch.tensor([PAD, PAD]), info={}, duplex=duplex)
    runner, fallback = _mrv2_sampler(model, monkeypatch)
    output, _ = _mrv2_frame(model, runner, [7])
    # This continuation raises the exact CI error before V2 registration.
    _, fused, _ = model._preprocess_duplex(
        request_id="one",
        input_ids=torch.tensor([PAD]),
        info={"_omni_num_computed_tokens": 2},
        duplex=dict(duplex, source_input_seq=2),
    )
    expected_function = 7 if use_function_head else PAD
    expected = model._fuse(torch.tensor([TEXT]), torch.zeros(1, 32), torch.tensor([expected_function]))
    torch.testing.assert_close(fused, expected)
    assert output.sampled_token_ids.tolist() == [[TEXT]]
    assert output.num_sampled.tolist() == [1] and output.num_rejected.tolist() == [0]
    assert fallback == []


def test_mrv2_duplex_function_feedback_stays_on_the_same_request_and_frame(monkeypatch):
    model = sampler_shell()
    runner, fallback = _mrv2_sampler(model, monkeypatch)
    model._sessions["one"].update(forced_function_token=30, forced_function_tokens=[30, 31])
    output, wire = _mrv2_frame(model, runner, [EOTC, SOTC], ("one", "two"))
    assert output.sampled_token_ids.tolist() == [[PAD], [PAD]]
    assert [int(t) for t in wire.multimodal_outputs["nvc_function_token"]] == [30, SOTC]
    assert [model._sessions[r]["func_token"] for r in ("one", "two")] == [30, SOTC]
    assert model._sessions["one"]["forced_function_tokens"] == [31]
    assert model._sessions["two"]["function_call_active"] is True
    model._sessions["one"]["forced_function_token"] = 31
    output, wire = _mrv2_frame(model, runner, [EOTC, SOTC], ("two", "one"))
    assert output.sampled_token_ids.tolist() == [[TEXT], [PAD]]
    assert [int(t) for t in wire.multimodal_outputs["nvc_function_token"]] == [EOTC, 31]
    assert model._sessions["one"]["forced_function_tokens"] == []
    assert model._duplex_previous_text_tokens == {"one": PAD, "two": TEXT}
    assert model._sessions["two"]["function_call_active"] is False
    # A discarded partial prefill cannot consume a queued response or commit
    # sampled text/function state, even though native sampling sees its row.
    session = model._sessions["one"]
    session.update(sampling_frame_eligible=False, forced_function_token=29, forced_function_tokens=[29])
    output, wire = _mrv2_frame(model, runner, [SOTC], partial=True)
    assert "nvc_function_token" not in wire.multimodal_outputs
    assert output.num_sampled.tolist() == [0] and output.num_rejected.tolist() == [0]
    assert session["func_token"] == 31 and session["forced_function_tokens"] == [29]
    assert model._duplex_previous_text_tokens["one"] == PAD
    assert fallback == []


def test_mrv2_offline_requests_use_the_stock_sampler_after_duplex_cleanup(monkeypatch):
    model = sampler_shell()
    runner, fallback = _mrv2_sampler(model, monkeypatch)
    model.on_requests_finished({"one"})
    model._sessions["one"] = {}  # The same native slot now holds an offline request.
    output, _ = _mrv2_frame(model, runner, [7])
    assert output.sampled_token_ids.tolist() == [[TEXT]]
    assert fallback == [("one",)]
    assert model._duplex_previous_text_tokens == {}
    assert model._duplex_sampling_rows == {} and model._nvc_function_sampling_tokens is None


def test_mrv2_turn_mode_keeps_native_sampler_registration():
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    model = sampler_shell()
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(session_mode="turn"))
    state = object.__new__(OmniModelState)
    state.model = model
    assert state.custom_sampler(object()) is None
    assert not getattr(model, "_mrv2_duplex_sampling", False)
