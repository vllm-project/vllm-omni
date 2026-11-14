# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 ordinary-turn Thinker length stop through the real async update path.

``update_from_output`` and the upstream AsyncScheduler request update run for
real. schedule() increments, sampled tokens and target rows are inputs; KV
cache calls are recorded, not physical blocks. Each step ends at the first
publication or grammar hook, after the stop decision.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_EOS = 151645
_TTS_EOS = 151704


class _StepCompleteError(Exception):
    pass


@dataclass
class _Request:
    request_id: str = "turn-0"
    client_index: int = 0
    prompt_token_ids: list[int] = field(default_factory=lambda: [1, 2])
    output_token_ids: list[int] = field(default_factory=lambda: [11])
    num_computed_tokens: int = 2
    num_in_flight_tokens: int = 0
    num_output_placeholders: int = 0
    num_stale_output_tokens: int = 0
    status: RequestStatus = RequestStatus.RUNNING
    drop_stale_output: bool = False
    has_encoder_inputs: bool = False
    use_structured_output: bool = False
    pooling_params: object = None
    stop_reason: object = None
    resumable: bool = False
    max_tokens: int = 128
    spec_token_ids: list[int] = field(default_factory=list)
    sampling_params: object = field(
        default_factory=lambda: SimpleNamespace(
            num_logprobs=None, eos_token_id=_EOS, stop_token_ids=[_TTS_EOS], min_tokens=0, repetition_detection=None
        )
    )

    @property
    def num_prompt_tokens(self):
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + self.num_output_tokens

    @property
    def num_output_tokens(self):
        return len(self.output_token_ids)

    def append_output_token_ids(self, token):
        self.output_token_ids.append(token)

    def take_prefill_stats(self):
        return None

    def is_finished(self):
        return RequestStatus.is_finished(self.status)


def _vllm_config(*, model_arch, session_mode, speculative):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model_arch=model_arch,
            model_stage="llm",
            session_mode=session_mode,
            use_v2_model_runner=True,
            async_chunk=False,
            stage_id=0,
            engine_output_type="latent",
        ),
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=3) if speculative else None,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )


def _scheduler(
    monkeypatch, *, model_arch="MiniCPMO45OmniForConditionalGeneration", session_mode="turn", speculative=False
):
    scheduler = object.__new__(OmniARAsyncScheduler)
    request = _Request()
    cache_calls = []
    scheduler.requests = {request.request_id: request}
    scheduler.vllm_config = _vllm_config(model_arch=model_arch, session_mode=session_mode, speculative=speculative)
    scheduler.use_v2_model_runner = True
    scheduler.num_spec_tokens = 3 if speculative else 0
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.perf_metrics = None
    scheduler.chunk_transfer_adapter = None
    scheduler.aux_output_connector = None
    scheduler._new_prompt_len_snapshot = {}
    scheduler._hold_payloadless_outputs = False
    scheduler._native_data_plane = False
    scheduler.max_model_len = 4096
    scheduler._minicpmo45_terminal_rows = {}
    scheduler._talker_kstep_armed = lambda: False
    scheduler.make_spec_decoding_stats = lambda stats, **kwargs: stats
    scheduler._maybe_decode_pooling_output = lambda *args: None
    scheduler._process_kv_transfer_trigger = lambda *args: False
    scheduler.kv_cache_manager = SimpleNamespace(
        cache_blocks=lambda req, boundary: cache_calls.append(boundary),
        estimate_cached_tokens=lambda req: 0,
    )

    def complete(*args, **kwargs):
        raise _StepCompleteError

    # Both are reached only after this step's stop decision.
    monkeypatch.setattr(OmniSchedulerMixin, "_reject_invalid_grammar_tokens", complete)
    monkeypatch.setattr(OmniSchedulerMixin, "_append_request_output", complete)
    return scheduler, request


def _payload(token, position):
    return {
        "latent": torch.tensor([[float(token), float(position)]]),
        "latent_input_ids": torch.tensor([[token]]),
        "latent_positions": torch.tensor([[position]]),
    }


def _step(scheduler, request, drafts, sampled, payload=None):
    # schedule()'s optimistic async increments for one decode step.
    scheduled = len(drafts) + 1
    request.num_computed_tokens += scheduled
    request.num_in_flight_tokens += scheduled
    request.num_output_placeholders += scheduled
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request.request_id: scheduled},
        scheduled_spec_decode_tokens={request.request_id: list(drafts)} if drafts else {},
        num_invalid_spec_tokens=0,
        finished_req_ids=set(),
    )
    runner_output = SimpleNamespace(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[list(sampled)],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        # MRv2 full-payload outputs carry the rows on both channels.
        multimodal_outputs=[payload] if payload is not None else None,
        inter_stage_outputs=[payload] if payload is not None else None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
    )
    with pytest.raises(_StepCompleteError):
        scheduler.update_from_output(scheduler_output, runner_output)


def _unconsumed(request):
    # Tokens appended to the request that no target forward has taken as input
    # (nothing else is in flight in these tests).
    return request.num_tokens - request.num_computed_tokens


@pytest.mark.parametrize(
    "speculative,drafts,sampled",
    [(False, [], [12]), (True, [12, 99, 98], [12, 13]), (True, [12, 13, 14], [12, 13, 14, 15])],
    ids=["ordinary", "correction", "bonus"],
)
def test_length_stop_leaves_the_final_token_unconsumed(monkeypatch, speculative, drafts, sampled):
    """The gap: at the length limit no target forward has the final token as input."""
    scheduler, request = _scheduler(monkeypatch, speculative=speculative)
    request.max_tokens = 1 + len(sampled)
    _step(scheduler, request, drafts, sampled)
    assert request.output_token_ids == [11, *sampled]
    assert _unconsumed(request) == 1


@pytest.mark.parametrize(
    "speculative,drafts,sampled",
    [(False, [], [12]), (True, [12, 99, 98], [12, 13]), (True, [12, 13, 14], [12, 13, 14, 15])],
    ids=["ordinary", "correction", "bonus"],
)
def test_length_stop_consumes_the_final_token_before_finishing(monkeypatch, speculative, drafts, sampled):
    """Contract: one draft-free target forward takes the final token; its sample is dropped."""
    scheduler, request = _scheduler(monkeypatch, speculative=speculative)
    request.max_tokens = 1 + len(sampled)
    _step(scheduler, request, drafts, sampled)
    committed = [11, *sampled]
    assert not request.is_finished()
    assert request.spec_token_ids in ([], None)
    position = request.num_tokens - 1
    _step(scheduler, request, [], [999], _payload(committed[-1], position))
    assert request.output_token_ids == committed
    assert _unconsumed(request) == 0
    assert request.num_output_placeholders == 0
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize("change", ["id", "position", "absent", "extra-row"])
def test_terminal_forward_must_return_the_committed_own_row(monkeypatch, change):
    """A wrong or missing terminal row fails the request instead of a silent shorter reply."""
    scheduler, request = _scheduler(monkeypatch)
    request.max_tokens = 2
    _step(scheduler, request, [], [12])
    assert not request.is_finished()
    payload = _payload(12, 3)
    if change == "id":
        payload["latent_input_ids"] += 1
    elif change == "position":
        payload["latent_positions"] -= 1
    elif change == "absent":
        payload = None
    else:
        payload = {k: torch.cat([v, v]) for k, v in payload.items()}
    _step(scheduler, request, [], [999], payload)
    assert request.output_token_ids == [11, 12]
    assert request.status == RequestStatus.FINISHED_ERROR


@pytest.mark.parametrize("eos", [_EOS, _TTS_EOS])
def test_eos_stop_finishes_without_a_terminal_forward(monkeypatch, eos):
    scheduler, request = _scheduler(monkeypatch)
    _step(scheduler, request, [], [eos])
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_end_token_at_the_length_limit_stops_without_a_terminal_forward(monkeypatch):
    # Stage 0 stops at the turn end ids, and vLLM checks stop ids before the length.
    scheduler, request = _scheduler(monkeypatch)
    request.max_tokens = 2
    _step(scheduler, request, [], [_TTS_EOS])
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert scheduler._minicpmo45_terminal_rows == {}


def test_length_stop_after_an_end_token_outside_the_stop_list_ends_length_capped(monkeypatch):
    # Without the end ids in the stop list, the reply runs on past one; the
    # terminal forward still returns the final token's own row.
    scheduler, request = _scheduler(monkeypatch)
    request.sampling_params.stop_token_ids = []
    request.output_token_ids = [11, _TTS_EOS]
    request.num_computed_tokens = 3
    request.max_tokens = 3
    _step(scheduler, request, [], [12])
    assert not request.is_finished()
    _step(scheduler, request, [], [999], _payload(12, 4))
    assert request.output_token_ids == [11, _TTS_EOS, 12]
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler._minicpmo45_terminal_rows == {}


def test_length_limit_inside_accepted_drafts_finishes_without_a_terminal_forward(monkeypatch):
    # Drafts 12, 13 accepted, bonus 14 trimmed by the limit: 13 was an input.
    scheduler, request = _scheduler(monkeypatch, speculative=True)
    request.max_tokens = 3
    _step(scheduler, request, [12, 13, 98], [12, 13, 14])
    assert request.output_token_ids == [11, 12, 13]
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize(
    "setting",
    [{"model_arch": "Qwen3OmniMoeForConditionalGeneration"}, {"session_mode": "duplex"}],
    ids=["other-model", "duplex"],
)
def test_length_stop_is_unchanged_outside_ordinary_turn_minicpmo(monkeypatch, setting):
    scheduler, request = _scheduler(monkeypatch, **setting)
    request.max_tokens = 2
    _step(scheduler, request, [], [12])
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def _held(scheduler, request):
    scheduler.running = [request]
    scheduler.current_step = 7
    request.next_decode_eligible_step = 0
    held = scheduler._hold_minicpmo45_terminal_steps()
    return held, request.next_decode_eligible_step


def test_step_that_may_end_the_reply_holds_the_next_schedule(monkeypatch):
    # In flight: last input 11 plus three drafts; all accepted would reach 4 outputs.
    scheduler, request = _scheduler(monkeypatch, speculative=True)
    request.max_tokens = 5
    request.num_output_placeholders = 4
    held, eligible = _held(scheduler, request)
    assert held == [(request, 0)]
    assert eligible > scheduler.current_step + 1


def test_step_far_from_the_limit_is_not_held(monkeypatch):
    scheduler, request = _scheduler(monkeypatch, speculative=True)
    request.num_output_placeholders = 4
    held, eligible = _held(scheduler, request)
    assert held == [] and eligible == 0


def test_terminal_forward_is_draft_free_and_never_scheduled_twice(monkeypatch):
    scheduler, request = _scheduler(monkeypatch)
    request.max_tokens = 2
    _step(scheduler, request, [], [12])
    request.spec_token_ids = [-1, -1, -1]
    held, _ = _held(scheduler, request)
    assert held == [] and request.spec_token_ids == []
    # schedule() booked the terminal forward; it is in flight now.
    request.num_output_placeholders = 1
    request.spec_token_ids = [-1, -1, -1]
    held, eligible = _held(scheduler, request)
    assert held == [(request, 0)] and eligible > scheduler.current_step + 1
    assert request.spec_token_ids == []


def test_other_models_are_never_held(monkeypatch):
    scheduler, request = _scheduler(monkeypatch, model_arch="Qwen3OmniMoeForConditionalGeneration")
    request.max_tokens = 2
    request.num_output_placeholders = 4
    assert _held(scheduler, request) == ([], 0)


@pytest.mark.parametrize(
    "speculative,drafts,sampled",
    [(False, [], [12]), (True, [12, 99, 98], [12, 13])],
    ids=["ordinary", "correction"],
)
def test_context_limit_stop_keeps_the_length_finish(monkeypatch, speculative, drafts, sampled):
    """No position is left for the final token as an input: finish as before, with a warning."""
    scheduler, request = _scheduler(monkeypatch, speculative=speculative)
    scheduler.max_model_len = 2 + 1 + len(sampled)
    warnings, debugs = [], []
    monkeypatch.setattr(ar_sched_mod.logger, "warning_once", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(ar_sched_mod.logger, "debug", lambda *args, **kwargs: debugs.append(args))
    _step(scheduler, request, drafts, sampled)
    assert request.output_token_ids == [11, *sampled]
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler._minicpmo45_terminal_rows == {}
    assert [w for w in warnings if "max_model_len" in w[0]] != []
    assert [d[1] for d in debugs if "max_model_len" in d[0]] == [request.request_id]


def test_release_keeps_an_eligible_step_written_during_schedule(monkeypatch):
    scheduler, request = _scheduler(monkeypatch, speculative=True)
    request.max_tokens = 5
    request.num_output_placeholders = 4
    held, _ = _held(scheduler, request)
    scheduler._release_minicpmo45_terminal_holds(held)
    assert request.next_decode_eligible_step == 0
    held, _ = _held(scheduler, request)
    request.next_decode_eligible_step = 9
    scheduler._release_minicpmo45_terminal_holds(held)
    assert request.next_decode_eligible_step == 9


def test_resumable_session_keeps_the_existing_length_stop(monkeypatch):
    scheduler, request = _scheduler(monkeypatch)
    request.max_tokens = 2
    request.resumable = True
    _step(scheduler, request, [], [12])
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize(
    "ids,positions,rows,ok",
    [
        ([12], [3], 1, True),
        ([1, 2, 11, 12], [0, 1, 2, 3], 4, True),
        ([12, 99], [3, 4], 2, False),
        ([12], [3], 2, False),
        ([11], [3], 1, False),
        ([12, 12], [3, 3], 2, False),
    ],
    ids=["single", "recompute", "draft-after", "row-count", "wrong-id", "duplicate"],
)
def test_own_row_check_needs_consecutive_rows_ending_at_the_final_token(ids, positions, rows, ok):
    from vllm_omni.model_executor.models.minicpmo_4_5.handoff import ends_with_own_row

    payload = {
        "latent_input_ids": torch.tensor(ids).reshape(-1, 1),
        "latent_positions": torch.tensor(positions).reshape(-1, 1),
        "latent": torch.zeros(rows, 2),
    }
    assert ends_with_own_row(payload, 3, 12) is ok
