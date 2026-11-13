# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streamed text (append_text) through the production AR scheduler and the real KV allocator.

Only the model configuration, the runner output and the clock are synthetic.
The stage uses the Qwen3-TTS text_stream_policy: a request with text lead k
waits for k + 1 ids before its prefill, and each later frame reads one id.
"""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.core.sched.output import TextStreamDelivery
from vllm_omni.core.sched.text_stream import TextStreams, load_text_stream_policy
from vllm_omni.core.sched.text_stream_mixin import restore_running_order
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

POLICY = "vllm_omni.model_executor.models.qwen3_tts.text_stream.text_stream_spec"
PROMPT_LEN = 4
BLOCK_SIZE = 16
TIMEOUT_S = 5.0
TEXT_VOCAB = 1000


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class _ModelConfig:
    """The ModelConfig fields this scheduler reads, for a stage with no checkpoint to load."""

    text_stream_policy: str | None
    hf_config: Qwen3TTSConfig
    max_model_len: int = 64
    runner_type: str = "generate"
    recompute_preemption: str = "allow"
    multimodal_config: None = None
    uses_mrope: bool = False
    is_encoder_decoder: bool = False
    is_diffusion: bool = False
    supports_multimodal_inputs: bool = False
    return_sampling_mask: bool = False


def _scheduler(
    *,
    max_num_seqs: int = 2,
    num_blocks: int = 8,
    policy: str | None = POLICY,
    test_clock: bool = True,
    scheduler_cls: type[OmniARScheduler] = OmniARScheduler,
    recompute_preemption: str = "allow",
):
    config = VllmConfig()
    config.model_config = _ModelConfig(  # type: ignore[assignment]
        text_stream_policy=policy,
        hf_config=Qwen3TTSConfig(talker_config={"text_vocab_size": TEXT_VOCAB}),
        recompute_preemption=recompute_preemption,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=64,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=64,
        is_encoder_decoder=False,
    )
    config.cache_config = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=False)
    config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(config)
    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.float16),
            )
        ],
    )
    scheduler = scheduler_cls(config, kv_config, Mock(spec=StructuredOutputManager), BLOCK_SIZE)
    clock = _Clock()
    if test_clock:
        # The scheduler's own state, with a clock the test moves.
        scheduler._text_streams = TextStreams(
            load_text_stream_policy(policy), TIMEOUT_S, model_config=config.model_config, clock=clock
        )
    return scheduler, clock


def _request(request_id: str, *, stream: bool = True, lead: int = 0, **info) -> Request:
    request = Request(request_id, list(range(PROMPT_LEN)), SamplingParams(max_tokens=32), None, client_index=0)
    if stream:
        request.additional_information = {
            "text_stream": [True],
            "task_type": ["CustomVoice"],
            "non_streaming_mode": [False],
            "text_lead": [lead],
            **info,
        }
    return request


def _output(scheduled) -> ModelRunnerOutput:
    ids = list(scheduled.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[[1] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _step(scheduler):
    scheduled = scheduler.schedule()
    outputs = scheduler.update_from_output(scheduled, _output(scheduled))
    return scheduled, outputs


def _free_blocks(scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


def _terminal_errors(outputs) -> dict[str, str]:
    return {
        output.request_id: output.stop_reason
        for client_outputs in outputs.values()
        for output in client_outputs.outputs
        if output.finish_reason == FinishReason.ERROR
    }


def test_a_request_with_no_text_yet_waits_before_its_prefill_and_holds_nothing():
    scheduler, _ = _scheduler()
    request = _request("r", lead=1)
    scheduler.add_request(request)
    free = _free_blocks(scheduler)

    scheduled, outputs = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}
    assert request in scheduler.waiting and request.status == RequestStatus.WAITING
    assert _free_blocks(scheduler) == free
    assert outputs == {} or not _terminal_errors(outputs)

    # A lead of 1 needs ids 0 and 1 before the prompt can be built.
    scheduler.append_text("r", [5], final=False)
    scheduled, _ = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}

    scheduler.append_text("r", [6, 7], final=False)
    scheduled, _ = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {"r": PROMPT_LEN}
    assert scheduled.text_stream_deliveries == {
        "r": TextStreamDelivery(start=0, token_ids=[5, 6, 7], ended=False, prompt_ids=2)
    }


def test_a_running_request_whose_next_text_is_missing_keeps_its_blocks_and_seat():
    scheduler, _ = _scheduler(max_num_seqs=1)
    request = _request("r", lead=0)
    scheduler.add_request(request)
    scheduler.append_text("r", [5, 6], final=False)
    _step(scheduler)  # prefill reads id 0, samples frame 0
    _step(scheduler)  # frame 1 reads id 1
    assert request.status == RequestStatus.RUNNING
    blocks = scheduler.kv_cache_manager.get_block_ids("r")
    free = _free_blocks(scheduler)

    # Frame 2 reads id 2, which has not arrived: the request runs no step and a
    # waiting request is not admitted into its seat.
    waiting = _request("w", stream=False)
    scheduler.add_request(waiting)
    scheduled, _ = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}
    assert scheduler.running == [request] and waiting in scheduler.waiting
    assert scheduler.kv_cache_manager.get_block_ids("r") == blocks and _free_blocks(scheduler) == free

    # The id arrives with the request's next scheduled step.
    scheduler.append_text("r", [7], final=False)
    scheduled, _ = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {"r": 1}
    assert scheduled.text_stream_deliveries == {
        "r": TextStreamDelivery(start=2, token_ids=[7], ended=False, prompt_ids=1)
    }


def test_after_the_end_a_request_runs_on_without_new_text():
    scheduler, _ = _scheduler(max_num_seqs=1)
    scheduler.add_request(_request("r"))
    scheduler.append_text("r", [5], final=True)
    for _ in range(4):
        scheduled, _ = _step(scheduler)
        assert scheduled.num_scheduled_tokens.get("r")


def test_held_requests_keep_their_place_in_running():
    scheduler, _ = _scheduler(max_num_seqs=3)
    requests = [_request(name) for name in ("a", "b", "c")]
    for request in requests:
        scheduler.add_request(request)
    scheduler.append_text("a", [1, 2, 3, 4], final=False)
    scheduler.append_text("b", [1], final=False)
    scheduler.append_text("c", [1, 2, 3, 4], final=False)
    _step(scheduler)  # all three prefill
    scheduled, _ = _step(scheduler)  # b is starved for frame 1 and held
    assert set(scheduled.num_scheduled_tokens) == {"a", "c"}
    assert [request.request_id for request in scheduler.running] == ["a", "b", "c"]


def test_parked_requests_keep_their_place_in_the_waiting_queue():
    """First come, first served: a request without text stays where it was, and so does a plain one before it."""
    scheduler, _ = _scheduler(max_num_seqs=1)
    scheduler.add_request(_request("busy", stream=False))
    _step(scheduler)  # takes the only seat
    waiting = [_request("plain", stream=False), _request("s1"), _request("s2")]
    for request in waiting:
        scheduler.add_request(request)
    for _ in range(2):
        _step(scheduler)
        assert list(scheduler.waiting) == waiting


def test_a_bad_deadline_setting_does_not_stop_a_stage_without_streamed_text(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_TEXT_STREAM_TIMEOUT_S", "soon")
    scheduler, _ = _scheduler(policy=None, test_clock=False)
    assert scheduler._text_streams.timeout_s == 0.0
    with pytest.raises(ValueError, match="VLLM_OMNI_TEXT_STREAM_TIMEOUT_S"):
        _scheduler(test_clock=False)


def test_restore_running_order_puts_held_requests_back_before_later_and_new_requests():
    a, b, c, d, new = (_request(n, stream=False) for n in "abcdn")
    # b and d were held; c was preempted by the step; new was admitted.
    assert restore_running_order([a, new], [b, d], [a, b, c, d]) == [a, b, d, new]
    assert restore_running_order([], [b], [a, b]) == [b]


def test_when_every_seat_waits_for_text_the_deadline_frees_them_and_a_waiting_request_runs():
    scheduler, clock = _scheduler(max_num_seqs=2)
    for name in ("s1", "s2"):
        scheduler.add_request(_request(name))
        scheduler.append_text(name, [1], final=False)
    _step(scheduler)  # both prefill on id 0; frame 1 needs id 1, which never comes
    free_before_hold = _free_blocks(scheduler)
    waiting = _request("w", stream=False)
    scheduler.add_request(waiting)

    scheduled, outputs = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}
    assert waiting in scheduler.waiting
    assert not _terminal_errors(outputs)

    clock.now += TIMEOUT_S + 0.1
    scheduled, outputs = _step(scheduler)
    errors = _terminal_errors(outputs)
    assert set(errors) == {"s1", "s2"}
    assert all("No text arrived" in reason for reason in errors.values())
    assert "s1" not in scheduler.requests and "s2" not in scheduler.requests
    assert "s1" not in scheduler._text_streams and "s2" not in scheduler._text_streams
    # Their blocks are back (one each), and the waiting request took a freed seat
    # (and one block) in the same step.
    assert scheduled.num_scheduled_tokens == {"w": PROMPT_LEN}
    assert _free_blocks(scheduler) == free_before_hold + 2 - 1


def test_the_hold_counts_frames_still_in_flight_under_async_scheduling():
    """Off-by-one guard: a step scheduled before its predecessor's output returns reads one row further."""
    scheduler, _ = _scheduler(max_num_seqs=1)
    request = _request("r")
    scheduler.add_request(request)
    scheduler.append_text("r", [5, 6], final=False)
    _step(scheduler)  # prefill: frame 0 sampled from id 0
    assert scheduler._text_stream_frame(request) == 1
    # Frame 1 (reads id 1) is scheduled but its output has not come back.
    request.num_computed_tokens += 1
    request.num_output_placeholders = 1
    assert scheduler._get_confirmed_num_computed_tokens(request) == PROMPT_LEN
    # Counting confirmed tokens alone would say frame 1 again, whose id is here;
    # the next step samples frame 2 and reads id 2, which is not.
    assert scheduler._text_stream_frame(request) == 2
    assert scheduler._hold_text_starved_running() == [request]
    scheduler.running.append(request)
    scheduler.append_text("r", [7], final=False)
    assert scheduler._hold_text_starved_running() == []


def test_the_async_scheduler_holds_a_step_whose_row_has_not_arrived_while_earlier_steps_are_in_flight():
    """The Talker stage's default scheduler schedules a step before the previous step's output returns."""
    scheduler, _ = _scheduler(max_num_seqs=1, scheduler_cls=OmniARAsyncScheduler)
    request = _request("r")
    scheduler.add_request(request)
    scheduler.append_text("r", [5, 6], final=False)
    assert "r" in scheduler.schedule().num_scheduled_tokens  # prefill: frame 0 reads id 0
    assert "r" in scheduler.schedule().num_scheduled_tokens  # frame 1 reads id 1; frame 0 is still in flight
    assert request.num_output_placeholders == 2
    # Frame 2 reads id 2, which has not arrived: no step, and the request keeps its seat.
    assert "r" not in scheduler.schedule().num_scheduled_tokens
    assert scheduler.running == [request]
    scheduler.append_text("r", [7], final=False)
    assert "r" in scheduler.schedule().num_scheduled_tokens


@pytest.mark.parametrize(
    ("policy", "info", "reason"),
    [
        (None, {}, "text_stream_policy"),
        (POLICY, {"non_streaming_mode": [True]}, "streaming mode"),
        (POLICY, {"task_type": ["VoiceDesign"]}, "only for CustomVoice"),
        (POLICY, {"text_lead": [-1]}, "text_lead"),
    ],
)
def test_a_request_that_cannot_stream_is_refused_at_admission(policy, info, reason):
    scheduler, _ = _scheduler(policy=policy)
    scheduler.add_request(_request("r", **info))
    scheduled, outputs = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}
    assert reason in _terminal_errors(outputs)["r"]
    assert "r" not in scheduler.requests


def test_multi_frame_talker_decode_refuses_streamed_text(monkeypatch):
    scheduler, _ = _scheduler()
    monkeypatch.setattr(scheduler, "_talker_kstep_armed", lambda: True)
    scheduler.add_request(_request("r"))
    _, outputs = _step(scheduler)
    assert "multi-frame" in _terminal_errors(outputs)["r"]


def test_a_streaming_input_session_cannot_stream_its_text():
    # Each session update re-prefills the request, which a streamed text cannot replay (see _preempt_request).
    scheduler, _ = _scheduler()
    request = _request("r")
    request.resumable = True
    scheduler.add_request(request)
    scheduled, outputs = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}
    assert "streaming input session" in _terminal_errors(outputs)["r"]
    assert "r" not in scheduler.requests and "r" not in scheduler._text_streams


def test_a_text_that_ends_shorter_than_its_lead_allows_fails_before_its_prefill():
    scheduler, _ = _scheduler()
    scheduler.add_request(_request("r", lead=4))  # the lead rows hold at least 3 ids and the END
    scheduler.append_text("r", [5, 6], final=True)
    scheduled, outputs = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}
    assert "needs at least 3" in _terminal_errors(outputs)["r"]


def test_an_append_that_breaks_a_live_stream_ends_the_request_with_its_reason():
    scheduler, _ = _scheduler()
    scheduler.add_request(_request("ended"))
    scheduler.add_request(_request("plain", stream=False))
    scheduler.append_text("ended", [5], final=True)
    with pytest.raises(ValueError, match="already ended"):
        scheduler.append_text("ended", [6], final=False)
    with pytest.raises(ValueError, match="no open text stream"):
        scheduler.append_text("plain", [6], final=False)
    _, outputs = _step(scheduler)
    errors = _terminal_errors(outputs)
    assert "already ended" in errors["ended"] and "no open text stream" in errors["plain"]
    # An append for a request that is gone changes nothing.
    with pytest.raises(ValueError, match="not running"):
        scheduler.append_text("ended", [7], final=False)


def test_an_id_outside_the_text_vocabulary_never_reaches_the_model():
    scheduler, _ = _scheduler()
    scheduler.add_request(_request("r"))
    with pytest.raises(ValueError, match=f"below {TEXT_VOCAB}"):
        scheduler.append_text("r", [5, TEXT_VOCAB], final=False)
    scheduled, outputs = _step(scheduler)
    assert scheduled.num_scheduled_tokens == {}
    assert f"below {TEXT_VOCAB}" in _terminal_errors(outputs)["r"]


def test_an_aborted_held_request_frees_its_blocks_and_its_stream():
    scheduler, _ = _scheduler(max_num_seqs=1)
    scheduler.add_request(_request("r"))
    scheduler.append_text("r", [5], final=False)
    _step(scheduler)
    free = _free_blocks(scheduler)
    _step(scheduler)  # held: frame 1 needs id 1
    scheduler.finish_requests("r", RequestStatus.FINISHED_ABORTED)
    assert "r" not in scheduler._text_streams
    assert _free_blocks(scheduler) > free


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
@pytest.mark.parametrize("recompute_preemption", ["allow", "fail"])
def test_a_preempted_streamed_request_ends_with_an_error_instead_of_replaying(recompute_preemption, scheduler_cls):
    # One usable block: the request's decode outgrows it mid-stream and the scheduler preempts it. A replay would
    # restart from the token stream, which holds none of the streamed text rows.
    scheduler, _ = _scheduler(
        max_num_seqs=1, num_blocks=2, recompute_preemption=recompute_preemption, scheduler_cls=scheduler_cls
    )
    request = _request("r")
    scheduler.add_request(request)
    scheduler.append_text("r", list(range(1, 30)), final=False)
    outputs: dict = {}
    for _ in range(20):
        _, outputs = _step(scheduler)
        if request.is_finished() or request in scheduler.waiting:
            break
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request not in scheduler.waiting and "r" not in scheduler.requests
    assert "r" not in scheduler._text_streams
    assert _free_blocks(scheduler) == 1
    [terminal] = outputs[0].outputs
    assert terminal.finish_reason == FinishReason.ERROR
    assert "streamed text cannot be replayed" in terminal.stop_reason


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
def test_a_prefix_cache_reset_ends_a_running_streamed_request_instead_of_replaying_it(scheduler_cls):
    # The Qwen3-TTS Talker sets recompute_preemption="fail", but a reset of running requests replays them anyway.
    scheduler, _ = _scheduler(max_num_seqs=1, recompute_preemption="fail", scheduler_cls=scheduler_cls)
    request = _request("r")
    scheduler.add_request(request)
    scheduler.append_text("r", [5, 6, 7], final=False)
    _step(scheduler)
    _step(scheduler)
    assert request.status == RequestStatus.RUNNING
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request not in scheduler.waiting and "r" not in scheduler._text_streams
    _, outputs = _step(scheduler)
    [terminal] = outputs[0].outputs
    assert terminal.finish_reason == FinishReason.ERROR and "streamed text cannot be replayed" in terminal.stop_reason


def test_requests_without_a_stream_never_reach_the_text_stream_rules(monkeypatch):
    scheduler, _ = _scheduler()

    def fail(*_args, **_kwargs):
        raise AssertionError("text stream rule called for a request without a stream")

    monkeypatch.setattr(TextStreams, "ready", fail)
    monkeypatch.setattr(TextStreams, "take_delivery", fail)
    scheduler.add_request(_request("plain", stream=False))
    for _ in range(3):
        scheduled, _ = _step(scheduler)
        assert scheduled.text_stream_deliveries == {}
    assert not scheduler._text_streams
