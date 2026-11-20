# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native resident-slot reuse while request-owned synthesis stays live."""

import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.model_executor.models.yue2.model_state import Yue2ModelState
from vllm_omni.model_executor.models.yue2.runtime import Yue2ForCausalLM, Yue2SchedulerBase, Yue2SchedulerOutput
from vllm_omni.model_executor.models.yue2.worker import Yue2ModelRunner
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.staged_writes import StagedWrites

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _schedule(**kwargs):
    fields = vars(SchedulerOutput.make_empty()) | kwargs
    return Yue2SchedulerOutput(**fields)


@pytest.fixture
def runner(mocker):
    runner = object.__new__(Yue2ModelRunner)
    runner.model = mocker.Mock(spec=Yue2ForCausalLM, acoustic_started=[])
    runner.model.poll_audio.return_value = {}
    runner.model_state = object.__new__(Yue2ModelState)
    runner.model_state.model = runner.model
    runner.model_state.intermediate_buffer = OmniIntermediateBuffer(4)
    runner.model_state._mtp_state = mocker.Mock()
    # Keep native add/remove and buffers' identity; only device allocation is stubbed.
    states = object.__new__(RequestState)
    states.req_id_to_index, states.index_to_req_id = {}, {}
    states.free_indices = list(range(4))
    states.num_speculative_steps = 0
    states.max_seq_len = np.zeros(4, dtype=np.int32)
    states.num_computed_prefill_tokens = np.zeros(4, dtype=np.int32)
    states.num_computed_tokens_np = np.zeros(4, dtype=np.int32)
    for name in ("prompt_len", "prefill_len"):
        setattr(states, name, SimpleNamespace(np=np.zeros(4, dtype=np.int32)))
    for name in ("total_len", "all_token_ids", "num_computed_tokens"):
        setattr(states, name, mocker.Mock())
    runner.req_states = states
    runner._staged_writes = StagedWrites()
    runner._mm_rng_state = None
    runner.pooling_runner = runner.pp_handler = runner.encoder_cache = None
    runner.prompt_logprobs_worker = runner.speculator = None
    runner.draft_tokens_handler = set()
    runner.lora_state = mocker.Mock()
    return runner


def _admit(runner, rid):
    runner.req_states.add_request(rid, 2, [1, 2], 0, 20)
    idx = runner.req_states.req_id_to_index[rid]
    data = NewRequestData(
        req_id=rid,
        prompt_token_ids=[1, 2],
        prefill_token_ids=[1, 2],
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
    )
    runner.model_state.intermediate_buffer.add_request(idx, data)
    runner.draft_tokens_handler.add(rid)
    return idx


@pytest.mark.parametrize("slots", [1, 4])
def test_native_slots_reused_while_old_audio_finishes(runner, slots):
    old = [f"old-{i}" for i in range(slots)]
    indices = {_admit(runner, rid) for rid in old}
    runner.finish_requests(_schedule(acoustic_req_ids=set(old)))
    assert not runner.req_states.req_id_to_index
    assert not runner.model_state.intermediate_buffer.req_id_to_index
    runner.model.on_requests_finished.assert_not_called()
    assert not runner.draft_tokens_handler
    new = [f"new-{i}" for i in range(slots)]
    assert {_admit(runner, rid) for rid in new} == indices
    runner.finish_requests(_schedule(acoustic_req_ids=set(old)))
    assert set(runner.req_states.req_id_to_index) == set(new)
    assert len(set(runner.req_states.free_indices)) == 4 - slots
    ready = {rid: {"audio": torch.full((2, 3), i)} for i, rid in enumerate(old)}
    runner.model.poll_audio.return_value = ready
    native = ModelRunnerOutput(req_ids=new, sampled_token_ids=[[7]] * slots)
    native.future_metadata = object()
    output = runner._with_audio(native)
    assert output.req_ids == new and output.sampled_token_ids == [[7]] * slots
    assert output.acoustic_ready is ready
    assert output.future_metadata is native.future_metadata
    assert not hasattr(native, "acoustic_ready")
    runner.finish_requests(_schedule(finished_req_ids=set(old)))
    runner.model.on_requests_finished.assert_called_once_with(set(old))
    assert set(runner.req_states.req_id_to_index) == set(new)


@pytest.mark.parametrize("transition", ["preempt", "replace", "abort"])
def test_retirement_cleanup_distinguishes_phase_and_history(runner, transition):
    slot = _admit(runner, "song")
    fields = {
        "preempt": {"preempted_req_ids": {"song"}},
        "replace": {"replaced_req_ids": {"song"}},
        "abort": {"finished_req_ids": {"song"}, "acoustic_req_ids": {"song"}},
    }
    schedule = _schedule(**fields[transition])
    original = schedule.finished_req_ids.copy()
    runner.finish_requests(schedule)
    assert schedule.finished_req_ids == original
    assert not runner.req_states.req_id_to_index
    if transition == "preempt":
        runner.model.on_requests_finished.assert_not_called()
        runner.model_state._mtp_state.suspend_audio.assert_called_once_with("song", slot)
    else:
        runner.model.on_requests_finished.assert_called_once_with({"song"})
    if transition != "abort":
        assert _admit(runner, "song") == slot


@pytest.mark.parametrize("admit_pair", [False, True])
def test_waiting_phase_replacements_release_slots_before_cfg_admission(runner, mocker, admit_pair):
    scheduler = object.__new__(Yue2SchedulerBase)
    scheduler.requests = {rid: SimpleNamespace(yue2_reset_phase=rid in {"a", "b"}) for rid in ("a", "b", "c", "d")}
    for rid in scheduler.requests:
        _admit(runner, rid)
    scheduler.requests["a-negative"] = SimpleNamespace(yue2_reset_phase=False)
    scheduled = {"c": 1, "d": 1} | ({"a": 2, "a-negative": 2} if admit_pair else {})
    native = OmniSchedulerOutput(**(vars(SchedulerOutput.make_empty()) | {"num_scheduled_tokens": scheduled}))
    mocker.patch.object(OmniARScheduler, "schedule", return_value=native)

    schedule = scheduler.schedule()
    runner.finish_requests(schedule)
    if admit_pair:
        _admit(runner, "a")
        _admit(runner, "a-negative")
    assert schedule.replaced_req_ids == {"a", "b"}
    assert "b" not in runner.req_states.req_id_to_index
    assert set(runner.req_states.req_id_to_index) == set(scheduled)
    runner.model.on_requests_finished.assert_called_once_with({"a", "b"})
    # A deferred semantic request must not clear a newly admitted phase later.
    assert not scheduler.schedule().replaced_req_ids
    runner.finish_requests(_schedule(finished_req_ids={"a", "a-negative"}))
    _admit(runner, "b")
    _admit(runner, "b-negative")
    assert len(runner.req_states.req_id_to_index) == 4


def test_parent_execute_retires_before_admission_on_zero_token_tick(runner, mocker):
    slot = _admit(runner, "a")
    # Execute the real shared parent through cleanup/admission/no-forward.
    for name in (
        "_prefix_cache_step_begin",
        "_prepare_native_data_plane",
        "free_states",
        "update_requests",
        "_apply_encoder_input_updates",
        "_sync_native_data_plane_payloads",
    ):
        setattr(runner, name, mocker.Mock())
    runner.add_requests = lambda _: _admit(runner, "b")
    runner.block_tables = mocker.Mock()
    runner.aux_output_connector = None
    native = ModelRunnerOutput(req_ids=[], sampled_token_ids=[])
    runner.kv_connector = mocker.Mock(no_forward=lambda _: native)
    runner._merge_ec_connector_no_forward = lambda schedule, output: output
    runner._attach_native_data_plane_signals = lambda output: output
    result = OmniGPUModelRunner.execute_model(runner, _schedule(acoustic_req_ids={"a"}))
    assert result is native
    assert runner.req_states.req_id_to_index == {"b": slot}
    runner.model.on_requests_finished.assert_not_called()


@pytest.mark.parametrize("mode", ["idle", "ar", "dummy", "profile"])
def test_progress_once_at_real_step_boundary_and_snapshot_output(runner, mocker, mode):
    native = ModelRunnerOutput(req_ids=[], sampled_token_ids=[])
    mocker.patch.object(OmniARModelRunner, "execute_model", return_value=None if mode == "ar" else native)
    mocker.patch.object(OmniARModelRunner, "sample_tokens", return_value=native)
    payload = {"old": {"audio": torch.ones(2, 4)}}
    runner.model.acoustic_started = ["a", "b"]
    runner.model.poll_audio.return_value = payload
    result = runner.execute_model(_schedule(), dummy_run=mode == "dummy", is_profile=mode == "profile")
    if mode == "ar":
        runner.model.poll_audio.assert_not_called()
        result = runner.sample_tokens(None)
    if mode in {"dummy", "profile"}:
        runner.model.poll_audio.assert_not_called()
        assert result is native
    else:
        runner.model.poll_audio.assert_called_once_with()
        assert result.acoustic_started == ["a", "b"] and result.acoustic_ready is payload
        assert runner.model.acoustic_started == []
        assert not hasattr(native, "acoustic_started")


@pytest.mark.parametrize("fails", [False, True])
def test_async_materialization_never_executes_model_and_preserves_kv(runner, mocker, fails):
    native = ModelRunnerOutput(req_ids=["new"], sampled_token_ids=[[8]])
    parent = mocker.Mock(spec=AsyncModelRunnerOutput)
    parent.copy_event = object()
    parent.get_output.side_effect = RuntimeError("copy failed") if fails else lambda: native
    kv = parent.get_kv_connector_output.return_value
    runner.model.poll_audio.return_value = {"old": {"audio": torch.ones(2, 1)}}
    output = runner._with_audio(parent)
    runner.model.poll_audio.side_effect = AssertionError("materialization advanced synthesis")
    assert output.copy_event is parent.copy_event
    if fails:
        with pytest.raises(RuntimeError, match="copy failed"):
            output.get_output()
    else:
        assert output.get_output().acoustic_ready["old"]["audio"].shape == (2, 1)
    assert output.get_kv_connector_output() is kv


def test_model_close_cleans_all_requests_even_when_cancellation_fails(mocker):
    model = mocker.Mock(spec=Yue2ForCausalLM, _states={"a": object(), "b": object()}, acoustic_started=["a"])
    event = mocker.Mock()
    model._pending = (event, torch.tensor([1]), [])
    host = weakref.ref(model._pending[1])

    def wait_for_copy():
        assert host() is not None

    event.synchronize.side_effect = wait_for_copy
    model.on_requests_finished.side_effect = [RuntimeError("release failed"), None]
    with pytest.raises(RuntimeError, match="release failed"):
        Yue2ForCausalLM.close(model)
    event.synchronize.assert_called_once_with()
    assert model.on_requests_finished.call_count == 2
    assert model._pending is None and not model.acoustic_started


def test_shutdown_always_runs_parent_cleanup(runner, mocker):
    parent = mocker.patch.object(OmniARModelRunner, "shutdown")
    runner.model.close.side_effect = RuntimeError("cancel failed")
    with pytest.raises(RuntimeError, match="cancel failed"):
        runner.shutdown()
    parent.assert_called_once_with()


@pytest.mark.parametrize("packed", [False, True])
def test_close_releases_active_waiting_retry_and_ready_work(packed, mocker):
    from vllm_omni.model_executor.models.yue2.batched_synthesis import BatchedSynthesisQueue
    from vllm_omni.model_executor.models.yue2.yue2 import _RequestState, _RowConstants, _SynthesisJob, _SynthesisQueue

    model = Yue2ForCausalLM.__new__(Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model._pending = None
    model.acoustic_started = []
    model._decode_t0 = {}
    model._step_rows = []
    model._states = {
        rid: _RequestState(rid, mocker.Mock(spec=_RowConstants), finished=True)
        for rid in ("active", "waiting", "retry", "ready")
    }
    queue = BatchedSynthesisQueue(model, 2, 1, 1024) if packed else _SynthesisQueue()
    model._synthesis = queue
    jobs = [_SynthesisJob(rid, lambda keep: iter(())) for rid in model._states]
    queue.active = jobs[0]
    if packed:
        queue.members = [model._states["active"]]
        queue.waiting = [model._states["waiting"]]
        queue.retrying = [model._states["retry"]]
        queue.ready = {"ready": {"audio": torch.ones(2, 4)}}
    else:
        for state, job in zip(model._states.values(), jobs):
            state.job = job
        queue.waiting = jobs[1:]
    model.close()
    assert not model._states and not queue.waiting and queue.active is None
    assert jobs[0].cancelled and jobs[0]._released
    if packed:
        assert not queue.members and not queue.retrying and not queue.ready
    else:
        assert all(job.cancelled and job._released for job in jobs)


def test_model_state_bypasses_legacy_sparse_audio(runner):
    hidden = torch.ones(4, 8)
    assert runner.model_state.postprocess_model_output(hidden, None, None) == (hidden, {})
    runner.model.make_omni_output.assert_not_called()
