# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from pytest_mock import MockerFixture
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionRequestAbortedError
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    BaseScheduler,
    DiffusionRequestStatus,
    RequestScheduler,
    Scheduler,
    StepScheduler,
)
from vllm_omni.diffusion.sched.interface import CachedRequestData, NewRequestData
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=f"prompt_{req_id}",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id=req_id,
    )


def _make_request_output(req_id: str, *, error: str | None = None, finished: bool = True):
    return RunnerOutput(
        request_id=req_id,
        step_index=None,
        finished=finished,
        result=DiffusionOutput(output=None, error=error),
    )


def _make_step_output(
    req_id: str,
    step_index: int,
    *,
    finished: bool = False,
    error: str | None = None,
):
    return RunnerOutput(
        request_id=req_id,
        step_index=step_index,
        finished=finished,
        result=DiffusionOutput(output=None, error=error) if error is not None else None,
    )


def _make_step_request(
    req_id: str,
    *,
    num_inference_steps: int = 4,
    step_index: int | None = None,
    sampling_params: OmniDiffusionSamplingParams | None = None,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=f"prompt_{req_id}",
        sampling_params=sampling_params
        or OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            step_index=step_index,
        ),
        request_id=req_id,
    )


def _new_ids(sched_output) -> list[str]:
    return [req.request_id for req in sched_output.scheduled_new_reqs]


def _cached_ids(sched_output) -> list[str]:
    return list(sched_output.scheduled_cached_reqs.request_ids)


def _initialize_paged_scheduler(
    scheduler: BaseScheduler,
    *,
    num_blocks: int = 64,
    max_num_seqs: int = 1,
) -> None:
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=["layer0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )
    scheduler.initialize(
        SimpleNamespace(
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            max_model_len=64,
            max_num_seqs=max_num_seqs,
        ),
        kv_cache_config=config,
        scheduler_block_size=4,
        hash_block_size=4,
        kv_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64),
            max_in_flight_tokens=64,
        ),
    )


def _attach_diffusion_kv(request: OmniDiffusionRequest, *, seq_len: int = 8) -> None:
    request.diffusion_kv_requests = (
        DiffusionKVRequest(
            f"{request.request_id}/diffusion-kv/0",
            sequence_id=0,
            prefix_len=4,
            target_len=4,
            seq_len=seq_len,
        ),
    )


class _StubScheduler:
    def __init__(self, request: OmniDiffusionRequest, output) -> None:
        self._request = request
        self._output = output
        self.initialized_with = None
        self._request_id = request.request_id
        self._state = None
        self._scheduled = False
        self.max_num_running_reqs = 1

    def initialize(self, od_config) -> None:
        self.initialized_with = od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        assert request is self._request
        self._state = SimpleNamespace(request_id=self._request_id, req=request)
        return self._request_id

    def schedule(self):
        if self._scheduled or self._state is None:
            return SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                scheduled_request_ids=[],
                is_empty=True,
            )
        self._scheduled = True
        return SimpleNamespace(
            scheduled_new_reqs=[NewRequestData.from_state(self._state)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            scheduled_request_ids=[self._state.request_id],
            is_empty=False,
        )

    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output
        assert output is self._output
        self._state.status = DiffusionRequestStatus.FINISHED_COMPLETED
        return {self._request_id}

    def has_requests(self) -> bool:
        return not self._scheduled

    def num_waiting_requests(self) -> int:
        return 0 if self._scheduled else 1

    def num_running_requests(self) -> int:
        return 1 if self._scheduled else 0

    def get_request_state(self, request_id: str):
        del request_id
        return self._state

    def pop_request_state(self, request_id: str):
        del request_id
        return self._state

    def preempt_request(self, request_id: str) -> bool:
        del request_id
        return False

    def finish_requests(self, request_ids, status) -> None:
        del request_ids, status
        return None

    def close(self) -> None:
        return None


class _ConcreteScheduler(BaseScheduler):
    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output, output
        return set()


class TestGetStepBatchSamplingParamsKey:
    """Tests for the step-batch compatibility key builder on BaseScheduler."""

    @staticmethod
    def _make(lora_int_id: int | None = None, lora_scale: float = 1.0) -> OmniDiffusionRequest:
        from vllm_omni.lora.request import LoRARequest

        sp = OmniDiffusionSamplingParams(num_inference_steps=2)
        if lora_int_id is not None:
            sp.lora_request = LoRARequest(
                lora_name=f"adapter-{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path=f"/tmp/lora-{lora_int_id}",
            )
        sp.lora_scale = lora_scale
        return OmniDiffusionRequest(
            prompt="prompt",
            sampling_params=sp,
            request_id=f"req-{lora_int_id}-{lora_scale}",
        )

    def test_distinguishes_lora_id(self) -> None:
        scheduler = _ConcreteScheduler()
        assert scheduler._build_sampling_params_key(self._make(lora_int_id=1)) != scheduler._build_sampling_params_key(
            self._make(lora_int_id=2)
        )

    def test_distinguishes_lora_scale(self) -> None:
        scheduler = _ConcreteScheduler()
        assert scheduler._build_sampling_params_key(
            self._make(lora_int_id=1, lora_scale=0.5)
        ) != scheduler._build_sampling_params_key(self._make(lora_int_id=1, lora_scale=1.0))

    def test_treats_no_lora_as_distinct_bucket(self) -> None:
        scheduler = _ConcreteScheduler()
        assert scheduler._build_sampling_params_key(
            self._make(lora_int_id=None)
        ) != scheduler._build_sampling_params_key(self._make(lora_int_id=1))

    def test_equal_for_same_lora_identity(self) -> None:
        scheduler = _ConcreteScheduler()
        a = scheduler._build_sampling_params_key(self._make(lora_int_id=1, lora_scale=0.5))
        b = scheduler._build_sampling_params_key(self._make(lora_int_id=1, lora_scale=0.5))
        assert a == b


class TestGetRequestBatchSamplingParamsKey:
    """Tests for the request-batch compatibility key builder on RequestScheduler."""

    @staticmethod
    def _make(
        *,
        num_inference_steps: int = 2,
        seed: int | None = 123,
        generator: torch.Generator | None = None,
        extra_args: dict | None = None,
        condition_key: tuple | None = None,
    ) -> OmniDiffusionRequest:
        sp = OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            seed=seed,
            generator=generator,
            extra_args=extra_args or {},
        )
        return OmniDiffusionRequest(
            prompt="prompt",
            sampling_params=sp,
            request_id=f"req-{num_inference_steps}",
            batch_compatibility_key=condition_key,
        )

    def test_distinguishes_num_inference_steps(self) -> None:
        scheduler = RequestScheduler()
        assert scheduler._build_sampling_params_key(
            self._make(num_inference_steps=2)
        ) != scheduler._build_sampling_params_key(self._make(num_inference_steps=4))

    def test_ignores_seed_and_generator(self) -> None:
        scheduler = RequestScheduler()
        gen_a = torch.Generator(device="cpu").manual_seed(1)
        gen_b = torch.Generator(device="cpu").manual_seed(2)

        assert scheduler._build_sampling_params_key(
            self._make(seed=1, generator=gen_a)
        ) == scheduler._build_sampling_params_key(self._make(seed=2, generator=gen_b))

    @pytest.mark.parametrize(
        ("first_extra_args", "second_extra_args"),
        [
            ({"sample_solver": "unipc"}, {"sample_solver": "euler"}),
            ({"flow_shift": 3.0}, {"flow_shift": 5.0}),
        ],
    )
    def test_distinguishes_wan_scheduler_structure(
        self,
        first_extra_args: dict,
        second_extra_args: dict,
    ) -> None:
        scheduler = RequestScheduler()

        assert scheduler._build_sampling_params_key(
            self._make(extra_args=first_extra_args)
        ) != scheduler._build_sampling_params_key(self._make(extra_args=second_extra_args))

    @pytest.mark.parametrize(
        ("first_extra_args", "second_extra_args"),
        [
            ({"sample_solver": " Euler "}, {"sample_solver": "euler"}),
            ({"flow_shift": "5.0"}, {"flow_shift": 5.0}),
        ],
    )
    def test_normalizes_equivalent_wan_scheduler_structure(
        self,
        first_extra_args: dict,
        second_extra_args: dict,
    ) -> None:
        scheduler = RequestScheduler()

        assert scheduler._build_sampling_params_key(
            self._make(extra_args=first_extra_args)
        ) == scheduler._build_sampling_params_key(self._make(extra_args=second_extra_args))

    def test_uses_none_for_unspecified_wan_scheduler_structure(self) -> None:
        scheduler = RequestScheduler()

        key = scheduler._build_sampling_params_key(self._make())

        assert key.sample_solver is None
        assert key.flow_shift is None

    def test_distinguishes_pipeline_condition_structure(self) -> None:
        scheduler = RequestScheduler()

        assert scheduler._build_sampling_params_key(
            self._make(condition_key=("wan22_s2v_condition", True))
        ) != scheduler._build_sampling_params_key(self._make(condition_key=("wan22_s2v_condition", False)))


class TestRequestScheduler:
    def setup_method(self) -> None:
        self.scheduler: RequestScheduler = RequestScheduler()
        self.scheduler.initialize(SimpleNamespace(request_batch_max_wait_ms=0.0))

    def test_admission_wait_disabled_with_zero_max_wait(self) -> None:
        self.scheduler.initialize(SimpleNamespace(request_batch_max_wait_ms=0.0))
        decision = self.scheduler.get_admission_wait_decision(now=10.0)

        assert decision.should_wait is False

    @pytest.mark.parametrize(
        ("dp_concurrent", "expected_stable_window_s"),
        [(False, 0.05), (True, 0.3)],
    )
    def test_admission_wait_decision_encodes_coalescing_policy(
        self,
        dp_concurrent: bool,
        expected_stable_window_s: float,
    ) -> None:
        self.scheduler.initialize(
            SimpleNamespace(
                max_num_seqs=4,
                request_batch_max_wait_ms=1000.0,
            )
        )

        decision = self.scheduler.get_admission_wait_decision(
            now=10.0,
            dp_concurrent=dp_concurrent,
        )

        assert decision.should_wait is True
        assert decision.deadline == 11.0
        assert decision.stable_window_s == expected_stable_window_s
        assert decision.max_batch == 4

    def test_admission_wait_disabled_while_wave_is_running(self) -> None:
        self.scheduler.initialize(
            SimpleNamespace(
                max_num_seqs=1,
                request_batch_max_wait_ms=1000.0,
            )
        )
        self.scheduler.add_request(_make_request("running"))
        self.scheduler.schedule()

        decision = self.scheduler.get_admission_wait_decision(now=10.0)

        assert decision.should_wait is False

    def test_admission_wait_end_conditions(self) -> None:
        self.scheduler.initialize(
            SimpleNamespace(
                max_num_seqs=2,
                request_batch_max_wait_ms=1000.0,
            )
        )
        decision = self.scheduler.get_admission_wait_decision(now=10.0)
        self.scheduler.add_request(_make_request("a"))

        assert not self.scheduler.should_end_admission_wait(
            decision,
            now=10.01,
            stable_since=10.0,
        )
        assert self.scheduler.should_end_admission_wait(
            decision,
            now=10.05,
            stable_since=10.0,
        )
        assert self.scheduler.should_end_admission_wait(
            decision,
            now=11.0,
            stable_since=11.0,
        )

        self.scheduler.add_request(_make_request("b"))
        assert self.scheduler.should_end_admission_wait(
            decision,
            now=10.01,
            stable_since=10.01,
        )

    def test_single_request_success_lifecycle(self) -> None:
        req_id = self.scheduler.add_request(_make_request("a"))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        assert _cached_ids(sched_output) == []
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_diffusion_kv_request_moves_to_scheduler_state(self) -> None:
        _initialize_paged_scheduler(self.scheduler)
        request = _make_request("diffusion-kv")
        prepared_layout = object()
        request.prepared_layout = prepared_layout
        kv_request = DiffusionKVRequest(
            "diffusion-kv/diffusion-kv/0",
            sequence_id=0,
            prefix_len=4,
            target_len=8,
            seq_len=16,
        )
        request.diffusion_kv_requests = (kv_request,)

        self.scheduler.add_request(request)
        state = self.scheduler.get_request_state(request.request_id)
        scheduler_output = self.scheduler.schedule()

        assert state is not None
        assert state.diffusion_kv_requests == (kv_request,)
        assert request.diffusion_kv_requests is None
        assert scheduler_output.scheduled_new_reqs[0].req.prepared_layout is prepared_layout
        metadata = scheduler_output.scheduled_new_reqs[0].diffusion_kv_metadata
        assert metadata is not None
        assert metadata.request_id == request.request_id
        assert len(metadata.sequences[0].block_ids[0]) == 4

    def test_diffusion_kv_capacity_backpressures_fifo_until_blocks_are_freed(self) -> None:
        _initialize_paged_scheduler(self.scheduler, num_blocks=3, max_num_seqs=2)
        first_request = _make_request("first")
        second_request = _make_request("second")
        _attach_diffusion_kv(first_request)
        _attach_diffusion_kv(second_request)
        self.scheduler.add_request(first_request)
        self.scheduler.add_request(second_request)

        first_output = self.scheduler.schedule()

        assert _new_ids(first_output) == ["first"]
        assert first_output.num_waiting_reqs == 1

        self.scheduler.update_from_output(first_output, _make_request_output("first"))
        second_output = self.scheduler.schedule()

        assert _new_ids(second_output) == ["second"]
        assert second_output.num_waiting_reqs == 0
        # The same output carries both the release and the newly admitted block
        # table; the future Worker data plane must process finished ids first.
        assert second_output.finished_req_ids == {"first"}
        assert second_output.scheduled_new_reqs[0].diffusion_kv_metadata is not None

    def test_impossible_diffusion_kv_capacity_finishes_only_that_request(self) -> None:
        _initialize_paged_scheduler(self.scheduler, num_blocks=2, max_num_seqs=2)
        impossible = _make_request("impossible")
        schedulable = _make_request("schedulable")
        _attach_diffusion_kv(impossible, seq_len=8)
        schedulable.diffusion_kv_requests = (
            DiffusionKVRequest(
                "schedulable/diffusion-kv/0",
                sequence_id=0,
                prefix_len=0,
                target_len=4,
                seq_len=4,
            ),
        )
        self.scheduler.add_request(impossible)
        self.scheduler.add_request(schedulable)

        sched_output = self.scheduler.schedule()

        failed_state = self.scheduler.get_request_state("impossible")
        assert failed_state is not None
        assert failed_state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert failed_state.error is not None
        assert "cannot fit even when the block pool is empty" in failed_state.error
        assert sched_output.finished_req_ids == {"impossible"}
        assert _new_ids(sched_output) == ["schedulable"]

        finished = self.scheduler.update_from_output(
            sched_output,
            _make_request_output("schedulable"),
        )

        assert finished == {"impossible", "schedulable"}

    def test_impossible_diffusion_kv_capacity_does_not_block_waiters_under_load(self) -> None:
        _initialize_paged_scheduler(self.scheduler, num_blocks=4, max_num_seqs=2)
        running = _make_request("running")
        _attach_diffusion_kv(running)
        self.scheduler.add_request(running)
        assert _new_ids(self.scheduler.schedule()) == ["running"]

        impossible = _make_request("impossible")
        impossible.diffusion_kv_requests = tuple(
            DiffusionKVRequest(
                f"impossible/diffusion-kv/{sequence_id}",
                sequence_id=sequence_id,
                prefix_len=0,
                target_len=4,
                seq_len=4,
            )
            for sequence_id in range(4)
        )
        schedulable = _make_request("schedulable")
        schedulable.diffusion_kv_requests = (
            DiffusionKVRequest(
                "schedulable/diffusion-kv/0",
                sequence_id=0,
                prefix_len=0,
                target_len=4,
                seq_len=4,
            ),
        )
        self.scheduler.add_request(impossible)
        self.scheduler.add_request(schedulable)

        sched_output = self.scheduler.schedule()

        failed_state = self.scheduler.get_request_state("impossible")
        assert failed_state is not None
        assert failed_state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert failed_state.error is not None
        assert "required_blocks=4, available_blocks=3" in failed_state.error
        assert sched_output.finished_req_ids == {"impossible"}
        assert _new_ids(sched_output) == ["schedulable"]

    def test_diffusion_kv_internal_allocation_error_finishes_request(self, monkeypatch) -> None:
        _initialize_paged_scheduler(self.scheduler)
        request = _make_request("native-error")
        _attach_diffusion_kv(request)
        self.scheduler.add_request(request)
        manager = self.scheduler._diffusion_kv_manager
        assert manager is not None

        def raise_native_error(*args, **kwargs):
            raise ValueError("native allocation bug")

        monkeypatch.setattr(
            manager,
            "reserve_request",
            raise_native_error,
        )

        sched_output = self.scheduler.schedule()

        state = self.scheduler.get_request_state("native-error")
        assert state is not None
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "native allocation bug"
        assert sched_output.finished_req_ids == {"native-error"}
        assert sched_output.scheduled_request_ids == []

    def test_diffusion_kv_preemption_retains_allocation(self) -> None:
        _initialize_paged_scheduler(self.scheduler, num_blocks=3)
        request = _make_request("preempted")
        _attach_diffusion_kv(request)
        self.scheduler.add_request(request)
        self.scheduler.schedule()
        manager = self.scheduler._diffusion_kv_manager
        assert manager is not None
        metadata = manager.get_metadata("preempted")

        assert self.scheduler.preempt_request("preempted") is True
        resumed_output = self.scheduler.schedule()

        assert resumed_output.scheduled_new_reqs == []
        assert _cached_ids(resumed_output) == ["preempted"]
        assert manager.get_metadata("preempted") == metadata

    @pytest.mark.parametrize(
        "status",
        [
            DiffusionRequestStatus.FINISHED_COMPLETED,
            DiffusionRequestStatus.FINISHED_ABORTED,
            DiffusionRequestStatus.FINISHED_ERROR,
        ],
    )
    def test_diffusion_kv_terminal_status_releases_allocation(self, status: DiffusionRequestStatus) -> None:
        _initialize_paged_scheduler(self.scheduler, num_blocks=3)
        request = _make_request("terminal")
        _attach_diffusion_kv(request)
        self.scheduler.add_request(request)
        self.scheduler.schedule()
        manager = self.scheduler._diffusion_kv_manager
        assert manager is not None
        assert manager.has_request("terminal") is True

        self.scheduler.finish_requests("terminal", status)

        assert manager.has_request("terminal") is False

    def test_paged_scheduler_rejects_missing_kv_request(self) -> None:
        _initialize_paged_scheduler(self.scheduler)
        request = _make_request("diffusion-kv")

        with pytest.raises(ValueError, match="did not produce DiffusionKVRequest"):
            self.scheduler.add_request(request)

    @pytest.mark.parametrize(
        ("owner_name", "field_name"),
        [
            ("request", "past_key_values"),
            ("sampling_params", "past_key_values"),
            ("sampling_params", "cfg_text_past_key_values"),
            ("sampling_params", "cfg_img_past_key_values"),
            ("sampling_params", "cfg_branch_past_key_values"),
        ],
    )
    def test_diffusion_kv_request_rejects_legacy_dense_kv(self, owner_name: str, field_name: str) -> None:
        _initialize_paged_scheduler(self.scheduler)
        request = _make_request("diffusion-kv")
        owner = request if owner_name == "request" else request.sampling_params
        setattr(owner, field_name, object())
        request.diffusion_kv_requests = (
            DiffusionKVRequest(
                "diffusion-kv/diffusion-kv/0",
                sequence_id=0,
                prefix_len=4,
                target_len=8,
                seq_len=16,
            ),
        )

        with pytest.raises(ValueError, match=rf"{owner_name}\.{field_name}"):
            self.scheduler.add_request(request)

    def test_dense_scheduler_rejects_scheduler_only_kv_request(self) -> None:
        request = _make_request("dense-kv-request")
        request.diffusion_kv_requests = (
            DiffusionKVRequest(
                "dense-kv-request/diffusion-kv/0",
                sequence_id=0,
                prefix_len=4,
                target_len=8,
                seq_len=16,
            ),
        )

        with pytest.raises(ValueError, match="dense_legacy request unexpectedly contains"):
            self.scheduler.add_request(request)

    def test_dense_scheduler_accepts_legacy_dense_kv(self) -> None:
        request = _make_request("dense-kv")
        request.sampling_params.past_key_values = object()

        request_id = self.scheduler.add_request(request)

        assert request_id == request.request_id

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_request("err"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_request_output(req_id, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"

    def test_empty_output_without_error_marks_completed(self) -> None:
        req_id = self.scheduler.add_request(_make_request("empty"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_streaming_output_keeps_request_running_until_final_chunk(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        req_id = scheduler.add_request(_make_request("stream"))

        sched_output = scheduler.schedule()
        chunk = RunnerOutput(
            request_id=req_id,
            step_index=1,
            finished=False,
            result=DiffusionOutput(output="chunk-0", finished=False, chunk_index=0, total_chunks=2),
        )
        finished = scheduler.update_from_output(sched_output, chunk)

        assert finished == set()
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.RUNNING
        assert scheduler.has_requests() is True

        final_chunk = RunnerOutput(
            request_id=req_id,
            step_index=2,
            finished=True,
            result=DiffusionOutput(output="chunk-1", finished=True, chunk_index=1, total_chunks=2),
        )
        finished = scheduler.update_from_output(sched_output, final_chunk)

        assert finished == {req_id}
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        # Request A is still running; scheduling again should not pull B.
        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        self.scheduler.update_from_output(first, _make_request_output(req_id_a))

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_batches_compatible_requests_up_to_max_num_seqs(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_id_a = scheduler.add_request(
            _make_step_request(
                "a",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, seed=123),
            )
        )
        req_id_b = scheduler.add_request(
            _make_step_request(
                "b",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, seed=123),
            )
        )

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_id_a, req_id_b]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 0

    def test_batches_incompatible_request_sampling_params_separately(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_id_a = scheduler.add_request(
            _make_step_request(
                "a", num_inference_steps=2, sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, seed=123)
            )
        )
        scheduler.add_request(
            _make_step_request(
                "b", num_inference_steps=4, sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4, seed=123)
            )
        )

        first = scheduler.schedule()

        assert _new_ids(first) == [req_id_a]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

    def test_batches_different_quality_levels_separately(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        high = scheduler.add_request(
            _make_step_request(
                "high",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=2,
                    quality="high",
                ),
            )
        )
        scheduler.add_request(
            _make_step_request(
                "lossless",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=2,
                    quality="lossless",
                ),
            )
        )

        first = scheduler.schedule()

        assert _new_ids(first) == [high]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

    def test_batches_omitted_and_explicit_lossless_separately(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        omitted = scheduler.add_request(
            _make_step_request(
                "omitted",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=2,
                ),
            )
        )
        scheduler.add_request(
            _make_step_request(
                "explicit",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=2,
                    quality="lossless",
                ),
            )
        )

        first = scheduler.schedule()

        assert _new_ids(first) == [omitted]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

    def test_batches_incompatible_pipeline_conditions_separately(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        request_a = OmniDiffusionRequest(
            prompt="a",
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, seed=123),
            request_id="a",
            batch_compatibility_key=("wan22_s2v_condition", True),
        )
        request_b = OmniDiffusionRequest(
            prompt="b",
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, seed=456),
            request_id="b",
            batch_compatibility_key=("wan22_s2v_condition", False),
        )
        scheduler.add_request(request_a)
        scheduler.add_request(request_b)

        first = scheduler.schedule()

        assert _new_ids(first) == ["a"]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        scheduler.update_from_output(first, _make_request_output("a"))
        second = scheduler.schedule()

        assert _new_ids(second) == ["b"]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 0

    def test_batches_different_request_local_seed_together(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_id_a = scheduler.add_request(
            _make_step_request(
                "a",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, seed=123),
            )
        )
        req_id_b = scheduler.add_request(
            _make_step_request(
                "b",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, seed=456),
            )
        )

        first = scheduler.schedule()

        assert _new_ids(first) == [req_id_a, req_id_b]
        assert first.num_running_reqs == 2
        assert first.num_waiting_reqs == 0

    @pytest.mark.parametrize(
        ("first_extra_args", "second_extra_args"),
        [
            ({"sample_solver": "unipc"}, {"sample_solver": "euler"}),
            ({"flow_shift": 3.0}, {"flow_shift": 5.0}),
            ({"sample_solver": "unipc"}, {}),
            ({"flow_shift": 3.0}, {}),
        ],
    )
    def test_batches_incompatible_wan_scheduler_structure_separately(
        self,
        first_extra_args: dict,
        second_extra_args: dict,
    ) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))
        first_id = scheduler.add_request(
            _make_step_request(
                "a",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=2,
                    extra_args=first_extra_args,
                ),
            )
        )
        scheduler.add_request(
            _make_step_request(
                "b",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=2,
                    extra_args=second_extra_args,
                ),
            )
        )

        first = scheduler.schedule()

        assert _new_ids(first) == [first_id]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

    def test_incompatible_waiting_head_blocks_later_compatible_request(self) -> None:
        scheduler = RequestScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=3))

        req_id_a = scheduler.add_request(_make_request("a"))
        req_id_b = scheduler.add_request(
            OmniDiffusionRequest(
                prompt="prompt_b",
                sampling_params=OmniDiffusionSamplingParams(width=768),
                request_id="b",
            )
        )
        scheduler.add_request(_make_request("c"))

        first = scheduler.schedule()

        assert _new_ids(first) == [req_id_a]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 2

        scheduler.update_from_output(first, _make_request_output(req_id_a))
        second = scheduler.schedule()

        assert _new_ids(second) == [req_id_b]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        # Abort waiting request.
        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        state_b = self.scheduler.get_request_state(req_id_b)
        assert state_b.status == DiffusionRequestStatus.FINISHED_ABORTED

        first = self.scheduler.schedule()
        assert first.finished_req_ids == {req_id_b}
        # A should still run normally.
        assert _new_ids(first) == [req_id_a]

        # B is already marked finished aborted, scheduling again should not pull it.
        second = self.scheduler.schedule()
        assert second.finished_req_ids == set()

        # Abort running request.
        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        state_a = self.scheduler.get_request_state(req_id_a)
        assert state_a.status == DiffusionRequestStatus.FINISHED_ABORTED

        assert self.scheduler.has_requests() is False
        assert self.scheduler.schedule().scheduled_request_ids == []

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_request("has"))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_request_id_is_scheduler_key(self) -> None:
        request = OmniDiffusionRequest(
            prompt="prompt_map_a",
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_id="map-parent",
        )

        request_id = self.scheduler.add_request(request)

        assert request_id == "map-parent"
        state = self.scheduler.get_request_state("map-parent")
        assert state.request_id == "map-parent"

        self.scheduler.pop_request_state("map-parent")

        assert self.scheduler.get_request_state("map-parent") is None

    def test_duplicate_request_id_is_rejected(self) -> None:
        self.scheduler.add_request(_make_request("dup"))

        with pytest.raises(ValueError, match="request_id 'dup' is already active"):
            self.scheduler.add_request(_make_request("dup"))


class TestDiffusionEngine:
    def test_add_req_and_wait_for_response_single_path(self, mocker: MockerFixture) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = SimpleNamespace(streaming_output=False)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(SimpleNamespace())
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._closed = False
        engine.abort_queue = queue.Queue()

        request = _make_request("engine")
        prepared_layout = object()

        def preprocess(req):
            req.prepared_layout = prepared_layout
            return req

        engine.pre_process_func = mocker.Mock(side_effect=preprocess)
        runner_output = _make_request_output("engine")
        engine.execute_fn = mocker.Mock(return_value=runner_output)

        output = engine.add_req_and_wait_for_response(request)

        assert output is runner_output.result
        engine.pre_process_func.assert_called_once_with(request)
        engine.execute_fn.assert_called_once()

    def test_supports_scheduler_interface_injection(self, mocker: MockerFixture) -> None:
        request = _make_request("engine_iface")
        runner_output = _make_request_output("engine_iface")
        scheduler = _StubScheduler(request, runner_output)

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = SimpleNamespace(streaming_output=False)
        engine.scheduler = scheduler
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._closed = False
        engine.abort_queue = queue.Queue()
        engine.execute_fn = mocker.Mock(return_value=runner_output)

        output = engine.add_req_and_wait_for_response(request)

        assert output is runner_output.result
        engine.execute_fn.assert_called_once()

    def test_initializes_default_request_scheduler(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        od_config = SimpleNamespace(model_class_name="mock_model", streaming_output=False)
        fake_executor_cls = mocker.Mock(return_value=mocker.Mock())

        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class",
            lambda *args, **kwargs: fake_executor_cls,
        )

        engine = DiffusionEngine(od_config)

        assert isinstance(engine.scheduler, RequestScheduler)
        assert engine.scheduler.max_num_running_reqs == 1
        fake_executor_cls.assert_called_once_with(od_config)

    def test_initializes_paged_scheduler_from_native_control_plane(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        od_config = SimpleNamespace(model_class_name="mock_model", streaming_output=False)
        fake_executor_cls = mocker.Mock(return_value=mocker.Mock())
        scheduler = mocker.Mock()
        kv_cache_config = object()
        kv_vllm_config = object()

        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class",
            lambda *args, **kwargs: fake_executor_cls,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.initialize_diffusion_kv_control_plane",
            lambda *args, **kwargs: (kv_cache_config, 16, 16, kv_vllm_config),
        )

        engine = DiffusionEngine(od_config, scheduler=scheduler)

        assert engine.scheduler is scheduler
        scheduler.initialize.assert_called_once_with(
            od_config,
            kv_cache_config=kv_cache_config,
            scheduler_block_size=16,
            hash_block_size=16,
            kv_vllm_config=kv_vllm_config,
        )

    def test_scheduler_alias_keeps_default_request_scheduler(self) -> None:
        scheduler = Scheduler()
        scheduler.initialize(SimpleNamespace())

        req_id = scheduler.add_request(_make_request("alias"))
        sched_output = scheduler.schedule()
        finished = scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert req_id in finished
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    @pytest.mark.asyncio
    async def test_step_streaming_raises_aborted_error(self, mocker: MockerFixture) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._check_and_start_background_loop = mocker.AsyncMock()
        request = _make_request("req-abort")
        prepared_layout = object()

        def preprocess(req):
            req.prepared_layout = prepared_layout
            return req

        engine.pre_process_func = mocker.Mock(side_effect=preprocess)

        async def _stream(_request):
            yield DiffusionOutput(aborted=True, abort_message="Request req-abort aborted.")

        engine._add_prepared_request = mocker.Mock(return_value=request.request_id)
        engine.get_output_stream = mocker.Mock(return_value=_stream(None))

        with pytest.raises(DiffusionRequestAbortedError, match="Request req-abort aborted"):
            async for _ in engine.step_streaming(request):
                pass

        engine.pre_process_func.assert_called_once_with(request)
        engine._add_prepared_request.assert_called_once_with(request)
        engine.get_output_stream.assert_called_once_with(request.request_id)

    def test_abort_queue_marks_request_finished_aborted(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._closed = False
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(SimpleNamespace())
        engine.abort_queue = queue.Queue()

        req_id = engine.scheduler.add_request(_make_request("req-abort"))
        engine.abort("req-abort")
        engine._process_aborts_queue()

        assert engine.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_ABORTED

    def test_finalize_finished_request_returns_aborted_output(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = StepScheduler()
        engine.scheduler.initialize(SimpleNamespace())

        req_id = engine.scheduler.add_request(_make_request("req-finalize"))
        engine.scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)

        output = engine._finalize_finished_request(req_id)

        assert output.aborted is True
        assert output.abort_message == "Request req-finalize aborted."

    def test_finalize_finished_request_returns_scheduler_admission_error(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = StepScheduler()
        engine.scheduler.initialize(SimpleNamespace())

        req_id = engine.scheduler.add_request(_make_request("req-error"))
        engine.scheduler._finish_requests(
            {req_id: DiffusionRequestStatus.FINISHED_ERROR},
            {req_id: "KV request cannot fit"},
        )

        output = engine._finalize_finished_request(req_id)

        assert output.error == "KV request cannot fit"

    @pytest.mark.asyncio
    async def test_streaming_runner_output_notifies_each_chunk(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = StepScheduler()
        engine.scheduler.initialize(SimpleNamespace())
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._out_streams = {}
        engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
        engine.main_loop = asyncio.get_running_loop()

        req_id = engine.scheduler.add_request(_make_request("stream-engine"))
        queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
        engine._out_streams[req_id] = queue
        sched_output = engine.scheduler.schedule()

        chunk = RunnerOutput(
            request_id=req_id,
            step_index=1,
            finished=False,
            result=DiffusionOutput(output="chunk-0", finished=False, chunk_index=0, total_chunks=2),
        )
        finished_req_ids = engine.scheduler.update_from_output(sched_output, chunk)
        engine._emit_outputs(finished_req_ids, sched_output.scheduled_request_ids, chunk)

        notified_chunk = await asyncio.wait_for(queue.get(), timeout=1)
        assert notified_chunk.output == "chunk-0"
        assert notified_chunk.finished is False
        assert engine.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.RUNNING

        final_chunk = RunnerOutput(
            request_id=req_id,
            step_index=2,
            finished=True,
            result=DiffusionOutput(output="chunk-1", finished=True, chunk_index=1, total_chunks=2),
        )
        finished_req_ids = engine.scheduler.update_from_output(sched_output, final_chunk)
        engine._emit_outputs(finished_req_ids, sched_output.scheduled_request_ids, final_chunk)

        notified_final = await asyncio.wait_for(queue.get(), timeout=1)
        assert notified_final.output == "chunk-1"
        assert notified_final.finished is True
        assert engine.scheduler.get_request_state(req_id) is None

    @pytest.mark.asyncio
    async def test_finished_streaming_request_without_runner_output_notifies_waiter(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(SimpleNamespace())
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)
        engine._out_streams = {}
        engine.main_loop = asyncio.get_running_loop()

        req_id = engine.scheduler.add_request(_make_request("stream-abort"))
        queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
        engine._out_streams[req_id] = queue
        engine.scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)

        engine._emit_finished_outputs({req_id})

        output = await asyncio.wait_for(queue.get(), timeout=1)
        assert output.aborted is True
        assert output.finished is True
        assert engine.scheduler.get_request_state(req_id) is None

    def test_initializes_step_scheduler_when_step_execution_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        od_config = SimpleNamespace(model_class_name="mock_model", streaming_output=False)
        od_config.step_execution = True
        fake_executor = mocker.Mock()
        fake_executor_cls = mocker.Mock(return_value=fake_executor)

        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class",
            lambda *args, **kwargs: fake_executor_cls,
        )
        engine = DiffusionEngine(od_config)

        assert engine.execution_mode == DiffusionExecutionMode.STEP_BATCH
        assert isinstance(engine.scheduler, StepScheduler)
        assert engine.execute_fn is fake_executor.execute_step
        fake_executor_cls.assert_called_once_with(od_config)

    def test_dummy_run_raises_on_output_error(self, mocker: MockerFixture) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = SimpleNamespace(model_class_name="mock_model", diffusion_load_format="default")
        engine.pre_process_func = None
        engine.add_req_and_wait_for_response = mocker.Mock(return_value=DiffusionOutput(error="boom"))

        with pytest.raises(RuntimeError, match="Dummy run failed: boom"):
            engine._dummy_run()

    def test_dummy_run_delegates_preprocessing_to_sync_admission(self, mocker: MockerFixture) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = SimpleNamespace(model_class_name="mock_model", diffusion_load_format="default")
        engine.pre_process_func = mocker.Mock()
        engine.add_req_and_wait_for_response = mocker.Mock(return_value=DiffusionOutput(output=None))

        engine._dummy_run()

        engine.pre_process_func.assert_not_called()
        admitted_request = engine.add_req_and_wait_for_response.call_args.args[0]
        assert admitted_request.request_id == "dummy_req_id"


class TestStepScheduler:
    def setup_method(self) -> None:
        self.scheduler: StepScheduler = StepScheduler()
        self.scheduler.initialize(SimpleNamespace())

    def test_admission_wait_is_not_supported(self) -> None:
        decision = self.scheduler.get_admission_wait_decision(
            now=10.0,
            dp_concurrent=True,
        )

        assert decision.should_wait is False

    def test_single_request_step_lifecycle(self) -> None:
        request = _make_step_request("step", num_inference_steps=3)
        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(first, _make_step_output(req_id, step_index=1))
        assert finished == set()
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.RUNNING
        assert request.sampling_params.step_index == 1
        assert self.scheduler.has_requests() is True

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(second, _make_step_output(req_id, step_index=2))
        assert finished == set()
        assert request.sampling_params.step_index == 2

        third = self.scheduler.schedule()
        assert _new_ids(third) == []
        assert _cached_ids(third) == [req_id]

        finished = self.scheduler.update_from_output(
            third,
            _make_step_output(req_id, step_index=3, finished=True),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert request.sampling_params.step_index == 3
        assert self.scheduler.has_requests() is False

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_id_b = self.scheduler.add_request(_make_step_request("b", num_inference_steps=2))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        finished = self.scheduler.update_from_output(first, _make_step_output(req_id_a, step_index=1))
        assert finished == set()

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        finished = self.scheduler.update_from_output(
            second,
            _make_step_output(req_id_a, step_index=2, finished=True),
        )
        assert finished == {req_id_a}

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("err", num_inference_steps=3))

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=1, finished=True, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"
        assert self.scheduler.has_requests() is False

    def test_missing_step_index_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("missing", num_inference_steps=3))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            RunnerOutput(
                request_id=req_id,
                step_index=None,
                finished=True,
                result=None,
            ),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "Missing step_index in RunnerOutput"

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_id_b = self.scheduler.add_request(_make_step_request("b", num_inference_steps=2))

        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        assert self.scheduler.get_request_state(req_id_b).status == DiffusionRequestStatus.FINISHED_ABORTED

        running = self.scheduler.schedule()
        assert _new_ids(running) == [req_id_a]

        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        assert self.scheduler.get_request_state(req_id_a).status == DiffusionRequestStatus.FINISHED_ABORTED
        assert self.scheduler.has_requests() is False

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_step_request("has", num_inference_steps=2))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=2, finished=True),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_scheduled_request_aborted_before_update_is_returned_finished(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("abort-late", num_inference_steps=2))

        sched_output = self.scheduler.schedule()
        self.scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)

        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=1),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_ABORTED

    def test_batches_compatible_step_requests(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_a = scheduler.add_request(_make_step_request("a"))
        req_b = scheduler.add_request(_make_step_request("b"))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_b]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 0

    def test_step_batch_allows_different_num_inference_steps(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        req_a = scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_b = scheduler.add_request(_make_step_request("b", num_inference_steps=4))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_b]
        assert sched_output.num_running_reqs == 2
        assert sched_output.num_waiting_reqs == 0

    def test_step_batch_rejects_different_sampling_key(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=3))

        req_a = scheduler.add_request(_make_step_request("a"))
        req_b = scheduler.add_request(
            _make_step_request(
                "b",
                sampling_params=OmniDiffusionSamplingParams(
                    height=768,
                    num_inference_steps=4,
                ),
            )
        )
        scheduler.add_request(_make_step_request("c"))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a]
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 2

        scheduler.update_from_output(
            sched_output,
            _make_step_output(req_a, step_index=4, finished=True),
        )
        second = scheduler.schedule()

        assert _new_ids(second) == [req_b]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

    def test_step_batch_rejects_different_quality_levels(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2))

        high = scheduler.add_request(
            _make_step_request(
                "high",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                    quality="high",
                ),
            )
        )
        scheduler.add_request(
            _make_step_request(
                "lossless",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                    quality="lossless",
                ),
            )
        )

        first = scheduler.schedule()

        assert _new_ids(first) == [high]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

    def test_step_batch_co_schedules_requests_sharing_lora(self) -> None:
        """Multiple requests with the same LoRA (id + scale) co-batch."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=3))

        lora = LoRARequest(lora_name="adapter", lora_int_id=42, lora_path="/tmp/lora")

        def _with_lora(req_id: str) -> OmniDiffusionRequest:
            sp = OmniDiffusionSamplingParams(num_inference_steps=4)
            sp.lora_request = lora
            sp.lora_scale = 0.5
            return _make_step_request(req_id, sampling_params=sp)

        req_a = scheduler.add_request(_with_lora("a"))
        req_b = scheduler.add_request(_with_lora("b"))
        req_c = scheduler.add_request(_with_lora("c"))

        sched_output = scheduler.schedule()

        assert _new_ids(sched_output) == [req_a, req_b, req_c]
        assert sched_output.num_running_reqs == 3
        assert sched_output.num_waiting_reqs == 0

    def test_step_batch_separates_requests_with_different_lora_ids(self) -> None:
        """Different LoRA adapters → distinct batches admitted in FIFO order."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=4))

        lora_a = LoRARequest(lora_name="adapter-A", lora_int_id=1, lora_path="/tmp/lora-a")
        lora_b = LoRARequest(lora_name="adapter-B", lora_int_id=2, lora_path="/tmp/lora-b")

        def _build(req_id: str, lora: LoRARequest) -> OmniDiffusionRequest:
            sp = OmniDiffusionSamplingParams(num_inference_steps=2)
            sp.lora_request = lora
            return _make_step_request(req_id, sampling_params=sp)

        req_a1 = scheduler.add_request(_build("a1", lora_a))
        req_b1 = scheduler.add_request(_build("b1", lora_b))
        req_a2 = scheduler.add_request(_build("a2", lora_a))

        # Strict FIFO admission: a1 starts; b1 (different LoRA) blocks the
        # queue head, so a2 (compatible with a1) is *not* skipped ahead.
        first = scheduler.schedule()
        assert _new_ids(first) == [req_a1]
        assert first.num_waiting_reqs == 2

        # Drain a1 → b1 becomes head-of-line and is admitted with its LoRA.
        scheduler.update_from_output(first, _make_step_output(req_a1, step_index=2, finished=True))
        second = scheduler.schedule()
        assert _new_ids(second) == [req_b1]
        assert second.num_waiting_reqs == 1

        # Drain b1 → a2 is admitted next; LoRA-A is re-activated for it.
        scheduler.update_from_output(second, _make_step_output(req_b1, step_index=2, finished=True))
        third = scheduler.schedule()
        assert _new_ids(third) == [req_a2]
        assert third.num_waiting_reqs == 0

    def test_step_batch_separates_requests_with_different_lora_scale(self) -> None:
        """Same adapter id but different scales → still separate batches."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=4))

        lora = LoRARequest(lora_name="adapter", lora_int_id=7, lora_path="/tmp/lora")

        def _build(req_id: str, scale: float) -> OmniDiffusionRequest:
            sp = OmniDiffusionSamplingParams(num_inference_steps=2)
            sp.lora_request = lora
            sp.lora_scale = scale
            return _make_step_request(req_id, sampling_params=sp)

        req_full = scheduler.add_request(_build("full", 1.0))
        req_half = scheduler.add_request(_build("half", 0.5))

        sched_output = scheduler.schedule()

        admitted = _new_ids(sched_output)
        assert admitted == [req_full]
        assert req_half not in admitted
        assert sched_output.num_waiting_reqs == 1

    def test_step_batch_separates_lora_from_no_lora(self) -> None:
        """A LoRA request and a no-LoRA request do not share a batch."""
        from vllm_omni.lora.request import LoRARequest

        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=4))

        lora = LoRARequest(lora_name="adapter", lora_int_id=3, lora_path="/tmp/lora")

        sp_with = OmniDiffusionSamplingParams(num_inference_steps=2)
        sp_with.lora_request = lora
        req_with = scheduler.add_request(_make_step_request("with", sampling_params=sp_with))
        req_without = scheduler.add_request(_make_step_request("without", num_inference_steps=2))

        sched_output = scheduler.schedule()

        admitted = _new_ids(sched_output)
        assert admitted == [req_with]
        assert req_without not in admitted
        assert sched_output.num_waiting_reqs == 1

    def test_preempt_request_preserves_step_index(self) -> None:
        request = _make_step_request("preempt", num_inference_steps=3)
        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert self.scheduler.update_from_output(first, _make_step_output(req_id, step_index=1)) == set()
        assert request.sampling_params.step_index == 1

        second = self.scheduler.schedule()
        assert _cached_ids(second) == [req_id]
        assert self.scheduler.preempt_request(req_id) is True
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.PREEMPTED
        assert request.sampling_params.step_index == 1

        third = self.scheduler.schedule()
        assert _cached_ids(third) == [req_id]
        assert request.sampling_params.step_index == 1

    @pytest.mark.parametrize(
        ("sampling_params", "expected_steps"),
        [
            (
                OmniDiffusionSamplingParams(
                    timesteps=torch.tensor([1.0, 0.5, 0.0]),
                    sigmas=[1.0, 0.5, 0.25, 0.0],
                    num_inference_steps=5,
                ),
                3,
            ),
            (
                OmniDiffusionSamplingParams(
                    sigmas=[1.0, 0.5],
                    num_inference_steps=5,
                ),
                2,
            ),
            (
                OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                ),
                4,
            ),
        ],
    )
    def test_total_steps_priority(self, sampling_params: OmniDiffusionSamplingParams, expected_steps: int) -> None:
        request = _make_step_request("priority", sampling_params=sampling_params)
        req_id = self.scheduler.add_request(request)

        for _ in range(expected_steps - 1):
            sched_output = self.scheduler.schedule()
            assert sched_output.scheduled_request_ids == [req_id]
            next_step = request.sampling_params.step_index + 1
            assert (
                self.scheduler.update_from_output(
                    sched_output,
                    _make_step_output(req_id, step_index=next_step),
                )
                == set()
            )

        final_output = self.scheduler.schedule()
        assert final_output.scheduled_request_ids == [req_id]
        assert self.scheduler.update_from_output(
            final_output,
            _make_step_output(req_id, step_index=expected_steps, finished=True),
        ) == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    @pytest.mark.parametrize(
        "sampling_params",
        [
            OmniDiffusionSamplingParams(num_inference_steps=0),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=3),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=-1),
        ],
    )
    def test_rejects_invalid_initial_step_state(self, sampling_params: OmniDiffusionSamplingParams) -> None:
        request = _make_step_request("invalid", sampling_params=sampling_params)

        with pytest.raises(ValueError):
            self.scheduler.add_request(request)
