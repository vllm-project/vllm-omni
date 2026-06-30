# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.diffusion.diffusion_engine as diffusion_engine_module
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, _move_tensor_tree_to_cpu
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    NewRequestData,
)
from vllm_omni.diffusion.sched.interface import (
    DiffusionSchedulerOutput as RealDiffusionSchedulerOutput,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


@dataclass
class DiffusionSchedulerOutput:
    step_id: int
    scheduled_new_reqs: list = field(default_factory=list)
    scheduled_cached_reqs: Any = None
    finished_req_ids: set = field(default_factory=set)
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    @property
    def scheduled_request_ids(self):
        ids = [req.request_id for req in self.scheduled_new_reqs]
        if self.scheduled_cached_reqs:
            ids.extend(self.scheduled_cached_reqs.request_ids)
        return ids

    @property
    def is_empty(self):
        return len(self.scheduled_request_ids) == 0


class MockScheduler:
    def __init__(self):
        self._waiting_queue = []
        self._step_id = 0

    def add_request(self, request):
        self._waiting_queue.append(request)
        return request.request_id

    def has_requests(self):
        return len(self._waiting_queue) > 0

    def schedule(self) -> DiffusionSchedulerOutput:
        if not self._waiting_queue:
            return DiffusionSchedulerOutput(step_id=self._step_id)

        batch = []
        while self._waiting_queue:
            req = self._waiting_queue.pop(0)
            batch.append(SimpleNamespace(request_id=req.request_id))

        output = DiffusionSchedulerOutput(step_id=self._step_id, scheduled_new_reqs=batch)
        self._step_id += 1
        return output

    def update_from_output(self, sched_output, runner_output):
        # assume all new req finished
        return [req.request_id for req in sched_output.scheduled_new_reqs]


class _BatchCapablePipeline:
    supports_request_batch = True


class _SingleRequestPipeline:
    pass


class _SingleRequestOverridePipeline(_BatchCapablePipeline):
    def forward(self, req, prompt_ids=None):
        return DiffusionOutput(output=None)


def _make_request_mode_sched_output(*request_ids: str) -> RealDiffusionSchedulerOutput:
    new_reqs = [
        NewRequestData(
            request_id=request_id,
            req=OmniDiffusionRequest(
                prompt=f"prompt_{request_id}",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                request_id=request_id,
            ),
        )
        for request_id in request_ids
    ]
    return RealDiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs),
        num_waiting_reqs=0,
    )


class TestRequestBatchCapability:
    pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

    def test_supports_request_batch_uses_registered_model_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        od_config = SimpleNamespace(model_class_name="BatchPipeline", custom_pipeline_args=None)

        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            lambda model_class_name: _BatchCapablePipeline if model_class_name == "BatchPipeline" else None,
        )

        assert diffusion_engine_module.supports_request_batch(od_config) is True

    def test_supports_request_batch_uses_custom_pipeline_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        od_config = SimpleNamespace(
            model_class_name="SinglePipeline",
            custom_pipeline_args={"pipeline_class": _BatchCapablePipeline},
        )

        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            lambda model_class_name: _SingleRequestPipeline,
        )

        assert diffusion_engine_module.supports_request_batch(od_config) is True

    def test_supports_request_batch_uses_custom_pipeline_class_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        od_config = SimpleNamespace(
            model_class_name="SinglePipeline",
            custom_pipeline_args={"pipeline_class": "test.module.BatchPipeline"},
        )

        monkeypatch.setattr(
            diffusion_engine_module,
            "resolve_obj_by_qualname",
            lambda qualname: _BatchCapablePipeline if qualname == "test.module.BatchPipeline" else None,
        )
        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            lambda model_class_name: _SingleRequestPipeline,
        )

        assert diffusion_engine_module.supports_request_batch(od_config) is True

    def test_supports_request_batch_uses_only_explicit_pipeline_attribute_for_custom_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        od_config = SimpleNamespace(
            model_class_name="BatchPipeline",
            custom_pipeline_args={"pipeline_class": _SingleRequestOverridePipeline},
        )
        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            lambda model_class_name: None,
        )

        assert diffusion_engine_module.supports_request_batch(od_config) is True

    def test_supports_request_batch_honors_explicit_false_on_custom_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _ExplicitlyUnsupportedOverride(_BatchCapablePipeline):
            supports_request_batch = False

            def forward(self, req, prompt_ids=None):
                return DiffusionOutput(output=None)

        od_config = SimpleNamespace(
            model_class_name="BatchPipeline",
            custom_pipeline_args={"pipeline_class": _ExplicitlyUnsupportedOverride},
        )
        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            lambda model_class_name: None,
        )

        assert diffusion_engine_module.supports_request_batch(od_config) is False

    def test_supports_request_batch_rejects_invalid_custom_pipeline_class_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        od_config = SimpleNamespace(
            model_class_name="BatchPipeline",
            custom_pipeline_args={"pipeline_class": "test.module.MissingPipeline"},
        )

        def fail_resolve(qualname):
            raise ImportError(qualname)

        monkeypatch.setattr(diffusion_engine_module, "resolve_obj_by_qualname", fail_resolve)
        registry_load = mocker.Mock(return_value=_BatchCapablePipeline)
        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            registry_load,
        )

        with pytest.raises(ValueError, match="Failed to resolve custom diffusion pipeline class"):
            diffusion_engine_module.supports_request_batch(od_config)
        registry_load.assert_not_called()

    def test_engine_disables_batch_dispatch_for_single_request_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        od_config = SimpleNamespace(
            model_class_name="SinglePipeline",
            custom_pipeline_args=None,
            streaming_output=False,
        )
        fake_executor = SimpleNamespace(
            execute_request=mocker.Mock(return_value="per-request"),
            execute_batch=mocker.Mock(return_value="batch"),
            execute_step=mocker.Mock(return_value="step"),
        )
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
        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            lambda model_class_name: _SingleRequestPipeline,
        )
        monkeypatch.setattr(DiffusionEngine, "_dummy_run", lambda self: None)

        engine = DiffusionEngine(od_config)
        output = engine.execute_fn(_make_request_mode_sched_output("req-a", "req-b"))

        assert engine.supports_request_batch is False
        assert output == "per-request"
        fake_executor.execute_request.assert_called_once()
        fake_executor.execute_batch.assert_not_called()

    @pytest.mark.parametrize("request_ids", [("req-a",), ("req-a", "req-b")])
    def test_engine_enables_batch_dispatch_for_request_batch_pipeline(
        self,
        request_ids: tuple[str, ...],
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
    ) -> None:
        od_config = SimpleNamespace(
            model_class_name="BatchPipeline",
            custom_pipeline_args=None,
            streaming_output=False,
        )
        fake_executor = SimpleNamespace(
            execute_request=mocker.Mock(return_value="per-request"),
            execute_batch=mocker.Mock(return_value="batch"),
            execute_step=mocker.Mock(return_value="step"),
        )
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
        monkeypatch.setattr(
            diffusion_engine_module.DiffusionModelRegistry,
            "_try_load_model_cls",
            lambda model_class_name: _BatchCapablePipeline,
        )
        monkeypatch.setattr(DiffusionEngine, "_dummy_run", lambda self: None)

        engine = DiffusionEngine(od_config)
        output = engine.execute_fn(_make_request_mode_sched_output(*request_ids))

        assert engine.supports_request_batch is True
        assert output == "batch"
        fake_executor.execute_batch.assert_called_once()
        fake_executor.execute_request.assert_not_called()


class TestRequestBatchAdmission:
    pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

    def test_config_rejects_negative_request_batch_max_wait_ms(self) -> None:
        with pytest.raises(ValueError, match="request_batch_max_wait_ms"):
            OmniDiffusionConfig(model="test", request_batch_max_wait_ms=-1.0)

    def test_config_normalizes_request_batch_max_wait_ms_to_float(self) -> None:
        config = OmniDiffusionConfig(model="test", request_batch_max_wait_ms=5)

        assert config.request_batch_max_wait_ms == 5.0
        assert isinstance(config.request_batch_max_wait_ms, float)

    def test_scheduler_exposes_waiting_and_running_counts(self) -> None:
        from vllm_omni.diffusion.sched import RequestScheduler

        od_config = SimpleNamespace(max_num_seqs=4)
        scheduler = RequestScheduler()
        scheduler.initialize(od_config)

        assert scheduler.num_waiting_requests() == 0
        assert scheduler.num_running_requests() == 0

        scheduler.add_request(
            OmniDiffusionRequest(
                prompt="prompt_a",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                request_id="req-a",
            )
        )
        scheduler.add_request(
            OmniDiffusionRequest(
                prompt="prompt_b",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                request_id="req-b",
            )
        )
        assert scheduler.num_waiting_requests() == 2
        assert scheduler.num_running_requests() == 0

        scheduler.schedule()
        assert scheduler.num_waiting_requests() == 0
        assert scheduler.num_running_requests() == 2

    def test_request_batch_admission_exits_early_when_waiting_queue_stable(self) -> None:
        from vllm_omni.diffusion.sched import RequestScheduler

        od_config = SimpleNamespace(
            max_num_seqs=32,
            request_batch_max_wait_ms=1000.0,
            step_execution=False,
        )
        scheduler = RequestScheduler()
        scheduler.initialize(od_config)
        for idx in range(2):
            scheduler.add_request(
                OmniDiffusionRequest(
                    prompt=f"prompt_{idx}",
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                    request_id=f"req-{idx}",
                )
            )

        engine = object.__new__(DiffusionEngine)
        engine.od_config = od_config
        engine.scheduler = scheduler
        engine.step_execution = False
        engine.supports_request_batch = True
        engine.stop_event = threading.Event()
        engine._rpc_lock = threading.RLock()
        engine._cv = threading.Condition(engine._rpc_lock)

        start = time.monotonic()
        with engine._cv:
            engine._wait_for_request_batch_admission_locked()
        waited_s = time.monotonic() - start

        # Stable-window exit (~50ms), not the full 1000ms deadline.
        assert waited_s < 0.5
        assert waited_s >= 0.04
        assert scheduler.num_waiting_requests() == 2
        assert scheduler.num_running_requests() == 0


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_move_tensor_tree_keeps_cpu_tensor_identity() -> None:
    tensor = torch.arange(8, dtype=torch.float32)

    moved = _move_tensor_tree_to_cpu(tensor)

    assert moved is tensor


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_move_tensor_tree_preserves_nested_structure_without_mutating_input() -> None:
    tensor = torch.arange(4, dtype=torch.float32)
    nested_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    sentinel = object()
    payload = {
        "tensor": tensor,
        "list": [nested_tensor, sentinel],
        "tuple": ({"inner": tensor}, "metadata"),
        "scalar": 3,
    }

    moved = _move_tensor_tree_to_cpu(payload)

    assert moved is not payload
    assert set(moved) == {"tensor", "list", "tuple", "scalar"}
    assert moved["list"] is not payload["list"]
    assert moved["tuple"] is not payload["tuple"]
    assert moved["tuple"][0] is not payload["tuple"][0]
    assert moved["tensor"] is tensor
    assert moved["list"][0] is nested_tensor
    assert moved["list"][1] is sentinel
    assert moved["tuple"][0]["inner"] is tensor
    assert moved["tuple"][1] == "metadata"
    assert moved["scalar"] == 3
    assert payload["list"][0] is nested_tensor
    assert payload["list"][1] is sentinel
    assert payload["tuple"][0]["inner"] is tensor
    assert payload["tuple"][1] == "metadata"


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_move_tensor_tree_returns_non_tensor_values_unchanged() -> None:
    value = object()

    moved = _move_tensor_tree_to_cpu(value)

    assert moved is value


@pytest.mark.diffusion
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_move_tensor_tree_moves_nested_cuda_tensors_to_cpu() -> None:
    tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    other = torch.arange(4, dtype=torch.int64, device="cuda")
    payload = {"tensor": tensor, "items": [other, ("keep", tensor)]}

    moved = _move_tensor_tree_to_cpu(payload)

    assert moved["tensor"].device.type == "cpu"
    assert moved["items"][0].device.type == "cpu"
    assert moved["items"][1][1].device.type == "cpu"
    torch.testing.assert_close(moved["tensor"], tensor.cpu())
    torch.testing.assert_close(moved["items"][0], other.cpu())
    torch.testing.assert_close(moved["items"][1][1], tensor.cpu())
    assert moved["items"][1][0] == "keep"


@pytest.mark.asyncio
async def test_async_add_req_and_wait_for_response():
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = MockScheduler()
    engine._out_queue = {}
    engine.abort_queue: queue.Queue[str] = queue.Queue()
    engine._rpc_queue = queue.Queue()
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._init_lock = asyncio.Lock()
    engine._closed = False
    engine.od_config = SimpleNamespace(streaming_output=False)
    engine._loop_started = False
    engine.main_loop = None
    engine.supports_request_batch = False

    engine._finalize_finished_request = lambda rid, out, err: out.result

    def mock_execute_batch(sched_output):
        request_ids = sched_output.scheduled_request_ids

        time.sleep(1)

        class MockRunnerOutput:
            def __init__(self, ids):
                self.request_id = ids
                self.step_index = 0
                self.finished = True
                self._results = {rid: SimpleNamespace(result_data=f"data_{rid}") for rid in ids}

            def get_request_output(self, rid):
                return SimpleNamespace(result=self._results[rid], step_index=0, finished=True)

        return MockRunnerOutput(request_ids)

    engine.execute_fn = mock_execute_batch

    await engine._check_and_start_background_loop()

    async def run_task(rid):
        req = SimpleNamespace(request_id=rid)
        start = time.time()
        res = await engine.async_add_req_and_wait_for_response(req)
        return rid, res, time.time() - start

    task_ids = [f"req_{i}" for i in range(5)]
    tasks = [run_task(rid) for rid in task_ids]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        with engine._cv:
            engine.stop_event.set()
            engine._cv.notify_all()
        engine.worker_thread.join(timeout=5)
    assert len(results) == 5
    for rid, res, elapsed in results:
        assert rid in res.result_data

    eps = 0.5
    latencies = [r[2] for r in results]
    assert max(latencies) - min(latencies) < eps
