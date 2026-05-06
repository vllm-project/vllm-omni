# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for step-level diffusion execution across runner / worker / executor / engine."""

import os
import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.comm import RingComm, SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.ipc import (
    pack_diffusion_output_shm,
    unpack_diffusion_output_shm,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


def _update_environment_variables(envs_dict: dict[str, str]) -> None:
    for key, value in envs_dict.items():
        os.environ[key] = value


class _StepPipeline:
    """Minimal pipeline stub that supports step-wise execution."""

    supports_step_execution = True

    def __init__(self):
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        state.timesteps = [torch.tensor(10), torch.tensor(5)]
        state.latents = torch.tensor([0.0])
        state.prompt_embeds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        return state

    def denoise_step(self, input_batch, **kwargs):
        self.denoise_calls += 1
        return torch.full_like(input_batch.prompt_embeds, fill_value=0.5)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        self.scheduler_calls += 1
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        return DiffusionOutput(output=torch.tensor([state.step_index], dtype=torch.float32))


class _InterruptingStepPipeline(_StepPipeline):
    interrupt = True

    def denoise_step(self, state, **kwargs):
        del state, kwargs
        self.denoise_calls += 1
        return None

    def step_scheduler(self, state, noise_pred, **kwargs):
        del state, noise_pred, kwargs
        raise AssertionError("step_scheduler should not run after interrupt")

    def post_decode(self, state, **kwargs):
        del state, kwargs
        raise AssertionError("post_decode should not run after interrupt")


class _IdentityNoiseTransformer(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs):
        del kwargs
        return (x,)


class _AdditiveScheduler:
    def step(self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, return_dict: bool = False):
        del t, return_dict
        return (latents + noise_pred,)


class _DistributedStepPipeline(CFGParallelMixin):
    supports_step_execution = True

    def __init__(self, mode: str, device: torch.device):
        self.mode = mode
        self.device = device
        self._interrupt = False
        self.scheduler = _AdditiveScheduler()
        self.transformer = _IdentityNoiseTransformer()

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_encode(self, state, **kwargs):
        del kwargs
        state.timesteps = [torch.tensor(1.0, device=self.device)]
        state.latents = torch.ones((1, 1), device=self.device)
        state.step_index = 0
        state.scheduler = self.scheduler
        state.do_true_cfg = self.mode == "cfg"
        state.prompt_embeds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        return state

    def denoise_step(self, state, **kwargs):
        del kwargs
        if self.mode == "ulysses":
            sp_group = get_sp_group().ulysses_group
            seq_world_size = torch.distributed.get_world_size(sp_group)
            input_tensor = torch.randn(1, 2, 2 * seq_world_size, 2, device=self.device)
            original = input_tensor.clone()
            intermediate = SeqAllToAll4D.apply(sp_group, input_tensor, 2, 1, False)
            output = SeqAllToAll4D.apply(sp_group, intermediate, 1, 2, False)
            torch.testing.assert_close(output, original, rtol=1e-5, atol=1e-5)
            return torch.ones_like(state.latents)

        if self.mode == "ring":
            ring_group = get_sp_group().ring_group
            rank = torch.distributed.get_rank(ring_group)
            world_size = torch.distributed.get_world_size(ring_group)
            comm = RingComm(ring_group)
            input_tensor = torch.full((1, 2, 2), float(rank + 1), device=self.device)
            recv_tensor = comm.send_recv(input_tensor)
            comm.commit()
            comm.wait()
            expected = torch.full_like(recv_tensor, float(((rank - 1) % world_size) + 1))
            torch.testing.assert_close(recv_tensor, expected, rtol=1e-5, atol=1e-5)
            return torch.ones_like(state.latents)

        positive_kwargs = {"x": state.latents + 1}
        negative_kwargs = {"x": state.latents - 1}
        return self.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=1.0,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )

    def step_scheduler(self, state, noise_pred, **kwargs):
        del kwargs
        if self.mode == "cfg":
            state.latents = self.scheduler_step_maybe_with_cfg(
                noise_pred,
                state.current_timestep,
                state.latents,
                do_true_cfg=True,
                per_request_scheduler=state.scheduler,
            )
        else:
            state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        return DiffusionOutput(output=state.latents.detach().cpu())


def _make_step_request(num_inference_steps: int = 2):
    return SimpleNamespace(
        prompts=["a prompt"],
        request_ids=["req-1"],
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=num_inference_steps,
        ),
    )


def _assert_aborted_output(output: DiffusionOutput, request_id: str) -> None:
    assert output.output is None
    assert output.error is None
    assert output.aborted is True
    assert output.abort_message == f"Request {request_id} aborted."


def _make_engine_request(req_id: str = "req-1", num_inference_steps: int = 2) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt-{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_inference_steps),
        request_ids=[req_id],
    )


def _make_runner():
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _StepPipeline()
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    return runner


def _make_distributed_runner(mode: str, device: torch.device):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = device
    runner.pipeline = _DistributedStepPipeline(mode=mode, device=device)
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    return runner


def _make_scheduler_output(req, sched_req_id="req-1", step_id=0, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(sched_req_id=sched_req_id, req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _make_cached_scheduler_output(sched_req_id="req-1", step_id=1, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(sched_req_ids=[sched_req_id]),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _make_engine(scheduler, execute_fn=None) -> DiffusionEngine:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(model_class_name="QwenImagePipeline")
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.scheduler = scheduler
    engine.execute_fn = execute_fn
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine.abort_queue = queue.Queue()
    return engine


def _expected_output_for_mode(mode: str) -> torch.Tensor:
    if mode == "cfg":
        return torch.tensor([[3.0]])
    return torch.tensor([[2.0]])


def _distributed_step_worker(local_rank: int, world_size: int, mode: str, master_port: str):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
    )
    model_runner_module.set_forward_context = _noop_forward_context

    try:
        init_distributed_environment()
        if mode == "ulysses":
            initialize_model_parallel(ulysses_degree=world_size)
        elif mode == "ring":
            initialize_model_parallel(ring_degree=world_size)
        elif mode == "cfg":
            initialize_model_parallel(cfg_parallel_size=world_size)
        else:
            raise ValueError(f"Unsupported distributed test mode: {mode}")

        runner = _make_distributed_runner(mode, device)
        result = DiffusionModelRunner.execute_stepwise(
            runner,
            _make_scheduler_output(_make_step_request(num_inference_steps=1), step_id=0),
        )
        output = result.get_req_output("req-1")

        assert output.finished is True
        assert output.result is not None
        torch.testing.assert_close(output.result.output, _expected_output_for_mode(mode), rtol=1e-5, atol=1e-5)
        assert "req-1" not in runner.state_cache
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# Runner / Worker
# ---------------------------------------------------------------------------


@pytest.mark.cpu
class TestRunner:
    """DiffusionModelRunner.execute_stepwise"""

    def test_completes_request_and_clears_state(self, monkeypatch):
        runner = _make_runner()
        req = _make_step_request()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        result = DiffusionModelRunner.execute_stepwise(runner, _make_scheduler_output(req, step_id=0))
        first = result.get_req_output("req-1")
        assert first.req_id == "req-1"
        assert first.step_index == 1
        assert first.finished is False
        assert first.result is None
        assert "req-1" in runner.state_cache

        result = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output(step_id=1))
        second = result.get_req_output("req-1")
        assert second.req_id == "req-1"
        assert second.step_index == 2
        assert second.finished is True
        assert second.result is not None
        assert second.result.error is None
        assert torch.equal(second.result.output, torch.tensor([2.0]))
        assert "req-1" not in runner.state_cache

        assert runner.pipeline.prepare_calls == 1
        assert runner.pipeline.denoise_calls == 2
        assert runner.pipeline.scheduler_calls == 2
        assert runner.pipeline.decode_calls == 1

    def test_rejects_multi_request_step_batch(self):
        runner = _make_runner()
        req_1 = _make_step_request()
        req_2 = _make_step_request()
        req_2.request_ids = ["req-2"]

        scheduler_output = DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=[
                NewRequestData(sched_req_id="req-1", req=req_1),
                NewRequestData(sched_req_id="req-2", req=req_2),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            num_running_reqs=2,
            num_waiting_reqs=0,
        )

        result = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
        assert len(result) == 2

    def test_rejects_missing_cached_state(self):
        runner = _make_runner()

        with pytest.raises(ValueError, match="Missing cached state"):
            DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output(sched_req_id="req-missing"))

    def test_interrupt_marks_request_finished_and_clears_state(self, monkeypatch):
        runner = _make_runner()
        runner.pipeline = _InterruptingStepPipeline()
        req = _make_step_request()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        result = DiffusionModelRunner.execute_stepwise(runner, _make_scheduler_output(req, step_id=0))
        output = result.get_req_output("req-1")
        assert output.req_id == "req-1"
        assert output.step_index == 0
        assert output.finished is True
        assert output.result is not None
        assert output.result.error == "stepwise denoise interrupted"
        assert "req-1" not in runner.state_cache
        assert runner.pipeline.prepare_calls == 1
        assert runner.pipeline.denoise_calls == 1
        assert runner.pipeline.scheduler_calls == 0
        assert runner.pipeline.decode_calls == 0

    def test_load_model_rejects_unsupported_step_execution(self, monkeypatch):
        class _RequestOnlyPipeline:
            pass

        class _FakeLoader:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def load_model(self, **kwargs):
                del kwargs
                return _RequestOnlyPipeline()

        class _FakeProfiler:
            consumed_memory = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        runner = object.__new__(DiffusionModelRunner)
        runner.vllm_config = object()
        runner.od_config = SimpleNamespace(
            enable_cpu_offload=False,
            enable_layerwise_offload=False,
            enforce_eager=True,
            cache_backend=None,
            cache_config=None,
            step_execution=True,
            model_class_name="RequestOnlyPipeline",
            parallel_config=SimpleNamespace(use_hsdp=False),
        )
        runner.device = torch.device("cpu")
        runner.pipeline = None
        runner.cache_backend = None
        runner.offload_backend = None
        runner.state_cache = {}
        runner.kv_transfer_manager = SimpleNamespace()

        monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _FakeLoader)
        monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _FakeProfiler)
        monkeypatch.setattr(model_runner_module, "get_offload_backend", lambda *args, **kwargs: None)
        monkeypatch.setattr(model_runner_module, "get_cache_backend", lambda *args, **kwargs: None)

        with pytest.raises(ValueError, match="RequestOnlyPipeline"):
            DiffusionModelRunner.load_model(runner)


@pytest.mark.cpu
class TestWorker:
    """DiffusionWorker.execute_stepwise"""

    def test_delegates_to_model_runner(self):
        worker = object.__new__(DiffusionWorker)
        expected = RunnerOutput(req_id="req-1", step_index=1, finished=False, result=None)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req=SimpleNamespace(
                        sampling_params=SimpleNamespace(lora_request=None),
                    )
                )
            ]
        )
        worker.lora_manager = None
        worker.model_runner = SimpleNamespace(
            execute_stepwise=lambda arg: expected if arg is scheduler_output else None
        )

        output = DiffusionWorker.execute_stepwise(worker, scheduler_output)

        assert output is expected

    def test_clears_active_lora_before_stepwise_execution(self):
        worker = object.__new__(DiffusionWorker)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req=SimpleNamespace(
                        sampling_params=SimpleNamespace(lora_request=None),
                    )
                )
            ]
        )
        calls: list[object | None] = []

        class _FakeLoRAManager:
            def set_active_adapter(self, adapter):
                calls.append(adapter)

        worker.lora_manager = _FakeLoRAManager()
        worker.model_runner = SimpleNamespace(execute_stepwise=lambda arg: RunnerOutput(req_id="req-1"))

        DiffusionWorker.execute_stepwise(worker, scheduler_output)

        assert calls == [None]

    def test_rejects_lora_requests_in_step_mode(self):
        worker = object.__new__(DiffusionWorker)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req=SimpleNamespace(
                        sampling_params=SimpleNamespace(lora_request=object()),
                    )
                )
            ]
        )
        worker.lora_manager = None
        worker.model_runner = SimpleNamespace(execute_stepwise=lambda arg: RunnerOutput(req_id="req-1"))

        with pytest.raises(ValueError, match="does not support LoRA"):
            DiffusionWorker.execute_stepwise(worker, scheduler_output)


@pytest.mark.cpu
class TestExecutor:
    """MultiprocDiffusionExecutor.execute_step"""

    def test_execute_step_passes_through_runner_output(self, mocker: MockerFixture):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._ensure_open = lambda: None
        expected = RunnerOutput(req_id="req-step", step_index=1, finished=False, result=None)
        executor.collective_rpc = mocker.Mock(return_value=expected)

        request = _make_engine_request("req-step", num_inference_steps=2)
        scheduler_output = _make_scheduler_output(request, sched_req_id="req-step")

        output = MultiprocDiffusionExecutor.execute_step(executor, scheduler_output)

        assert output is expected


@pytest.mark.cpu
class TestEngine:
    """Step-execution paths in DiffusionEngine.add_req_and_wait_for_response"""

    @pytest.mark.parametrize(
        ("execute_fn", "expected_error"),
        [
            (
                lambda _: RunnerOutput(
                    req_id="req-error",
                    step_index=1,
                    finished=True,
                    result=DiffusionOutput(error="boom"),
                ),
                "boom",
            ),
            (
                lambda _: (_ for _ in ()).throw(RuntimeError("gpu on fire")),
                "gpu on fire",
            ),
        ],
    )
    def test_step_engine_returns_error(self, execute_fn, expected_error, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler, execute_fn=execute_fn)

        output = engine.add_req_and_wait_for_response(_make_engine_request("req-error", num_inference_steps=2))

        assert output.output is None
        assert expected_error in output.error

    def test_step_execution_completes(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-step", num_inference_steps=2)

        call_count = {"n": 0}

        def execute_fn(_):
            call_count["n"] += 1
            finished = call_count["n"] == 2
            return RunnerOutput(
                req_id="req-step",
                step_index=call_count["n"],
                finished=finished,
                result=(DiffusionOutput(output=torch.tensor([2.0])) if finished else None),
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert call_count["n"] == 2
        assert output.error is None
        assert torch.equal(output.output, torch.tensor([2.0]))

    def test_step_abort_stops_rescheduling_after_first_step(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-stop", num_inference_steps=4)

        step = {"n": 0}

        def execute_fn(_):
            step["n"] += 1
            engine.abort("req-stop")
            return RunnerOutput(
                req_id="req-stop",
                step_index=1,
                finished=False,
                result=None,
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert step["n"] == 1
        _assert_aborted_output(output, "req-stop")

    def test_step_abort_after_reschedule_returns_aborted_output(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-mid", num_inference_steps=4)

        step = {"n": 0}

        def execute_fn(sched_output):
            step["n"] += 1
            if step["n"] == 2:
                assert sched_output == _make_cached_scheduler_output("req-mid", step_id=1)
                engine.abort("req-mid")
            return RunnerOutput(
                req_id="req-mid",
                step_index=step["n"],
                finished=False,
                result=None,
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert step["n"] == 2
        _assert_aborted_output(output, "req-mid")

    def test_finished_step_without_result_returns_error(self, mocker: MockerFixture):
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace())
        engine = _make_engine(
            scheduler,
            execute_fn=lambda _: RunnerOutput(
                req_id="req-missing",
                step_index=1,
                finished=True,
                result=None,
            ),
        )

        output = engine.add_req_and_wait_for_response(_make_engine_request("req-missing", num_inference_steps=1))

        assert output.output is None
        assert output.error == "Diffusion execution finished without a final output."


@pytest.mark.cpu
class TestIPC:
    def test_pack_unpack_runner_output_shm(self):
        tensor = torch.zeros(300_000, dtype=torch.float32)
        output = RunnerOutput(req_id="req-1", finished=True, result=DiffusionOutput(output=tensor))

        packed = pack_diffusion_output_shm(output)
        assert isinstance(packed.result.output, dict)
        assert packed.result.output["__tensor_shm__"] is True

        unpacked = unpack_diffusion_output_shm(packed)
        assert isinstance(unpacked.result.output, torch.Tensor)
        torch.testing.assert_close(unpacked.result.output, tensor)


@pytest.mark.cpu
class TestSupportedPipelines:
    """Step-execution protocol checks for supported pipelines."""

    def test_default_stage_config_includes_step_execution(self):
        stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
            {
                "step_execution": True,
            }
        )[0]

        assert stage_cfg["engine_args"]["step_execution"] is True

    def test_qwen_image_supports_step_execution(self):
        from vllm_omni.diffusion.models.interface import SupportsStepExecution, supports_step_execution
        from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline

        # Avoid loading model weights; protocol membership depends on the class contract.
        pipeline = object.__new__(QwenImagePipeline)

        assert pipeline.supports_step_execution is True
        assert supports_step_execution(pipeline) is True
        assert isinstance(pipeline, SupportsStepExecution) is True


@hardware_test(
    res={"cuda": "L4"},
    num_cards=2,
)
def test_execute_stepwise_with_ulysses_parallel():
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} devices")

    torch.multiprocessing.spawn(
        _distributed_step_worker,
        args=(world_size, "ulysses", "29540"),
        nprocs=world_size,
    )


@hardware_test(
    res={"cuda": "L4"},
    num_cards=2,
)
def test_execute_stepwise_with_ring_parallel():
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} devices")

    torch.multiprocessing.spawn(
        _distributed_step_worker,
        args=(world_size, "ring", "29541"),
        nprocs=world_size,
    )


@hardware_test(
    res={"cuda": "L4"},
    num_cards=2,
)
def test_execute_stepwise_with_cfg_parallel():
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} devices")

    torch.multiprocessing.spawn(
        _distributed_step_worker,
        args=(world_size, "cfg", "29542"),
        nprocs=world_size,
    )
