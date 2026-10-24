# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for diffusion worker lifecycle and message contracts."""

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker import diffusion_worker as worker_module
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("method", ["handle_sleep_task", "handle_wake_task"])
@pytest.mark.parametrize("task", [{"task_id": "bad-task", "unknown": True}, {"unknown": True}])
def test_invalid_lifecycle_task_returns_error_ack(mocker: MockerFixture, method, task):
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    mocker.patch.object(worker_module.dist, "is_initialized", return_value=False)
    sleep = mocker.patch.object(worker, "sleep")
    wake = mocker.patch.object(worker, "wake_up")

    ack = getattr(worker, method)(task)

    assert ack.status == "ERROR"
    assert ack.task_id == task.get("task_id", "local")
    assert "unknown" in ack.error_msg
    sleep.assert_not_called()
    wake.assert_not_called()


@pytest.mark.parametrize("message_kind", ["request", "invalid"])
def test_busy_loop_handles_direct_request_and_invalid_message(mocker: MockerFixture, message_kind):
    request = OmniDiffusionRequest(request_id="request-0", prompt="test", sampling_params=OmniDiffusionSamplingParams())
    proc = object.__new__(WorkerProc)
    proc.gpu_id = 0
    proc._running = True
    proc.od_config = OmniDiffusionConfig()
    proc.worker = mocker.Mock()
    output = DiffusionOutput(output="result")
    proc.worker.execute_model.return_value = output
    mocker.patch.object(
        proc, "recv_message", side_effect=[request if message_kind == "request" else 42, {"type": "shutdown"}]
    )
    respond = mocker.patch.object(proc, "_return_result")
    mocker.patch.object(worker_module.current_omni_platform, "empty_cache")

    proc._worker_busy_loop()

    if message_kind == "request":
        proc.worker.execute_model.assert_called_once_with(request, proc.od_config)
        respond.assert_called_once_with(output)
    else:
        proc.worker.execute_model.assert_not_called()
        assert respond.call_count == 1
        assert respond.call_args.args[0].error is not None
    assert not proc._running


@pytest.fixture
def lora_worker(mocker: MockerFixture):
    worker = object.__new__(DiffusionWorker)
    worker.od_config = OmniDiffusionConfig()
    worker.device = torch.device("cpu")
    worker.lora_manager = None
    worker.model_runner = mocker.Mock(spec=DiffusionModelRunner)
    worker.model_runner.pipeline = mocker.Mock(spec=torch.nn.Module)
    worker.model_runner.pipeline.lora_is_fused = False
    return worker


def test_lora_manager_normalizes_single_path_and_scale_lists(lora_worker, mocker: MockerFixture):
    lora_worker.od_config.lora_path = ["unused-lora"]
    lora_worker.od_config.lora_scale = [0.4]
    manager = mocker.patch.object(worker_module, "DiffusionLoRAManager")

    lora_worker.init_lora_manager()

    assert manager.call_args.kwargs["lora_path"] == "unused-lora"
    assert manager.call_args.kwargs["lora_scale"] == 0.4
    assert manager.call_args.kwargs["max_cached_adapters"] == 1


@pytest.mark.parametrize(
    "field,value", [("lora_path", []), ("lora_path", ["a", "b"]), ("lora_scale", []), ("lora_scale", [0.4, 0.7])]
)
def test_lora_manager_rejects_non_scalar_inputs_before_mutation(lora_worker, mocker: MockerFixture, field, value):
    setattr(lora_worker.od_config, field, value)
    manager = mocker.patch.object(worker_module, "DiffusionLoRAManager")

    with pytest.raises(ValueError, match=field):
        lora_worker.init_lora_manager()

    manager.assert_not_called()
    assert lora_worker.lora_manager is None


def test_fused_lora_keeps_model_owned_path_and_scale_lists(lora_worker, mocker: MockerFixture):
    lora_worker.od_config.lora_path = ["a", "b"]
    lora_worker.od_config.lora_scale = [0.4, 0.7]
    lora_worker.model_runner.pipeline.lora_is_fused = True
    manager = mocker.patch.object(worker_module, "DiffusionLoRAManager")

    lora_worker.init_lora_manager()

    manager.assert_not_called()


def test_custom_pipeline_reload_failure_leaves_unloaded_runner(mocker: MockerFixture):
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = object.__new__(DiffusionModelRunner)
    worker.model_runner.pipeline = torch.nn.Identity()
    failure = RuntimeError("custom loader failed")
    mocker.patch.object(worker, "load_model", side_effect=failure)
    initialize_lora = mocker.patch.object(worker, "init_lora_manager")
    mocker.patch.object(torch.accelerator, "empty_cache")

    with pytest.raises(RuntimeError) as caught:
        worker_module.CustomPipelineWorkerExtension.re_init_pipeline(worker, {"pipeline_class": "unused"})

    assert caught.value is failure
    assert worker.model_runner.pipeline is None
    initialize_lora.assert_not_called()


def test_distilled_lora_preserves_multiple_transformer_paths(lora_worker, mocker: MockerFixture):
    lora_worker.od_config.lora_backend = worker_module.LoRABackend.DISTILL
    lora_worker.od_config.lora_path = ["high-noise", "low-noise"]
    lora_worker.od_config.lora_scale = [1.0]
    lora_worker.model_runner.pipeline.load_lora_weights = mocker.Mock()

    lora_worker.init_lora_manager()

    lora_worker.model_runner.pipeline.load_lora_weights.assert_called_once_with(["high-noise", "low-noise"])
    assert lora_worker.model_runner.pipeline.lora_is_fused


def test_missing_runner_kv_probe_still_gathers_rank_status(mocker: MockerFixture):
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = None
    gather = mocker.patch.object(worker_module, "_all_gather_rank_values", side_effect=lambda status: [status])

    with pytest.raises(RuntimeError, match="Diffusion KV receive failed.*Model runner not initialized"):
        worker.prepare_kv_for_forward(mocker.Mock(spec=DiffusionSchedulerOutput))

    gather.assert_called_once()
    assert gather.call_args.args[0][0] is False
