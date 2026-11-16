# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject unsupported AlignCache batches before worker-owned weight changes."""

import pytest
from pytest_mock import MockerFixture

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("count,with_lora,cached", [(2, False, False), (1, True, False), (1, False, True)])
def test_aligncache_batch_admission_precedes_lora_activation(
    mocker: MockerFixture, count: int, with_lora: bool, cached: bool
):
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = mocker.Mock(spec=DiffusionModelRunner)
    worker.model_runner.execute_model_batch.return_value = BatchRunnerOutput.from_list([])
    worker.lora_manager = mocker.Mock(spec=DiffusionLoRAManager)
    requests = []
    for index in range(count):
        sampling = OmniDiffusionSamplingParams(seed=11)
        if with_lora:
            sampling.lora_request = LoRARequest("test", 1, "unused-lora")
        request = OmniDiffusionRequest(prompt="test", sampling_params=sampling, request_id=f"request-{index}")
        requests.append(NewRequestData(request_id=request.request_id, req=request))
    scheduled = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=requests,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=count,
        num_waiting_reqs=0,
    )
    if cached:
        scheduled.scheduled_cached_reqs.request_ids.append("cached-request")
    config = OmniDiffusionConfig(cache_backend="align_cache", enforce_eager=True)
    with pytest.raises(OmniClientError, match="align_cache") as caught:
        worker.execute_model_batch(scheduled, config)
    assert caught.value.status_code == 400
    worker.lora_manager.set_active_adapter.assert_not_called()
    worker.model_runner.execute_model_batch.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("enforce_eager", False),
        ("enable_cpu_offload", True),
        ("enable_layerwise_offload", True),
        ("enable_distributed_layerwise_offload", True),
        ("diffusion_offload_config", {"dit": "layerwise"}),
        ("streaming_output", True),
        ("step_execution", True),
        ("dtype", "float16"),
        ("lora_path", "unused-lora"),
        ("quantization_config", "fp8"),
        ("enable_sleep_mode", True),
        ("num_gpus", 2),
        ("stage_role", "denoise"),
        ("diffusion_kv_mode", "paged_scheduler"),
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.data_parallel_size", 2),
        ("parallel_config.tensor_parallel_size", 2),
        ("parallel_config.sequence_parallel_size", 2),
        ("parallel_config.ulysses_degree", 2),
        ("parallel_config.ring_degree", 2),
        ("parallel_config.allgather_degree", 2),
        ("parallel_config.cfg_parallel_size", 2),
        ("parallel_config.vae_patch_parallel_size", 2),
        ("parallel_config.enable_expert_parallel", True),
        ("parallel_config.use_hsdp", True),
    ],
)
def test_aligncache_static_admission_precedes_platform_and_model_loading(mocker: MockerFixture, field, value):
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module

    config = OmniDiffusionConfig(cache_backend="align_cache", enforce_eager=True)
    # Mutate the typed config after construction to test the loading boundary.
    if field.startswith("parallel_config."):
        setattr(config.parallel_config, field.split(".", 1)[1], value)
    else:
        setattr(config, field, value)
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = config
    init = mocker.patch.object(runner_module.current_omni_platform, "init_diffusion_model_runner_runtime")
    loader = mocker.patch.object(runner_module, "DiffusersPipelineLoader")
    with pytest.raises(ValueError, match=field):
        runner.load_model()
    init.assert_not_called()
    loader.assert_not_called()
