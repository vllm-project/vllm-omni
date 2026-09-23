# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of Ray placement, rank assignment, RPC output, and cleanup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor import ray_executor as ray_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_cross_node_workers_get_global_rank_and_explicit_rendezvous(monkeypatch):
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,5")
    actors = []
    for ip in ["10.0.0.2", "10.0.0.1", "10.0.0.2"]:
        actor = Mock()
        actor.get_node_ip.remote.return_value = ip
        actor.get_open_port.remote.return_value = 23456
        actors.append(actor)
    factory = Mock()
    factory.remote.side_effect = actors
    fake_ray = Mock()
    fake_ray.remote.return_value = lambda cls: factory
    fake_ray.get.side_effect = lambda values, **kwargs: values
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    monkeypatch.setattr(ray_module, "get_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(ray_module, "PlacementGroupSchedulingStrategy", Mock())
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor.od_config = SimpleNamespace(num_gpus=3, ray_worker_env={"CUSTOM_PLUGIN_SETTING": "enabled"})
    executor.workers = []
    executor._init_workers(object())
    assert fake_ray.remote.call_args.kwargs["max_concurrency"] == 1
    assert fake_ray.remote.call_args.kwargs["concurrency_groups"] == {"health": 1}
    worker_env = fake_ray.remote.call_args.kwargs["runtime_env"]["env_vars"]
    assert worker_env["NCCL_SOCKET_IFNAME"] == "eth-test"
    assert worker_env["CUSTOM_PLUGIN_SETTING"] == "enabled"
    assert "CUDA_VISIBLE_DEVICES" not in worker_env
    assert len(fake_ray.get.call_args_list) == 3
    assert all(call.kwargs["timeout"] == ray_module._WORKER_INIT_TIMEOUT_S for call in fake_ray.get.call_args_list)
    assert [worker.ip for worker in executor.workers] == ["10.0.0.1", "10.0.0.2", "10.0.0.2"]
    for rank, metadata in enumerate(executor.workers):
        metadata.worker.init_worker.remote.assert_called_once_with(executor.od_config, rank, "tcp://10.0.0.1:23456")


def test_worker_env_forwards_stage_overrides_without_driver_identity(monkeypatch):
    from unittest.mock import patch

    driver_env = {
        "HF_TOKEN": "test-token",
        "VLLM_PLUGINS": "test-plugin",
        "NCCL_DEBUG": "WARN",
        "DIFFUSION_ATTENTION_BACKEND": "FLASH_ATTN",
        "PYTHONPATH": "/test/modules",
        "UNRELATED_SECRET": "do-not-copy",
        "HOME": "/driver/home",
    }
    protected = {
        "CUDA_VISIBLE_DEVICES": "7",
        "VLLM_HOST_IP": "10.0.0.1",
        "VLLM_HOST_PORT": "1234",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "10.0.0.1",
        "LOCAL_RANK": "7",
        "RANK": "7",
        "WORLD_SIZE": "8",
        "MASTER_ADDR": "driver-host",
        "MASTER_PORT": "1234",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
    config = SimpleNamespace(ray_worker_env={**protected, "NCCL_DEBUG": "INFO", "CUSTOM_PLUGIN_SETTING": "stage"})
    with patch.dict(ray_module.os.environ, {**driver_env, **protected}, clear=True):
        env = ray_module._worker_env(config)
        assert env == {key: value for key, value in driver_env.items() if key not in {"UNRELATED_SECRET", "HOME"}} | {
            "NCCL_DEBUG": "INFO",
            "CUSTOM_PLUGIN_SETTING": "stage",
        }
        assert ray_module.os.environ["NCCL_DEBUG"] == "WARN"


@pytest.mark.parametrize("rank", [0, 1])
def test_collective_executes_all_ranks_but_only_output_rank_returns_tensors(rank):
    actor = ray_module.RayDiffusionWorkerWrapper(rank)
    value = torch.ones(2, requires_grad=True)
    actor.worker = Mock()
    actor.worker.execute_method.return_value = DiffusionOutput(output=value)
    result = actor.execute_rpc("execute_model", output_rank=0, exec_all_ranks=True)
    actor.worker.execute_method.assert_called_once()
    assert result["replied"] == (rank == 0)
    if rank == 0:
        assert not result["result"].output.requires_grad
        torch.testing.assert_close(result["result"].output, value)
    else:
        assert result["result"] is None


def test_shutdown_kills_workers_without_waiting_and_releases_owned_group(monkeypatch):
    events = []
    fake_ray = Mock()
    fake_ray.get.side_effect = lambda futures, **kw: events.append("wait")
    fake_ray.kill.side_effect = lambda worker, **kw: events.append("kill")
    fake_ray.util.remove_placement_group.side_effect = lambda group: events.append("remove")
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    actor = Mock()
    actor.shutdown.remote.side_effect = lambda: events.append("shutdown")
    resources = ray_module._RayExecutorResources([ray_module._RayWorkerMetadata(actor)], object(), True)
    resources()
    assert events == ["kill", "remove"]
    fake_ray.kill.assert_called_once_with(actor, no_restart=True)
    actor.shutdown.remote.assert_not_called()
    fake_ray.get.assert_not_called()
    resources()
    assert events == ["kill", "remove"]
    assert resources.workers == []


def test_shutdown_preserves_borrowed_placement_group(monkeypatch):
    fake_ray = Mock()
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    ray_module._RayExecutorResources([], object(), False)()
    fake_ray.util.remove_placement_group.assert_not_called()


@pytest.mark.parametrize("failure_kind", ["GetTimeoutError", "RayActorError", "RayTaskError"])
@pytest.mark.parametrize("output_rank", [None, 0])
def test_rpc_failure_kills_peers_and_rejects_subsequent_work(monkeypatch, failure_kind, output_rank):
    fake_ray = Mock()
    fake_ray.exceptions = SimpleNamespace(
        **{name: type(name, (Exception,), {}) for name in ("GetTimeoutError", "RayActorError", "RayTaskError")}
    )
    failure = getattr(fake_ray.exceptions, failure_kind)("rank failed while peers are still running")
    fake_ray.get.side_effect = failure
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    actors = [Mock(), Mock()]
    group = object()
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    callback = Mock()
    executor._failure_callbacks = [callback]
    executor.workers = [ray_module._RayWorkerMetadata(actor, rank=rank) for rank, actor in enumerate(actors)]
    executor._finalizer = ray_module._RayExecutorResources(executor.workers, group, True)

    expected_error = TimeoutError if failure_kind == "GetTimeoutError" else ray_module.EngineDeadError
    with pytest.raises(expected_error) as raised:
        executor.collective_rpc("execute_model", unique_reply_rank=output_rank, exec_all_ranks=True)

    assert raised.value.__cause__ is failure
    assert executor.is_dead
    assert executor._is_failed
    assert executor._closed
    callback.assert_called_once_with()
    assert [call.args[0] for call in fake_ray.kill.call_args_list] == actors
    fake_ray.util.remove_placement_group.assert_called_once_with(group)

    with pytest.raises(RuntimeError, match="closed"):
        executor.collective_rpc("execute_model", unique_reply_rank=output_rank, exec_all_ranks=True)
    for actor in actors:
        actor.execute_rpc.remote.assert_called_once()
    fake_ray.get.assert_called_once()
    executor.shutdown()
    assert fake_ray.kill.call_count == len(actors)
    callback.assert_called_once_with()


def test_actor_death_during_shutdown_does_not_report_failure():
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor._closed = True
    executor._is_failed = False
    callback = Mock()
    executor._failure_callbacks = [callback]
    executor._mark_failed(RuntimeError("actor terminated"))
    assert not executor._is_failed
    callback.assert_not_called()


@pytest.mark.parametrize(
    "bundles, required, error",
    [
        ([{"GPU": 1}] * 4, 4, None),
        ([{"GPU": 1}] * 8, 4, None),
        ([{"CPU": 1}, {"GPU": 0.5}, {"GPU": 1}], 1, None),
        ([{"GPU": 1}] * 2, 4, "requires 4 full GPUs.*only 2"),
        ([{"GPU": 0.5}] * 4, 4, "requires 4 full GPUs.*only 0"),
        ([{"CPU": 4}], 1, "requires 1 full GPUs.*only 0"),
        ([{"GPU": 4}], 4, None),
        ([{"GPU": 2}, {"GPU": 2}], 4, None),
        ([{"GPU": 1.5}, {"GPU": 1.5}], 3, "requires 3 full GPUs.*only 2"),
    ],
)
def test_reused_placement_group_requires_full_gpu_capacity(monkeypatch, bundles, required, error):
    group = SimpleNamespace(bundle_specs=bundles)
    fake_ray = Mock()
    fake_ray.util.get_current_placement_group.return_value = group
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor.od_config = SimpleNamespace(num_gpus=required)

    if error is not None:
        with pytest.raises(ValueError, match=error):
            executor._get_or_create_placement_group()
    else:
        actual_group, owned = executor._get_or_create_placement_group()
        assert actual_group is group
        assert not owned

    fake_ray.util.placement_group.assert_not_called()
    fake_ray.util.remove_placement_group.assert_not_called()


@pytest.mark.parametrize("failure_phase", ["creation", "discovery", "port", "initialization"])
def test_worker_startup_failure_cleans_up_registered_actors(monkeypatch, failure_phase):
    actors = [Mock(), Mock()]
    for actor in actors:
        actor.get_node_ip.remote.return_value = "10.0.0.1"
        actor.get_open_port.remote.return_value = 23456
    factory = Mock()
    error = RuntimeError("startup failed")
    factory.remote.side_effect = [actors[0], error] if failure_phase == "creation" else actors
    fake_ray = Mock()
    fake_ray.is_initialized.return_value = True
    fake_ray.remote.return_value = lambda cls: factory
    call_count = 0

    def get(values, **kwargs):
        nonlocal call_count
        call_count += 1
        assert kwargs["timeout"] == ray_module._WORKER_INIT_TIMEOUT_S
        if call_count == {"discovery": 1, "port": 2, "initialization": 3}.get(failure_phase):
            raise error
        return values

    fake_ray.get.side_effect = get
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    monkeypatch.setattr(ray_module, "get_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(ray_module, "PlacementGroupSchedulingStrategy", Mock())
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor.od_config = SimpleNamespace(num_gpus=2)
    group = object()
    executor._get_or_create_placement_group = Mock(return_value=(group, True))

    with pytest.raises(RuntimeError, match="startup failed"):
        executor._init_executor()

    expected = actors[:1] if failure_phase == "creation" else actors
    assert [call.args[0] for call in fake_ray.kill.call_args_list] == expected
    fake_ray.util.remove_placement_group.assert_called_once_with(group)
    assert executor.workers == []
    assert executor._closed


@pytest.mark.parametrize("failure_type", [TimeoutError, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_placement_group_wait_failure_cleans_up_and_preserves_error(monkeypatch, failure_type, cleanup_fails):
    fake_ray = Mock()
    fake_ray.exceptions.GetTimeoutError = TimeoutError
    fake_ray.util.get_current_placement_group.return_value = None
    group = fake_ray.util.placement_group.return_value
    original_error = failure_type("readiness failed")
    fake_ray.get.side_effect = original_error
    if cleanup_fails:
        fake_ray.util.remove_placement_group.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor.od_config = SimpleNamespace(num_gpus=2)

    expected_type = ValueError if failure_type is TimeoutError else failure_type
    with pytest.raises(expected_type) as raised:
        executor._get_or_create_placement_group()

    if failure_type is TimeoutError:
        assert raised.value.__cause__ is original_error
        assert "Cannot reserve 2 Ray GPU bundle(s)" in str(raised.value)
    else:
        assert raised.value is original_error
    fake_ray.util.remove_placement_group.assert_called_once_with(group)


def test_health_probes_use_separate_concurrency_group(monkeypatch):
    fake_ray = Mock()
    fake_ray.get.return_value = [True, True]
    monkeypatch.setattr(ray_module, "ray", fake_ray)
    actors = [Mock(), Mock()]
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    executor.workers = [ray_module._RayWorkerMetadata(actor, rank=rank) for rank, actor in enumerate(actors)]

    executor.check_health()

    for actor in actors:
        actor.check_alive.options.assert_called_once_with(concurrency_group="health")
        actor.check_alive.options.return_value.remote.assert_called_once_with()
        actor.check_alive.remote.assert_not_called()
    fake_ray.get.assert_called_once_with(
        [actor.check_alive.options.return_value.remote.return_value for actor in actors],
        timeout=ray_module._HEALTH_CHECK_TIMEOUT_S,
    )
    assert not executor._is_failed


def _make_request_executor(monkeypatch, prompts, component="text_encoder", allgather=True):
    executor = object.__new__(ray_module.RayDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    executor._failure_callbacks = []
    executor._finalizer = Mock()
    executor.od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=len(prompts)),
        diffusion_offload_config={
            "mode": "layer",
            "components": [component],
            "layer_options": {component: {"weight_transfer": "allgather" if allgather else "rank-local"}},
        },
    )
    monkeypatch.setattr(ray_module, "build_request_batch_sampling_params_key", lambda req: ())
    requests = []
    for index, prompt in enumerate(prompts):
        request_id = str(index)
        req = SimpleNamespace(request_id=request_id, prompt=prompt, sampling_params=SimpleNamespace(extra_args=None))
        requests.append(SimpleNamespace(request_id=request_id, req=req, diffusion_kv_metadata=None))
    return executor, SimpleNamespace(scheduled_new_reqs=requests, kv_prefetch_job=None)


@pytest.mark.parametrize("field", ["prompt_embeds", "negative_prompt_embeds"])
def test_text_encoder_allgather_rejects_mismatched_embedding_presence(monkeypatch, field):
    executor, scheduled = _make_request_executor(monkeypatch, ["first", {"prompt": "second", field: object()}])
    executor.collective_rpc = Mock()
    with pytest.raises(ValueError, match="same positive/negative prompt embedding fields"):
        executor.execute_request(scheduled)
    executor.collective_rpc.assert_not_called()


def test_text_encoder_allgather_accepts_matching_precomputed_embedding_paths(monkeypatch):
    executor, scheduled = _make_request_executor(
        monkeypatch, [{"prompt_embeds": object()}, {"prompt_embeds": object()}]
    )
    results = [DiffusionOutput(output="first"), DiffusionOutput(output="second")]
    executor.collective_rpc = Mock(return_value=[{"dp_rank": rank, "output": out} for rank, out in enumerate(results)])
    output = executor.execute_request(scheduled)
    assert [item.result for item in output.runner_outputs] == results
    executor.collective_rpc.assert_called_once()


@pytest.mark.parametrize("allgather", [False, True])
def test_single_request_allgather_timeout_option(monkeypatch, allgather):
    executor, scheduled = _make_request_executor(monkeypatch, ["prompt"], allgather=allgather)
    executor.collective_rpc = Mock(return_value=DiffusionOutput(output="done"))
    executor.execute_request(scheduled)
    kwargs = executor.collective_rpc.call_args.kwargs
    if allgather:
        assert kwargs["timeout"] == ray_module._DLO_DP_WAVE_TIMEOUT_S
    else:
        assert "timeout" not in kwargs


def test_single_request_allgather_timeout_fails_executor_and_cleans_up(monkeypatch):
    executor, scheduled = _make_request_executor(monkeypatch, ["prompt"])
    callback = Mock()
    executor.register_failure_callback(callback)
    executor.collective_rpc = Mock(side_effect=TimeoutError("AllGather timed out"))
    output = executor.execute_request(scheduled)
    assert executor._is_failed
    assert executor._closed
    executor._finalizer.assert_called_once()
    callback.assert_called_once()
    assert "AllGather timed out" in output.runner_outputs[0].result.error


@pytest.mark.parametrize("wrapper", ["media", "output", "batch"])
def test_move_to_cpu_handles_typed_video_media(wrapper):
    from vllm_omni.diffusion.media import (
        DiffusionMediaOutput,
        VideoMediaOutput,
        VideoTensorEncoding,
        VideoTensorLayout,
        VideoTensorSpec,
        VideoValueRange,
    )
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    tensor = torch.ones((1, 3, 2, 4, 4), requires_grad=True)
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=tensor,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.ZERO_TO_ONE,
            ),
        ),
        prepared_for_transport=True,
    )
    value = media
    if wrapper != "media":
        value = DiffusionOutput(media=media, to_cpu=False)
    if wrapper == "batch":
        value = BatchRunnerOutput.from_list([RunnerOutput(request_id="video", finished=True, result=value)])

    moved = ray_module._move_to_cpu(value)
    if wrapper == "batch":
        moved = moved.runner_outputs[0].result
    if wrapper != "media":
        moved = moved.media

    moved.validate()
    assert moved.video.tensor.device.type == "cpu"
    assert not moved.video.tensor.requires_grad
    assert moved.video.spec == media.video.spec
    assert moved.prepared_for_transport
    torch.testing.assert_close(moved.video.tensor, tensor.detach())
    assert media.video.tensor is tensor
    assert tensor.requires_grad
