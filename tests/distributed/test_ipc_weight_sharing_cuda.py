# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import replace
from queue import Empty
from typing import Any

import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.distributed.ipc_weight_sharing import (
    ChannelAddress,
    WeightManifest,
    WeightSharingConsumer,
    WeightSharingEndpoint,
    WeightSharingProvider,
    build_weight_manifest,
    get_cuda_device_uuid,
    rebuild_cuda_tensor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.parallel]


def _consume_cuda_tensor(
    endpoint_payload: tuple[ChannelAddress, str, bytes],
    manifest_payload: dict[str, object],
    command_queue: Any,
    result_queue: Any,
    consumer_device_index: int | None = None,
) -> None:
    import torch

    try:
        endpoint = WeightSharingEndpoint(
            address=endpoint_payload[0],
            family=endpoint_payload[1],
            authkey=endpoint_payload[2],
        )
        provider_manifest = WeightManifest.from_payload(manifest_payload)
        local_device_index = provider_manifest.device_index if consumer_device_index is None else consumer_device_index
        manifest = replace(
            provider_manifest,
            device_index=local_device_index,
            tensors=tuple(replace(entry, device_index=local_device_index) for entry in provider_manifest.tensors),
        )
        torch.cuda.set_device(local_device_index)
        local_device_uuid = get_cuda_device_uuid(local_device_index)
        if local_device_uuid != provider_manifest.device_uuid:
            raise AssertionError(
                "consumer CUDA device UUID does not match the provider manifest: "
                f"{local_device_uuid!r} != {provider_manifest.device_uuid!r}"
            )
        allocated_before_mapping = torch.cuda.memory_allocated(local_device_index)
        ref_counter_key: tuple[str, int] | None = None

        def rebuild_and_record(handle: tuple[Any, ...]) -> object:
            nonlocal ref_counter_key
            ref_counter_handle, ref_counter_offset = handle[-4:-2]
            assert isinstance(ref_counter_handle, bytes)
            assert isinstance(ref_counter_offset, int)
            ref_counter_key = (ref_counter_handle.hex(), ref_counter_offset)
            return rebuild_cuda_tensor(handle)

        mapped = WeightSharingConsumer.connect(
            endpoint,
            manifest,
            rebuild=rebuild_and_record,
            heartbeat_interval=0.1,
        )
        try:
            mapped_tensor = mapped["layer.weight"]
            allocated_after_mapping = torch.cuda.memory_allocated(local_device_index)
            result_queue.put(
                {
                    "initial": mapped_tensor[:8].cpu().tolist(),
                    "consumer_allocation_delta_bytes": allocated_after_mapping - allocated_before_mapping,
                    "device_index": mapped_tensor.device.index,
                    "device_uuid": local_device_uuid,
                    "ref_counter_key": ref_counter_key,
                }
            )
            command_queue.get(timeout=30)
            torch.accelerator.synchronize(mapped_tensor.device.index)
            result_queue.put(
                {
                    "updated": mapped_tensor[:8].cpu().tolist(),
                    "device_index": mapped_tensor.device.index,
                    "device_uuid": local_device_uuid,
                }
            )
        finally:
            mapped.close()
    except Exception as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})


def _result(result_queue: Any) -> dict[str, object]:
    try:
        result = result_queue.get(timeout=30)
    except Empty as exc:
        raise AssertionError("CUDA IPC consumer did not report a result") from exc
    assert isinstance(result, dict)
    if "error" in result:
        raise AssertionError(result["error"])
    return result


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_cuda_ipc_maps_existing_allocation_across_processes() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device_index = 0
    with torch.cuda.device(device_index):
        tensor = torch.arange(4 * 1024 * 1024, dtype=torch.float32, device=f"cuda:{device_index}")
        manifest = build_weight_manifest(
            {"layer.weight": tensor},
            model_id="ipc-test-model",
            checkpoint_revision="revision-a",
            component="decoder",
            device_index=device_index,
            device_uuid=get_cuda_device_uuid(device_index),
            parallel_rank=0,
            parallel_size=1,
        )
    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"cuda-ipc-test-authkey"),
        manifest,
        {"layer.weight": tensor},
        heartbeat_interval=0.1,
    )
    provider.start()
    context = mp.get_context("spawn")
    command_queue = context.Queue()
    result_queue = context.Queue()
    endpoint_payload = (
        provider.endpoint.address,
        provider.endpoint.family,
        provider.endpoint.authkey,
    )
    process = context.Process(
        target=_consume_cuda_tensor,
        args=(endpoint_payload, manifest.to_payload(), command_queue, result_queue),
    )
    process.start()
    try:
        initial = _result(result_queue)
        assert initial["initial"] == list(range(8))
        assert initial["consumer_allocation_delta_bytes"] == 0
        assert initial["device_index"] == device_index
        assert initial["device_uuid"] == manifest.device_uuid

        tensor.add_(100)
        torch.accelerator.synchronize(tensor.device.index)
        command_queue.put("read-again")

        updated = _result(result_queue)
        assert updated["updated"] == list(range(100, 108))
        assert updated["device_index"] == device_index
        assert updated["device_uuid"] == manifest.device_uuid
    finally:
        provider.stop()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
        command_queue.close()
        result_queue.close()
    assert process.exitcode == 0


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_cuda_ipc_exports_fresh_ref_counter_for_each_consumer() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device_index = 0
    with torch.cuda.device(device_index):
        tensor = torch.arange(8, dtype=torch.float32, device=f"cuda:{device_index}")
        manifest = build_weight_manifest(
            {"layer.weight": tensor},
            model_id="ipc-test-model",
            checkpoint_revision="revision-a",
            component="decoder",
            device_index=device_index,
            device_uuid=get_cuda_device_uuid(device_index),
            parallel_rank=0,
            parallel_size=1,
        )

    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"cuda-ipc-multi-consumer-authkey"),
        manifest,
        {"layer.weight": tensor},
        heartbeat_interval=0.1,
    )
    provider.start()
    context = mp.get_context("spawn")
    command_queues = [context.Queue() for _ in range(2)]
    result_queues = [context.Queue() for _ in range(2)]
    endpoint_payload = (
        provider.endpoint.address,
        provider.endpoint.family,
        provider.endpoint.authkey,
    )
    processes = [
        context.Process(
            target=_consume_cuda_tensor,
            args=(endpoint_payload, manifest.to_payload(), command_queue, result_queue),
        )
        for command_queue, result_queue in zip(command_queues, result_queues, strict=True)
    ]
    for process in processes:
        process.start()

    try:
        initial_results = [_result(result_queue) for result_queue in result_queues]
        assert all(result["initial"] == list(range(8)) for result in initial_results)
        ref_counter_keys = [result["ref_counter_key"] for result in initial_results]
        assert len(set(ref_counter_keys)) == len(processes)

        tensor.add_(100)
        torch.accelerator.synchronize(tensor.device.index)
        for command_queue in command_queues:
            command_queue.put("read-again")

        updated_results = [_result(result_queue) for result_queue in result_queues]
        assert all(result["updated"] == list(range(100, 108)) for result in updated_results)
    finally:
        for command_queue in command_queues:
            command_queue.put("stop")
        provider.stop()
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
        for command_queue, result_queue in zip(command_queues, result_queues, strict=True):
            command_queue.close()
            result_queue.close()

    assert all(process.exitcode == 0 for process in processes)


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_cuda_ipc_maps_provider_device_to_consumer_local_ordinal() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("The device-ordinal remapping test needs at least two CUDA devices")

    provider_device_index = 1
    with torch.cuda.device(provider_device_index):
        tensor = torch.arange(8, dtype=torch.float32, device=f"cuda:{provider_device_index}")
        manifest = build_weight_manifest(
            {"layer.weight": tensor},
            model_id="ipc-test-model",
            checkpoint_revision="revision-a",
            component="decoder",
            device_index=provider_device_index,
            device_uuid=get_cuda_device_uuid(provider_device_index),
            parallel_rank=0,
            parallel_size=1,
        )

    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"cuda-ipc-device-remap-authkey"),
        manifest,
        {"layer.weight": tensor},
        heartbeat_interval=0.1,
    )
    provider.start()
    context = mp.get_context("spawn")
    command_queue = context.Queue()
    result_queue = context.Queue()
    process = context.Process(
        target=_consume_cuda_tensor,
        args=(
            (provider.endpoint.address, provider.endpoint.family, provider.endpoint.authkey),
            manifest.to_payload(),
            command_queue,
            result_queue,
            0,
        ),
    )
    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(provider_device_index)
    try:
        process.start()
    finally:
        if previous_visible_devices is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices
    try:
        initial = _result(result_queue)
        assert initial["initial"] == list(range(8))
        assert initial["device_index"] == 0
        assert initial["device_uuid"] == manifest.device_uuid
    finally:
        command_queue.put("stop")
        provider.stop()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
        command_queue.close()
        result_queue.close()

    assert process.exitcode == 0


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_cuda_ipc_preserves_parallel_rank_and_device_mapping() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("The dual-GPU IPC test needs at least two CUDA devices")

    context = mp.get_context("spawn")
    providers: list[WeightSharingProvider] = []
    tensors: list[Any] = []
    manifests: list[WeightManifest] = []
    processes: list[mp.Process] = []
    command_queues: list[Any] = []
    result_queues: list[Any] = []

    try:
        for parallel_rank in range(2):
            with torch.cuda.device(parallel_rank):
                tensor = torch.arange(
                    8,
                    dtype=torch.float32,
                    device=f"cuda:{parallel_rank}",
                )
                manifest = build_weight_manifest(
                    {"layer.weight": tensor},
                    model_id="ipc-test-model",
                    checkpoint_revision="revision-a",
                    component="decoder",
                    device_index=parallel_rank,
                    device_uuid=get_cuda_device_uuid(parallel_rank),
                    parallel_rank=parallel_rank,
                    parallel_size=2,
                )
            provider = WeightSharingProvider(
                WeightSharingEndpoint.tcp(
                    0,
                    f"cuda-ipc-test-authkey-{parallel_rank}".encode("ascii"),
                ),
                manifest,
                {"layer.weight": tensor},
                heartbeat_interval=0.1,
            )
            provider.start()
            providers.append(provider)
            tensors.append(tensor)
            manifests.append(manifest)

        for provider, manifest in zip(providers, manifests, strict=True):
            command_queue = context.Queue()
            result_queue = context.Queue()
            endpoint_payload = (
                provider.endpoint.address,
                provider.endpoint.family,
                provider.endpoint.authkey,
            )
            process = context.Process(
                target=_consume_cuda_tensor,
                args=(endpoint_payload, manifest.to_payload(), command_queue, result_queue),
            )
            command_queues.append(command_queue)
            result_queues.append(result_queue)
            process.start()
            processes.append(process)

        initial_results = [_result(result_queue) for result_queue in result_queues]
        for parallel_rank, result in enumerate(initial_results):
            assert result["initial"] == list(range(8))
            assert result["device_index"] == parallel_rank
            assert result["device_uuid"] == manifests[parallel_rank].device_uuid

        for parallel_rank, tensor in enumerate(tensors):
            with torch.cuda.device(parallel_rank):
                tensor.add_(100 * (parallel_rank + 1))
                torch.accelerator.synchronize(parallel_rank)

        for command_queue in command_queues:
            command_queue.put("read-again")
        updated_results = [_result(result_queue) for result_queue in result_queues]
        for parallel_rank, result in enumerate(updated_results):
            start = 100 * (parallel_rank + 1)
            expected = list(range(start, start + 8))
            assert result["updated"] == expected
            assert result["device_index"] == parallel_rank
            assert result["device_uuid"] == manifests[parallel_rank].device_uuid
    finally:
        for provider in providers:
            provider.stop()
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
        for command_queue, result_queue in zip(command_queues, result_queues, strict=True):
            command_queue.close()
            result_queue.close()

    assert all(process.exitcode == 0 for process in processes)
