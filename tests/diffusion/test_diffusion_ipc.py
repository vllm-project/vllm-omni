# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib
import multiprocessing as mp
import queue
import threading
import time
from collections import OrderedDict
from multiprocessing import shared_memory
from multiprocessing.connection import Connection

import numpy as np
import pytest
import torch
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.ipc import (
    _SHM_TENSOR_THRESHOLD,
    DIFFUSION_RPC_RESULT_ENVELOPE,
    _pack_value_if_large,
    _unpack_if_shm_handle,
    pack_diffusion_output_shm,
    unpack_diffusion_output_shm,
)
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _large_numel(dtype: torch.dtype) -> int:
    return int(_SHM_TENSOR_THRESHOLD // torch.empty((), dtype=dtype).element_size()) + 1


def _cleanup_shm_handle(value: object) -> None:
    if isinstance(value, dict) and value.get("__tensor_shm__"):
        with contextlib.suppress(FileNotFoundError):
            _unpack_if_shm_handle(value)


def _result_queue_worker(connection: Connection, rank: int) -> None:
    """Send one nested NumPy result through a worker-owned MessageQueue."""
    result_mq = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
    try:
        connection.send(result_mq.export_handle())
        assert connection.recv() == "reader-ready"

        frames = np.full(300_000, rank, dtype=np.float32)
        audio = np.arange(300_000, dtype=np.float32) + rank
        output = DiffusionOutput(
            output={
                "rank": rank,
                "nested": {"frames": frames},
                "audio": [audio],
            }
        )
        pack_diffusion_output_shm(output)
        connection.send(
            [
                output.output["nested"]["frames"]["name"],
                output.output["audio"][0]["name"],
            ]
        )
        result_mq.enqueue(output)
        assert connection.recv() == "shutdown"
    finally:
        result_mq.shutdown()
        connection.close()


def test_per_worker_result_queues_release_nested_numpy_shm_and_processes() -> None:
    """Exercise the production per-worker queue/pump/SHM lifecycle."""
    ctx = mp.get_context("spawn")
    parent_connections = []
    processes = []
    result_mqs = []
    shm_names = []
    executor = None

    try:
        for rank in range(2):
            parent_connection, child_connection = ctx.Pipe()
            process = ctx.Process(
                target=_result_queue_worker,
                args=(child_connection, rank),
                name=f"IPCResultWorker-{rank}",
            )
            process.start()
            child_connection.close()
            parent_connections.append(parent_connection)
            processes.append(process)

        for connection in parent_connections:
            handle = connection.recv()
            result_mqs.append(MessageQueue.create_from_handle(handle, 0))
            connection.send("reader-ready")
            shm_names.extend(connection.recv())

        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._result_mqs = result_mqs
        executor._result_mq = result_mqs[0]
        executor._pump_running = False
        executor._pump_stop = threading.Event()
        executor._sync_result_buffer = queue.Queue()
        executor._is_failed = False
        executor._futures_lock = threading.RLock()
        executor._rpc_futures = {}
        executor._output_futures = {}
        executor._completed_outputs = {}
        executor._dropped_output_ids = OrderedDict()
        executor._batch_split_map = {}
        executor._start_result_pump()

        deadline = time.monotonic() + 10.0
        while executor._sync_result_buffer.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert executor._sync_result_buffer.qsize() == 2

        outputs = []
        for _ in range(2):
            output = executor._sync_result_buffer.get_nowait()
            unpack_diffusion_output_shm(output)
            outputs.append(output)

        assert sorted(output.output["rank"] for output in outputs) == [0, 1]
        for output in outputs:
            rank = output.output["rank"]
            assert output.output["nested"]["frames"][0] == rank
            assert output.output["audio"][0][0] == rank

        executor._pump_stop.set()
        for thread in executor._result_pump_threads:
            thread.join(timeout=3.0)
        assert all(not thread.is_alive() for thread in executor._result_pump_threads)
    finally:
        if executor is not None:
            executor._pump_stop.set()
            for thread in getattr(executor, "_result_pump_threads", []):
                thread.join(timeout=3.0)

        for connection in parent_connections:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                connection.send("shutdown")
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for result_mq in result_mqs:
            with contextlib.suppress(Exception):
                result_mq.shutdown()
        for connection in parent_connections:
            connection.close()

    assert all(process.exitcode == 0 for process in processes)
    for name in shm_names:
        try:
            leaked = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            continue
        leaked.close()
        pytest.fail(f"shared memory segment {name} still exists after unpack")


def test_diffusion_output_dict_tensors_round_trip_through_shm() -> None:
    image = torch.arange(300_000, dtype=torch.float32)
    video = torch.arange(300_000, dtype=torch.float32) * 2
    output = DiffusionOutput(output={"image": image, "video": video, "metadata": {"keep": "inline"}})

    pack_diffusion_output_shm(output)

    assert output.output["image"]["__tensor_shm__"] is True
    assert output.output["video"]["__tensor_shm__"] is True
    assert output.output["metadata"] == {"keep": "inline"}

    unpack_diffusion_output_shm(output)

    torch.testing.assert_close(output.output["image"], image)
    torch.testing.assert_close(output.output["video"], video)
    assert output.output["metadata"] == {"keep": "inline"}


def test_diffusion_output_tuple_tensors_round_trip_through_shm() -> None:
    # LTX2 returns (video, audio) tuples as DiffusionOutput.output.
    video = torch.arange(300_000, dtype=torch.float32)
    audio = torch.arange(300_000, dtype=torch.float32) * 3
    output = DiffusionOutput(output=(video, audio))

    pack_diffusion_output_shm(output)

    assert isinstance(output.output, tuple)
    assert output.output[0]["__tensor_shm__"] is True
    assert output.output[1]["__tensor_shm__"] is True

    unpack_diffusion_output_shm(output)

    assert isinstance(output.output, tuple)
    torch.testing.assert_close(output.output[0], video)
    torch.testing.assert_close(output.output[1], audio)


def test_diffusion_output_list_tensors_round_trip_through_shm() -> None:
    frames = [torch.arange(300_000, dtype=torch.float32), torch.arange(300_000, dtype=torch.float32) + 1]
    output = DiffusionOutput(output=list(frames))

    pack_diffusion_output_shm(output)

    assert isinstance(output.output, list)
    assert all(isinstance(item, dict) and item["__tensor_shm__"] is True for item in output.output)

    unpack_diffusion_output_shm(output)

    assert isinstance(output.output, list)
    torch.testing.assert_close(output.output[0], frames[0])
    torch.testing.assert_close(output.output[1], frames[1])


def test_diffusion_output_numpy_array_round_trips_through_shm() -> None:
    frames = np.arange(300_000, dtype=np.float32)
    output = DiffusionOutput(output=frames)

    pack_diffusion_output_shm(output)

    assert output.output["__ndarray_shm__"] is True

    unpack_diffusion_output_shm(output)

    assert isinstance(output.output, np.ndarray)
    np.testing.assert_array_equal(output.output, frames)


def test_numpy_object_array_stays_inline() -> None:
    values = np.empty(200_000, dtype=object)
    values[:] = "safe-to-pickle"

    packed = _pack_value_if_large(values)

    assert packed is values


def test_rpc_result_envelope_diffusion_output_round_trips_through_shm() -> None:
    tensor = torch.arange(300_000, dtype=torch.float32)
    envelope = {
        "type": DIFFUSION_RPC_RESULT_ENVELOPE,
        "result": DiffusionOutput(output=tensor),
        "rank_statuses": [{"rank": 0, "ok": True}],
    }

    packed = pack_diffusion_output_shm(envelope)

    assert packed is envelope
    result = packed["result"]
    assert isinstance(result, DiffusionOutput)
    assert result.output["__tensor_shm__"] is True
    assert packed["rank_statuses"] == [{"rank": 0, "ok": True}]

    unpacked = unpack_diffusion_output_shm(packed)

    assert unpacked is envelope
    result = unpacked["result"]
    assert isinstance(result, DiffusionOutput)
    torch.testing.assert_close(result.output, tensor)
    assert unpacked["rank_statuses"] == [{"rank": 0, "ok": True}]


def test_rpc_result_envelope_dp_tagged_output_round_trips_through_shm() -> None:
    frames = np.arange(300_000, dtype=np.float32)
    envelope = {
        "type": DIFFUSION_RPC_RESULT_ENVELOPE,
        "result": {"dp_rank": 1, "output": DiffusionOutput(output=frames)},
        "rank_statuses": [{"rank": 1, "ok": True}],
    }

    packed = pack_diffusion_output_shm(envelope)
    tagged = packed["result"]
    assert tagged["dp_rank"] == 1
    assert tagged["output"].output["__ndarray_shm__"] is True

    unpacked = unpack_diffusion_output_shm(packed)
    tagged = unpacked["result"]
    assert tagged["dp_rank"] == 1
    np.testing.assert_array_equal(tagged["output"].output, frames)


def test_batch_runner_output_round_trips_nested_results_through_shm() -> None:
    first = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    second = torch.arange(_large_numel(torch.float32), dtype=torch.float32) + 1
    output = BatchRunnerOutput.from_list(
        [
            RunnerOutput(request_id="req-0", finished=True, result=DiffusionOutput(output=first)),
            RunnerOutput(request_id="req-1", finished=True, result=DiffusionOutput(output={"image": second})),
            RunnerOutput(request_id="req-error", finished=True, result=DiffusionOutput(error="boom")),
        ]
    )

    pack_diffusion_output_shm(output)

    assert output.runner_outputs[0].result.output["__tensor_shm__"] is True
    assert output.runner_outputs[1].result.output["image"]["__tensor_shm__"] is True
    assert output.runner_outputs[2].result.error == "boom"

    unpack_diffusion_output_shm(output)

    torch.testing.assert_close(output["req-0"].result.output, first)
    torch.testing.assert_close(output["req-1"].result.output["image"], second)
    assert output["req-error"].result.error == "boom"


def test_per_request_views_of_a_batch_tensor_do_not_ship_whole_storage() -> None:
    """Regression: per-request slices used to pickle the whole batch storage.

    Each view is under the packing threshold, but pickle writes the entire
    shared storage per view. Packing must key off storage nbytes so those
    views become SHM handles instead of going on the wire.
    """
    num_requests = 16
    # Each row stays under the threshold; the shared storage is well over it.
    row_numel = _SHM_TENSOR_THRESHOLD // (2 * torch.empty((), dtype=torch.float32).element_size())
    batch = torch.arange(num_requests * row_numel, dtype=torch.float32).reshape(num_requests, row_numel)
    per_request = batch[0]
    assert per_request.nelement() * per_request.element_size() <= _SHM_TENSOR_THRESHOLD
    assert per_request.untyped_storage().nbytes() > _SHM_TENSOR_THRESHOLD

    packed_view = _pack_value_if_large(per_request)
    try:
        assert isinstance(packed_view, dict)
        assert packed_view["__tensor_shm__"] is True
    finally:
        _cleanup_shm_handle(packed_view)

    output = BatchRunnerOutput.from_list(
        [
            RunnerOutput(
                request_id=f"req-{i}",
                finished=True,
                result=DiffusionOutput(trajectory_latents=batch[i]),
            )
            for i in range(num_requests)
        ]
    )

    pack_diffusion_output_shm(output)

    assert output.runner_outputs[0].result.trajectory_latents["__tensor_shm__"] is True

    unpack_diffusion_output_shm(output)
    for i in range(num_requests):
        torch.testing.assert_close(output[f"req-{i}"].result.trajectory_latents, batch[i])


def test_pack_value_keeps_tensor_at_threshold_inline() -> None:
    tensor = torch.arange(
        _SHM_TENSOR_THRESHOLD // torch.empty((), dtype=torch.float32).element_size(),
        dtype=torch.float32,
    )

    packed = _pack_value_if_large(tensor)

    assert packed is tensor


def test_pack_value_packs_large_tensor_and_round_trips() -> None:
    tensor = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    packed = _pack_value_if_large(tensor)

    try:
        assert isinstance(packed, dict)
        assert packed["__tensor_shm__"] is True
        assert packed["shape"] == [tensor.numel()]
        assert packed["torch_dtype"] == "torch.float32"

        unpacked = _unpack_if_shm_handle(packed)
        assert isinstance(unpacked, torch.Tensor)
        torch.testing.assert_close(unpacked, tensor)
    finally:
        _cleanup_shm_handle(packed)


def test_pack_value_recurses_nested_dicts_and_lists_without_mutating_inline_values() -> None:
    large = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    small = torch.arange(8, dtype=torch.float32)
    list_tensor = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    original_list = [list_tensor]
    payload = {
        "media": {
            "large": large,
            "small": small,
        },
        "list_value": original_list,
        "metadata": {"prompt": "keep inline"},
    }

    packed = _pack_value_if_large(payload)

    try:
        assert isinstance(packed, dict)
        assert packed is not payload
        media = packed["media"]
        list_value = packed["list_value"]
        assert isinstance(media, dict)
        assert isinstance(list_value, list)
        assert media is not payload["media"]
        assert media["large"]["__tensor_shm__"] is True
        assert media["small"] is small
        # Lists are recursed too: the large tensor inside is packed and a new
        # list is returned, while the input payload is left untouched.
        assert list_value is not original_list
        assert list_value[0]["__tensor_shm__"] is True
        assert original_list[0] is list_tensor
        assert packed["metadata"] == {"prompt": "keep inline"}

        torch.testing.assert_close(_unpack_if_shm_handle(media["large"]), large)
        torch.testing.assert_close(_unpack_if_shm_handle(list_value[0]), list_tensor)
    finally:
        if isinstance(packed, dict):
            _cleanup_shm_handle(packed.get("media", {}).get("large"))
            list_value = packed.get("list_value")
            if isinstance(list_value, list) and list_value:
                _cleanup_shm_handle(list_value[0])


def test_pack_value_preserves_dtype_shape_and_values_for_bfloat16() -> None:
    tensor = torch.arange(_large_numel(torch.bfloat16), dtype=torch.float32).to(torch.bfloat16).reshape(1, -1)
    packed = _pack_value_if_large(tensor)

    try:
        assert isinstance(packed, dict)
        assert packed["__tensor_shm__"] is True
        assert packed["shape"] == list(tensor.shape)
        assert packed["torch_dtype"] == "torch.bfloat16"
        assert packed["numpy_dtype"] == "float32"

        unpacked = _unpack_if_shm_handle(packed)
        assert isinstance(unpacked, torch.Tensor)
        assert unpacked.dtype == torch.bfloat16
        torch.testing.assert_close(unpacked, tensor)
    finally:
        _cleanup_shm_handle(packed)


def test_pack_value_packs_non_contiguous_large_tensor_values() -> None:
    tensor = torch.arange(_large_numel(torch.float32) * 2, dtype=torch.float32).reshape(-1, 2)[:, 0]
    assert not tensor.is_contiguous()

    packed = _pack_value_if_large(tensor)

    try:
        assert isinstance(packed, dict)
        assert packed["__tensor_shm__"] is True
        assert packed["shape"] == list(tensor.shape)

        unpacked = _unpack_if_shm_handle(packed)
        assert isinstance(unpacked, torch.Tensor)
        torch.testing.assert_close(unpacked, tensor)
    finally:
        _cleanup_shm_handle(packed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_tensor_with_d2h_stream_never_allocates_pinned_copy(monkeypatch, dtype, strided) -> None:
    from typing import cast

    from vllm_omni.diffusion import ipc

    count = _large_numel(dtype)
    source = torch.arange(count * 2, dtype=torch.float32).to(dtype)[::2]
    if not strided:
        source = source.contiguous()

    def forbidden_accelerator_access(*args, **kwargs):
        pytest.fail("Packing a CPU tensor must not access accelerator streams or allocate a pinned CPU copy")

    monkeypatch.setattr(torch.accelerator, "current_stream", forbidden_accelerator_access)
    monkeypatch.setattr(torch.accelerator, "set_stream", forbidden_accelerator_access)
    original_empty = torch.empty

    def no_pinned_empty(*args, pin_memory=False, **kwargs):
        if pin_memory:
            pytest.fail("A CPU output must not be duplicated into pinned storage")
        return original_empty(*args, pin_memory=pin_memory, **kwargs)

    monkeypatch.setattr(torch, "empty", no_pinned_empty)
    handle = None
    try:
        handle = ipc._tensor_to_shm(source, d2h_stream=cast(torch.Stream, object()))
        actual = ipc._tensor_from_shm(handle)
        handle = None
        assert actual.device.type == "cpu" and actual.dtype == dtype
        torch.testing.assert_close(actual, source, rtol=0, atol=0)
    finally:
        _cleanup_shm_handle(handle)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("async_copy", [False, True])
def test_cpu_bfloat16_transport_avoids_full_float_staging(monkeypatch, strided, async_copy) -> None:
    """A BF16 video must not coexist with a full widened staging tensor and SHM."""
    from typing import cast

    from torch.utils._python_dispatch import TorchDispatchMode

    from vllm_omni.diffusion import ipc

    bits = torch.tensor([0, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC1, 0x7F81], dtype=torch.uint16)
    source = bits.view(torch.bfloat16).repeat(64).reshape(1, 2, 8, 32)
    if strided:
        source = source[..., ::2]
    expected_wire = source.float().numpy().copy()
    expected_restored_bits = source.float().bfloat16().view(torch.uint16)

    class ForbidFullFloatStaging(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if (
                func == torch.ops.aten._to_copy.default
                and args[0].device.type == "cpu"
                and args[0].dtype == torch.bfloat16
                and kwargs.get("dtype") == torch.float32
                and args[0].numel() == source.numel()
            ):
                pytest.fail("BF16 transport allocated a full FP32 staging tensor before SHM")
            return func(*args, **kwargs)

    handle = None
    try:
        with ForbidFullFloatStaging():
            handle = ipc._tensor_to_shm(source, d2h_stream=cast(torch.Stream, object()) if async_copy else None)
        assert handle["numpy_dtype"] == "float32"
        assert handle["torch_dtype"] == "torch.bfloat16"
        shm = shared_memory.SharedMemory(name=handle["name"])
        try:
            wire = np.ndarray(handle["shape"], dtype=np.float32, buffer=shm.buf)
            assert wire.tobytes() == expected_wire.tobytes()
            del wire
        finally:
            shm.close()
        restored = ipc._tensor_from_shm(handle)
        handle = None
        assert torch.equal(restored.view(torch.uint16), expected_restored_bits)
    finally:
        _cleanup_shm_handle(handle)


def test_cpu_bfloat16_shm_conversion_failure_unlinks_segment(monkeypatch) -> None:
    from vllm_omni.diffusion import ipc

    source = torch.ones(32, dtype=torch.bfloat16)
    created_names: list[str] = []
    original_new = ipc._new_shm_array
    original_copy = torch.Tensor.copy_

    def track_segment(*args, **kwargs):
        shm, array, handle = original_new(*args, **kwargs)
        created_names.append(shm.name)
        return shm, array, handle

    def fail_conversion(destination, value, *args, **kwargs):
        if destination.dtype == torch.float32 and value.dtype == torch.bfloat16:
            raise RuntimeError("injected BF16 conversion failure")
        return original_copy(destination, value, *args, **kwargs)

    monkeypatch.setattr(ipc, "_new_shm_array", track_segment)
    monkeypatch.setattr(torch.Tensor, "copy_", fail_conversion)
    try:
        with pytest.raises(RuntimeError, match="injected BF16 conversion failure"):
            ipc._tensor_to_shm(source)
        assert torch.equal(source, torch.ones_like(source))
        assert created_names
        for name in created_names:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)
    finally:
        for name in created_names:
            try:
                shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                continue
            shm.close()
            shm.unlink()
