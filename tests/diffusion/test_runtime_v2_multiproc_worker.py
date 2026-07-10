# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import pickle
import queue
import signal
import threading
from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.runtime_v2.multiproc_worker import (
    FetchArtifactsCommand,
    FetchArtifactsResult,
    MultiprocWorkerPool,
    WorkerProcessHandle,
    _clone_diffusion_output_for_transport,
    _deserialize_artifact_value,
    _serialize_artifact_value,
    _WorkerProcessRuntime,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactValue,
    ParallelSpec,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _minimal_od_config() -> Mock:
    od_config = Mock()
    od_config.master_port = 30005
    return od_config


def test_clone_for_transport_preserves_all_fields():
    """SHM transport must preserve every carried DiffusionOutput field."""
    original = DiffusionOutput(
        output=torch.zeros(2, 3),
        trajectory_timesteps=torch.ones(4),
        trajectory_latents=torch.ones(2, 2),
        trajectory_log_probs=torch.ones(3),
        trajectory_decoded=[],
        error="boom",
        error_status_code=503,
        error_type="service_unavailable",
        aborted=True,
        abort_message="user aborted",
        custom_output={"k": "v"},
        finished=False,
        chunk_index=7,
        total_chunks=9,
        stage_durations={"dit": 1.5},
        peak_memory_mb=123.0,
    )

    clone = _clone_diffusion_output_for_transport(original)

    # Every dataclass field must survive the clone (no silent drops). Exclude
    # to_cpu, which is a construction-time directive rather than carried state.
    for f in fields(DiffusionOutput):
        if f.name == "to_cpu":
            continue
        orig_val = getattr(original, f.name)
        clone_val = getattr(clone, f.name)
        if isinstance(orig_val, torch.Tensor):
            assert torch.equal(clone_val, orig_val), f"tensor field {f.name} not preserved"
        else:
            assert clone_val == orig_val, f"field {f.name} not preserved: {clone_val!r} != {orig_val!r}"


def test_serialize_round_trip_pickle_transport():
    handle = ArtifactHandle(
        request_id="req-1",
        artifact_id="latent",
        kind=ArtifactKind.TENSOR,
        layout=ArtifactLayout.WORKER_LOCAL,
    )
    original = ArtifactValue(handle=handle, value={"data": [1, 2, 3]})

    serialized = _serialize_artifact_value(original)
    assert serialized.transport == "pickle"
    assert serialized.handle == handle

    restored = _deserialize_artifact_value(serialized)
    assert restored.handle == handle
    assert restored.value == {"data": [1, 2, 3]}


# 1.2 MB > the 1 MB _SHM_TENSOR_THRESHOLD, so packing actually creates a segment.
_LARGE_NUMEL = 300_000


def _serialized_shm_names(serialized) -> set[str]:
    names = set()

    def visit(value):
        if isinstance(value, dict):
            if value.get("__tensor_shm__"):
                names.add(value["name"])
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, DiffusionOutput):
            for field in fields(DiffusionOutput):
                visit(getattr(value, field.name))

    visit(pickle.loads(serialized.payload))
    return names


def _shm_exists(name: str) -> bool:
    from multiprocessing import shared_memory

    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    shm.close()
    return True


def _unlink_shm_by_name(name: str) -> None:
    """Best-effort unlink of a POSIX SHM segment by name (test cleanup)."""
    from multiprocessing import shared_memory

    with contextlib.suppress(FileNotFoundError):
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()


def _output_handle(request_id: str) -> ArtifactHandle:
    return ArtifactHandle(
        request_id=request_id,
        artifact_id="out",
        kind=ArtifactKind.OUTPUT,  # prefer_shm_output only fires for OUTPUT kind
        layout=ArtifactLayout.HOST,
    )


@pytest.mark.parametrize(
    "build, check",
    [
        pytest.param(
            lambda t: DiffusionOutput(output={"image": t}),
            lambda out, t: torch.testing.assert_close(out.output["image"], t),
            id="nested-dict-output",
        ),
        pytest.param(
            lambda t: DiffusionOutput(output=torch.zeros(2), trajectory_timesteps=t),
            lambda out, t: torch.testing.assert_close(out.trajectory_timesteps, t),
            id="trajectory-timesteps",
        ),
    ],
)
def test_serialize_shm_transport_round_trips_without_leak(build, check):
    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-shm"), value=build(tensor))

    serialized = _serialize_artifact_value(original, prefer_shm_output=True)
    created = _serialized_shm_names(serialized)
    try:
        assert serialized.transport == "shm"
        assert created

        restored = _deserialize_artifact_value(serialized, unpack_shm=True)
        assert isinstance(restored.value, DiffusionOutput)
        check(restored.value, tensor)
        assert not any(_shm_exists(name) for name in created)
    finally:
        for name in created:
            _unlink_shm_by_name(name)


def test_serialize_small_output_stays_pickle_and_creates_no_segment():
    # Everything is below threshold: no handles, so the SHM path is skipped and
    # the value is pickled inline -- and crucially no segment is orphaned.
    original = ArtifactValue(
        handle=_output_handle("req-small"),
        value=DiffusionOutput(output={"image": torch.zeros(2, 3)}),
    )

    serialized = _serialize_artifact_value(original, prefer_shm_output=True)
    assert serialized.transport == "pickle"
    restored = _deserialize_artifact_value(serialized, unpack_shm=True)
    torch.testing.assert_close(restored.value.output["image"], torch.zeros(2, 3))


def test_worker_installs_parent_death_signal():
    """The runtime_v2 GPU worker must tie its lifetime to its parent process:
    register a SIGTERM handler AND arm PR_SET_PDEATHSIG(SIGTERM), so it dies with
    its owner instead of holding GPU memory / hanging mid-collective."""
    from vllm_omni.diffusion.runtime_v2.multiproc_worker import _WorkerProcessRuntime

    rt = object.__new__(_WorkerProcessRuntime)
    with (
        patch("signal.signal") as sig_signal,
        patch("vllm_omni.engine.stage_init_utils.set_death_signal") as set_death,
    ):
        rt._install_parent_death_signal()

    assert any(c.args and c.args[0] == signal.SIGTERM for c in sig_signal.call_args_list)
    set_death.assert_called_once_with(signal.SIGTERM)


def test_health_check_reports_pipe_forwarder_failure():
    from vllm_omni.diffusion.runtime_v2.multiproc_worker import MultiprocWorkerPool

    pool = object.__new__(MultiprocWorkerPool)
    pool._reader_error = EOFError("event pipe closed")
    pool.worker_handles = {}

    with pytest.raises(RuntimeError, match="pipe forwarder failed"):
        pool.check_health()


def test_fetch_thread_reports_serialization_failure():
    runtime = object.__new__(_WorkerProcessRuntime)
    runtime.worker_rank = 0
    runtime._fetch_stop = threading.Event()
    runtime._fetch_queue = queue.Queue()
    command = FetchArtifactsCommand(request_id="r", group_id="g0", artifact_ids=("out",), fetch_id="f")
    runtime._fetch_queue.put(command)
    runtime._handle_fetch_artifacts = Mock(side_effect=RuntimeError("cannot serialize"))
    results = []

    def record_result(result):
        results.append(result)
        runtime._fetch_stop.set()

    runtime._send_result = record_result
    runtime._fetch_loop()

    assert len(results) == 1
    assert results[0].fetch_id == "f"
    assert "cannot serialize" in results[0].error


def test_discard_fetch_result_shm_unlinks_stale_segment():
    """A stale fetch result (request aborted before draining) must have its
    packed POSIX-SHM segment unlinked, not leaked until worker exit."""
    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-stale"), value=DiffusionOutput(output=tensor))

    sav = _serialize_artifact_value(original, prefer_shm_output=True)
    assert sav.transport == "shm"
    created = _serialized_shm_names(sav)
    try:
        result = FetchArtifactsResult(request_id="req-stale", worker_rank=0, artifacts=(sav,))
        MultiprocWorkerPool._discard_fetch_result_shm(result)
        assert not any(_shm_exists(name) for name in created)
    finally:
        for name in created:
            _unlink_shm_by_name(name)


def test_discard_fetch_unlinks_completed_result_shm():
    """discard_fetch (abort/cleanup) on an ALREADY-completed fetch must unlink its
    packed POSIX-SHM segment, not just drop the bookkeeping (leak until exit)."""
    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())

    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-abort"), value=DiffusionOutput(output=tensor))
    sav = _serialize_artifact_value(original, prefer_shm_output=True)
    assert sav.transport == "shm"
    created = _serialized_shm_names(sav)
    try:
        pool._completed_fetches["fetch-1"] = FetchArtifactsResult(
            request_id="req-abort", worker_rank=0, artifacts=(sav,)
        )
        pool.discard_fetch("fetch-1")
        assert not any(_shm_exists(name) for name in created)
        assert "fetch-1" not in pool._completed_fetches
    finally:
        for name in created:
            _unlink_shm_by_name(name)


def test_shutdown_unlinks_stranded_result_queue_shm():
    """A late fetch result the reader queued but no poll ever drained (e.g. the
    last request was aborted, so its rank is never polled) must have its SHM
    reclaimed at shutdown, not dropped by _result_queues.clear()."""
    import queue as _queue

    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())

    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    original = ArtifactValue(handle=_output_handle("req-q"), value=DiffusionOutput(output=tensor))
    sav = _serialize_artifact_value(original, prefer_shm_output=True)
    created = _serialized_shm_names(sav)
    try:
        rank_queue: _queue.Queue = _queue.Queue()
        rank_queue.put(FetchArtifactsResult(request_id="req-q", worker_rank=0, artifacts=(sav,)))
        pool._result_queues[0] = rank_queue
        pool.shutdown()
        assert not any(_shm_exists(name) for name in created)
    finally:
        for name in created:
            _unlink_shm_by_name(name)


def test_shutdown_unlinks_result_left_in_worker_pipe():
    import multiprocessing as mp

    topo = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    pool = MultiprocWorkerPool(topology=topo, od_config=_minimal_od_config())
    tensor = torch.arange(_LARGE_NUMEL, dtype=torch.float32)
    sav = _serialize_artifact_value(
        ArtifactValue(handle=_output_handle("req-pipe"), value=DiffusionOutput(output=tensor)),
        prefer_shm_output=True,
    )
    created = _serialized_shm_names(sav)
    result_r, result_w = mp.Pipe(duplex=False)
    result_w.send(FetchArtifactsResult(request_id="req-pipe", worker_rank=0, artifacts=(sav,)))
    result_w.close()
    process = Mock()
    process.is_alive.return_value = False
    pool.worker_handles[0] = WorkerProcessHandle(
        process=process,
        worker_rank=0,
        command_pipe_w=Mock(),
        event_pipe_r=Mock(),
        result_pipe_r=result_r,
    )

    try:
        pool.shutdown()
        assert not any(_shm_exists(name) for name in created)
    finally:
        result_r.close()
        for name in created:
            _unlink_shm_by_name(name)
