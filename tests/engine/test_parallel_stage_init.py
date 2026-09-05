"""Unit tests for parallel stage initialization (CPU-only).

Covers the pure/lock logic added for ``parallel_stage_init``:
  * init-group device keying (resolved physical set; parallel unique keys),
  * admission control arithmetic + graph reserve,
  * SH/EX device phase locks (real flock on a temp dir),
  * the phase-locked executor wrapper ordering,
  * a regression guard that the NVML process-scope path stayed removed.

The GPU-only same-device concurrency (actual overlapping init) requires a
hardware soak and is out of scope for these unit tests.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import subprocess
import threading
import types
from pathlib import Path

import pytest

from vllm_omni.engine.stage_admission import (
    ADMISSION_EXEMPT,
    DeviceLedger,
    StageAdmissionError,
    StageDemand,
    check_admission,
    evaluate,
    graph_reserve_bytes,
)
from vllm_omni.engine.stage_engine_core_proc import _install_phase_locks
from vllm_omni.engine.stage_init_utils import (
    LogicalStageInitPlan,
    ReplicaInitPlan,
    device_init_lock_path,
    open_device_lock_file,
)
from vllm_omni.engine.stage_phase_lock import (
    DeviceLockTimeoutError,
    DevicePhaseLock,
    resolve_driver_device_ids,
    wrap_executor_with_phase_locks,
)
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _llm_replica(stage_id: int, replica_id: int, devices: str | None, *, vllm_config=None) -> ReplicaInitPlan:
    metadata = types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="llm",
        runtime_cfg={"devices": devices} if devices is not None else {},
        replica_id=replica_id,
    )
    stage_cfg = types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="llm",
        runtime=types.SimpleNamespace(devices=devices),
        engine_args={},
    )
    return ReplicaInitPlan(
        replica_id=replica_id,
        num_replicas=1,
        launch_mode="local",
        stage_cfg=stage_cfg,
        metadata=metadata,
        stage_connector_spec={},
        omni_kv_connector=(None, None, None),
        stage_vllm_config=vllm_config,
        executor_class=object,
    )


def _runtime(parallel_stage_init: bool = False) -> StageRuntime:
    return StageRuntime(
        stage_configs=[],
        model="dummy",
        config_path="dummy",
        stage_init_timeout=5,
        diffusion_batch_size=1,
        async_chunk=False,
        parallel_stage_init=parallel_stage_init,
    )


def _fake_vllm_config(util: float, *, model="m", dtype="bf16", cudagraph_mode="FULL", capture_sizes=(1, 2, 4)):
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(gpu_memory_utilization=util),
        model_config=types.SimpleNamespace(model=model, dtype=dtype, enforce_eager=False),
        compilation_config=types.SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=list(capture_sizes),
        ),
    )


# --------------------------------------------------------------------------- #
# A1: init-group device key
# --------------------------------------------------------------------------- #
def test_init_group_key_resolves_physical_devices_serial():
    runtime = _runtime(parallel_stage_init=False)
    runtime._init_visible_devices_baseline = None  # no CUDA_VISIBLE_DEVICES mapping
    r0 = _llm_replica(0, 0, "0")
    r1 = _llm_replica(1, 0, "1")
    r0b = _llm_replica(2, 0, "0")

    k0 = runtime._replica_init_group_key(r0)
    k1 = runtime._replica_init_group_key(r1)
    k0b = runtime._replica_init_group_key(r0b)

    assert k0 == "device:0"
    assert k1 == "device:1"
    # Different physical devices -> different groups (parallel).
    assert k0 != k1
    # Same physical device -> same group (serialized under LOCK_EX).
    assert k0 == k0b


def test_init_group_key_parallel_is_unique_per_replica():
    runtime = _runtime(parallel_stage_init=True)
    runtime._init_visible_devices_baseline = None
    # Both on the SAME device, but parallel mode must give distinct keys so they
    # run concurrently, coordinated by the child SH/EX locks + admission.
    keys = [
        runtime._replica_init_group_key(_llm_replica(0, 0, "0")),
        runtime._replica_init_group_key(_llm_replica(0, 1, "0")),
        runtime._replica_init_group_key(_llm_replica(1, 0, "0")),
    ]
    assert keys == ["parallel:0:0", "parallel:0:1", "parallel:1:0"]
    assert len(set(keys)) == 3


def test_init_group_key_remote_and_diffusion_unchanged():
    runtime = _runtime(parallel_stage_init=True)
    remote = _llm_replica(1, 0, "0")
    remote.launch_mode = "remote"
    remote.metadata.runtime_cfg = None
    assert runtime._replica_init_group_key(remote) == "remote:1:0"

    diffusion = _llm_replica(2, 0, "0")
    diffusion.metadata.stage_type = "diffusion"
    assert runtime._replica_init_group_key(diffusion) == "inline:diffusion"


# --------------------------------------------------------------------------- #
# B1: admission
# --------------------------------------------------------------------------- #
def test_evaluate_fits():
    cap = 40 * 1024**3
    demands = [StageDemand(0, 0, [0], utilization=0.5, graph_reserve_bytes=1 * 1024**3)]
    ledgers = evaluate(demands, {0: cap}, external_reserve_bytes=0, safety_margin_bytes=0)
    assert ledgers[0].fits
    assert ledgers[0].kv_budget_bytes == int(cap * 0.5)


def test_evaluate_oversubscription_raises():
    cap = 40 * 1024**3
    demands = [
        StageDemand(0, 0, [0], utilization=0.7, graph_reserve_bytes=0),
        StageDemand(1, 0, [0], utilization=0.7, graph_reserve_bytes=0),
    ]
    with pytest.raises(StageAdmissionError):
        evaluate(demands, {0: cap}, external_reserve_bytes=0, safety_margin_bytes=0)


def test_evaluate_unknown_capacity_raises():
    with pytest.raises(StageAdmissionError):
        evaluate([StageDemand(0, 0, [3], 0.5, 0)], {0: 1})


def test_graph_reserve_positive_when_enabled_zero_when_disabled():
    # cudagraph enabled -> conservative positive reserve
    reserve = graph_reserve_bytes(_fake_vllm_config(0.9, cudagraph_mode="FULL"))
    assert reserve > 0
    # cudagraph disabled -> exactly 0
    assert graph_reserve_bytes(_fake_vllm_config(0.9, cudagraph_mode="NONE")) == 0
    eager = _fake_vllm_config(0.9)
    eager.model_config.enforce_eager = True
    assert graph_reserve_bytes(eager) == 0


def test_check_admission_multi_device_and_grouping():
    cap = 80 * 1024**3
    plan_a = LogicalStageInitPlan(
        stage_idx=0,
        stage_id=0,
        replicas=[_llm_replica(0, 0, "0,1", vllm_config=_fake_vllm_config(0.4, cudagraph_mode="NONE"))],
    )
    plan_b = LogicalStageInitPlan(
        stage_idx=1,
        stage_id=1,
        replicas=[_llm_replica(1, 0, "1", vllm_config=_fake_vllm_config(0.4, cudagraph_mode="NONE"))],
    )
    ledgers = check_admission(
        [plan_a, plan_b],
        resolve_physical_devices=lambda r: [int(x) for x in r.metadata.runtime_cfg["devices"].split(",")],
        device_total_memory=lambda _d: cap,
        external_reserve_bytes=0,
        safety_margin_bytes=0,
    )
    # device 0: only stage 0 (0.4). device 1: stage 0 + stage 1 (0.8).
    assert ledgers[0].kv_budget_bytes == int(cap * 0.4)
    assert ledgers[1].kv_budget_bytes == int(cap * 0.4) + int(cap * 0.4)
    assert ledgers[1].fits


def _local_diffusion_replica(stage_id: int, replica_id: int, devices: str, *, util: float | None) -> ReplicaInitPlan:
    replica = _llm_replica(stage_id, replica_id, devices)
    replica.metadata.stage_type = "diffusion"
    replica.stage_vllm_config = None
    replica.stage_cfg.engine_args = {} if util is None else {"gpu_memory_utilization": util}
    return replica


def test_check_admission_admits_diffusion_with_explicit_util():
    cap = 80 * 1024**3
    diff = _local_diffusion_replica(0, 0, "0", util=0.3)
    plan = LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=[diff])
    ledgers = check_admission(
        [plan],
        resolve_physical_devices=lambda _r: [0],
        device_total_memory=lambda _d: cap,
        external_reserve_bytes=0,
        safety_margin_bytes=0,
    )
    assert set(ledgers) == {0}
    assert ledgers[0].kv_budget_bytes == int(cap * 0.3)


def test_check_admission_unresolved_local_raises():
    """A local replica invisible to the ledger must fail admission, not skip it:
    it would otherwise initialize concurrently with unbounded demand."""
    unresolved = _llm_replica(1, 0, None, vllm_config=_fake_vllm_config(0.9, cudagraph_mode="NONE"))
    plan = LogicalStageInitPlan(stage_idx=0, stage_id=1, replicas=[unresolved])
    with pytest.raises(StageAdmissionError, match="stage1/replica0.*unresolved"):
        check_admission(
            [plan],
            resolve_physical_devices=lambda _r: None,
            device_total_memory=lambda _d: 80 * 1024**3,
        )


def test_check_admission_local_diffusion_without_util_raises():
    """The reviewer's repro: local LLM + local diffusion with no explicit util
    used to leave the diffusion stage out of the ledger silently."""
    llm = _llm_replica(0, 0, "0", vllm_config=_fake_vllm_config(0.45, cudagraph_mode="NONE"))
    diff = _local_diffusion_replica(1, 0, "0", util=None)
    plans = [
        LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=[llm]),
        LogicalStageInitPlan(stage_idx=1, stage_id=1, replicas=[diff]),
    ]
    with pytest.raises(StageAdmissionError, match="diffusion, no gpu_memory_utilization"):
        check_admission(
            plans,
            resolve_physical_devices=lambda _r: [0],
            device_total_memory=lambda _d: 80 * 1024**3,
        )


def test_check_admission_exempt_replicas_skip_without_error():
    """Only ADMISSION_EXEMPT (remote/operator-isolated) may skip the ledger."""
    cap = 80 * 1024**3
    local = _llm_replica(0, 0, "0", vllm_config=_fake_vllm_config(0.4, cudagraph_mode="NONE"))
    remote = _llm_replica(1, 0, None, vllm_config=_fake_vllm_config(0.9, cudagraph_mode="NONE"))
    remote.launch_mode = "remote"
    plan = LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=[local, remote])
    ledgers = check_admission(
        [plan],
        resolve_physical_devices=lambda r: ADMISSION_EXEMPT if r.launch_mode == "remote" else [0],
        device_total_memory=lambda _d: cap,
        external_reserve_bytes=0,
        safety_margin_bytes=0,
    )
    assert set(ledgers) == {0}
    assert ledgers[0].contributors == ["stage0/replica0"]
    assert ledgers[0].kv_budget_bytes == int(cap * 0.4)


def test_run_stage_admission_excludes_remote_replicas(monkeypatch):
    """Remote replicas consume a remote node's memory — never the local ledger.

    Remote LLM plans already carry runtime_cfg=None, but remote diffusion
    replicas keep their cfg + utilization, so without an explicit launch_mode
    guard they would be counted against local devices. They must resolve to
    ADMISSION_EXEMPT (not None): None now means an unaccountable LOCAL replica
    and fails admission.
    """
    import vllm_omni.engine.stage_admission as admission_mod

    runtime = _runtime(parallel_stage_init=True)
    runtime._init_visible_devices_baseline = None

    local = _llm_replica(0, 0, "0", vllm_config=_fake_vllm_config(0.4, cudagraph_mode="NONE"))
    remote_diff = _llm_replica(1, 0, "0")
    remote_diff.launch_mode = "remote"
    remote_diff.metadata.stage_type = "diffusion"
    remote_diff.stage_vllm_config = None
    remote_diff.stage_cfg.engine_args = {"gpu_memory_utilization": 0.3}

    captured = {}

    def _fake_check_admission(stage_plans, *, resolve_physical_devices, device_total_memory, **kw):
        captured["resolve"] = resolve_physical_devices
        return {}

    monkeypatch.setattr(admission_mod, "check_admission", _fake_check_admission)
    runtime._run_stage_admission([LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=[local, remote_diff])])

    assert captured["resolve"](local) == [0]
    assert captured["resolve"](remote_diff) is ADMISSION_EXEMPT


# --------------------------------------------------------------------------- #
# B1b: admission capacity is queried in the visible-ordinal namespace
# --------------------------------------------------------------------------- #
def _capture_admission(monkeypatch, runtime, replicas):
    """Run _run_stage_admission with check_admission stubbed; return its callables."""
    import vllm_omni.engine.stage_admission as admission_mod

    captured: dict = {}

    def _fake_check_admission(stage_plans, *, resolve_physical_devices, device_total_memory, **kw):
        captured["resolve"] = resolve_physical_devices
        captured["capacity"] = device_total_memory
        return {}

    monkeypatch.setattr(admission_mod, "check_admission", _fake_check_admission)
    runtime._run_stage_admission([LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=replicas)])
    return captured


def _stub_platform(monkeypatch, seen: list[int], total: int = 80 * 1024**3):
    import vllm_omni.engine.stage_runtime as runtime_mod

    monkeypatch.setattr(
        runtime_mod,
        "current_omni_platform",
        types.SimpleNamespace(
            device_control_env_var="CUDA_VISIBLE_DEVICES",
            get_device_total_memory=lambda d: (seen.append(d), total)[1],
        ),
    )


@pytest.mark.parametrize(
    ("baseline", "devices", "physical", "ordinal"),
    [
        ("3,4,5", "1", 4, 1),  # restricted visibility
        ("1,0", "0", 1, 0),  # reordered visibility
        ("5,GPU-deadbeef", "0", 5, 0),  # mixed identifiers, numeric first
        ("GPU-deadbeef,5", "1", 5, 1),  # mixed identifiers, numeric second
        ("0,1,2,3", "2", 2, 2),  # identity
    ],
    ids=["restricted", "reordered", "mixed-numeric-first", "mixed-numeric-second", "identity"],
)
def test_admission_capacity_uses_visible_ordinal(monkeypatch, baseline, devices, physical, ordinal):
    """The ledger stays keyed by physical id, but the capacity query is translated.

    ``get_device_total_memory``/``torch.cuda.get_device_properties`` read their
    argument as an ordinal in the *current* process's visible-device namespace.
    With CUDA_VISIBLE_DEVICES=3,4,5 the orchestrator only has ordinals 0..2, so
    passing physical id 4 straight through would raise — or, when visibility is
    reordered, silently measure a different GPU.
    """
    runtime = _runtime(parallel_stage_init=True)
    runtime._init_visible_devices_baseline = baseline
    replica = _llm_replica(0, 0, devices, vllm_config=_fake_vllm_config(0.4, cudagraph_mode="NONE"))

    seen: list[int] = []
    _stub_platform(monkeypatch, seen)
    captured = _capture_admission(monkeypatch, runtime, [replica])

    assert captured["resolve"](replica) == [physical], "ledger key must stay physical"
    assert captured["capacity"](physical) == 80 * 1024**3
    assert seen == [ordinal], f"expected the capacity query on ordinal {ordinal}, got {seen}"


def test_admission_capacity_rejects_device_outside_baseline(monkeypatch):
    """A physical id absent from the captured baseline cannot be measured here."""
    runtime = _runtime(parallel_stage_init=True)
    runtime._init_visible_devices_baseline = "3,4,5"
    replica = _llm_replica(0, 0, "0", vllm_config=_fake_vllm_config(0.4, cudagraph_mode="NONE"))

    seen: list[int] = []
    _stub_platform(monkeypatch, seen)
    captured = _capture_admission(monkeypatch, runtime, [replica])

    with pytest.raises(StageAdmissionError, match="not in the visibility baseline"):
        captured["capacity"](7)
    assert seen == [], "must not query the platform with an unmappable id"


def test_admission_capacity_identity_when_no_baseline(monkeypatch):
    """With no device-control env captured, ordinals and physical ids coincide."""
    runtime = _runtime(parallel_stage_init=True)
    runtime._init_visible_devices_baseline = None
    replica = _llm_replica(0, 0, "2", vllm_config=_fake_vllm_config(0.4, cudagraph_mode="NONE"))

    seen: list[int] = []
    _stub_platform(monkeypatch, seen)
    captured = _capture_admission(monkeypatch, runtime, [replica])

    assert captured["capacity"](2) == 80 * 1024**3
    assert seen == [2]


# --------------------------------------------------------------------------- #
# B1c: fail closed for executors the phase locks cannot guard
# --------------------------------------------------------------------------- #
class _RayBackedExecutor:
    uses_ray = True


class _LocalExecutor:
    uses_ray = False


def test_parallel_init_refuses_ray_backed_executor():
    """Phase locks are node-local flocks and the ledger only sees this host.

    A Ray executor may place the workers that load weights, profile, allocate KV
    and capture graphs on other nodes, which participate in neither mechanism —
    and the parent skips its own device lock when parallel_stage_init is on.
    """
    runtime = _runtime(parallel_stage_init=True)
    replica = _llm_replica(0, 0, "0")
    replica.executor_class = _RayBackedExecutor
    plans = [LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=[replica])]

    with pytest.raises(RuntimeError, match="cannot guard Ray-backed or multi-node executors"):
        runtime._reject_unguardable_executors(plans)


@pytest.mark.parametrize(
    ("use_ray", "nnodes", "reason"),
    [
        (True, 1, "Ray-backed"),
        (False, 2, r"multi-node \(nnodes=2\)"),
    ],
    ids=["parallel-config-ray", "multi-node-mp"],
)
def test_parallel_init_refuses_unguardable_parallel_config(use_ray, nnodes, reason):
    """The config guard also covers non-Ray executors spread across nodes."""
    runtime = _runtime(parallel_stage_init=True)
    vllm_config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(use_ray=use_ray, nnodes=nnodes),
    )
    replica = _llm_replica(0, 0, "0", vllm_config=vllm_config)
    replica.executor_class = _LocalExecutor
    plans = [LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=[replica])]

    with pytest.raises(RuntimeError, match=reason):
        runtime._reject_unguardable_executors(plans)


def test_parallel_init_allows_non_ray_executor():
    runtime = _runtime(parallel_stage_init=True)
    replicas = [_llm_replica(0, 0, "0"), _llm_replica(1, 0, "1")]
    replicas[0].executor_class = _LocalExecutor
    replicas[1].executor_class = object  # no uses_ray attribute at all
    runtime._reject_unguardable_executors([LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=replicas)])


def test_parallel_init_ignores_remote_ray_replicas():
    """Remote replicas are launched by the runtime owning their node."""
    runtime = _runtime(parallel_stage_init=True)
    replica = _llm_replica(0, 0, "0")
    replica.executor_class = _RayBackedExecutor
    replica.launch_mode = "remote"
    runtime._reject_unguardable_executors([LogicalStageInitPlan(stage_idx=0, stage_id=0, replicas=[replica])])


# --------------------------------------------------------------------------- #
# B2: SH/EX device phase locks (real flock)
# --------------------------------------------------------------------------- #
def _lock(tmp: Path, **kw) -> DevicePhaseLock:
    return DevicePhaseLock([0], lock_dir=str(tmp), timeout_s=kw.pop("timeout_s", 0.3), **kw)


def test_shared_locks_coexist(tmp_path):
    a, b = _lock(tmp_path), _lock(tmp_path)
    with a.shared():
        with b.shared():
            pass  # no timeout


def test_exclusive_excludes_shared_fail_closed(tmp_path):
    a, b = _lock(tmp_path, timeout_s=0.2), _lock(tmp_path, timeout_s=0.2)
    with a.exclusive():
        with pytest.raises(DeviceLockTimeoutError):
            with b.shared():
                pass


def test_exclusive_excludes_exclusive(tmp_path):
    a, b = _lock(tmp_path, timeout_s=0.2), _lock(tmp_path, timeout_s=0.2)
    with a.exclusive():
        with pytest.raises(DeviceLockTimeoutError):
            with b.exclusive():
                pass


def test_empty_device_set_is_noop(tmp_path):
    lock = DevicePhaseLock([], lock_dir=str(tmp_path))
    with lock.shared():
        pass
    with lock.exclusive():
        pass


def _dead_pid() -> int:
    """A PID that has certainly exited and been reaped."""
    proc = subprocess.Popen(["/bin/true"])
    proc.wait()
    return proc.pid


def test_legacy_device_lock_never_unlinks_the_path():
    """The stale-PID unlink path is gone from the legacy acquirer.

    ``flock`` is released by the kernel when the holder exits (SIGKILL included),
    so a dead holder never keeps the lock and unlink-based cleanup is never
    needed. It was also unsafe: the cleanup only ran on ``BlockingIOError``, i.e.
    only while the lock *was* held, so every case it fired in was a case where
    unlinking broke a live holder.
    """
    import vllm_omni.engine.stage_init_utils as siu

    assert not hasattr(siu, "_cleanup_stale_lock_if_dead")
    assert "unlink" not in inspect.getsource(siu.acquire_device_locks)

    # Test cleanup runs in many autouse/runtime fixtures and must not unlink the
    # shared production lock path either. Persistent unlocked flock files are
    # harmless; deleting one can split live holders across different inodes.
    helper_source = (Path(__file__).parents[1] / "helpers" / "clean.py").read_text()
    assert "_cleanup_stale_device_locks" not in helper_source
    assert "vllm_omni_device_*_init.lock" not in helper_source


def test_legacy_lock_cannot_break_a_live_shared_holder(tmp_path, monkeypatch):
    """A live LOCK_SH holder must keep excluding the legacy LOCK_EX acquirer
    even when the PID recorded in the lock file belongs to a dead process.

    Several LOCK_SH holders overwrite that PID, so the recorded value can name a
    process that has exited while another holder still owns the inode. Unlinking
    on that signal let the legacy acquirer create a fresh file under the same
    name and take LOCK_EX on a *different* inode — conflicting with nobody.
    """
    import vllm_omni.engine.stage_init_utils as siu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        siu,
        "device_init_lock_path",
        lambda device_id, lock_dir="/tmp": device_init_lock_path(device_id, str(tmp_path)),
    )

    path = Path(device_init_lock_path(0, str(tmp_path)))
    holder = DevicePhaseLock([0], lock_dir=str(tmp_path), timeout_s=5)

    with holder.shared():
        path.write_text(f"{_dead_pid()}\n")  # peer SH holder exited after recording its PID
        inode_before = path.stat().st_ino

        lock_fds = siu.acquire_device_locks(0, {"tensor_parallel_size": 1}, stage_init_timeout=1)

        assert lock_fds == [], "legacy LOCK_EX must not be granted while a shared holder is live"
        assert path.exists(), "the lock path must never be unlinked"
        assert path.stat().st_ino == inode_before, "the lock inode must stay stable"

    siu.release_device_locks(lock_fds)


def test_phase_lock_reuses_the_existing_lock_inode(tmp_path):
    """Repeated acquisitions coordinate on one stable inode per device."""
    path = Path(device_init_lock_path(0, str(tmp_path)))
    lock = DevicePhaseLock([0], lock_dir=str(tmp_path), timeout_s=1)

    with lock.shared():
        first = path.stat().st_ino
    with lock.exclusive():
        assert path.stat().st_ino == first
    assert os.path.exists(path)


def test_device_lock_file_never_visible_at_restrictive_mode(tmp_path):
    """Concurrent creators must never observe the lock at a umask-filtered mode.

    Creating in place with O_EXCL and widening afterwards publishes the file at
    0600 under a restrictive umask; a second user opening it in that window is
    locked out and parallel init aborts. Publishing atomically closes the window,
    so every concurrent creator either wins the link or opens a readable file.
    """
    path = device_init_lock_path(0, str(tmp_path))
    results: list[tuple[int, bool]] = []
    errors: list[BaseException] = []
    modes: list[int] = []
    barrier = threading.Barrier(8)

    def _worker() -> None:
        barrier.wait()
        try:
            fd, writable = open_device_lock_file(path)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)
            return
        try:
            modes.append(os.stat(path).st_mode & 0o777)
            results.append((fd, writable))
        finally:
            os.close(fd)

    previous = os.umask(0o077)
    try:
        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        os.umask(previous)

    assert not errors, f"concurrent creators were locked out: {errors!r}"
    assert len(results) == 8
    assert all(m & 0o044 == 0o044 for m in modes), f"lock was visible at {[oct(m) for m in modes]}"
    assert len({os.stat(path).st_ino}) == 1


@pytest.mark.parametrize("umask_value", [0o022, 0o027, 0o077], ids=["022", "027", "077"])
def test_device_lock_file_is_readable_by_other_users(tmp_path, umask_value):
    """The lock file must stay group/other readable whatever the creator's umask.

    These files coordinate GPU init *across* users, so a second user has to be
    able to open the first user's file. The mode passed to ``os.open`` is filtered
    by umask, so a restrictive umask (0027, 0077) would produce 0640 or 0600 and
    lock every other user out -- fatal for parallel_stage_init, and a silent loss
    of serialization in the legacy path.
    """
    path = device_init_lock_path(0, str(tmp_path))
    previous = os.umask(umask_value)
    try:
        fd, writable = open_device_lock_file(path)
    finally:
        os.umask(previous)
    try:
        assert writable is True
        mode = os.stat(path).st_mode & 0o777
        assert mode & 0o044 == 0o044, f"lock file mode {mode:04o} is not readable by other users"
    finally:
        os.close(fd)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permission bits")
def test_phase_lock_tolerates_lock_file_owned_by_another_user(tmp_path):
    """A lock file created by a different user must not break locking.

    These files coordinate *across* users -- two people on the same physical GPU
    must contend on the same path -- so whoever creates one owns it at 0644 and
    every later user's O_RDWR open fails with EACCES. That made parallel_stage_init
    die at startup on any shared machine. flock attaches to the open file
    description and needs no write access, so the read-only fallback keeps full
    mutual exclusion and only forfeits the diagnostic PID write.
    """
    path = Path(device_init_lock_path(0, str(tmp_path)))
    path.write_text("999999\n")
    path.chmod(0o444)

    fd, writable = open_device_lock_file(str(path))
    try:
        assert writable is False, "expected the read-only fallback on an unwritable file"
    finally:
        os.close(fd)

    holder = DevicePhaseLock([0], lock_dir=str(tmp_path), timeout_s=0.3)
    contender = DevicePhaseLock([0], lock_dir=str(tmp_path), timeout_s=0.2)
    with holder.exclusive():
        with pytest.raises(DeviceLockTimeoutError):
            with contender.shared():
                pass

    assert path.read_text() == "999999\n", "must not clobber a file it cannot own"


def test_resolve_driver_device_ids_dp_slice(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    env_var = current_omni_platform.device_control_env_var
    monkeypatch.setenv(env_var, "4,5,6,7")
    pc = types.SimpleNamespace(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        sequence_parallel_size=1,
        cfg_parallel_size=1,
    )
    vc = types.SimpleNamespace(parallel_config=pc)
    assert resolve_driver_device_ids(vc, local_dp_rank=0) == [4, 5]
    assert resolve_driver_device_ids(vc, local_dp_rank=1) == [6, 7]


# --------------------------------------------------------------------------- #
# B2: executor wrapper ordering + retry
# --------------------------------------------------------------------------- #
class _RecordingLocker:
    def __init__(self):
        self.events: list[str] = []

    @contextlib.contextmanager
    def shared(self):
        self.events.append("SH-enter")
        try:
            yield
        finally:
            self.events.append("SH-exit")

    @contextlib.contextmanager
    def exclusive(self):
        self.events.append("EX-enter")
        try:
            yield
        finally:
            self.events.append("EX-exit")


class _FakeExecutor:
    instances: list[object] = []

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.dam_calls = 0
        _FakeExecutor.instances.append(self)

    def determine_available_memory(self):
        self.dam_calls += 1
        return 123

    def initialize_from_config(self, kv_cache_configs):
        return "ok"

    def compile_or_warm_up_model(self):
        return "warm"


def test_wrapper_phase_ordering():
    locker = _RecordingLocker()
    wrapped_cls = wrap_executor_with_phase_locks(_FakeExecutor, locker)
    exec_ = wrapped_cls(vllm_config=object())
    assert exec_.determine_available_memory() == 123
    assert exec_.initialize_from_config([]) == "ok"
    assert exec_.compile_or_warm_up_model() == "warm"
    assert locker.events == [
        "SH-enter",
        "SH-exit",  # load
        "EX-enter",
        "EX-exit",  # profile
        "SH-enter",
        "SH-exit",  # KV cache allocation
        "SH-enter",
        "SH-exit",  # kernel warmup + CUDA-graph capture
    ]


def test_wrapper_locks_warmup_and_capture_separately():
    """Regression guard for the upstream split.

    ``Executor.initialize_from_config`` used to fuse ``compile_or_warm_up_model``
    into itself; since vLLM v0.20.0 ``EngineCore._initialize_kv_caches`` calls the
    two separately. If the wrapper only brackets ``initialize_from_config``, one
    stage captures CUDA graphs — mutating device memory — while a peer holds the
    exclusive lock for its profiling measurement.
    """
    locker = _RecordingLocker()
    wrapped_cls = wrap_executor_with_phase_locks(_FakeExecutor, locker)
    exec_ = wrapped_cls(vllm_config=object())
    locker.events.clear()

    assert exec_.compile_or_warm_up_model() == "warm"

    assert locker.events == ["SH-enter", "SH-exit"], "compile_or_warm_up_model must run under the shared device lock"
    assert "compile_or_warm_up_model" in vars(wrapped_cls), (
        "the wrapper must override compile_or_warm_up_model, not inherit it unguarded"
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"vllm_config": object()}, {"executor_class": object}],
    ids=["both-missing", "executor-missing", "config-missing"],
)
def test_install_phase_locks_fails_closed_on_missing_kwargs(kwargs):
    """parallel_stage_init must never proceed with the phase-lock guard
    uninstalled — a missing vllm_config/executor_class aborts the child."""
    with pytest.raises(RuntimeError, match="phase-lock guard cannot be installed"):
        _install_phase_locks(kwargs, local_dp_rank=0)


def test_install_phase_locks_wraps_executor(monkeypatch):
    import vllm_omni.engine.stage_phase_lock as phase_lock_mod

    locker = _RecordingLocker()
    locker.device_ids = [0]
    monkeypatch.setattr(
        phase_lock_mod.DevicePhaseLock,
        "from_child",
        classmethod(lambda _cls, _cfg, _rank: locker),
    )
    kwargs = {"vllm_config": object(), "executor_class": _FakeExecutor}

    _install_phase_locks(kwargs, local_dp_rank=0)

    exec_ = kwargs["executor_class"](vllm_config=object())
    assert exec_.determine_available_memory() == 123
    assert locker.events[:2] == ["SH-enter", "SH-exit"]


# --------------------------------------------------------------------------- #
# A2: NVML process-scope removal regression guard
# --------------------------------------------------------------------------- #
def test_base_worker_dropped_nvml_process_scope():
    import vllm_omni.worker.base as base

    src = Path(base.__file__).read_text()
    assert "get_process_gpu_memory" not in src
    assert "is_process_scoped_memory_available" not in src


def test_device_ledger_required_and_fits():
    led = DeviceLedger(device_id=0, capacity_bytes=100, kv_budget_bytes=40, graph_reserve_bytes=20)
    led.external_reserve_bytes = 10
    led.safety_margin_bytes = 10
    assert led.required_bytes == 80
    assert led.fits
    led.kv_budget_bytes = 90
    assert not led.fits
