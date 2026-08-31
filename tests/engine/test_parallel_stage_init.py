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
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan, ReplicaInitPlan
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


def test_wrapper_phase_ordering():
    locker = _RecordingLocker()
    wrapped_cls = wrap_executor_with_phase_locks(_FakeExecutor, locker)
    exec_ = wrapped_cls(vllm_config=object())
    assert exec_.determine_available_memory() == 123
    assert exec_.initialize_from_config([]) == "ok"
    assert locker.events == [
        "SH-enter",
        "SH-exit",  # load
        "EX-enter",
        "EX-exit",  # profile
        "SH-enter",
        "SH-exit",  # KV alloc + capture
    ]


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
