# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU CPU binding respects device visibility, CPU limits and worker startup ownership."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.offloader import npu_cpu_affinity as affinity

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def config(**options):
    return SimpleNamespace(
        enable_layerwise_offload=True,
        cache_adaln=True,
        additional_config=options,
        parallel_config=SimpleNamespace(
            ulysses_degree=8, allgather_degree=2, ring_degree=1, tensor_parallel_size=1, world_size=16
        ),
        diffusion_attention_config=SimpleNamespace(
            default=SimpleNamespace(
                backend="RAINFUSION_ATTN", block_sparse=SimpleNamespace(parallel=SimpleNamespace())
            ),
            per_role={},
        ),
    )


@pytest.fixture
def runtime(monkeypatch):
    current = {"cpus": {0, 1, 2, 3}}
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda pid: set(current["cpus"]))
    monkeypatch.setattr(affinity.os, "sched_setaffinity", lambda pid, cpus: current.update(cpus=set(cpus)))
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {2, 3, 99}))
    monkeypatch.setattr(affinity, "_worker_threads", lambda: [0], raising=False)
    return current


def test_auto_binding_intersects_incoming_cpu_limits(runtime):
    assert affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {2, 3}


def test_worker_start_failure_restores_parent(runtime):
    with pytest.raises(RuntimeError, match="spawn failed"):
        with affinity.npu_worker_start_affinity(config(), 0, is_npu=True):
            assert runtime["cpus"] == {2, 3}
            raise RuntimeError("spawn failed")
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_explicit_false_skips_topology_and_preserves_affinity(runtime, monkeypatch):
    def unexpected(index):
        raise AssertionError("disabled binding queried topology")

    monkeypatch.setattr(affinity, "_npu_affinity", unexpected)
    assert not affinity.maybe_bind_npu_offload_cpu(
        config(enable_cpu_binding=False), SimpleNamespace(type="npu", index=0)
    )
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_partial_thread_binding_failure_restores_earlier_threads(monkeypatch):
    masks = {0: {0, 1, 2, 3}, 7: {0, 1, 2, 3}}
    monkeypatch.setattr(affinity, "_worker_threads", lambda: [0, 7], raising=False)
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda tid: set(masks[tid]))

    def bind(tid, cpus):
        if tid == 7:
            raise PermissionError("thread affinity denied")
        masks[tid] = set(cpus)

    monkeypatch.setattr(affinity.os, "sched_setaffinity", bind)
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {2, 3, 99}))
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert masks == {0: {0, 1, 2, 3}, 7: {0, 1, 2, 3}}


@pytest.mark.parametrize("explicit", [False, True])
def test_remote_only_live_thread_rolls_back_binding(monkeypatch, explicit):
    masks = {0: {0, 1, 2, 3}, 7: {0, 1}}
    monkeypatch.setattr(affinity, "_worker_threads", lambda: [0, 7])
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda tid: set(masks[tid]))
    monkeypatch.setattr(affinity.os, "sched_setaffinity", lambda tid, cpus: masks.update({tid: set(cpus)}))
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {2, 3, 99}))
    device = SimpleNamespace(type="npu", index=0)
    if explicit:
        with pytest.raises(RuntimeError, match="no allowed CPUs.*thread 7"):
            affinity.maybe_bind_npu_offload_cpu(config(enable_cpu_binding=True), device)
    else:
        assert not affinity.maybe_bind_npu_offload_cpu(config(), device)
    assert masks == {0: {0, 1, 2, 3}, 7: {0, 1}}


def test_no_allowed_local_cpu_does_not_widen_limits(runtime, monkeypatch):
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {99}))
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_topology_unavailable_falls_back_for_auto(runtime, monkeypatch):
    def unavailable(index):
        raise OSError("driver unavailable")

    monkeypatch.setattr(affinity, "_npu_affinity", unavailable)
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_loader_binds_before_entering_weight_loading(monkeypatch):
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    calls = []
    loader = object.__new__(DiffusersPipelineLoader)
    loader.od_config, loader.quant_config = config(), None

    def bind(cfg, device):
        calls.append("bind")
        return True

    def enter_loading():
        assert calls == ["bind"]
        raise RuntimeError("weight-load sentinel")

    monkeypatch.setattr(affinity, "maybe_bind_npu_offload_cpu", bind)
    monkeypatch.setattr(loader, "_get_shared_weight_group", enter_loading)
    with pytest.raises(RuntimeError, match="weight-load sentinel"):
        loader.load_model("cpu", device=SimpleNamespace(type="npu", index=0))
    assert loader.cpu_affinity_bound


def test_worker_binds_before_device_runtime_initialization(monkeypatch):
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, current_omni_platform

    worker = object.__new__(DiffusionWorker)
    worker.od_config = config()
    worker.od_config.num_gpus = 16
    worker.rank = worker.local_rank = 0
    worker.distributed_init_method = "env://"
    calls = []

    def bind(cfg, device):
        calls.append("bind")
        return True

    monkeypatch.setattr(affinity, "maybe_bind_npu_offload_cpu", bind)
    monkeypatch.setattr(current_omni_platform, "get_torch_device", lambda rank: SimpleNamespace(type="npu", index=0))

    def initialize(device):
        assert calls == ["bind"]
        raise RuntimeError("device-init sentinel")

    monkeypatch.setattr(current_omni_platform, "set_device", initialize)
    with pytest.raises(RuntimeError, match="device-init sentinel"):
        worker.init_device()


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("visible", [None, "12,8,15,9"])
def test_dcmi_version_and_visible_device_mapping(monkeypatch, v2, visible):
    calls = []

    def integer(name, value):
        def call(index, output):
            calls.append((name, index))
            output._obj.value = value
            return 0

        return Function(call)

    def pcie(*args):
        calls.append(("pcie", *args[:-1]))
        info = args[-1]._obj
        info.domain, info.bus, info.slot, info.function = 0, 0x9F, 0, 0
        return 0

    def card(card, device, logical):
        calls.append(("card", logical.value))
        card._obj.value, device._obj.value = 4, 0
        return 0

    lib = SimpleNamespace(
        dcmi_init=Function(lambda: -8255 if v2 else 0),
        dcmiv2_init=Function(lambda: 0),
        dcmi_get_device_phyid_from_logicid=integer("legacy_phy", 8),
        dcmi_get_device_logicid_from_phyid=integer("legacy_logic", 18),
        dcmi_get_card_id_device_id_from_logicid=Function(card),
        dcmi_get_device_pcie_info_v2=Function(pcie),
        dcmiv2_get_chip_phy_id_by_dev_id=integer("v2_phy", 8),
        dcmiv2_get_dev_id_by_chip_phy_id=integer("v2_logic", 18),
        dcmiv2_get_device_pcie_info=Function(pcie),
    )
    monkeypatch.setattr(affinity.ctypes, "CDLL", lambda path: lib)
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    values = {"vendor": "0x19e5", "numa_node": "1", "local_cpulist": "64-65,192-193"}

    def read(path, *args, **kwargs):
        assert str(path.parent) == "/sys/bus/pci/devices/0000:9f:00.0"
        return values[path.name]

    monkeypatch.setattr(affinity.Path, "read_text", read)
    assert affinity._npu_affinity(1) == (8, 1, {64, 65, 192, 193})
    prefix = "v2" if v2 else "legacy"
    assert (prefix + "_logic", 8) in calls
    assert ((prefix + "_phy", 1) in calls) is (visible is None)
    assert not any(name.startswith("legacy" if v2 else "v2") for name, *_ in calls)


@pytest.mark.parametrize("has_v2", [False, True])
def test_failed_dcmi_versions_do_not_guess_topology(monkeypatch, has_v2):
    lib = SimpleNamespace(dcmi_init=Function(lambda: -8255))
    if has_v2:
        lib.dcmiv2_init = Function(lambda: -1)
    monkeypatch.setattr(affinity.ctypes, "CDLL", lambda path: lib)
    with pytest.raises(RuntimeError, match="DCMI initialization failed"):
        affinity._npu_affinity(0)
