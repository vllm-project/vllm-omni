# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import uuid
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import ulysses_transport, ulysses_transport_tables
from vllm_omni.diffusion.distributed.ulysses_transport import (
    UlyssesTopology,
    UlyssesTransportPlan,
    auto_plan,
    eligible_transports,
    explicit_plan,
)
from vllm_omni.platforms.interface import OmniPlatformEnum, UnspecifiedOmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

MiB = 1024 * 1024
# The shipped tables, captured before the fixture below swaps in synthetic ones.
_SHIPPED_TABLES = {
    name: getattr(ulysses_transport_tables, name)
    for name in ("FLASHINFER_MIN_BYTES", "COPY_ENGINE_MIN_BYTES", "COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES")
}


@pytest.fixture(autouse=True)
def measured_tables(monkeypatch):
    """Synthetic tables so the tests do not depend on the measured values."""
    monkeypatch.setattr(ulysses_transport_tables, "FLASHINFER_MIN_BYTES", {"10.0": {8: 32 * MiB}})
    monkeypatch.setattr(ulysses_transport_tables, "COPY_ENGINE_MIN_BYTES", {"10.0": {2: MiB, 4: MiB, 8: MiB}})
    monkeypatch.setattr(
        ulysses_transport_tables, "COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES", {"10.0": {2: 4 * MiB, 4: 4 * MiB}}
    )


def _topology(world_size=2, **overrides):
    values = dict(
        cuda=True, single_node=True, nvlink=True, p2p=True, capability="10.0", symmetric_memory=True, flashinfer=True
    )
    values.update(overrides)
    return UlyssesTopology(world_size, **values)


def _plan(topology, heads=28, invariant=True, pure_strict=True, chunks=0):
    return auto_plan(
        topology, pure_strict=pure_strict, heads_per_rank=heads, per_head_invariant=invariant, overlap_chunks=chunks
    )


@pytest.mark.parametrize(
    "world_size, heads, expected",
    [
        (2, 28, UlyssesTransportPlan(None, MiB, 14, 4 * MiB)),
        (4, 14, UlyssesTransportPlan(None, MiB, 7, 4 * MiB)),
        (8, 7, UlyssesTransportPlan(32 * MiB, MiB, 0, 0)),
    ],
)
def test_measured_entries_enable_transports(world_size, heads, expected):
    assert _plan(_topology(world_size), heads=heads) == expected


@pytest.mark.parametrize(
    "overrides, pure_strict, reason",
    [
        ({"cuda": False}, True, "not CUDA"),
        ({}, False, "not strict pure Ulysses"),
        ({"single_node": False}, True, "spans nodes"),
        ({"nvlink": False}, True, "NVLink"),
        ({"p2p": False}, True, "peer access"),
        ({"capability": "9.0"}, True, "not measured for sm9.0"),
        ({"capability": "12.0"}, True, "not measured for sm12.0"),
    ],
)
def test_ineligible_groups_fall_through_to_nccl(overrides, pure_strict, reason):
    topology = _topology(**overrides)
    reasons = eligible_transports(topology, pure_strict=pure_strict)
    assert all(reason in r for r in reasons.values())
    assert _plan(topology, pure_strict=pure_strict) == ulysses_transport.NCCL_PLAN


@pytest.mark.parametrize("world_size, h3_q_mib", [(2, 259), (4, 129.5), (8, 64.75)])
def test_shipped_sm90_rows_pick_the_copy_engine_for_h3_only(monkeypatch, world_size, h3_q_mib):
    for name, table in _SHIPPED_TABLES.items():
        monkeypatch.setattr(ulysses_transport_tables, name, table)
    topology = _topology(world_size, capability="9.0")
    reasons = eligible_transports(topology, pure_strict=True)
    assert reasons["copy_engine"] is None
    assert "not measured" in reasons["flashinfer"] and "not measured" in reasons["copy_engine_overlap"]
    plan = _plan(topology, heads=56 // world_size)
    assert plan.exchange(int(h3_q_mib * MiB)) == "copy_engine"
    assert plan.exchange(int(h3_q_mib * MiB) // 4) == "nccl"  # smaller messages were not measured on sm9.0


def test_unmeasured_world_size_is_not_eligible():
    reasons = eligible_transports(_topology(3), pure_strict=True)
    assert reasons == dict.fromkeys(reasons, "not measured for sm10.0 x 3 GPUs")


def test_missing_libraries_disable_only_their_transport():
    reasons = eligible_transports(_topology(8, flashinfer=False), pure_strict=True)
    assert reasons["flashinfer"] == "FlashInfer UlyssesCommunicator unavailable"
    assert reasons["copy_engine"] is None
    reasons = eligible_transports(_topology(2, symmetric_memory=False), pure_strict=True)
    assert reasons["copy_engine"] == reasons["copy_engine_overlap"] == "no symmetric memory"


def test_overlap_requires_per_head_invariant_attention():
    plan = _plan(_topology(2), invariant=False)
    assert plan.overlap_chunks == 0 and plan.copy_engine_min_bytes == MiB


@pytest.mark.parametrize("heads, requested, chunks", [(1, 0, 0), (3, 0, 0), (28, 4, 4), (28, 3, 0)])
def test_overlap_chunk_count(heads, requested, chunks):
    assert _plan(_topology(2), heads=heads, chunks=requested).overlap_chunks == chunks


def test_dispatch_falls_through_by_message_size():
    plan = UlyssesTransportPlan(flashinfer_min_bytes=32 * MiB, copy_engine_min_bytes=MiB)
    assert plan.exchange(64 * MiB) == "flashinfer"
    assert plan.exchange(64 * MiB, flashinfer_ok=False) == "copy_engine"
    assert plan.exchange(8 * MiB) == "copy_engine"
    assert plan.exchange(MiB // 2) == "nccl"
    overlap = UlyssesTransportPlan(copy_engine_min_bytes=0, overlap_chunks=14, overlap_min_chunk_bytes=4 * MiB)
    assert overlap.use_overlap(4 * MiB) and not overlap.use_overlap(4 * MiB - 1)


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("nccl", UlyssesTransportPlan()),
        ("copy_engine", UlyssesTransportPlan(copy_engine_min_bytes=0)),
        ("flashinfer", UlyssesTransportPlan(flashinfer_min_bytes=0, copy_engine_min_bytes=0)),
        ("copy_engine_overlap", UlyssesTransportPlan(copy_engine_min_bytes=0, overlap_chunks=14)),
    ],
)
def test_explicit_backends_have_no_size_gates(backend, expected):
    assert explicit_plan(backend, heads_per_rank=28) == expected


class _FakeDist:
    """The peer's contributions to the collective topology probe."""

    def __init__(self, hosts, uuids, remote_features):
        self.hosts, self.uuids, self.remote_features = hosts, uuids, remote_features
        self.ReduceOp = SimpleNamespace(MIN="min")

    def all_gather_into_tensor(self, out, local, group):
        rows = [torch.tensor([h, *u.bytes], dtype=torch.int64) for h, u in zip(self.hosts, self.uuids)]
        rows[0] = local.cpu()
        out.copy_(torch.cat(rows))

    def all_reduce(self, tensor, op, group):
        assert op == "min"
        tensor.copy_(torch.minimum(tensor, torch.tensor(self.remote_features, dtype=tensor.dtype)))


class _CudaPlatform(UnspecifiedOmniPlatform):
    _omni_enum = OmniPlatformEnum.CUDA


def _probe(monkeypatch, *, hosts_differ=False, nvlink=True, remote_features=(1, 1, 1, 100, 1, 1)):
    local_uuid, peer_uuid = uuid.uuid4(), uuid.uuid4()
    host = ulysses_transport.zlib.crc32(ulysses_transport.socket.gethostname().encode()) & 0x7FFFFFFF
    fake = _FakeDist([host, host + 1 if hosts_differ else host], [local_uuid, peer_uuid], list(remote_features))
    checked = []
    monkeypatch.setattr(ulysses_transport, "dist", fake)
    monkeypatch.setattr(ulysses_transport, "current_omni_platform", _CudaPlatform())
    monkeypatch.setattr(ulysses_transport.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        ulysses_transport.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(uuid=str(local_uuid), major=10, minor=0),
    )
    monkeypatch.setattr(ulysses_transport, "_probe_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ulysses_transport, "_nvlink_all_pairs", lambda uuids: checked.append(uuids) or nvlink)
    monkeypatch.setattr(ulysses_transport, "_peer_access", lambda device, uuids: True)
    monkeypatch.setattr(ulysses_transport, "_symmetric_memory_available", lambda: True)
    monkeypatch.setattr(ulysses_transport, "_flashinfer_available", lambda: True)
    monkeypatch.setattr(ulysses_transport, "_TOPOLOGIES", {})
    topology = ulysses_transport.probe_ulysses_topology(SimpleNamespace(group_name="ulysses"), 2)
    return topology, checked, (local_uuid, peer_uuid)


def test_probe_checks_every_pair_and_agrees_on_the_weakest_rank(monkeypatch):
    topology, checked, uuids = _probe(monkeypatch, remote_features=(1, 1, 1, 100, 1, 0))
    assert checked == [[f"GPU-{u}" for u in uuids]]
    # The peer lacks FlashInfer, so neither rank selects it.
    assert topology == _topology(2, flashinfer=False)


def test_probe_detects_multi_node_groups(monkeypatch):
    topology, checked, _ = _probe(monkeypatch, hosts_differ=True)
    assert checked == []
    assert not topology.single_node and not topology.nvlink


def test_probe_takes_the_lowest_compute_capability(monkeypatch):
    topology, _, _ = _probe(monkeypatch, remote_features=(1, 1, 1, 90, 1, 1))
    assert topology.capability == "9.0"


def test_probe_is_cached_per_group(monkeypatch):
    first, _, _ = _probe(monkeypatch)
    monkeypatch.setattr(ulysses_transport, "dist", None)
    assert ulysses_transport.probe_ulysses_topology(SimpleNamespace(group_name="ulysses"), 2) is first


class _FakeNVML:
    class NVMLError(Exception):
        pass

    NVML_P2P_CAPS_INDEX_NVLINK = 0
    NVML_P2P_STATUS_OK = 0

    def __init__(self, init_fails=False, status=0):
        self.init_fails = init_fails
        self.status = status
        self.shutdowns = 0

    def nvmlInit(self):
        if self.init_fails:
            raise self.NVMLError("NVML not available in this container")

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetHandleByUUID(self, gpu):
        return gpu

    def nvmlDeviceGetP2PStatus(self, a, b, index):
        return self.status


def _patch_nvml(monkeypatch, nvml):
    import vllm.utils.import_utils as import_utils

    def load():
        if isinstance(nvml, BaseException):
            raise nvml
        return nvml

    monkeypatch.setattr(import_utils, "import_pynvml", load)


@pytest.mark.parametrize("status, expected", [(0, True), (1, False)])
def test_nvlink_probe_checks_pairs_and_shuts_nvml_down(monkeypatch, status, expected):
    nvml = _FakeNVML(status=status)
    _patch_nvml(monkeypatch, nvml)
    assert ulysses_transport._nvlink_all_pairs(["GPU-a", "GPU-b", "GPU-c"]) is expected
    assert nvml.shutdowns == 1


def test_nvlink_probe_treats_failed_nvml_init_as_no_nvlink(monkeypatch):
    nvml = _FakeNVML(init_fails=True)
    _patch_nvml(monkeypatch, nvml)
    assert ulysses_transport._nvlink_all_pairs(["GPU-a", "GPU-b"]) is False
    assert nvml.shutdowns == 0


@pytest.mark.parametrize("error", [ImportError("no pynvml"), OSError("libnvidia-ml.so.1 not found")])
def test_nvlink_probe_treats_missing_pynvml_as_no_nvlink(monkeypatch, error):
    _patch_nvml(monkeypatch, error)
    assert ulysses_transport._nvlink_all_pairs(["GPU-a", "GPU-b"]) is False


def _resolve(requested, monkeypatch, topology, backend="FLASHINFER_ATTN", impl="cute-dsl"):
    messages = []
    monkeypatch.setattr(ulysses_transport, "_LOGGED", set())
    monkeypatch.setattr(ulysses_transport.logger, "info", lambda m, *args: messages.append(("info", m % args)))
    monkeypatch.setattr(ulysses_transport.logger, "warning", lambda m, *args: messages.append(("warning", m % args)))
    monkeypatch.setattr(ulysses_transport, "probe_ulysses_topology", lambda group, world_size: topology)
    plans = [
        ulysses_transport.resolve_ulysses_transports(
            requested=requested,
            group=SimpleNamespace(group_name="sp0"),
            world_size=2,
            pure_strict=True,
            num_heads=56,
            attention_backend=backend,
            attention_impl=impl,
        )
        for _ in range(3)
    ]
    return plans[0], messages


def test_auto_logs_dispatch_order_once(monkeypatch):
    plan, messages = _resolve("auto", monkeypatch, _topology(2))
    assert plan.overlap_chunks == 14
    assert messages == [
        ("warning", "Ulysses a2a transport 'flashinfer' disabled on group 'sp0': not measured for sm10.0 x 2 GPUs"),
        (
            "info",
            "Using ['copy_engine_overlap', 'copy_engine', 'nccl'] Ulysses a2a transports (in dispatch order) "
            "for group 'sp0' out of potential: ['flashinfer', 'copy_engine_overlap', 'copy_engine', 'nccl']",
        ),
    ]


def test_auto_skips_overlap_for_variant_attention(monkeypatch):
    plan, messages = _resolve("auto", monkeypatch, _topology(2), backend="TORCH_SDPA", impl=None)
    assert plan.overlap_chunks == 0
    skipped = (
        "Ulysses a2a transport 'copy_engine_overlap' skipped for TORCH_SDPA attention on group 'sp0': "
        "not per-head invariant"
    )
    assert ("info", skipped) in messages


def test_explicit_backend_skips_the_probe(monkeypatch):
    plan, messages = _resolve("nccl", monkeypatch, None)
    assert plan == ulysses_transport.NCCL_PLAN
    assert messages == [("info", "Using Ulysses a2a transport 'nccl' for group 'sp0' (explicit)")]


@pytest.mark.parametrize(
    "backend, permute, expected",
    [("auto", False, "nccl"), ("nccl", False, "nccl"), ("copy_engine", False, None), ("auto", True, None)],
)
def test_require_default_transport_pins_auto_to_nccl(backend, permute, expected):
    parallel = SimpleNamespace(ulysses_a2a_backend=backend, ulysses_a2a_permute=permute)
    if expected is None:
        with pytest.raises(ValueError, match="supports only ulysses_a2a_backend='nccl'"):
            ulysses_transport.require_default_ulysses_transport(parallel, "Model")
    else:
        ulysses_transport.require_default_ulysses_transport(parallel, "Model")
        assert parallel.ulysses_a2a_backend == expected
