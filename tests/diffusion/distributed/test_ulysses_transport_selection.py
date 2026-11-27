# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed import ulysses_transport
from vllm_omni.diffusion.distributed.ulysses_transport import select_transport

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

GROUP = SimpleNamespace(group_name="sp0")


class _FakeDist:
    """The peer ranks' contributions to the probe's collectives."""

    ReduceOp = SimpleNamespace(MIN="min")

    def __init__(self, world_size, hosts=None, peer_features=None):
        self.world_size, self.hosts, self.peer_features = world_size, hosts, peer_features

    def get_process_group_ranks(self, group):
        return getattr(group, "ranks", list(range(self.world_size)))

    def all_gather_into_tensor(self, out, local, group):
        hosts = self.hosts or [int(local[0])] * self.world_size
        out.copy_(torch.tensor([[h, i] for i, h in enumerate(hosts)]).flatten())

    def all_reduce(self, tensor, op, group):
        if self.peer_features is not None:
            tensor.copy_(torch.minimum(tensor, torch.tensor(self.peer_features)))


@pytest.fixture
def probe(monkeypatch):
    """Run the probe on a fake group; returns the reason (None: copy engine) and what was logged under auto."""

    def run(world_size, *, nvlink=True, p2p=True, nvml_error=False, hosts=None, peer_features=None, **select):
        def fully_connected(ids):
            if nvml_error:
                raise RuntimeError("NVML Shared Library Not Found")
            return nvlink

        platform = SimpleNamespace(is_fully_connected=fully_connected, visible_device_id_to_physical_device_id=int)
        monkeypatch.setattr(ulysses_transport, "current_platform", platform)
        monkeypatch.setattr(ulysses_transport, "dist", _FakeDist(world_size, hosts, peer_features))
        monkeypatch.setattr(ulysses_transport, "_is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(ulysses_transport, "_probe_device", lambda: SimpleNamespace(index=0, type="cpu"))
        monkeypatch.setattr(ulysses_transport.torch, "tensor", _cpu(torch.tensor))
        monkeypatch.setattr(ulysses_transport.torch, "empty", _cpu(torch.empty))
        monkeypatch.setattr(ulysses_transport.torch.accelerator, "device_count", lambda: world_size)
        monkeypatch.setattr(ulysses_transport.torch.cuda, "can_device_access_peer", lambda a, b: p2p)
        monkeypatch.setattr(ulysses_transport, "_symmetric_memory_available", lambda: True)
        monkeypatch.setattr(ulysses_transport, "_GROUPS", {})
        monkeypatch.setattr(ulysses_transport, "_LOGGED", set())
        messages = []
        monkeypatch.setattr(ulysses_transport.logger, "info", messages.append)
        options = dict(cuda_graphs=False, strict_pure=True) | select
        requested = options.pop("requested", "auto")
        transports = [select_transport(requested, GROUP, world_size, **options) for _ in range(3)]
        assert len(set(transports)) == 1 and len(messages) <= 1  # decided and logged once per group
        return transports[0], messages

    return run


def _cpu(make):
    return lambda *args, device=None, **kwargs: make(*args, **kwargs)


@pytest.mark.parametrize(
    "world_size, settings, transport, log",
    [
        (2, {}, "copy_engine", "(auto: 2 GPUs fully NVLink-connected)"),
        (8, {}, "copy_engine", "(auto: 8 GPUs fully NVLink-connected)"),
        (2, {"nvlink": False}, "copy_engine", "(auto: 2 GPUs with PCIe P2P)"),
        (4, {"nvlink": False}, "nccl", "copy engine needs full NVLink for more than two PCIe GPUs"),
        (4, {"nvml_error": True}, "nccl", "copy engine needs full NVLink for more than two PCIe GPUs"),
        (2, {"nvml_error": True}, "copy_engine", "(auto: 2 GPUs with PCIe P2P)"),
        (2, {"p2p": False}, "nccl", "GPU P2P access unavailable"),
        (2, {"hosts": [1, 2]}, "nccl", "group spans nodes"),
        (2, {"peer_features": [1, 1, 1, 0]}, "nccl", "torch symmetric memory unavailable"),  # agreed with the peer
        (2, {"cuda_graphs": True}, "nccl", "CUDA-graphed denoising"),
        (2, {"strict_pure": False}, "nccl", "layer is not strict pure Ulysses"),
        (2, {"p2p": False, "overlap_chunks": 4}, "nccl", "ulysses_overlap_chunks ignored"),
        (2, {"requested": "nccl"}, "nccl", None),
        (2, {"requested": "copy_engine"}, "copy_engine", "(explicit)"),
        (2, {"requested": "copy_engine", "strict_pure": False}, "copy_engine", "(explicit)"),  # per-call fallback
        (2, {"requested": "copy_engine", "p2p": False}, ValueError, None),
        (4, {"requested": "copy_engine", "nvlink": False}, ValueError, None),
    ],
)
def test_select_transport(probe, world_size, settings, transport, log):
    if transport is ValueError:
        with pytest.raises(ValueError, match="ulysses_a2a_backend='copy_engine' is not supported here"):
            probe(world_size, **settings)
        return
    chosen, messages = probe(world_size, **settings)
    assert chosen == transport
    assert (log is None and not messages) or (log in messages[0] and f"Using '{transport}'" in messages[0])


def test_outside_nvidia_cuda_auto_is_nccl_without_a_probe(monkeypatch):
    monkeypatch.setattr(ulysses_transport, "_is_nvidia_cuda", lambda: False)
    monkeypatch.setattr(ulysses_transport, "_group_probe", pytest.fail)
    assert select_transport("auto", GROUP, 2, cuda_graphs=False, strict_pure=True) == "nccl"
    # Explicit symmetric memory on other platforms (NPU) keeps its own validation.
    assert select_transport("copy_engine", GROUP, 2, cuda_graphs=False, strict_pure=True) == "copy_engine"


def test_probe_is_cached_per_group_and_members(monkeypatch):
    probes = []
    monkeypatch.setattr(ulysses_transport, "_GROUPS", {})
    monkeypatch.setattr(ulysses_transport, "dist", _FakeDist(2))
    monkeypatch.setattr(ulysses_transport, "_probe", lambda group, world_size: probes.append(group) or (None, ""))
    for group in (GROUP, GROUP, SimpleNamespace(group_name="sp0", ranks=[2, 3])):
        ulysses_transport._group_probe(group, 2)
    assert len(probes) == 2
    ulysses_transport.clear_ulysses_transport_state()
    ulysses_transport._group_probe(GROUP, 2)
    assert len(probes) == 3


@pytest.mark.parametrize(
    "values, backend",
    [
        ({}, "auto"),
        ({"ulysses_a2a_backend": "nccl"}, "nccl"),
        ({"ulysses_overlap_chunks": 7}, "auto"),
        ({"ulysses_overlap_chunks": 7, "ulysses_a2a_backend": "copy_engine"}, "copy_engine"),
        ({"ulysses_overlap_chunks": 7, "ulysses_a2a_backend": "nccl"}, ValueError),
        ({"ulysses_a2a_backend": "copy_engine_overlap"}, ValueError),
    ],
)
def test_config(values, backend):
    if backend is ValueError:
        with pytest.raises(ValueError):
            DiffusionParallelConfig(ulysses_degree=2, **values)
    else:
        assert DiffusionParallelConfig(ulysses_degree=2, **values).ulysses_a2a_backend == backend


def test_removed_permute_field_is_rejected():
    # A plain dataclass raises TypeError; vLLM's pydantic-wrapped config raises a ValidationError (a ValueError).
    with pytest.raises((TypeError, ValueError)):
        DiffusionParallelConfig(ulysses_degree=2, ulysses_a2a_permute=True)


@pytest.mark.parametrize("backend, pinned", [("auto", "nccl"), ("nccl", "nccl"), ("copy_engine", ValueError)])
def test_models_without_copy_engine_support(backend, pinned):
    parallel = SimpleNamespace(ulysses_a2a_backend=backend)
    if pinned is ValueError:
        with pytest.raises(ValueError, match="supports only ulysses_a2a_backend='nccl'"):
            ulysses_transport.require_default_ulysses_transport(parallel, "Model")
    else:
        ulysses_transport.require_default_ulysses_transport(parallel, "Model")
        assert parallel.ulysses_a2a_backend == pinned
