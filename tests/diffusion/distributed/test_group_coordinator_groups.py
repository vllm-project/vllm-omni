# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The broadcast helpers must use the group the caller asked for."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator, TensorMetadata

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model, pytest.mark.cpu]


def _make_coordinator(rank: int = 1) -> GroupCoordinator:
    ranks = [1, 3]
    coordinator = object.__new__(GroupCoordinator)
    coordinator.rank = rank
    coordinator.ranks = ranks
    coordinator.rank_in_group = ranks.index(rank)
    coordinator.world_size = len(ranks)
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    coordinator.shm_broadcaster = None
    return coordinator


def test_broadcast_object_list_honours_explicit_group(monkeypatch):
    coordinator = _make_coordinator()
    explicit_group = object()
    seen = {}

    def fake_broadcast(obj_list, src=0, group=None):
        seen["src"] = src
        seen["group"] = group
        return obj_list

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast)

    payload = [{"step": 1}]
    result = coordinator.broadcast_object_list(payload, src=1, group=explicit_group)

    assert result is payload
    assert seen == {"src": 3, "group": explicit_group}


def test_broadcast_object_list_defaults_to_cpu_group(monkeypatch):
    coordinator = _make_coordinator()
    seen = {}

    def fake_broadcast(obj_list, src=0, group=None):
        seen["group"] = group
        return obj_list

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast)

    coordinator.broadcast_object_list([{"step": 1}], src=0)

    assert seen["group"] is coordinator.cpu_group


def test_broadcast_tensor_dict_honours_explicit_groups(monkeypatch):
    coordinator = _make_coordinator()
    explicit_group = object()
    explicit_metadata_group = object()
    metadata = [("payload", TensorMetadata("cpu", torch.float32, (2,)))]
    seen_groups = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(coordinator, "broadcast_object", lambda value, src: metadata)

    def fake_broadcast(tensor, src=0, group=None, async_op=False):
        seen_groups.append(group)
        tensor.fill_(7)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    result = coordinator.broadcast_tensor_dict(
        {"payload": torch.full((2,), 7.0)},
        src=0,
        group=explicit_group,
        metadata_group=explicit_metadata_group,
    )

    assert torch.equal(result["payload"], torch.full((2,), 7.0))
    assert seen_groups == [explicit_metadata_group]


def test_broadcast_tensor_dict_metadata_only_uses_host_path(monkeypatch):
    """Metadata-only dicts go through broadcast_object, not a tensor collective."""
    coordinator = _make_coordinator(rank=3)
    metadata = [("step", 42)]
    seen = {}

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_broadcast_object(value, src):
        seen["src"] = src
        return metadata

    def fail_tensor_broadcast(*args, **kwargs):
        raise AssertionError("metadata-only dict must not issue a tensor broadcast")

    monkeypatch.setattr(coordinator, "broadcast_object", fake_broadcast_object)
    monkeypatch.setattr(torch.distributed, "broadcast", fail_tensor_broadcast)

    result = coordinator.broadcast_tensor_dict(
        None,
        src=0,
        group=object(),
        metadata_group=object(),
    )

    assert result == {"step": 42}
    assert seen == {"src": 0}


def test_broadcast_tensor_dict_uses_group_for_device_tensors(monkeypatch):
    coordinator = _make_coordinator(rank=3)
    explicit_group = object()
    explicit_metadata_group = object()
    metadata = [("payload", TensorMetadata("meta", torch.float32, (2,)))]
    seen_groups = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(coordinator, "broadcast_object", lambda value, src: metadata)

    def fake_broadcast(tensor, src=0, group=None, async_op=False):
        assert src == 1
        seen_groups.append(group)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    coordinator.broadcast_tensor_dict(
        None,
        src=0,
        group=explicit_group,
        metadata_group=explicit_metadata_group,
    )

    assert seen_groups == [explicit_group]
