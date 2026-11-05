# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.distributed import ce_ulysses

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def isolated_groups(monkeypatch):
    monkeypatch.setattr(ce_ulysses, "_GROUPS", {})
    monkeypatch.setattr(ce_ulysses, "_OUTPUT_BUFFERS", {})


def _sp_group(name="ulysses-subgroup", size=4):
    return SimpleNamespace(ulysses_group=SimpleNamespace(group_name=name), ulysses_world_size=size)


def test_initialization_reuses_the_actual_ulysses_subgroup(monkeypatch):
    created = []
    backend_changes: list[str] = []
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    monkeypatch.setattr(ce_ulysses.symm_mem, "get_backend", lambda device: "CUDA")
    monkeypatch.setattr(ce_ulysses.symm_mem, "set_backend", backend_changes.append)
    monkeypatch.setattr(
        ce_ulysses.dist,
        "all_gather_object",
        lambda hosts, host, group: hosts.__setitem__(slice(None), [host] * len(hosts)),
    )

    def create(pg, device):
        group = SimpleNamespace(pg=pg, device=device)
        created.append(group)
        return group

    monkeypatch.setattr(ce_ulysses, "_load_fast_ulysses", lambda: create)
    first = _sp_group()
    second = _sp_group("other-subgroup")
    ce_ulysses.initialize_ulysses_groups(first)
    ce_ulysses.initialize_ulysses_groups(first)
    ce_ulysses.initialize_ulysses_groups(second)

    assert [group.pg for group in created] == [first.ulysses_group, second.ulysses_group]
    assert all(group.device == torch.device("cuda:2") for group in created)
    assert not backend_changes  # Setting the already-active CUDA backend fails on some PyTorch versions.


def test_cross_node_group_fails_before_cuda_group_creation(monkeypatch):
    monkeypatch.setattr(ce_ulysses, "_load_fast_ulysses", lambda: pytest.fail)
    monkeypatch.setattr(
        ce_ulysses.dist,
        "all_gather_object",
        lambda hosts, host, group: hosts.__setitem__(slice(None), ["host-a", "host-b"]),
    )
    with pytest.raises(ValueError, match="one NVLink-connected node"):
        ce_ulysses._group_for(_sp_group(size=2), torch.device("cuda:0"))
    assert not ce_ulysses._GROUPS


def test_single_rank_initialization_does_not_load_cuda_extension(monkeypatch):
    monkeypatch.setattr(ce_ulysses, "_load_fast_ulysses", pytest.fail)
    ce_ulysses.initialize_ulysses_groups(_sp_group(size=1))


def test_groups_are_destroyed_once():
    destroyed = []
    ce_ulysses._GROUPS.update(
        {name: SimpleNamespace(destroy=lambda name=name: destroyed.append(name)) for name in ("a", "b")}
    )
    ce_ulysses.clear_ulysses_groups()
    ce_ulysses.clear_ulysses_groups()
    assert destroyed == ["a", "b"]
    assert not ce_ulysses._GROUPS


def test_output_buffers_are_reused_per_slot_and_refreshed_for_new_shapes(monkeypatch):
    class _OutputGroup:
        def __init__(self):
            self.created = []

        def empty_output(self, tensor, *, mode):
            batch, sequence, heads, head_size = tensor.shape
            shape = (
                (batch, sequence * 2, heads // 2, head_size)
                if mode == 0
                else (batch, sequence // 2, heads * 2, head_size)
            )
            output = torch.empty(shape, dtype=tensor.dtype)
            self.created.append(output)
            return output

    group = _OutputGroup()
    ce_ulysses._GROUPS["group"] = group
    monkeypatch.setattr(ce_ulysses.torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.empty(1, 8, 4, 16)

    first = ce_ulysses._get_output_buffer(x, "group", 2, 0, 0)
    repeated = ce_ulysses._get_output_buffer(x, "group", 2, 0, 0)
    other_slot = ce_ulysses._get_output_buffer(x, "group", 2, 0, 1)
    changed_shape = ce_ulysses._get_output_buffer(torch.empty(1, 16, 4, 16), "group", 2, 0, 0)

    assert first is repeated
    assert first is not other_slot
    assert changed_shape is not first
    assert changed_shape.shape == (1, 32, 2, 16)
    assert len(group.created) == 3
    assert len(ce_ulysses._OUTPUT_BUFFERS) == 2


def test_output_buffer_cache_miss_falls_back_during_cuda_graph_capture(monkeypatch):
    class _OutputGroup:
        def empty_output(self, *_args, **_kwargs):
            pytest.fail("symmetric output allocation must not happen during capture")

    ce_ulysses._GROUPS["group"] = _OutputGroup()
    monkeypatch.setattr(ce_ulysses.torch.cuda, "is_current_stream_capturing", lambda: True)
    assert ce_ulysses._get_output_buffer(torch.empty(1, 8, 4, 16), "group", 2, 0, 0) is None
    assert not ce_ulysses._OUTPUT_BUFFERS


def test_custom_op_fake_metadata_roundtrip():
    # FakeTensorMode never allocates or initializes a CUDA device.
    with FakeTensorMode():
        x = torch.empty(1, 17, 8, 64, device="cuda", dtype=torch.float16)
        y = ce_ulysses.ulysses_all_to_all_4d(x, "fake-only", 2, 0)
        z = ce_ulysses.ulysses_all_to_all_4d(y, "fake-only", 2, 1)
        assert y.shape == (1, 34, 4, 64)
        assert z.shape == x.shape
        assert z.dtype == x.dtype
        assert z.device == x.device
        y_reuse = ce_ulysses.ulysses_all_to_all_4d(x, "fake-only", 2, 0, slot=0)
        assert y_reuse.shape == y.shape
        assert y_reuse.dtype == x.dtype
