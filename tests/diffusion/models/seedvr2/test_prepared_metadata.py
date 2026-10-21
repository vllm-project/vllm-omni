# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packing parity, request isolation and retained-storage budget contracts."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.models.seedvr2 import nadit as n

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class NoDeviceScalar(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            assert args[0].device.type == "cpu", "device scalar read during plan preparation"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("grid", [(2, 3, 5), (1, 1, 1), (3, 11, 7)])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("text_len", [0, 3])
def test_prepared_context_matches_independent_window_packing(device, grid, world, text_len):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    for method in n.DEFAULT_WINDOW_METHODS:
        layout = n.build_window_layout(grid, method)
        assignment = n.build_window_assignment(layout, world)
        for rank in range(world):
            plan = n.build_rank_window_plan(layout, assignment, rank)
            with NoDeviceScalar():
                ctx = n.build_local_window_context(
                    plan, text_len=text_len, global_windows=layout.num_windows, device=device
                )
                assert ctx.max_joint_len == max(
                    (
                        b - a + text_len
                        for a, b in zip(plan.video_cu_seqlens.tolist(), plan.video_cu_seqlens.tolist()[1:])
                    ),
                    default=0,
                )
            video = torch.arange(plan.num_local_tokens * 2, device=device).reshape(-1, 2)
            text = -torch.arange(text_len * 2, device=device).reshape(-1, 2) - 1
            pieces: list[torch.Tensor] = []
            offsets = plan.video_cu_seqlens.tolist()
            for a, b in zip(offsets, offsets[1:]):
                pieces.extend((video[a:b], text))
            expected = torch.cat(pieces) if pieces else torch.empty((0, 2), device=device, dtype=video.dtype)
            packed = n.pack_joint_windows(video, text, ctx)
            torch.testing.assert_close(packed, expected, rtol=0, atol=0)
            unpacked, repeated = n.unpack_joint_windows(packed, ctx)
            torch.testing.assert_close(unpacked, video, rtol=0, atol=0)
            if plan.num_local_windows:
                torch.testing.assert_close(repeated, text.expand(plan.num_local_windows, -1, -1), rtol=0, atol=0)
            assert ctx.joint_cu_seqlens.cpu().tolist() == [v + i * text_len for i, v in enumerate(offsets)]


@pytest.mark.cpu
def test_cache_budget_retirement_and_oversized_bypass():
    cache = n.DeviceWindowPlanCache(max_bytes=64)
    held = cache.get("a", lambda: torch.arange(8))
    same = cache.get("a", lambda: pytest.fail("cache miss"))
    assert held is same
    cache.get("b", lambda: torch.zeros(8, dtype=torch.long))
    assert "a" not in cache.entries and cache.resident_bytes == 64
    assert held.tolist() == list(range(8))  # active consumer survives eviction
    oversized = cache.get("c", lambda: torch.ones(9, dtype=torch.long))
    assert oversized.numel() == 9 and "c" not in cache.entries
    cache.clear()
    assert cache.resident_bytes == 0 and not cache.entries


@pytest.mark.cpu
def test_new_request_reuses_only_metadata_and_shape_identity():
    cache = n.DeviceWindowPlanCache()

    def runtime(grid=(2, 3, 5)):
        return n.SeedVR2WindowRuntime(grid, text_len=3, num_layers=2, device_cache=cache)

    a, b, again = runtime(), runtime(), runtime()
    contexts = [r.context(r.layout_for_layer(0), "cpu") for r in (a, b, again)]
    ca, cb, cc = contexts
    assert ca is not cb and ca.joint_order is cb.joint_order is cc.joint_order
    assert ca.rotary_cache is not cb.rotary_cache and a.stats is not b.stats
    a.stats["text_all_reduces"] = 8
    assert b.stats["text_all_reduces"] == 0
    video = torch.randn(ca.num_video_tokens, 2)
    text = torch.randn(3, 2)
    before = n.pack_joint_windows(video, text, ca)
    other = n.pack_joint_windows(video + 3, text - 2, cb)
    assert not torch.equal(before, other)
    torch.testing.assert_close(before, n.pack_joint_windows(video, text, cc), rtol=0, atol=0)
    swapped = runtime((2, 5, 3))
    different = swapped.context(swapped.layout_for_layer(0), "cpu")
    assert different.joint_order is not ca.joint_order
    assert cache.resident_bytes <= cache.max_bytes


@pytest.mark.cpu
def test_cpu_geometry_and_legacy_tensor_callers_agree():
    model = n.SeedVR2NaDiT.__new__(n.SeedVR2NaDiT)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    assert model.token_grid_for((3, 6, 10)) == model.token_grid_for(torch.tensor([[3, 6, 10]])) == (3, 3, 5)
    with pytest.raises(ValueError):
        model.token_grid_for((0, 6, 10))


@pytest.mark.cpu
def test_cpu_plan_cache_churn_is_byte_bounded():
    cache = n.WindowPlanCache(capacity=2, max_bytes=2048)
    for grid in [(1, 2, 3), (2, 3, 5), (3, 7, 11), (1, 2, 3)]:
        layout = cache.layout(grid, n.DEFAULT_WINDOW_METHODS[0])
        assignment = cache.assignment(layout, 4)
        for rank in range(4):
            plan = cache.rank_plan(layout, assignment, rank)
            assert plan.num_local_tokens == assignment.rank_token_counts[rank]
            assert cache.resident_bytes <= cache.max_bytes
    cache.clear()
    assert cache.resident_bytes == 0 and not cache._order


@pytest.mark.cpu
def test_zero_byte_entries_have_an_entry_bound():
    cache = n.DeviceWindowPlanCache(max_bytes=0, capacity=2)
    for index in range(10):
        cache.get(index, lambda: torch.empty(0))
    assert len(cache.entries) == 2 and cache.resident_bytes == 0


def _canonical_worker(rank, world, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    real_gather = dist.all_gather
    calls = []

    def rows_only(outputs, values, **kwargs):
        assert values.dtype == torch.float32, "unexpected metadata count collective"
        calls.append(tuple(values.shape))
        return real_gather(outputs, values, **kwargs)

    dist.all_gather = rows_only
    try:
        for grid in [(1, 1, 1), (3, 11, 7)]:
            runtime = n.SeedVR2WindowRuntime(grid, text_len=3, group=dist.group.WORLD, world_size=world, rank=rank)
            for layer in [0, 1]:
                layout = runtime.layout_for_layer(layer)
                canonical = torch.arange(layout.num_tokens * 3, dtype=torch.float32).reshape(-1, 3)
                for shift in [0, 7, 0]:
                    local = runtime.local_rows_for(canonical + shift, layout)
                    before = len(calls)
                    actual = runtime.to_canonical_rows(local, layout)
                    assert len(calls) == before + 1
                    torch.testing.assert_close(actual, canonical + shift, rtol=0, atol=0)
    finally:
        dist.all_gather = real_gather
        dist.destroy_process_group()


@pytest.mark.cpu
@pytest.mark.parametrize("world", [2, 4])
def test_canonical_rows_real_distributed_uneven_and_empty(world, tmp_path):
    torch.multiprocessing.spawn(
        _canonical_worker, args=(world, "file://" + str(tmp_path / "gloo")), nprocs=world, join=True
    )


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_implicit_cuda_device_has_a_concrete_cache_identity():
    if torch.accelerator.device_count() < 2:
        pytest.skip("two CUDA devices required")
    cache = n.DeviceWindowPlanCache()
    runtime = n.SeedVR2WindowRuntime((2, 3, 5), text_len=3, device_cache=cache)
    layout = runtime.layout_for_layer(0)
    with torch.accelerator.device_index(0):
        first = runtime.context(layout, "cuda")
        assert first.joint_order.device == torch.device("cuda", 0)
    with torch.accelerator.device_index(1):
        second = runtime.context(layout, "cuda")
        assert second.joint_order.device == torch.device("cuda", 1)
    with torch.accelerator.device_index(0):
        third = runtime.context(layout, "cuda")
        assert third.joint_order is first.joint_order
