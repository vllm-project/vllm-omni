# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared-metadata and text-reduction tests for the SeedVR2 window path.

These tests exercise the implementations the production path calls
(``joint_cu_seqlens``, ``build_local_window_context``,
``SeedVR2WindowRuntime.reduce_text`` / ``global_window_mean``) rather than
private copies.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.seedvr2.na_ops import (
    SeedVR2WindowRuntime,
    build_local_window_context,
)
from vllm_omni.diffusion.models.seedvr2.window_sp import (
    INT32_MAX,
    RankWindowPlan,
    WindowLayoutKey,
    global_window_mean,
    joint_cu_seqlens,
    to_cu_seqlens,
)

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.parallel,
    pytest.mark.sp,
    pytest.mark.core_model,
    pytest.mark.cpu,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

REGULAR = "720pwin_by_size_bysize"


def _rank_plan(lengths: list[int]) -> RankWindowPlan:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    total = offsets[-1]
    return RankWindowPlan(
        layout_key=WindowLayoutKey(
            token_grid=(max(total, 1), 1, 1),
            window_method=REGULAR,
            window_shape=(1, 1, 1),
            geometry_version=1,
            geometry_fingerprint="test-fixture",
        ),
        rank=0,
        window_ids=torch.arange(len(lengths), dtype=torch.int64),
        global_token_ids=torch.arange(total, dtype=torch.int64),
        video_cu_seqlens=to_cu_seqlens(torch.tensor(offsets, dtype=torch.int64), context="test fixture"),
        window_shapes=torch.tensor([[length, 1, 1] for length in lengths], dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# joint offsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text_len", [0, 1, 7])
def test_joint_cu_seqlens_matches_hand_computation(text_len):
    video = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    joint = joint_cu_seqlens(video, text_len)
    expected = [value + index * text_len for index, value in enumerate([0, 2, 5, 7])]
    assert joint.dtype == torch.int32
    assert joint.tolist() == expected


def test_joint_cu_seqlens_device_follows_the_input():
    video = torch.tensor([0, 3, 4], dtype=torch.int32)
    assert joint_cu_seqlens(video, 2).device == video.device
    if torch.cuda.is_available():
        cuda_video = video.to("cuda")
        joint = joint_cu_seqlens(cuda_video, 2)
        assert joint.device == cuda_video.device and joint.tolist() == [0, 5, 8]


def test_joint_cu_seqlens_empty_rank_returns_device_local_zero():
    empty = torch.zeros(1, dtype=torch.int64)
    joint = joint_cu_seqlens(empty, 4)
    assert joint.dtype == torch.int32 and joint.tolist() == [0]
    assert joint.device == empty.device


def test_joint_cu_seqlens_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="text_len"):
        joint_cu_seqlens(torch.tensor([0, 1], dtype=torch.int64), -1)
    with pytest.raises(ValueError, match="leading zero|empty"):
        joint_cu_seqlens(torch.empty(0, dtype=torch.int64), 1)
    with pytest.raises(ValueError, match="start at 0"):
        joint_cu_seqlens(torch.tensor([1, 2], dtype=torch.int64), 1)
    with pytest.raises(ValueError, match="non-decreasing"):
        joint_cu_seqlens(torch.tensor([0, 5, 3], dtype=torch.int64), 1)


def test_joint_cu_seqlens_rejects_int32_overflow_before_conversion():
    # A window that ends exactly at the int32 bound is representable; one past it
    # must be rejected while the metadata is still int64.
    ok = joint_cu_seqlens(torch.tensor([0, INT32_MAX], dtype=torch.int64), 0)
    assert ok.tolist() == [0, INT32_MAX]
    with pytest.raises(ValueError, match="int32"):
        joint_cu_seqlens(torch.tensor([0, INT32_MAX], dtype=torch.int64), 1)
    with pytest.raises(ValueError, match="int32"):
        joint_cu_seqlens(torch.tensor([0, INT32_MAX + 1], dtype=torch.int64), 0)


# ---------------------------------------------------------------------------
# production context uses the same offsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lengths", [[1], [2, 3, 2], [4, 4, 4, 1]])
@pytest.mark.parametrize("text_len", [0, 1, 5])
def test_context_joint_offsets_match_the_helper(lengths, text_len):
    plan = _rank_plan(lengths)
    ctx = build_local_window_context(plan, text_len=text_len, global_windows=len(lengths), device="cpu")
    assert torch.equal(ctx.joint_cu_seqlens, joint_cu_seqlens(plan.video_cu_seqlens, text_len))


def test_context_packing_indices_follow_the_joint_offsets():
    lengths = [2, 3, 2]
    text_len = 2
    plan = _rank_plan(lengths)
    ctx = build_local_window_context(plan, text_len=text_len, global_windows=len(lengths), device="cpu")

    num_video = int(plan.global_token_ids.numel())
    video = torch.arange(num_video, dtype=torch.float32).unsqueeze(1)
    text = torch.full((text_len, 1), -1.0)
    from vllm_omni.diffusion.models.seedvr2.na_ops import pack_joint_windows, unpack_joint_windows

    joint = pack_joint_windows(video, text, ctx)
    cu = ctx.joint_cu_seqlens.tolist()
    for window, length in enumerate(lengths):
        start, end = cu[window], cu[window + 1]
        assert end - start == length + text_len
        assert torch.equal(
            joint[start : start + length, 0], video[sum(lengths[:window]) : sum(lengths[: window + 1]), 0]
        )
        assert torch.all(joint[start + length : end] == -1.0)

    vid_out, txt_windows = unpack_joint_windows(joint, ctx)
    assert torch.equal(vid_out, video)
    assert txt_windows.shape == (len(lengths), text_len, 1)


# ---------------------------------------------------------------------------
# global_window_mean helper
# ---------------------------------------------------------------------------


def test_global_window_mean_local_branch_has_no_collective(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("group=None must not reach all_reduce")

    monkeypatch.setattr(dist, "all_reduce", explode)
    local = torch.tensor([[4.0, 10.0]])
    result = global_window_mean(local, 4)
    # Local branch: the caller's sum over the global count, no collective.
    assert torch.allclose(result, torch.tensor([[1.0, 2.5]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_global_window_mean_accumulates_low_precision_in_fp32(dtype):
    local = torch.tensor([[4.0, 10.0]], dtype=dtype)
    result = global_window_mean(local, 2, dtype=dtype)
    assert result.dtype == dtype
    assert torch.allclose(result.float(), torch.tensor([[2.0, 5.0]]), atol=1e-3)


def test_global_window_mean_preserves_float64_and_does_not_mutate_input():
    local = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    snapshot = local.clone()
    result = global_window_mean(local, 3)
    assert result.dtype == torch.float64
    assert torch.equal(local, snapshot), "the caller's tensor must not be modified"
    tiny = torch.tensor([[1.0 + 1e-12]], dtype=torch.float64)
    assert global_window_mean(tiny, 1).item() != 1.0, "float64 increments must not be flattened"


def test_global_window_mean_rejects_non_positive_count():
    with pytest.raises(ValueError, match="positive"):
        global_window_mean(torch.zeros(1, 1), 0)
    with pytest.raises(ValueError, match="positive"):
        global_window_mean(torch.zeros(1, 1), -2)


# ---------------------------------------------------------------------------
# production reduce_text
# ---------------------------------------------------------------------------


def _runtime(world_size: int = 1, group=None) -> SeedVR2WindowRuntime:
    return SeedVR2WindowRuntime((8, 1, 1), text_len=1, group=group, world_size=world_size, rank=0, num_layers=1)


def test_reduce_text_local_branch_matches_the_global_mean(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("SP=1 must not issue a collective")

    monkeypatch.setattr(dist, "all_reduce", explode)
    runtime = _runtime()
    result = runtime.reduce_text(torch.tensor([[3.0, 5.0]]), 2)
    assert torch.allclose(result, torch.tensor([[1.5, 2.5]]))
    assert runtime.stats["fused_text_mean_calls"] == 1
    assert runtime.stats["text_all_reduces"] == 0


def test_reduce_text_sums_across_ranks_not_rank_means(monkeypatch):
    """Three windows [1, 3, 11]; rank0 holds two (local sum 4), rank1 holds one."""
    peer_sum = 11.0

    def fake_all_reduce(tensor, op=None, group=None):
        assert op == dist.ReduceOp.SUM
        tensor.add_(peer_sum)

    class _FakeGroup:
        pass

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    runtime = _runtime(world_size=2, group=_FakeGroup())
    result = runtime.reduce_text(torch.tensor([[4.0]]), 3)
    assert result.item() == pytest.approx(15.0 / 3.0)
    assert result.item() != pytest.approx((4.0 + 11.0) / 2), "must not be the mean of rank means"
    assert runtime.stats["text_all_reduces"] == 1
    assert runtime.stats["fused_text_mean_calls"] == 0


def test_reduce_text_empty_rank_contributes_zeros_and_joins_the_collective(monkeypatch):
    seen: list[float] = []

    def fake_all_reduce(tensor, op=None, group=None):
        seen.append(float(tensor.sum()))
        tensor.add_(7.0)

    class _FakeGroup:
        pass

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    runtime = _runtime(world_size=2, group=_FakeGroup())
    empty = torch.zeros((0, 4), dtype=torch.float16)
    result = runtime.reduce_text(empty, 2)
    assert seen == [0.0], "an empty rank must submit zeros in the peers' dtype"
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.full((1, 4), 3.5))
    assert runtime.stats["text_all_reduces"] == 1


def test_multi_rank_runtime_is_rejected_without_a_group():
    """A multi-rank runtime without a group cannot even be constructed."""
    with pytest.raises(ValueError, match="group is required"):
        _runtime(world_size=2, group=None)


def test_reduce_text_rejects_a_group_of_the_wrong_size(monkeypatch):
    class _FakeGroup:
        pass

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    runtime = _runtime(world_size=2, group=_FakeGroup())
    # The group must still match the runtime's world size at reduction time.
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)
    with pytest.raises(RuntimeError, match="does not match world_size"):
        runtime.reduce_text(torch.ones(1, 1), 2)
