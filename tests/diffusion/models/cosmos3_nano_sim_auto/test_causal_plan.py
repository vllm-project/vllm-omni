# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal maskless planner: token-level oracle, hand-derived reference rules, numerics and the ring remap.

The oracle below re-derives the reference replay rules per token in plain
Python (independently of the vectorised run-level predicate), and the
hand-written expectations encode specific consequences of those rules read
from the imaginaire4 predicate and its maskless replay partitions: condition
frames read only themselves, their own controls and captions; controls never
read current targets and read clean history of their own view only (the
maskless cross-view partition excludes controls); cross-view reads are
past-only within 0.4 s; chunk 1 sees only chunk 0 and chunk k >= 2 sees chunks
0 and k-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import multiview_maskless_attention as m
from vllm_omni.diffusion.models.cosmos3.multiview_maskless_attention import maskless_attention_streams
from vllm_omni.diffusion.models.cosmos3.multiview_maskless_plan import META_K_START, META_KEY_STREAM, META_Q_START
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.causal_plan import (
    STREAM_CURRENT,
    STREAM_HISTORY,
    STREAM_STAGED,
    STREAM_UND,
    CausalMasklessPlan,
    build_causal_maskless_plan,
    dense_token_mask,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import SensorGrid, build_chunk_schedule
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import (
    CAPTION_SCOPE_ALL,
    CAPTION_SCOPE_SAME_VIEW,
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    CaptionDocument,
    RingSlot,
    TokenMetadata,
    build_chunk_items,
    caption_metadata,
    chunk_token_metadata,
    committed_metadata,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

RGB = SensorGrid(2, 2, 2, 2)  # 1 token per camera frame
LIDAR = SensorGrid(2, 4, 2, 2)  # 2 tokens per sweep
RGB_SPF, LIDAR_SPF, WINDOW, UNIT = 0.2, 0.1, 0.4, 32
LIDAR_VIEW = 2
HORIZON_SECONDS = 1.4


@dataclass
class Scenario:
    plan: CausalMasklessPlan
    current: TokenMetadata
    slots: list[RingSlot]
    captions_docs: tuple[CaptionDocument, ...]

    def q(self, **fields) -> int:
        return _single(self.current, **fields)

    def k_current(self, **fields) -> int:
        return _single(self.current, **fields)

    def k_history(self, slot: int, **fields) -> int:
        return self.plan.num_gen_tokens + self.slots[slot].base + _single(self.slots[slot].metadata, **fields)

    def k_und(self, token: int) -> int:
        return self.plan.num_gen_tokens + self.plan.num_history_tokens + token


def _single(metadata: TokenMetadata, **fields) -> int:
    mask = torch.ones(metadata.num_tokens, dtype=torch.bool)
    for name, value in fields.items():
        mask &= getattr(metadata, name) == value
    index = torch.nonzero(mask).flatten()
    assert index.numel() >= 1, fields
    return int(index[0])


def captions(num_views: int) -> tuple[CaptionDocument, ...]:
    docs = []
    for view in range(num_views):
        docs.append(CaptionDocument(view, 2, is_sink=True))
        docs.append(CaptionDocument(view, 3 + view, start_seconds=0.0, end_seconds=HORIZON_SECONDS))
    return tuple(docs)


def scenario(
    chunk_index: int,
    *,
    pass_kind: str = "noisy",
    joint: bool = True,
    num_views: int = 2,
    lidar_condition: bool = False,
    device: str = "cpu",
) -> Scenario:
    schedule = build_chunk_schedule(
        rgb_latents=7,
        lidar_sweeps=13 if joint else None,
        frames_per_chunk=2,
        rgb_seconds_per_frame=RGB_SPF,
        lidar_seconds_per_frame=LIDAR_SPF if joint else None,
    )

    def items(index: int):
        return build_chunk_items(
            schedule[index],
            num_views=num_views,
            rgb_grid=RGB,
            rgb_seconds_per_frame=RGB_SPF,
            lidar_grid=LIDAR if joint else None,
            lidar_seconds_per_frame=LIDAR_SPF if joint else None,
            rgb_condition_frames=(0,),
            lidar_condition_sweeps=(0,) if lidar_condition else (),
        )

    slots: list[RingSlot] = []
    if chunk_index >= 1:
        slots.append(RingSlot(0, 0, committed_metadata(chunk_token_metadata(items(0), step=0, pass_kind="clean")[0])))
    if chunk_index >= 2:
        previous = chunk_token_metadata(items(chunk_index - 1), step=chunk_index - 1, pass_kind="clean")[0]
        slots.append(RingSlot(1, UNIT, committed_metadata(previous)))
    current, _ = chunk_token_metadata(items(chunk_index), step=chunk_index, pass_kind=pass_kind)
    docs = captions(num_views)
    plan = build_causal_maskless_plan(
        current=current, history=slots, captions=caption_metadata(docs), window_seconds=WINDOW, device=device
    )
    return Scenario(plan, current, slots, docs)


# ---------------------------------------------------------------------------
# Independent token-level oracle of the reference replay predicate.
# ---------------------------------------------------------------------------


def _token_oracle(scn: Scenario) -> torch.Tensor:
    cur = scn.current
    gen = cur.num_tokens
    hist_len = scn.plan.num_history_tokens
    und_meta = caption_metadata(scn.captions_docs)
    keys: list[dict | None] = []
    for i in range(gen):
        keys.append(_sensor_token(cur, i))
    hist_tokens: list[dict | None] = [None] * hist_len
    for slot in scn.slots:
        for i in range(slot.metadata.num_tokens):
            hist_tokens[slot.base + i] = _sensor_token(slot.metadata, i)
    keys.extend(hist_tokens)
    for i in range(und_meta.num_tokens):
        keys.append(
            {
                "und": True,
                "view": int(und_meta.view_id[i]),
                "start": float(und_meta.start_seconds[i]),
                "end": float(und_meta.end_seconds[i]),
                "role": ROLE_UND,
            }
        )
    allowed = torch.zeros(gen, len(keys), dtype=torch.bool)
    for qi in range(gen):
        q = _sensor_token(cur, qi)
        for ki, k in enumerate(keys):
            allowed[qi, ki] = k is not None and _pair(q, k)
    return allowed


def _sensor_token(meta: TokenMetadata, i: int) -> dict:
    return {
        "und": False,
        "view": int(meta.view_id[i]),
        "frame": int(meta.frame_id[i]),
        "t": float(meta.timestamp[i]),
        "role": int(meta.role[i]),
        "step": int(meta.step[i]),
        "scope": int(meta.caption_scope[i]),
    }


def _pair(q: dict, k: dict) -> bool:
    eps = 1e-4
    if k["und"]:
        reaches = q["scope"] == CAPTION_SCOPE_ALL or (q["scope"] == CAPTION_SCOPE_SAME_VIEW and q["view"] == k["view"])
        reaches = reaches and (q["t"] >= k["start"] - eps) and (q["t"] < k["end"] - eps)
        # Targets, controls and conditions all read captions the same way.
        return bool(reaches)
    same_view = q["view"] == k["view"]
    gap = q["t"] - k["t"]
    in_window = -eps <= gap <= WINDOW + eps
    in_scope = same_view or in_window
    q_role, k_role = q["role"], k["role"]
    q_target = q_role in (ROLE_CURRENT_TARGET, ROLE_CLEAN_TARGET)
    q_current = q_role == ROLE_CURRENT_TARGET
    k_clean_sensor = k_role in (ROLE_TARGET_CONDITION, ROLE_CLEAN_TARGET)
    step_le = k["step"] <= q["step"]
    if q_role == ROLE_CONTROL:
        if k_role == ROLE_CONTROL:
            return same_view and step_le
        # Reference maskless replay: the cross-view partition excludes control queries.
        return k_clean_sensor and k["step"] < q["step"] and same_view
    if q_role == ROLE_TARGET_CONDITION:
        if k_role == ROLE_TARGET_CONDITION:
            return same_view and q["frame"] == k["frame"]
        if k_role == ROLE_CONTROL:
            return same_view and step_le
        if k_role == ROLE_CURRENT_TARGET:
            return k["step"] < q["step"] and in_scope
        return False
    assert q_target
    if k_role == ROLE_CURRENT_TARGET:
        return q_current and k["step"] == q["step"] and in_scope
    if k_role == ROLE_CLEAN_TARGET:
        causal = k["step"] < q["step"] if q_current else step_le
        return causal and in_scope
    if k_role == ROLE_CONTROL:
        return same_view and step_le
    if k_role == ROLE_TARGET_CONDITION:
        return in_scope and step_le
    return False


@pytest.mark.parametrize("pass_kind", ["noisy", "clean"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("lidar_condition", [False, True])
@pytest.mark.parametrize("chunk_index", [0, 1, 2, 3])
def test_plan_matches_token_level_oracle(chunk_index: int, pass_kind: str, joint: bool, lidar_condition: bool) -> None:
    if lidar_condition and not joint:
        pytest.skip("LiDAR conditioning needs a joint request")
    scn = scenario(chunk_index, pass_kind=pass_kind, joint=joint, lidar_condition=lidar_condition)
    planned = dense_token_mask(scn.plan)
    expected = _token_oracle(scn)
    assert planned.shape == expected.shape
    assert torch.equal(planned, expected), (planned ^ expected).nonzero()
    # The varlen calls the executor runs cover exactly the same pairs, once each.
    assert torch.equal(_token_pair_counts(scn.plan), expected.to(torch.int64))
    # Every GEN row is seeded by the first pass and attends at least itself; at most one merge follows.
    assert [attention_pass.name for attention_pass in scn.plan.passes] in (["same_view"], ["same_view", "cross_view"])
    assert int(scn.plan.passes[0].meta[META_Q_START]) == 0
    assert scn.plan.passes[0].q_index.numel() == scn.plan.num_gen_tokens
    assert bool(planned.diagonal()[: scn.plan.num_gen_tokens].all())


def _token_pair_counts(plan: CausalMasklessPlan) -> torch.Tensor:
    """Expand every varlen segment through the staging rows into ``[GEN, current + history + und]`` counts."""
    offsets = {
        STREAM_CURRENT: 0,
        STREAM_HISTORY: plan.num_gen_tokens,
        STREAM_UND: plan.num_gen_tokens + plan.num_history_tokens,
    }
    column = torch.cat(
        [torch.arange(start, start + length) + offsets[stream] for stream, start, length in plan.staging.tolist()]
    )
    counts = torch.zeros(plan.num_gen_tokens, offsets[STREAM_UND] + plan.num_und_tokens, dtype=torch.int64)
    for attention_pass in plan.passes:
        assert int(attention_pass.meta[META_KEY_STREAM]) == STREAM_STAGED
        for field, index in ((META_Q_START, attention_pass.q_index), (META_K_START, attention_pass.k_index)):
            first = int(index[0])
            contiguous = torch.equal(index, torch.arange(first, first + index.numel()))
            assert int(attention_pass.meta[field]) == (first if contiguous else -1)
        q_bounds, k_bounds = attention_pass.cu_seqlens_q.tolist(), attention_pass.cu_seqlens_k.tolist()
        for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
            assert qb > qa and kb > ka, "no empty varlen segment"
            counts[attention_pass.q_index[qa:qb, None], column[attention_pass.k_index[ka:kb]][None, :]] += 1
    return counts


def test_chunk0_condition_frames_read_only_their_own_view() -> None:
    scn = scenario(0)
    mask = dense_token_mask(scn.plan)
    cond0 = scn.q(view_id=0, role=ROLE_TARGET_CONDITION)
    row = mask[cond0]
    assert row[scn.k_current(view_id=0, role=ROLE_CONTROL)]
    assert row[cond0]
    assert not row[scn.k_current(view_id=1, role=ROLE_TARGET_CONDITION)]
    assert not row[scn.k_current(view_id=1, role=ROLE_CONTROL)]
    assert not row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CURRENT_TARGET)]
    assert not row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CONTROL)]
    # Own captions (sink + caption) only.
    und = row[scn.k_und(0) :]
    assert und.tolist() == [True] * 5 + [False] * 6
    # The noisy initial LiDAR sweep reads both cameras' condition frames and every caption.
    lidar = scn.q(view_id=LIDAR_VIEW, role=ROLE_CURRENT_TARGET)
    row = mask[lidar]
    assert row[scn.k_current(view_id=0, role=ROLE_TARGET_CONDITION)]
    assert row[scn.k_current(view_id=1, role=ROLE_TARGET_CONDITION)]
    assert row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CONTROL)]
    assert not row[scn.k_current(view_id=0, role=ROLE_CONTROL)]
    assert row[scn.k_und(0) :].all()


def test_controls_never_read_current_targets_and_read_only_own_view_clean_history() -> None:
    scn = scenario(2)
    mask = dense_token_mask(scn.plan)
    control = scn.q(view_id=0, role=ROLE_CONTROL, frame_id=3)  # t = 0.6, step 2
    row = mask[control]
    assert not row[scn.k_current(view_id=0, role=ROLE_CURRENT_TARGET, frame_id=3)]
    assert not row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CURRENT_TARGET)]
    assert not row[scn.k_current(view_id=1, role=ROLE_CONTROL, frame_id=3)]
    assert row[scn.k_current(view_id=0, role=ROLE_CONTROL, frame_id=4)]
    assert row[scn.k_history(0, view_id=0, role=ROLE_CONTROL)]
    assert row[scn.k_history(1, view_id=0, role=ROLE_CONTROL, frame_id=1)]
    # Clean history: own view unrestricted. The reference maskless cross-view partition
    # excludes control queries, so other views stay hidden even inside the 0.4 s window.
    assert row[scn.k_history(0, view_id=0, role=ROLE_TARGET_CONDITION)]
    assert row[scn.k_history(1, view_id=0, role=ROLE_CLEAN_TARGET, frame_id=1)]
    assert row[scn.k_history(1, view_id=0, role=ROLE_CLEAN_TARGET, frame_id=2)]
    assert not row[scn.k_history(0, view_id=1, role=ROLE_TARGET_CONDITION)]  # gap 0.6
    assert not row[scn.k_history(1, view_id=1, role=ROLE_CLEAN_TARGET, frame_id=1)]  # gap 0.4
    assert not row[scn.k_history(1, view_id=1, role=ROLE_CLEAN_TARGET, frame_id=2)]  # gap 0.2
    assert not row[scn.k_history(1, view_id=LIDAR_VIEW, role=ROLE_CLEAN_TARGET, frame_id=2)]  # gap 0.4
    assert not row[scn.k_history(1, view_id=LIDAR_VIEW, role=ROLE_CLEAN_TARGET, frame_id=4)]  # gap 0.2
    # Control rows never enter a cross-view pass.
    control_runs = scn.plan.query_runs.role == ROLE_CONTROL
    for attention_pass in scn.plan.passes:
        if attention_pass.name.startswith("cross_view"):
            assert not any(bool(control_runs[list(q_runs)].any()) for q_runs, _ in attention_pass.segments)


def test_cross_view_window_is_past_only_for_current_and_history() -> None:
    scn = scenario(1)
    mask = dense_token_mask(scn.plan)
    frame1 = scn.q(view_id=0, role=ROLE_CURRENT_TARGET, frame_id=1)  # t = 0.2
    row = mask[frame1]
    assert row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CURRENT_TARGET, frame_id=1)]  # t = 0.1
    assert row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CURRENT_TARGET, frame_id=2)]  # t = 0.2
    assert not row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CURRENT_TARGET, frame_id=3)]  # t = 0.3 (future)
    assert not row[scn.k_current(view_id=LIDAR_VIEW, role=ROLE_CURRENT_TARGET, frame_id=4)]
    assert row[scn.k_current(view_id=0, role=ROLE_CURRENT_TARGET, frame_id=2)]  # own view, any frame
    assert not row[scn.k_current(view_id=1, role=ROLE_CURRENT_TARGET, frame_id=2)]  # other view, future
    assert row[scn.k_history(0, view_id=1, role=ROLE_TARGET_CONDITION)]  # gap 0.2
    assert row[scn.k_history(0, view_id=0, role=ROLE_CONTROL)]
    assert not row[scn.k_history(0, view_id=1, role=ROLE_CONTROL)]
    frame2 = scn.q(view_id=0, role=ROLE_CURRENT_TARGET, frame_id=2)  # t = 0.4
    assert mask[frame2][scn.k_history(0, view_id=1, role=ROLE_TARGET_CONDITION)]  # gap exactly 0.4


def test_chunk_two_reads_sink_and_previous_chunk_with_window() -> None:
    scn = scenario(2)
    mask = dense_token_mask(scn.plan)
    frame4 = scn.q(view_id=0, role=ROLE_CURRENT_TARGET, frame_id=4)  # t = 0.8
    row = mask[frame4]
    assert row[scn.k_history(0, view_id=0, role=ROLE_TARGET_CONDITION)]
    assert not row[scn.k_history(0, view_id=1, role=ROLE_TARGET_CONDITION)]  # gap 0.8
    assert row[scn.k_history(1, view_id=0, role=ROLE_CLEAN_TARGET, frame_id=1)]
    assert not row[scn.k_history(1, view_id=1, role=ROLE_CLEAN_TARGET, frame_id=1)]  # gap 0.6
    assert row[scn.k_history(1, view_id=1, role=ROLE_CLEAN_TARGET, frame_id=2)]  # gap 0.4
    # Both ring slots are staged, in slot order, between the current chunk and the captions.
    gen, und = scn.plan.num_gen_tokens, scn.plan.num_und_tokens
    assert scn.plan.staging.tolist() == [
        [STREAM_CURRENT, 0, gen],
        [STREAM_HISTORY, 0, scn.slots[0].metadata.num_tokens],
        [STREAM_HISTORY, UNIT, scn.slots[1].metadata.num_tokens],
        [STREAM_UND, 0, und],
    ]


def test_clean_pass_is_chunk_causal() -> None:
    scn = scenario(2, pass_kind="clean")
    mask = dense_token_mask(scn.plan)
    frame3 = scn.q(view_id=0, role=ROLE_CLEAN_TARGET, frame_id=3)  # t = 0.6
    row = mask[frame3]
    assert row[scn.k_current(view_id=0, role=ROLE_CLEAN_TARGET, frame_id=4)]  # same chunk, own view
    assert row[scn.k_current(view_id=1, role=ROLE_CLEAN_TARGET, frame_id=3)]  # gap 0
    assert not row[scn.k_current(view_id=1, role=ROLE_CLEAN_TARGET, frame_id=4)]  # future, other view


def test_rgb_only_plan_has_no_lidar_and_matches_oracle() -> None:
    scn = scenario(2, joint=False)
    assert LIDAR_VIEW not in scn.current.view_id.tolist()
    assert torch.equal(dense_token_mask(scn.plan), _token_oracle(scn))


def _random_streams(scn: Scenario, *, heads: int = 2, kv_heads: int = 1, dim: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    gen, hist, und = scn.plan.num_gen_tokens, scn.plan.num_history_tokens, scn.plan.num_und_tokens
    q = torch.randn(1, gen, heads, dim, generator=generator)
    streams = []
    for length in (gen, max(hist, 1), und):
        streams.append(
            (
                torch.randn(length, kv_heads, dim, generator=generator),
                torch.randn(length, kv_heads, dim, generator=generator),
            )
        )
    return q, streams


def _dense_softmax(scn: Scenario, q: torch.Tensor, streams) -> torch.Tensor:
    mask = dense_token_mask(scn.plan)
    hist = scn.plan.num_history_tokens
    keys = torch.cat([streams[0][0], streams[1][0][:hist], streams[2][0]]).double()
    values = torch.cat([streams[0][1], streams[1][1][:hist], streams[2][1]]).double()
    ratio = q.shape[2] // keys.shape[1]
    scores = torch.einsum("qhd,khd->hqk", q[0].double(), keys.repeat_interleave(ratio, 1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), values.repeat_interleave(ratio, 1))


def _run(plan: CausalMasklessPlan, q: torch.Tensor, streams, kernel: str = "torch") -> torch.Tensor:
    keys = [k for k, _ in streams]
    values = [v for _, v in streams]
    return maskless_attention_streams(q, keys, values, plan.flatten(), len(plan.passes), kernel)


@pytest.mark.parametrize("chunk_size", [m.MERGE_CHUNK_SIZE, 3])
@pytest.mark.parametrize(("chunk_index", "joint"), [(0, True), (1, True), (3, True), (3, False)])
def test_streams_op_matches_masked_softmax(
    monkeypatch: pytest.MonkeyPatch, chunk_index: int, joint: bool, chunk_size: int
) -> None:
    # Small merge chunks split the sliced cross-view rows mid-segment.
    monkeypatch.setattr(m, "MERGE_CHUNK_SIZE", chunk_size)
    scn = scenario(chunk_index, joint=joint)
    q, streams = _random_streams(scn)
    out = _run(scn.plan, q, streams)
    torch.testing.assert_close(out[0].double(), _dense_softmax(scn, q, streams), rtol=1e-5, atol=1e-5)


def test_rgb_cross_view_pass_slices_its_query_rows() -> None:
    scn = scenario(3, joint=False)
    same_view, cross_view = scn.plan.passes
    # Packed [control | target]: the cross-view pass reads exactly the target rows.
    targets = int((scn.current.role != ROLE_CONTROL).sum())
    assert int(same_view.meta[META_Q_START]) == 0
    assert int(cross_view.meta[META_Q_START]) == scn.plan.num_gen_tokens - targets
    assert cross_view.q_index.numel() == targets


def test_history_remap_moves_only_staged_slots_to_their_pages() -> None:
    scn = scenario(2)
    bases = [100, 300]
    remapped = scn.plan.with_history_offsets(bases, UNIT)
    # Pass tensors address the staged stream and are shared untouched.
    assert all(a is b for a, b in zip(scn.plan.passes, remapped.passes, strict=True))
    expected = [
        [stream, bases[start // UNIT] + start % UNIT if stream == STREAM_HISTORY else start, length]
        for stream, start, length in scn.plan.staging.tolist()
    ]
    assert remapped.staging.tolist() == expected
    # A physical pool holding each slot at its page reproduces the virtual-ring result.
    q, streams = _random_streams(scn)
    pool_k = torch.randn(400 + UNIT, *streams[1][0].shape[1:])
    pool_v = torch.randn_like(pool_k)
    for slot, base in zip(scn.slots, bases, strict=True):
        length = slot.metadata.num_tokens
        pool_k[base : base + length] = streams[1][0][slot.base : slot.base + length]
        pool_v[base : base + length] = streams[1][1][slot.base : slot.base + length]
    physical = [streams[0], (pool_k, pool_v), streams[2]]
    torch.testing.assert_close(_run(remapped, q, physical), _run(scn.plan, q, streams), rtol=0, atol=0)
    with pytest.raises(ValueError, match="without a physical page"):
        scn.plan.with_history_offsets([100], UNIT)


def test_streams_op_rejects_staging_outside_its_stream() -> None:
    scn = scenario(2)
    q, streams = _random_streams(scn)
    short_history = (streams[1][0][:UNIT], streams[1][1][:UNIT])
    with pytest.raises(ValueError, match="exceeds key stream 1"):
        _run(scn.plan, q, [streams[0], short_history, streams[2]])


def test_chunk_one_stages_one_ring_slot_and_reads_it_in_both_passes() -> None:
    scn = scenario(1)
    gen, und = scn.plan.num_gen_tokens, scn.plan.num_und_tokens
    assert [attention_pass.name for attention_pass in scn.plan.passes] == ["same_view", "cross_view"]
    assert scn.plan.staging.tolist() == [
        [STREAM_CURRENT, 0, gen],
        [STREAM_HISTORY, 0, scn.slots[0].metadata.num_tokens],
        [STREAM_UND, 0, und],
    ]
    history_columns = range(scn.plan.query_runs.num_runs, scn.plan.query_runs.num_runs + scn.plan.key_runs[1].num_runs)
    und_columns = range(scn.plan.allowed.shape[1] - scn.plan.key_runs[2].num_runs, scn.plan.allowed.shape[1])
    for attention_pass in scn.plan.passes:
        k_runs = {run for _, runs in attention_pass.segments for run in runs}
        assert k_runs & set(history_columns)
        # Captions are same-view keys: the cross-view pass never reads them.
        assert bool(k_runs & set(und_columns)) == (attention_pass.name == "same_view")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and vLLM's bundled FlashAttention")
@pytest.mark.parametrize("fa_version", [2, 3, 4])
@pytest.mark.parametrize("chunk_index", [0, 3])
def test_cuda_streams_op_matches_masked_softmax(fa_version: int, chunk_index: int) -> None:
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    if not is_fa_version_supported(fa_version):
        pytest.skip(f"FlashAttention {fa_version} is unavailable on this device")
    # Staging, the sliced cross-view rows and the compiled single merge on real varlen kernels.
    scn = scenario(chunk_index, device="cuda")
    q, streams = _random_streams(scn, heads=4, kv_heads=2, dim=128)
    q, streams = q.bfloat16(), [(k.bfloat16(), v.bfloat16()) for k, v in streams]
    with torch.inference_mode():
        out = _run(scn.plan, q.cuda(), [(k.cuda(), v.cuda()) for k, v in streams], kernel=f"fa{fa_version}")
    torch.testing.assert_close(out[0].double().cpu(), _dense_softmax(scn, q, streams), atol=2e-2, rtol=2e-2)
