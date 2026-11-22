# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal maskless planner: token-level oracle, hand-derived reference rules, numerics and the ring remap.

The oracle below re-derives the reference replay rules per token in plain
Python (independently of the vectorised run-level predicate), and the
hand-written expectations encode specific consequences of those rules read
from the imaginaire4 predicate: condition frames read only themselves, their
own controls and captions; controls never read current targets; cross-view
reads are past-only within 0.4 s; chunk 1 sees only chunk 0 and chunk k >= 2
sees chunks 0 and k-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.multiview_maskless_attention import maskless_attention_streams
from vllm_omni.diffusion.models.cosmos3.multiview_maskless_plan import META_IDENTITY_K, META_KEY_STREAM
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.causal_plan import (
    STREAM_HISTORY,
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
        current=current, history=slots, captions=caption_metadata(docs), window_seconds=WINDOW, device="cpu"
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
        return k_clean_sensor and k["step"] < q["step"] and in_scope
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
    # Every GEN row is seeded by the first pass and attends at least itself.
    assert scn.plan.passes[0].name == "same_view_current"
    assert scn.plan.passes[0].q_index.numel() == scn.plan.num_gen_tokens
    assert bool(planned.diagonal()[: scn.plan.num_gen_tokens].all())


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


def test_controls_never_read_current_targets_but_read_windowed_clean_history() -> None:
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
    # Clean history: own view unrestricted, other views only inside the 0.4 s window.
    assert row[scn.k_history(0, view_id=0, role=ROLE_TARGET_CONDITION)]
    assert not row[scn.k_history(0, view_id=1, role=ROLE_TARGET_CONDITION)]  # gap 0.6
    assert row[scn.k_history(1, view_id=1, role=ROLE_CLEAN_TARGET, frame_id=1)]  # gap 0.4
    assert row[scn.k_history(1, view_id=1, role=ROLE_CLEAN_TARGET, frame_id=2)]  # gap 0.2
    assert not row[scn.k_history(1, view_id=LIDAR_VIEW, role=ROLE_CLEAN_TARGET, frame_id=1)]  # gap 0.5
    assert row[scn.k_history(1, view_id=LIDAR_VIEW, role=ROLE_CLEAN_TARGET, frame_id=2)]  # gap 0.4
    assert row[scn.k_history(1, view_id=LIDAR_VIEW, role=ROLE_CLEAN_TARGET, frame_id=4)]  # gap 0.2


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
    assert any(name.endswith("history") for name in (attention_pass.name for attention_pass in scn.plan.passes))


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


@pytest.mark.parametrize("chunk_index", [0, 1, 3])
def test_streams_op_matches_masked_softmax(chunk_index: int) -> None:
    scn = scenario(chunk_index)
    q, streams = _random_streams(scn)
    keys = [k for k, _ in streams]
    values = [v for _, v in streams]
    out = maskless_attention_streams(q, keys, values, scn.plan.flatten(), len(scn.plan.passes), "torch")
    mask = dense_token_mask(scn.plan)
    hist = scn.plan.num_history_tokens
    all_k = torch.cat([keys[0], keys[1][:hist], keys[2]]).double()
    all_v = torch.cat([values[0], values[1][:hist], values[2]]).double()
    ratio = q.shape[2] // all_k.shape[1]
    scores = torch.einsum("qhd,khd->hqk", q[0].double(), all_k.repeat_interleave(ratio, 1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), all_v.repeat_interleave(ratio, 1))
    torch.testing.assert_close(out[0].double(), expected, rtol=1e-5, atol=1e-5)


def test_history_remap_shifts_each_slot_to_its_page() -> None:
    scn = scenario(2)
    bases = [100, 300]
    remapped = scn.plan.with_history_offsets(bases, UNIT)
    for before, after in zip(scn.plan.passes, remapped.passes, strict=True):
        if int(before.meta[META_KEY_STREAM]) != STREAM_HISTORY:
            assert torch.equal(before.k_index, after.k_index)
            continue
        slots = torch.div(before.k_index, UNIT, rounding_mode="floor")
        expected = before.k_index - slots * UNIT + torch.tensor(bases)[slots]
        assert torch.equal(after.k_index, expected)
        assert int(after.meta[META_IDENTITY_K]) == 0
    with pytest.raises(ValueError, match="without a physical page"):
        scn.plan.with_history_offsets([100], UNIT)


def test_caption_pass_marks_und_stream() -> None:
    scn = scenario(1)
    names = [attention_pass.name for attention_pass in scn.plan.passes]
    assert names[0] == "same_view_current" and names[-1] == "caption"
    assert int(scn.plan.passes[-1].meta[META_KEY_STREAM]) == STREAM_UND
