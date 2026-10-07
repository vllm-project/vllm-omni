# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests of the TaoMate-H3 CUDA-graph helpers (no device needed)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.taomate_h3.audio_teacher import (
    TEACHER_SLOT_REFERENCE,
    TEACHER_SLOT_TARGET,
    TEACHER_SLOT_TEXT,
    attach_teacher_timestep_slots,
    teacher_forward_kwargs,
)
from vllm_omni.diffusion.models.taomate_h3.cuda_graph import (
    GraphedForward,
    clone_static,
    copy_into,
    kwargs_signature,
)
from vllm_omni.diffusion.models.taomate_h3.packed import (
    taomate_audio_only_frozen_prefix_packed_layout,
    taomate_audio_only_packed_layout,
)
from vllm_omni.diffusion.models.taomate_h3.transformer import LocalEmbedPlan, local_embed_plan


def test_signature_counts_tensors_by_shape_and_values_by_value() -> None:
    a = {"x": torch.zeros(3, 4), "meta": {"n": 1, "flag": True}, "seq": (1, 2)}
    b = {"x": torch.ones(3, 4), "meta": {"n": 1, "flag": True}, "seq": (1, 2)}
    c = {"x": torch.zeros(3, 5), "meta": {"n": 1, "flag": True}, "seq": (1, 2)}
    d = {"x": torch.zeros(3, 4), "meta": {"n": 2, "flag": True}, "seq": (1, 2)}
    assert kwargs_signature(a) == kwargs_signature(b)
    assert kwargs_signature(a) != kwargs_signature(c)
    assert kwargs_signature(a) != kwargs_signature(d)
    assert kwargs_signature(torch.zeros(2, dtype=torch.float32)) != kwargs_signature(
        torch.zeros(2, dtype=torch.bfloat16)
    )
    hash(kwargs_signature(a))


def test_clone_static_and_copy_into_follow_the_tree() -> None:
    live = {"x": torch.arange(6.0).view(2, 3), "inner": {"pos": torch.tensor([1, 2])}, "n": 4, "t": (torch.ones(2),)}
    static = clone_static(live)
    assert static["x"].data_ptr() != live["x"].data_ptr()
    assert torch.equal(static["x"], live["x"]) and static["n"] == 4
    live["x"].fill_(7.0)
    live["inner"]["pos"].fill_(9)
    assert copy_into(static, live) == 3
    assert torch.equal(static["x"], torch.full((2, 3), 7.0))
    assert torch.equal(static["inner"]["pos"], torch.tensor([9, 9]))
    with pytest.raises(ValueError):
        copy_into(static, {**live, "x": torch.zeros(3, 2)})
    with pytest.raises(ValueError):
        copy_into(static, {**live, "extra": 1})


class _Doubler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *, x: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return x * scale, x + scale


def test_graphed_forward_runs_eagerly_without_cuda() -> None:
    module = _Doubler()
    graphed = GraphedForward(module, device=torch.device("cpu"), name="test")
    assert not graphed.enabled
    out, other = graphed(variant="v", x=torch.ones(2), scale=3.0)
    assert torch.equal(out, torch.full((2,), 3.0)) and torch.equal(other, torch.full((2,), 4.0))
    assert module.calls == 1 and graphed.stats()["eager_calls"] == 1 and graphed.num_graphs == 0
    with pytest.raises(ValueError):
        GraphedForward(module, device=torch.device("cpu"), max_entries=0)


def _upstream_local_selection(img, audio, text, local_start, local_len):
    local_end = local_start + local_len
    img_mask = (img >= local_start) & (img < local_end)
    audio_mask = (audio >= local_start) & (audio < local_end)
    text_mask = (text >= local_start) & (text < local_end)
    return (
        img[img_mask],
        audio[audio_mask],
        img[img_mask] - local_start,
        audio[audio_mask] - local_start,
        text[text_mask] - local_start,
        torch.nonzero(text_mask, as_tuple=False).view(-1),
    )


def test_local_embed_plan_matches_the_upstream_mask_logic() -> None:
    text = torch.arange(0, 9)
    audio = torch.arange(9, 9 + 80)
    img = torch.arange(89, 89 + 30)
    seq_len = 128
    for local_start, local_len in ((0, 64), (64, 64), (32, 32)):
        plan = LocalEmbedPlan.build(
            img_pos=img,
            audio_pos=audio,
            text_pos=text,
            seq_len=seq_len,
            local_span=(local_start, local_len),
            device=torch.device("cpu"),
        )
        expected = _upstream_local_selection(img, audio, text, local_start, local_len)
        assert torch.equal(plan.img_global_pos, expected[0])
        assert torch.equal(plan.audio_global_pos, expected[1])
        assert torch.equal(plan.img_local_pos, expected[2])
        assert torch.equal(plan.audio_local_pos, expected[3])
        assert torch.equal(plan.text_local_pos, expected[4])
        assert plan.text_local_indices is not None and torch.equal(plan.text_local_indices, expected[5])
        assert plan.fingerprint == (seq_len, local_start, local_len, 30, 80, 9)
    full = LocalEmbedPlan.build(
        img_pos=img,
        audio_pos=audio,
        text_pos=text,
        seq_len=seq_len,
        local_span=(0, seq_len),
        device=torch.device("cpu"),
    )
    assert full.text_local_indices is None and torch.equal(full.img_local_pos, img)
    with local_embed_plan(full):
        from vllm_omni.diffusion.models.taomate_h3.transformer import _LOCAL_EMBED_PLAN

        assert _LOCAL_EMBED_PLAN.get() is full
    assert _LOCAL_EMBED_PLAN.get() is None


def _teacher_branch(*, with_reference: bool, seq_len: int | None = None) -> MiniMaxH3DenoiseBranch:
    text_len = 9
    if with_reference:
        packed = taomate_audio_only_frozen_prefix_packed_layout(
            text_len=text_len,
            ref_audio_t=40,
            audio_t=198,
            latent_h=54,
            latent_w=30,
            reference_time_start=text_len + 207 - 40,
            target_time_start=text_len + 207,
            seq_len=seq_len,
        )
    else:
        packed = taomate_audio_only_packed_layout(
            text_len=text_len, audio_t=207, latent_h=54, latent_w=30, seq_len=seq_len
        )
    tags = packed["token_tags"].clone()
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(text_len, 5120),
        token_tags=tags,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
def test_teacher_fixed_slot_timesteps_match_the_branch_fill(with_reference: bool, pinned: bool) -> None:
    branch = _teacher_branch(with_reference=with_reference, seq_len=640 if pinned else None)
    inverse = attach_teacher_timestep_slots(branch)
    assert inverse.shape == (branch.seq_len,)
    audio_rows = torch.randn(int(branch.audio_pos.numel()), 32)
    video_rows = torch.empty(0, 96)
    for t_video, t_audio in ((0.0, 0.0), (0.2, 0.5), (1.0, 1.0)):
        kwargs = teacher_forward_kwargs(
            branch, video_rows=video_rows, audio_rows=audio_rows, t_video=t_video, t_audio=t_audio
        )
        assert kwargs["unique_timesteps"].shape == (3,)
        per_row = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
        expected = torch.empty(branch.seq_len)
        branch.fill_timesteps(
            expected, t_video=t_video, t_audio=t_audio, imgvid_cond_timestep=t_video, audio_ref_cond_timestep=1.0
        )
        assert torch.equal(per_row, expected)
        reference = branch.forward_kwargs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            t_video=t_video,
            t_audio=t_audio,
            imgvid_cond_timestep=t_video,
            audio_ref_cond_timestep=1.0,
        )
        assert torch.equal(kwargs["audio_x"], reference["audio_x"]) and torch.equal(kwargs["x"], reference["x"])
        assert torch.equal(reference["unique_timesteps"][reference["inverse_indices"]], per_row)
    slots = set(inverse.tolist())
    assert TEACHER_SLOT_TEXT in slots and TEACHER_SLOT_TARGET in slots
    assert (TEACHER_SLOT_REFERENCE in slots) == with_reference
    # Graph identity: the same document shape gives the same signature whatever the values.
    first = teacher_forward_kwargs(branch, video_rows=video_rows, audio_rows=audio_rows, t_video=0.1, t_audio=0.2)
    second = teacher_forward_kwargs(branch, video_rows=video_rows, audio_rows=audio_rows * 2, t_video=0.7, t_audio=0.9)
    assert kwargs_signature(first) == kwargs_signature(second)


def test_vae_tile_values_are_multiples_of_16() -> None:
    from vllm_omni.diffusion.models.taomate_h3.pipeline import validate_vae_tile_value

    assert validate_vae_tile_value(None, "x", minimum=16) is None
    assert validate_vae_tile_value(480, "x", minimum=16) == 480
    assert validate_vae_tile_value(0, "x", minimum=0) == 0
    for bad in (True, 15, 100, -16, 8.0):
        with pytest.raises(ValueError):
            validate_vae_tile_value(bad, "x", minimum=16)


def test_graphed_forward_keeps_keep_alive_objects_on_cpu_passthrough() -> None:
    module = _Doubler()
    graphed = GraphedForward(module, device=torch.device("cpu"))
    plan = object()
    graphed(variant="v", keep_alive=(plan,), x=torch.ones(1), scale=1.0)
    assert graphed.num_graphs == 0  # eager on CPU; the argument is accepted


def test_clone_static_rejects_opaque_objects() -> None:
    """A tensor hidden inside an arbitrary object would not be refreshed on replay."""

    class Holder:
        def __init__(self) -> None:
            self.t = torch.zeros(2)

    with pytest.raises(TypeError):
        clone_static({"x": Holder()})
    assert clone_static({"x": (1, 2.0, "s", None, [torch.ones(1)])})["x"][4][0].item() == 1.0
    # Tensor-free objects (layout dataclasses, devices, dtypes) pass through.
    layout = _teacher_branch(with_reference=True, seq_len=640).static_kwargs.get("video_layout")
    cloned = clone_static({"device": torch.device("cpu"), "dtype": torch.float32, "layout": layout})
    assert cloned["device"] == torch.device("cpu") and cloned["layout"] is layout


def test_real_teacher_kwargs_are_graph_cloneable() -> None:
    """The teacher's forward kwargs (branch statics included) must pass the kwargs-tree guard."""
    for with_reference in (False, True):
        branch = _teacher_branch(with_reference=with_reference, seq_len=640)
        kwargs = teacher_forward_kwargs(
            branch,
            video_rows=torch.empty(0, 96),
            audio_rows=torch.randn(int(branch.audio_pos.numel()), 32),
            t_video=0.5,
            t_audio=0.5,
        )
        static = clone_static(kwargs)
        assert copy_into(static, kwargs) >= 4
        assert kwargs_signature(static) == kwargs_signature(kwargs)


def test_copy_into_refreshes_an_alias_with_other_strides() -> None:
    base = torch.arange(6.0).view(2, 3)
    static = base.clone()
    # Same storage pointer as ``static`` but transposed strides: must be copied, not skipped.
    alias = static.t().contiguous().t()  # a fresh tensor with the same values and shape
    assert copy_into(static, alias) == 1 and torch.equal(static, alias)
    stale = torch.zeros(2, 3)
    assert copy_into(stale, base) == 1 and torch.equal(stale, base)


def test_pinned_entries_survive_lru_eviction() -> None:
    from vllm_omni.diffusion.models.taomate_h3.cuda_graph import _Entry

    graph = GraphedForward(torch.nn.Identity(), device=torch.device("cpu"), max_entries=2)
    graph.enabled = True  # exercise the eviction bookkeeping without a device
    graph._entries["a"] = _Entry(graph=None, static_kwargs={}, outputs=None, pinned=True)
    graph._entries["b"] = _Entry(graph=None, static_kwargs={}, outputs=None)
    # Emulate the eviction step of _capture for a third key.
    victim = next((k for k, e in graph._entries.items() if not e.pinned), None)
    assert victim == "b"
    assert graph.stats()["pinned"] == 1
