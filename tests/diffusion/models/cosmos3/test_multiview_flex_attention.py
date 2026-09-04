# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# UND is padded to a fixed capacity, not to the nearest block above each
# prompt's length, so the compiled kernel sees one shape per layout.
#
# Production uses DEFAULT_MAX_UND_TOKENS (4098). These are fixture sizes: the
# smallest capacity that holds the largest real UND length each group of tests
# exercises, chosen so round_up(capacity, kv_block) equals the padded length the
# old prompt-dependent rule produced for those same lengths. The geometry
# assertions below therefore still assert what they asserted before capacity
# padding, and the CPU fixtures stay small.
#
#   Triton (kv_block 64):  real UND 3 and 7  -> round_up(64, 64)   == 64
#   FA4    (kv_block 128): real UND 96 and 128 -> round_up(128, 128) == 128
_TRITON_MAX_UND = 64
_FA4_MAX_UND = 128


def _tiny_metadata(attention_scope: str = "same_view_or_frame"):
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MaskItem,
        build_multiview_flex_metadata,
    )

    control = MaskItem(
        token_shape=(4, 1, 1),
        condition_mask=torch.ones(4, dtype=torch.bool),
        num_views=2,
        is_control=True,
    )
    target = MaskItem(
        token_shape=(4, 1, 1),
        condition_mask=torch.tensor([True, False, True, False]),
        num_views=2,
    )
    return build_multiview_flex_metadata(
        seq_len=10,
        full_q_offsets=(2, 6, 10),
        items_per_sample=(control, target),
        device="cpu",
        num_und=2,
        attention_scope=attention_scope,
    )


def test_expand_multiview_condition_indexes_camera_major() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        expand_multiview_condition_frame_indexes,
        resolve_item_condition_frames,
    )

    assert expand_multiview_condition_frame_indexes([1, 0, 1, -1, 99], 3, 12) == [0, 1, 4, 5, 8, 9]
    assert resolve_item_condition_frames(0, 2, [0, 4], 8) == list(range(8))
    assert resolve_item_condition_frames(1, 2, [4, 0, 4, 99], 8) == [0, 4]


def test_metadata_is_camera_major_and_marks_padding() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MaskItem,
        build_multiview_flex_metadata,
    )

    item = MaskItem(
        token_shape=(6, 1, 2),
        condition_mask=torch.tensor([True, False, False, True, False, False]),
        num_views=2,
    )
    metadata = build_multiview_flex_metadata(
        seq_len=18,
        full_q_offsets=(4, 16),
        items_per_sample=(item,),
        device="cpu",
        num_und=3,
        attention_scope="same_view",
    )

    assert metadata.sample_id.tolist() == [0, 0, 0, -1] + [0] * 12 + [-1, -1]
    assert metadata.is_und.tolist() == [True, True, True] + [False] * 15
    assert metadata.frame_id[4:16].tolist() == [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
    assert metadata.view_id[4:16].tolist() == [0] * 6 + [1] * 6
    assert metadata.is_noisy[4:16].tolist() == [False, False, True, True, True, True] * 2


# Truth table transcribed from the Multiview-AV visibility spec, one row per
# (query role, key role).  Deliberately NOT the boolean expression the
# implementation evaluates: the spec states the rules as a case analysis over
# token roles, so the oracle is a literal table plus the scope definition, and
# a sign error in the implementation's masked-OR form cannot cancel out here.
#
# Values name the condition under which the pair is visible:
#   "never"     - never visible
#   "always"    - visible whenever both tokens belong to the same sample
#   "same_view" - visible only when the two tokens share a view
#   "in_scope"  - visible when the pair falls inside the attention scope
_SPEC_VISIBILITY = {
    # Control tokens are own-view-only and never read RGB.
    ("control", "und"): "always",
    ("control", "control"): "same_view",
    ("control", "rgb_clean"): "never",
    ("control", "rgb_noisy"): "never",
    # A clean (conditioned) RGB token must never see a noisy RGB token.
    ("rgb_clean", "und"): "always",
    ("rgb_clean", "control"): "same_view",
    ("rgb_clean", "rgb_clean"): "in_scope",
    ("rgb_clean", "rgb_noisy"): "never",
    # A noisy RGB token sees every in-scope RGB token, clean or noisy.
    ("rgb_noisy", "und"): "always",
    ("rgb_noisy", "control"): "same_view",
    ("rgb_noisy", "rgb_clean"): "in_scope",
    ("rgb_noisy", "rgb_noisy"): "in_scope",
}


def _token_role(vectors: tuple[torch.Tensor, ...], index: int) -> str:
    """Classify one token into the role vocabulary the spec table is keyed on."""
    _sample, _frame, _view, is_noisy, is_control, is_und = (bool(vector[index]) for vector in vectors)
    if is_und:
        return "und"
    if is_control:
        return "control"
    return "rgb_noisy" if is_noisy else "rgb_clean"


def _spec_visible(
    q_role: str,
    k_role: str,
    *,
    same_sample: bool,
    same_view: bool,
    same_frame: bool,
    attention_scope: str,
) -> bool:
    """Resolve one table cell; the scope wording is also taken from the spec."""
    if not same_sample:
        return False
    rule = _SPEC_VISIBILITY[(q_role, k_role)]
    if rule == "never":
        return False
    if rule == "always":
        return True
    if rule == "same_view":
        return same_view
    if attention_scope == "all_views":
        return True
    if attention_scope == "same_view":
        return same_view
    return same_view or same_frame


@pytest.mark.parametrize("attention_scope", ["all_views", "same_view", "same_view_or_frame"])
def test_visibility_predicate_matches_spec_truth_table(attention_scope: str) -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import multiview_pair_predicate

    metadata = _tiny_metadata(attention_scope)
    q_index = torch.arange(metadata.q_len)[:, None]
    kv_index = torch.arange(metadata.kv_len)[None, :]
    actual = multiview_pair_predicate(metadata, q_index, kv_index)

    q_vectors = metadata.query_vectors()
    k_vectors = metadata.key_vectors()
    expected = torch.zeros_like(actual)
    covered = set()
    for q in range(metadata.q_len):
        q_sample, q_frame, q_view = (int(vector[q]) for vector in q_vectors[:3])
        q_role = _token_role(q_vectors, q)
        for k in range(metadata.kv_len):
            k_sample, k_frame, k_view = (int(vector[k]) for vector in k_vectors[:3])
            k_role = _token_role(k_vectors, k)
            covered.add((q_role, k_role))
            expected[q, k] = _spec_visible(
                q_role,
                k_role,
                same_sample=q_sample == k_sample,
                same_view=q_view == k_view,
                same_frame=q_frame == k_frame,
                attention_scope=attention_scope,
            )

    # The fixture must exercise every row of the table, or the table is not
    # actually being checked.
    assert covered == set(_SPEC_VISIBILITY)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("attention_scope", ["all_views", "same_view", "same_view_or_frame"])
def test_padding_queries_attend_only_padding(attention_scope: str) -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MaskItem,
        build_multiview_flex_metadata,
        multiview_pair_predicate,
    )

    item = MaskItem(
        token_shape=(2, 1, 2),
        condition_mask=torch.tensor([True, False]),
        num_views=1,
    )
    metadata = build_multiview_flex_metadata(
        seq_len=9,
        full_q_offsets=(3, 7),
        items_per_sample=(item,),
        device="cpu",
        num_und=2,
        attention_scope=attention_scope,
    )
    allowed = multiview_pair_predicate(
        metadata,
        torch.arange(metadata.q_len)[:, None],
        torch.arange(metadata.kv_len)[None, :],
    )
    q_padding = metadata.query_vectors()[0] == -1
    kv_padding = metadata.sample_id == -1

    assert allowed[q_padding][:, kv_padding].all()
    assert not allowed[q_padding][:, ~kv_padding].any()
    assert not allowed[~q_padding][:, kv_padding].any()
    assert allowed[q_padding].any(dim=-1).all()


def test_visibility_rule_examples() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import multiview_pair_predicate

    metadata = _tiny_metadata()

    def visible(q: int, keys: list[int]) -> list[bool]:
        return multiview_pair_predicate(metadata, torch.tensor(q), torch.tensor(keys)).tolist()

    # Control view zero: UND + own-view control only.
    assert visible(0, [0, 1, 2, 3, 4, 6]) == [True, True, True, True, False, False]
    # Clean RGB view zero/frame zero cannot see noisy RGB, but can see the
    # clean same-frame token in view one and its own-view control.
    assert visible(4, [2, 3, 6, 7, 8, 9]) == [True, True, True, False, True, False]
    # Noisy RGB view zero/frame one sees every RGB token in its view plus the
    # same-frame token in view one, but not the other-view/different-frame token.
    assert visible(5, [2, 3, 6, 7, 8, 9]) == [True, True, True, True, False, True]


def test_compressed_block_mask_and_request_cache() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        get_multiview_block_mask,
    )

    layout = MultiviewLayout(
        num_views=2,
        latent_frames=4,
        patch_height=1,
        patch_width=1,
        condition_frame_indexes=(0, 2),
        max_und_tokens=_TRITON_MAX_UND,
    )
    cache = {}
    context = MultiviewAttentionContext(layout, cache)
    first, geometry = get_multiview_block_mask(context, real_und_len=3, real_q_len=8, device=torch.device("cpu"))
    second, second_geometry = get_multiview_block_mask(
        context, real_und_len=3, real_q_len=8, device=torch.device("cpu")
    )
    different_branch, _ = get_multiview_block_mask(context, real_und_len=7, real_q_len=8, device=torch.device("cpu"))

    assert first is second
    assert first is not different_branch
    assert len(cache) == 2
    assert geometry == second_geometry
    assert geometry.padded_q_len == 64
    assert geometry.padded_und_len == 64
    assert first.BLOCK_SIZE == (64, 64)
    assert first.shape == (1, 1, 64, 128)
    assert int(first.kv_num_blocks.sum() + first.full_kv_num_blocks.sum()) > 0
    # The two CFG branches get different masks but must keep one shape, so they
    # share a compiled kernel rather than forcing a recompile.
    assert different_branch.shape == first.shape


def test_und_padding_is_independent_of_prompt_length() -> None:
    """UND pads to the layout capacity, never to the individual prompt length.

    A prompt-dependent pad resizes the packed key tensor, and the flex kernel is
    compiled with ``dynamic=False``, so each distinct prompt-length bucket costs
    a recompile; past Dynamo's default limit of eight the frame drops to eager
    FlexAttention, which materializes a multi-terabyte score matrix at the
    released geometry.  Everything the compiled call is specialized on must
    therefore be a function of the layout alone.
    """
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        get_multiview_block_mask,
    )

    layout = MultiviewLayout(2, 4, 1, 1, condition_frame_indexes=(0, 2), max_und_tokens=200)
    context = MultiviewAttentionContext(layout, {})

    shapes = set()
    for real_und_len in (1, 63, 64, 65, 199, 200):
        block_mask, geometry = get_multiview_block_mask(
            context,
            real_und_len=real_und_len,
            real_q_len=layout.gen_tokens,
            device=torch.device("cpu"),
        )
        assert geometry.padded_und_len == 256  # round_up(200, 64), for every prompt
        # block_mask.shape is (batch, heads, q_len, kv_len); kv_len - q_len is
        # the UND offset the mask_mod's captured metadata is sliced at, and the
        # captured tensors are all kv_len long, so this pins every shape the
        # compiled flex call is specialized on.
        assert block_mask.shape[3] - block_mask.shape[2] == geometry.padded_und_len
        shapes.add(
            (
                tuple(block_mask.shape),
                tuple(block_mask.kv_indices.shape),
                tuple(block_mask.full_kv_indices.shape),
            )
        )
    assert len(shapes) == 1, f"prompt length changed a compile-relevant shape: {shapes}"

    with pytest.raises(ValueError, match="exceeds the layout capacity"):
        get_multiview_block_mask(
            context,
            real_und_len=201,
            real_q_len=layout.gen_tokens,
            device=torch.device("cpu"),
        )


def test_und_capacity_padding_does_not_change_attention_output() -> None:
    """Padding the UND stream to a capacity must be numerically free.

    Pad keys carry ``sample_id == -1`` and are already excluded from every real
    query by the predicate, so growing the capacity cannot move the output.  The
    tolerance is tight rather than exact only because a larger pad shifts the
    absolute position of the real keys and hence the float reduction order; a
    pad key actually leaking in would contribute a whole softmax term, orders of
    magnitude above this.
    """
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        padded_multiview_flex_attention,
    )

    torch.manual_seed(5)
    q = torch.randn(1, 8, 4, 8)
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)
    k_und = torch.randn(1, 5, 2, 8)
    v_und = torch.randn(1, 5, 2, 8)

    outputs = []
    for max_und_tokens in (5, 64, 200):
        layout = MultiviewLayout(2, 4, 1, 1, condition_frame_indexes=(0, 2), max_und_tokens=max_und_tokens)
        outputs.append(
            padded_multiview_flex_attention(q, k, v, k_und, v_und, MultiviewAttentionContext(layout, {}, {}))
        )

    for other in outputs[1:]:
        torch.testing.assert_close(outputs[0], other, atol=1e-6, rtol=1e-6)


def test_block_mask_uses_canonical_full_width_contiguous_layout() -> None:
    """The Triton flex template is only exercised upstream with the layout
    ``create_block_mask`` emits: index tensors spanning every KV block and
    contiguous memory. Trimmed column slices are non-contiguous and violate
    that contract (see pytorch/pytorch#153344)."""
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        get_multiview_block_mask,
    )

    layout = MultiviewLayout(2, 4, 8, 8, condition_frame_indexes=(0, 2), max_und_tokens=_TRITON_MAX_UND)
    block_mask, geometry = get_multiview_block_mask(
        MultiviewAttentionContext(layout, {}),
        real_und_len=3,
        real_q_len=layout.gen_tokens,
        device=torch.device("cpu"),
    )
    num_kv_blocks = (geometry.padded_und_len + geometry.padded_q_len) // 64
    for counts, indices in (
        (block_mask.kv_num_blocks, block_mask.kv_indices),
        (block_mask.full_kv_num_blocks, block_mask.full_kv_indices),
    ):
        assert indices.shape[-1] == num_kv_blocks
        assert indices.is_contiguous()
        assert counts.is_contiguous()
        assert counts.dtype == torch.int32
        assert indices.dtype == torch.int32
        assert int(indices.min()) >= 0
        assert int(indices.max()) < num_kv_blocks
        assert int(counts.max()) <= num_kv_blocks


def test_compressed_block_mask_matches_dense_token_projection() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        build_multiview_flex_metadata,
        get_multiview_block_mask,
        multiview_pair_predicate,
    )

    q_block_size = 64
    kv_block_size = 64
    layout = MultiviewLayout(2, 4, 8, 8, condition_frame_indexes=(0, 2), max_und_tokens=_TRITON_MAX_UND)
    block_mask, geometry = get_multiview_block_mask(
        MultiviewAttentionContext(layout, {}),
        real_und_len=3,
        real_q_len=layout.gen_tokens,
        device=torch.device("cpu"),
    )
    metadata = build_multiview_flex_metadata(
        seq_len=geometry.padded_und_len + geometry.padded_q_len,
        full_q_offsets=(
            geometry.padded_und_len,
            geometry.padded_und_len + layout.item_tokens,
            geometry.padded_und_len + 2 * layout.item_tokens,
        ),
        items_per_sample=layout.mask_items(torch.device("cpu")),
        device="cpu",
        num_und=3,
        attention_scope=layout.attention_scope,
    )
    dense = multiview_pair_predicate(
        metadata,
        torch.arange(metadata.q_len)[:, None],
        torch.arange(metadata.kv_len)[None, :],
    )
    q_blocks = metadata.q_len // q_block_size
    kv_blocks = metadata.kv_len // kv_block_size
    tiles = dense.view(q_blocks, q_block_size, kv_blocks, kv_block_size).permute(0, 2, 1, 3)
    expected_visible = tiles.any(dim=(-2, -1))
    expected_full = tiles.all(dim=(-2, -1))
    expected_partial = expected_visible & ~expected_full

    def unpack(counts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        result = torch.zeros((q_blocks, kv_blocks), dtype=torch.bool)
        for row in range(q_blocks):
            count = int(counts[0, 0, row])
            result[row, indices[0, 0, row, :count].to(torch.int64)] = True
        return result

    actual_partial = unpack(block_mask.kv_num_blocks, block_mask.kv_indices)
    actual_full = unpack(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
    assert expected_partial.any()
    assert expected_full.any()
    torch.testing.assert_close(actual_partial, expected_partial)
    torch.testing.assert_close(actual_full, expected_full)


def test_flex_attention_matches_dense_masked_gqa_oracle() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        build_multiview_flex_metadata,
        multiview_pair_predicate,
        padded_multiview_flex_attention,
    )

    torch.manual_seed(4)
    layout = MultiviewLayout(2, 4, 1, 1, condition_frame_indexes=(0, 2), max_und_tokens=_TRITON_MAX_UND)
    context = MultiviewAttentionContext(layout, {})
    q = torch.randn(1, 8, 4, 8)
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)
    k_und = torch.randn(1, 3, 2, 8)
    v_und = torch.randn(1, 3, 2, 8)
    actual = padded_multiview_flex_attention(q, k, v, k_und, v_und, context)

    metadata = build_multiview_flex_metadata(
        seq_len=11,
        full_q_offsets=(3, 7, 11),
        items_per_sample=layout.mask_items(torch.device("cpu")),
        device="cpu",
        num_und=3,
        attention_scope=layout.attention_scope,
    )
    allowed = multiview_pair_predicate(
        metadata,
        torch.arange(8)[:, None],
        torch.arange(11)[None, :],
    )
    dense_k = torch.cat([k_und, k], dim=1).repeat_interleave(2, dim=2)
    dense_v = torch.cat([v_und, v], dim=1).repeat_interleave(2, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, dense_k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), dense_v)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_packing_buffers_are_reused_and_never_leak_stale_rows() -> None:
    """Layers re-pack identical shapes, so the padded q/k/v buffers are reused.

    Reuse is only sound because the padding rows are masked out: a shorter UND
    branch must not see the rows a longer branch left behind in the shared
    buffer.  Since UND pads to a fixed capacity, every branch shares one buffer,
    so that collision is the normal case rather than an edge case.
    """
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        padded_multiview_flex_attention,
    )

    torch.manual_seed(11)
    layout = MultiviewLayout(2, 4, 1, 1, condition_frame_indexes=(0, 2), max_und_tokens=_TRITON_MAX_UND)
    q = torch.randn(1, 8, 4, 8)
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)
    long_und = (torch.randn(1, 7, 2, 8), torch.randn(1, 7, 2, 8))
    short_und = (torch.randn(1, 3, 2, 8), torch.randn(1, 3, 2, 8))

    shared = MultiviewAttentionContext(layout, {}, {})
    padded_multiview_flex_attention(q, k, v, *long_und, shared)
    pointers = sorted(tensor.data_ptr() for tensor in shared.buffer_cache.values())
    assert len(pointers) == 3  # one buffer each for q, k, v

    reused = padded_multiview_flex_attention(q, k, v, *short_und, shared)
    assert sorted(tensor.data_ptr() for tensor in shared.buffer_cache.values()) == pointers

    fresh = padded_multiview_flex_attention(q, k, v, *short_und, MultiviewAttentionContext(layout, {}, {}))
    torch.testing.assert_close(reused, fresh)


def test_flex_attention_explicitly_pins_triton_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.diffusion.models.cosmos3.multiview_flex_attention as module

    captured: dict[str, object] = {}

    def fake_flex_attention(q, k, v, **kwargs):
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        captured.update(kwargs)
        return q

    monkeypatch.setattr(module, "torch_flex_attention", fake_flex_attention)
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn_like(k)
    output = module.flex_attention(q, k, v, block_mask=object(), backend="triton")

    assert output.shape == q.shape
    assert captured["kernel_options"] == {
        "BACKEND": "TRITON",
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "num_stages": 1,
        "num_warps": 4,
        "USE_TMA": False,
    }


def test_same_view_or_frame_rejects_mixed_view_offsets() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MaskItem,
        build_multiview_flex_metadata,
    )

    condition = torch.ones(2, dtype=torch.bool)
    items = (
        MaskItem((2, 1, 1), condition, num_views=1, view_offset=0, is_control=True),
        MaskItem((2, 1, 1), condition, num_views=1, view_offset=10),
    )
    with pytest.raises(ValueError, match="mixed view offsets"):
        build_multiview_flex_metadata(
            seq_len=5,
            full_q_offsets=(1, 3, 5),
            items_per_sample=items,
            device="cpu",
            num_und=1,
            attention_scope="same_view_or_frame",
        )


# --- FlashAttention-4 backend: host-side mask encoding ----------------------
#
# These cover the half of the FA4 port that needs no GPU: that the packed run
# truth table the CuTe mask_mod reads reproduces multiview_pair_predicate
# exactly, and that the coarser (256, 128) block map FA4 mandates is still a
# correct classification of the dense mask.


def _fa4_plan(layout, real_und_len: int):
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        get_multiview_attention_plan,
    )

    return get_multiview_attention_plan(
        MultiviewAttentionContext(layout, {}),
        real_und_len=real_und_len,
        real_q_len=layout.gen_tokens,
        device=torch.device("cpu"),
    )


def test_fa4_layout_uses_the_block_geometry_the_sm100_kernel_demands() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewBlockSparsity,
        MultiviewLayout,
        get_multiview_attention_plan,
        get_multiview_block_mask,
    )

    layout = MultiviewLayout(2, 4, 1, 1, condition_frame_indexes=(0, 2), backend="fa4", max_und_tokens=_FA4_MAX_UND)
    context = MultiviewAttentionContext(layout, {})
    first, geometry = get_multiview_attention_plan(context, real_und_len=3, real_q_len=8, device=torch.device("cpu"))
    second, _ = get_multiview_attention_plan(context, real_und_len=3, real_q_len=8, device=torch.device("cpu"))

    assert first is second
    assert isinstance(first, MultiviewBlockSparsity)
    assert (first.q_block_size, first.kv_block_size) == (256, 128)
    assert geometry.padded_q_len == 256
    assert geometry.padded_und_len == 128
    # 256 query rows is one q block; 128 + 256 keys is three kv blocks.
    assert first.partial_counts.shape == (1,)
    assert first.partial_indices.shape == (1, 3)
    assert first.partial_indices.dtype == torch.int32
    assert first.partial_indices.is_contiguous()

    with pytest.raises(TypeError, match="backend='triton'"):
        get_multiview_block_mask(context, real_und_len=3, real_q_len=8, device=torch.device("cpu"))


def test_fa4_rejects_unknown_backend() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import MultiviewLayout

    with pytest.raises(ValueError, match="backend must be one of"):
        MultiviewLayout(2, 4, 1, 1, backend="cudnn")


def test_run_table_reproduces_the_pair_predicate_exactly() -> None:
    """The CuTe mask_mod is a lookup of group_allowed[q_run, k_run].

    If that composition is not bit-identical to multiview_pair_predicate, every
    partially masked tile is resolved wrongly, so this is the load-bearing
    correctness claim for the FA4 mask_mod.
    """
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewLayout,
        multiview_pair_predicate,
    )

    layout = MultiviewLayout(2, 4, 2, 2, condition_frame_indexes=(0, 2), backend="fa4", max_und_tokens=_FA4_MAX_UND)
    sparsity, _ = _fa4_plan(layout, real_und_len=3)
    metadata = sparsity.metadata

    dense = multiview_pair_predicate(
        metadata,
        torch.arange(metadata.q_len)[:, None],
        torch.arange(metadata.kv_len)[None, :],
    )
    q_runs = sparsity.q_word_base.to(torch.int64) // sparsity.words_per_row
    k_runs = sparsity.k_group_ids.to(torch.int64)
    reconstructed = sparsity.group_allowed[q_runs][:, k_runs]

    assert dense.any() and not dense.all()
    torch.testing.assert_close(reconstructed, dense)


def test_packed_allowed_bits_round_trip() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import MultiviewLayout

    layout = MultiviewLayout(2, 4, 2, 2, condition_frame_indexes=(0, 2), backend="fa4", max_und_tokens=_FA4_MAX_UND)
    sparsity, _ = _fa4_plan(layout, real_und_len=3)

    num_q_groups, num_k_groups = sparsity.group_allowed.shape
    assert sparsity.allowed_words.numel() == num_q_groups * sparsity.words_per_row
    assert sparsity.words_per_row == (num_k_groups + 31) // 32

    # Reproduce the kernel's read: word (run * words_per_row + k // 32), bit k % 32.
    words = sparsity.allowed_words.to(torch.int64).view(num_q_groups, sparsity.words_per_row)
    words = words & 0xFFFFFFFF
    bits = (words[:, :, None] >> torch.arange(32, dtype=torch.int64)) & 1
    unpacked = bits.reshape(num_q_groups, -1)[:, :num_k_groups].bool()

    torch.testing.assert_close(unpacked, sparsity.group_allowed)


def test_fa4_block_map_matches_dense_token_projection() -> None:
    """The (256, 128) map must still separate full from partial tiles exactly."""
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewLayout,
        multiview_pair_predicate,
    )

    # A UND length that fills its KV block exactly is what makes fully-allowed
    # tiles reachable here: every real query attends every real UND token, but a
    # block padded out with sentinels is partial by construction.
    layout = MultiviewLayout(2, 4, 8, 8, condition_frame_indexes=(0, 2), backend="fa4", max_und_tokens=_FA4_MAX_UND)
    sparsity, geometry = _fa4_plan(layout, real_und_len=128)
    metadata = sparsity.metadata

    dense = multiview_pair_predicate(
        metadata,
        torch.arange(metadata.q_len)[:, None],
        torch.arange(metadata.kv_len)[None, :],
    )
    q_blocks = metadata.q_len // sparsity.q_block_size
    kv_blocks = metadata.kv_len // sparsity.kv_block_size
    tiles = dense.view(q_blocks, sparsity.q_block_size, kv_blocks, sparsity.kv_block_size)
    tiles = tiles.permute(0, 2, 1, 3)
    expected_visible = tiles.any(dim=(-2, -1))
    expected_full = tiles.all(dim=(-2, -1))
    expected_partial = expected_visible & ~expected_full

    def unpack(counts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        result = torch.zeros((q_blocks, kv_blocks), dtype=torch.bool)
        for row in range(q_blocks):
            count = int(counts[row])
            result[row, indices[row, :count].to(torch.int64)] = True
        return result

    assert geometry.padded_und_len == 128
    assert expected_partial.any()
    assert expected_full.any()
    torch.testing.assert_close(unpack(sparsity.partial_counts, sparsity.partial_indices), expected_partial)
    torch.testing.assert_close(unpack(sparsity.full_counts, sparsity.full_indices), expected_full)


def test_backend_validation_is_reusable_and_lists_choices() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MULTIVIEW_BACKENDS,
        validate_multiview_backend,
    )

    assert MULTIVIEW_BACKENDS == ("fa4", "triton")
    assert validate_multiview_backend("fa4") == "fa4"
    assert validate_multiview_backend("triton") == "triton"
    with pytest.raises(ValueError, match=r"must be one of \['fa4', 'triton'\]"):
        validate_multiview_backend("tirton")


def test_fa4_layout_registers_the_opaque_kernel_op() -> None:
    """The FA4 launch must reach Dynamo as one custom op, not as traced CuTe.

    Calling ``flash_attn.cute``'s entry point directly from the regionally
    compiled GEN block makes Dynamo trace FA4's JIT compile-cache lookup and
    guard on that cache's contents; the CUTLASS ``arith.const`` frame then hits
    the recompile limit and the process falls back to eager.  Constructing an
    fa4 layout is what registers the boundary, and it happens host-side so the
    registration never runs under Dynamo.
    """
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import MultiviewLayout

    MultiviewLayout(
        num_views=2,
        latent_frames=8,
        patch_height=4,
        patch_width=4,
        backend="fa4",
        max_und_tokens=_FA4_MAX_UND,
    )

    assert hasattr(torch.ops.vllm_omni, "cosmos3_multiview_fa4")
