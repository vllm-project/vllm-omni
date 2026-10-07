# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the single architecture to an independent dense capture-time reference."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    PaddedAttentionGeometry,
    _pack_padded_bshd,
    build_multiview_block_sparsity,
    build_multiview_flex_metadata,
    get_multiview_attention_plan,
    multiview_pair_predicate,
    padded_multiview_flex_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _layout(window: float = 0.4, backend: str = "triton") -> MultiviewLayout:
    return MultiviewLayout(
        items=(
            MaskItem((10, 1, 2), 2, is_control=True, seconds_per_frame=0.2),
            MaskItem((10, 1, 2), 2, seconds_per_frame=0.2),
            MaskItem((11, 1, 1), 1, view_offset=2, is_control=True, is_lidar=True, seconds_per_frame=0.1),
            MaskItem((11, 1, 1), 1, view_offset=2, is_lidar=True, seconds_per_frame=0.1),
        ),
        cross_view_past_window_seconds=window,
        caption_lengths=(2, 3),
        max_und_tokens=64,
        backend=backend,
    )


def _dense_reference(layout: MultiviewLayout, geometry: PaddedAttentionGeometry) -> torch.Tensor:
    # Construct records from sensor geometry, without consulting production token metadata.
    # A record is (real, control, text, sensor, capture_time); None marks padding.
    keys: list[tuple[bool, bool, bool, int, float] | None] = []
    for view, length in enumerate(layout.caption_lengths):
        keys.extend([(True, False, True, view, 0.0)] * length)
    keys.extend([None] * (geometry.padded_und_len - len(keys)))
    queries: list[tuple[bool, bool, bool, int, float] | None] = []
    for item in layout.items:
        for view in range(item.num_views):
            for frame in range(item.token_shape[0] // item.num_views):
                queries.extend(
                    [
                        (
                            True,
                            item.is_control,
                            False,
                            -2 if item.is_lidar else item.view_offset + view,
                            frame * item.seconds_per_frame,
                        )
                    ]
                    * (item.token_shape[1] * item.token_shape[2])
                )
    queries.extend([None] * (geometry.padded_q_len - len(queries)))
    keys.extend(queries)
    expected = torch.zeros(len(queries), len(keys), dtype=torch.bool)
    for qi, q in enumerate(queries):
        for ki, k in enumerate(keys):
            if q is None or k is None:
                allowed = q is None and k is None
            elif k[2]:
                allowed = q[3] == -2 or q[3] == k[3]
            elif q[3] == k[3]:
                allowed = True  # All same-sensor target and control edges span the clip.
            else:
                allowed = not q[1] and not k[1] and -1e-4 <= q[4] - k[4] <= layout.cross_view_past_window_seconds + 1e-4
            expected[qi, ki] = allowed
    return expected


@pytest.mark.cpu
@pytest.mark.parametrize("window", [0.0, 0.4, 0.40001])
def test_attention_pairs_match_independent_dense_reference(window: float) -> None:
    layout = _layout(window)
    geometry = PaddedAttentionGeometry(layout.gen_tokens, 64, 5, 64)
    metadata = build_multiview_flex_metadata(layout, geometry, "cpu")
    actual = multiview_pair_predicate(metadata, torch.arange(64)[:, None], torch.arange(128)[None, :])
    expected = _dense_reference(layout, geometry)
    torch.testing.assert_close(actual, expected)
    assert metadata.timestamp.dtype == torch.float32
    sparsity = build_multiview_block_sparsity(metadata)
    mask = sparsity.to_block_mask()
    torch.testing.assert_close(
        mask.mask_mod(torch.tensor(0), torch.tensor(0), torch.arange(64)[:, None], torch.arange(128)[None, :]), expected
    )
    # Both cross-view boundaries are inclusive; future keys are excluded, while
    # same-view future keys and same-view controls remain unrestricted.
    if window == 0.4:
        query = geometry.padded_und_len + 20 + 4  # camera 0 target, t=0.4
        other_target = geometry.padded_und_len + 20 + 10  # camera 1 target, t=0
        assert actual[query - 64, other_target]
        assert actual[query - 64, other_target + 4]  # t=0.4 upper boundary
        assert not actual[query - 64, other_target + 6]  # t=0.6 future cross-view
        assert actual[query - 64, query + 4]  # t=0.8 same-view future
        assert actual[query - 64, geometry.padded_und_len + 8]  # same-view future control
        assert not actual[: layout.gen_tokens, 5:64].any()  # text padding
        assert not actual[: layout.gen_tokens, 64 + layout.gen_tokens :].any()  # GEN padding


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("camera_rate", "lidar_rate", "lidar_frame", "visible"),
    [
        (0.20004, 0.1, 0, True),  # oldest key lies 8e-5 beyond the nominal past bound
        (0.20006, 0.1, 0, False),
        (0.2, 0.10002, 4, True),  # key lies 8e-5 beyond the nominal current-time bound
        (0.2, 0.10003, 4, False),
    ],
)
def test_capture_time_tolerance_at_both_boundaries(
    camera_rate: float, lidar_rate: float, lidar_frame: int, visible: bool
) -> None:
    layout = MultiviewLayout(
        items=(
            MaskItem((3, 1, 1), 1, seconds_per_frame=camera_rate),
            MaskItem((5, 1, 1), 1, view_offset=1, is_lidar=True, seconds_per_frame=lidar_rate),
        ),
        cross_view_past_window_seconds=0.4,
        caption_lengths=(2,),
        max_und_tokens=64,
    )
    metadata = build_multiview_flex_metadata(layout, PaddedAttentionGeometry(8, 64, 2, 64), "cpu")
    visible_pair = multiview_pair_predicate(metadata, torch.tensor(2), torch.tensor(64 + 3 + lidar_frame))
    assert bool(visible_pair) is visible


@pytest.mark.cpu
@pytest.mark.parametrize("window", [None, True, -0.1, float("inf"), float("nan")])
def test_layout_rejects_invalid_window(window: float) -> None:
    with pytest.raises(ValueError, match="finite.*non-negative"):
        _layout(window)


@pytest.mark.cpu
@pytest.mark.parametrize("backend", ["maskless", "unknown"])
def test_layout_rejects_removed_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        _layout(backend=backend)


def _numerical_comparison(device: str, backend: str, dtype: torch.dtype, *, compiled: bool = False) -> None:
    torch.manual_seed(37)
    layout = _layout(backend=backend)
    q_block, kv_block = layout.block_sizes
    q_len = math.ceil(layout.gen_tokens / q_block) * q_block
    und_len = math.ceil(layout.max_und_tokens / kv_block) * kv_block
    geometry = PaddedAttentionGeometry(layout.gen_tokens, q_len, 5, und_len)
    q = torch.randn(1, layout.gen_tokens, 4, 128, device=device, dtype=dtype)
    k = torch.randn(1, layout.gen_tokens, 2, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    ku = torch.randn(1, 5, 2, 128, device=device, dtype=dtype)
    vu = torch.randn_like(ku)
    keys, values = torch.cat([ku, k], dim=1), torch.cat([vu, v], dim=1)
    dense_mask = _dense_reference(layout, geometry)[: layout.gen_tokens]
    dense_mask = torch.cat([dense_mask[:, :5], dense_mask[:, und_len : und_len + layout.gen_tokens]], dim=1).to(device)
    scores = q.float().transpose(1, 2) @ keys.float().repeat_interleave(2, dim=2).transpose(1, 2).transpose(-1, -2)
    scores = scores / math.sqrt(128)
    scores.masked_fill_(~dense_mask, -float("inf"))
    expected = (scores.softmax(-1) @ values.float().repeat_interleave(2, dim=2).transpose(1, 2)).transpose(1, 2)
    context = MultiviewAttentionContext(layout, {})
    if compiled:
        from vllm_omni.diffusion.models.cosmos3.multiview_fa4 import multiview_fa4_attention

        plan, padded = get_multiview_attention_plan(
            context, real_und_len=5, real_q_len=layout.gen_tokens, device=q.device
        )
        attention = torch.compile(multiview_fa4_attention, fullgraph=True)
        actual = attention(
            _pack_padded_bshd((q, padded.padded_q_len)),
            _pack_padded_bshd((ku, padded.padded_und_len), (k, padded.padded_q_len)),
            _pack_padded_bshd((vu, padded.padded_und_len), (v, padded.padded_q_len)),
            plan,
        )[:, : layout.gen_tokens]
    else:
        actual = padded_multiview_flex_attention(q, k, v, ku, vu, context)
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        atol=0.01 if dtype == torch.bfloat16 else 1e-5,
    )


@pytest.mark.cpu
def test_triton_attention_numerically_matches_dense_on_cpu() -> None:
    _numerical_comparison("cpu", "triton", torch.float32)


@pytest.mark.gpu
@pytest.mark.parametrize(("backend", "compiled"), [("triton", False), ("fa4", False), ("fa4", True)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical comparison")
def test_cuda_attention_numerically_matches_dense(backend: str, compiled: bool) -> None:
    # Triton's production entrypoint compiles the dynamic-shape Flex kernel.
    _numerical_comparison("cuda", backend, torch.bfloat16, compiled=compiled)
