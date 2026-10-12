# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.vdnh3_attn import (
    VDNAttentionBackend,
    VDNLayout,
    window_plan,
    windowed_attention,
)
from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _dense_mask(window: VDNLayout) -> torch.Tensor:
    """The documented window as a boolean [used, used] mask, built independently."""
    frame = torch.full((window.used,), -1)
    frame[window.video_start : window.video_end] = torch.arange(window.num_frames).repeat_interleave(
        window.tokens_per_frame
    )
    bounds = torch.tensor(window.window_bounds())
    q_frame, k_frame = frame[:, None], frame[None, :]
    lo, hi = bounds[q_frame.clamp(min=0), 0], bounds[q_frame.clamp(min=0), 1]
    keep = (q_frame < 0) | (k_frame < 0) | ((k_frame >= lo) & (k_frame <= hi))
    for anchor in window.dense_row_frames:
        keep |= q_frame == anchor
    for anchor in window.dense_column_frames:
        keep |= k_frame == anchor
    return keep


def _sdpa(q, k, v, mask=None):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2)


def test_vdn_backend_is_registered():
    assert DiffusionAttentionBackendEnum.VDNH3_ATTN.get_path().endswith("vdnh3_attn.VDNAttentionBackend")
    assert DiffusionAttentionBackendEnum.VDNH3_ATTN.get_class().supports_multi_doc_packed_varlen() is False


@pytest.mark.parametrize("anchor_frames", ["none", "columns", "rows", "both"])
@pytest.mark.parametrize(("chunk", "radius"), [(3, 1), (0, 1), (2, 0)])
def test_vdn_window_matches_dense_mask(anchor_frames, chunk, radius):
    torch.manual_seed(0)
    # [text 5 | audio 4 | 9 frames of 2x3 | 7 padding rows]
    window = VDNLayout(
        used=63,
        text_len=5,
        video_start=9,
        num_frames=9,
        frame_height=2,
        frame_width=3,
        chunk=chunk,
        radius=radius,
        anchor_frames=anchor_frames,
    )
    q, k, v = (torch.randn(1, 70, 2, 8) for _ in range(3))
    out = windowed_attention(q, k, v, window, _sdpa)
    used = window.used
    ref = _sdpa(q[:, :used], k[:, :used], v[:, :used], _dense_mask(window))
    torch.testing.assert_close(out[:, :used], ref, atol=1e-5, rtol=1e-5)
    assert torch.all(out[:, used:] == 0)
    # Anything narrower than the clip is genuinely sparse.
    assert not torch.allclose(ref, _sdpa(q[:, :used], k[:, :used], v[:, :used]), atol=1e-3)


def test_vdn_window_full_cover_is_one_dense_call():
    window = VDNLayout(
        used=20, text_len=2, video_start=4, num_frames=4, frame_height=2, frame_width=2, chunk=0, radius=3
    )
    assert window.full_cover
    assert window_plan(window, torch.device("cpu")) == ((0, 20, None),)


def test_vdn_window_rejects_inconsistent_layout():
    with pytest.raises(ValueError, match="invalid VDN layout"):
        VDNLayout(used=10, text_len=0, video_start=4, num_frames=4, frame_height=2, frame_width=2, chunk=0, radius=1)


def test_vdnh3_attn_does_not_inherit_flash_attn_capabilities():
    context = ExecutionContext(platform="cuda", kernel_variant="fa4", dtype="bfloat16", causal=False)
    result = VDNAttentionBackend.resolve_capabilities(context)
    assert result.backend == "VDNH3_ATTN"
    assert result.support.status is SupportStatus.UNMIGRATED
