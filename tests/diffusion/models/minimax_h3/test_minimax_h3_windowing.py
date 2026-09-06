# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sliding-window shape math and history-block packing for MiniMax H3."""

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# --------------------------------------------------------------------------- #
# Overlap / window frame math
# --------------------------------------------------------------------------- #
def _frames_from_latent_t(out_t: int) -> int:
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
    )

    return MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(out_t)


def test_windowing_plan_30s_is_two_windows_with_overlap_drop():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    plan = _resolve_minimax_h3_windowing(
        duration=30.0,
        fps=24,
        num_segments=None,  # auto-activates because duration > 15
        overlap_frames=None,
        window_duration=None,
    )
    assert plan is not None
    assert plan.is_active
    # Default window is the native ceiling (15 s -> 362 frames = 17*21 + 5).
    assert plan.window_num_frames == 362
    # Default overlap is 58 frames -> 17 latents (56 decoded frames, the
    # cross-fade span). overlap_latent_t must satisfy
    # (wt - overlap_latent_t) % 15 == 0 so the concatenated latent stays on the
    # VAE's 5n+2 grid AND each continuation window contributes an integral
    # number of frames and audio latents.
    assert plan.overlap_latent_t == 17
    assert plan.overlap_frames == 56
    # Audio overlap is derived from the same wall-clock span as the video
    # contribution, not converted independently from overlap_frames.
    assert plan.overlap_audio_t == 93
    # 30 s rounds to two windows.
    assert plan.num_windows == 2
    # total_num_frames is what the concatenated latent actually decodes to:
    # frames(107 + 90) = 668 = 362 + 306.
    assert plan.total_num_frames == 668


def test_windowing_contribution_is_av_exact():
    """Every continuation window must add the same wall-clock span of video
    and audio, or the A/V desync accumulates per window."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    for window_duration in (8.0, 12.0, 15.0):
        plan = _resolve_minimax_h3_windowing(
            duration=45.0,
            fps=24,
            num_segments=3,
            overlap_frames=None,
            window_duration=window_duration,
        )
        wt = _video_latent_t(plan.window_num_frames)
        contributed_frames = _frames_from_latent_t(wt + (wt - plan.overlap_latent_t)) - plan.window_num_frames
        video_seconds = contributed_frames / 24.0
        audio_seconds = (plan.window_audio_t - plan.overlap_audio_t) / 40.0
        assert video_seconds == audio_seconds, window_duration
        # And the plan's total matches what the latent decodes to.
        total_t = wt + (plan.num_windows - 1) * (wt - plan.overlap_latent_t)
        assert plan.total_num_frames == _frames_from_latent_t(total_t)


def test_windowing_overlap_snaps_to_latent_grid():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    # 100 requested frames -> 27 latents, which is off the wt=107 grid
    # (27 % 15 != 107 % 15); nearest valid is 32.
    plan = _resolve_minimax_h3_windowing(
        duration=30.0,
        fps=24,
        num_segments=2,
        overlap_frames=100,
        window_duration=None,
    )
    assert plan.overlap_latent_t == 32
    # An 8 s window (192 frames, wt=57) needs overlap_latent_t ≡ 57
    # (mod 15) = 12; the default 58-frame request (17 latents) snaps down
    # to 12.
    plan = _resolve_minimax_h3_windowing(
        duration=20.0,
        fps=24,
        num_segments=2,
        overlap_frames=None,
        window_duration=8.0,
    )
    assert plan.window_num_frames == 192
    assert plan.overlap_latent_t == 12
    assert plan.overlap_audio_t == 65
    assert plan.total_num_frames == 192 + 153


def test_windowing_explicit_num_segments():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    plan = _resolve_minimax_h3_windowing(
        duration=45.0,
        fps=24,
        num_segments=3,
        overlap_frames=58,
        window_duration=None,
    )
    assert plan.num_windows == 3
    # Window 0 contributes 362; windows 1-2 each contribute 306 frames
    # (90 latents); frames(107 + 180) = 974.
    assert plan.total_num_frames == 974


def test_windowing_inactive_for_single_window():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    # Within the native contract and no num_segments -> single window.
    assert (
        _resolve_minimax_h3_windowing(
            duration=8.0,
            fps=24,
            num_segments=None,
            overlap_frames=None,
            window_duration=None,
        )
        is None
    )
    # num_segments=1 is an explicit single-window request (no windowing).
    assert (
        _resolve_minimax_h3_windowing(
            duration=8.0,
            fps=24,
            num_segments=1,
            overlap_frames=None,
            window_duration=None,
        )
        is None
    )


def test_windowing_rejects_overlap_larger_than_window():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError):
        _resolve_minimax_h3_windowing(
            duration=30.0,
            fps=24,
            num_segments=2,
            overlap_frames=400,
            window_duration=None,
        )


def _video_latent_t(frame_count: int) -> int:
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
    )

    return MINIMAX_H3_SHAPE_PLANNER.video_latent_t(frame_count)


def _audio_latent_t(duration_seconds: float) -> int:
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
    )

    return MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(duration_seconds)


# --------------------------------------------------------------------------- #
# History-block packing (video_audio ref block)
# --------------------------------------------------------------------------- #
def test_history_video_audio_block_emits_frozen_rows():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    latent_h, latent_w = 48, 80  # 768x1280-ish canvas at /16
    overlap_latent_t = 17
    overlap_audio_t = 93
    window_latent_t = 107
    window_audio_t = 603
    history_block = {
        "kind": "video_audio",
        "ref_audio_t": overlap_audio_t,
        "latent_t": overlap_latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
    }
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=128,
        latent_t=window_latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=window_audio_t,
        ref_blocks=[history_block],
    )
    frame_rows = (latent_h // 2) * (latent_w // 2)
    ref_visual_rows = overlap_latent_t * frame_rows
    ref_audio_rows = overlap_audio_t * 2

    # Visual: history rows frozen, target rows updated.
    assert int(packed["update_mask"][:ref_visual_rows].sum()) == 0
    assert bool(packed["update_mask"][ref_visual_rows:].all())
    # Audio: history rows frozen, target rows updated.
    assert int(packed["audio_update_mask"][:ref_audio_rows].sum()) == 0
    assert bool(packed["audio_update_mask"][ref_audio_rows:].all())
    # The history block is advertised as a reference span.
    roles = [span["role"] for span in packed["video_spans"]]
    assert "reference" in roles
    assert roles[-1] == "target"


def test_audio_history_rows_round_trip_through_latent_tail():
    """Continuation windows pack the previous audio tail via pack_audio_latent."""
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_pack_audio_latent,
        minimax_h3_unpack_audio_tokens,
    )

    audio_latent = torch.arange(2 * 32 * 602).reshape(2, 32, 602).float()
    overlap_audio_t = 93
    tail = audio_latent[:, :, -overlap_audio_t:]
    rows = minimax_h3_pack_audio_latent(tail)
    assert rows.shape == (overlap_audio_t * 2, 32)
    restored = minimax_h3_unpack_audio_tokens(rows, audio_t=overlap_audio_t * 2, audio_channel=2)
    torch.testing.assert_close(restored, tail)


def test_video_history_rows_round_trip_through_latent_tail():
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
        minimax_h3_unpatchify_video_tokens,
    )

    latent = torch.arange(1 * 24 * 107 * 48 * 80).reshape(1, 24, 107, 48, 80).float()
    overlap_latent_t = 17
    tail = latent[:, :, -overlap_latent_t:, :, :]
    rows = minimax_h3_patchify_video_latent(tail, patch_size=(1, 2, 2))
    frame_rows = (48 // 2) * (80 // 2)
    assert rows.shape == (overlap_latent_t * frame_rows, 96)
    restored = minimax_h3_unpatchify_video_tokens(
        rows, latent_shape=(overlap_latent_t, 24, 40, 24), patch_size=(1, 2, 2)
    )
    torch.testing.assert_close(restored, tail)


def test_identity_anchor_plus_history_block_frozen_row_split():
    """Ref2VA continuation layout: [image identity-anchor, video_audio history] + target.

    Mirrors the ref_blocks the ref2va window loop assembles for a continuation
    window: a 1-frame image anchor (window 0's first frame) followed by the
    video+audio history block (17 latents / 93 audio latents when the overlap
    is 58 frames), then the target. t2va/fl2va windows take the first-frame
    path instead (see ``test_continuation_window_is_a_first_frame_request``).
    """
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    latent_h, latent_w = 48, 80
    frame_rows = (latent_h // 2) * (latent_w // 2)
    window_latent_t = 107
    window_audio_t = 603
    overlap_latent_t = 17
    overlap_audio_t = 93
    ref_blocks = [
        {"kind": "image", "latent_h": latent_h, "latent_w": latent_w},
        {
            "kind": "video_audio",
            "ref_audio_t": overlap_audio_t,
            "latent_t": overlap_latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
        },
    ]
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=128,
        latent_t=window_latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=window_audio_t,
        ref_blocks=ref_blocks,
    )
    # Anchor image (1 frame) + history video (overlap_latent_t frames) are the
    # frozen visual rows; the target window is the rest.
    anchor_rows = 1 * frame_rows
    history_video_rows = overlap_latent_t * frame_rows
    ref_visual_rows = anchor_rows + history_video_rows
    assert int(packed["update_mask"][:ref_visual_rows].sum()) == 0
    assert bool(packed["update_mask"][ref_visual_rows:].all())
    # Both reference spans are advertised; the target is last.
    roles = [span["role"] for span in packed["video_spans"]]
    assert roles.count("reference") == 1
    assert roles[-1] == "target"


# --------------------------------------------------------------------------- #
# Window-level bookkeeping for the first-frame continuation path
# --------------------------------------------------------------------------- #
def test_continuation_window_is_a_first_frame_request():
    """A continuation window is conditioned like a user fl2va request whose
    first frame is the handoff still, plus the request's own last frame when
    this is the final window."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        MINIMAX_H3_IMGVID_COND_ID,
        minimax_h3_packed_sequence,
    )
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _continuation_keyframes,
    )

    assert _continuation_keyframes(None) == [0]
    assert _continuation_keyframes([-1]) == [0, -1]
    # The layout of such a window is byte-identical to a standalone
    # first-frame (or first+last) fl2va request of the same size.
    for keyframes in ([0], [0, -1]):
        window = minimax_h3_packed_sequence(
            text_len=128,
            latent_t=107,
            latent_h=30,
            latent_w=52,
            audio_t=603,
            include_keyframe_cond=True,
            keyframe_frame_indices=_continuation_keyframes(keyframes[1:] or None),
            frame_count=362,
        )
        standalone = minimax_h3_packed_sequence(
            text_len=128,
            latent_t=107,
            latent_h=30,
            latent_w=52,
            audio_t=603,
            include_keyframe_cond=True,
            keyframe_frame_indices=keyframes,
            frame_count=362,
        )
        for key in ("input_ids", "update_mask", "img_position_ids", "token_tags", "cu_seqlens"):
            torch.testing.assert_close(window[key], standalone[key], rtol=0, atol=0)
        frame_rows = (30 // 2) * (52 // 2)
        cond_ids = window["input_ids"][128 : 128 + len(keyframes) * frame_rows]
        assert bool((cond_ids == MINIMAX_H3_IMGVID_COND_ID).all())
        # Every target video row is generated; only the stills are frozen.
        assert int(window["update_mask"].sum()) == 107 * frame_rows
        assert int((~window["update_mask"]).sum()) == len(keyframes) * frame_rows


def test_window_keyframes_split_first_and_last_across_windows():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _window_keyframe_indices,
    )

    # fl2va with both anchors: first frame pins window 0, last frame pins the
    # final window, middle windows carry only the handoff still.
    assert _window_keyframe_indices([0, -1], window_index=0, num_windows=3) == [0]
    assert _window_keyframe_indices([0, -1], window_index=1, num_windows=3) is None
    assert _window_keyframe_indices([0, -1], window_index=2, num_windows=3) == [-1]
    assert _window_keyframe_indices([-1], window_index=0, num_windows=2) is None
    assert _window_keyframe_indices([-1], window_index=1, num_windows=2) == [-1]
    assert _window_keyframe_indices([0], window_index=1, num_windows=2) is None
    # t2va has no keyframes anywhere.
    assert _window_keyframe_indices(None, window_index=0, num_windows=2) is None
    # A single window is untouched.
    assert _window_keyframe_indices([0, -1], window_index=0, num_windows=1) == [0, -1]


def test_window_trim_matches_plan_contribution():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
        _window_trim,
    )

    plan = _resolve_minimax_h3_windowing(
        duration=30.0, fps=24, num_segments=None, overlap_frames=None, window_duration=None
    )
    trim_frames, trim_samples = _window_trim(plan, sample_rate=32000)
    # A continuation window's shared span (cross-faded video, dropped audio)
    # must leave exactly the planned contribution: 362 - 56 = 306 frames,
    # 12.75 s of audio (93 latents = 74400 samples dropped).
    assert plan.window_num_frames - trim_frames == 306
    assert trim_frames == plan.overlap_frames == 56
    assert trim_samples == plan.overlap_audio_t * 800 == 74400
    assert (plan.window_audio_t - plan.overlap_audio_t) * 800 == 12.75 * 32000
    # The handoff still is the frame the next window's frame 0 reproduces.
    assert plan.window_num_frames - trim_frames == plan.total_num_frames - plan.window_num_frames


def test_history_reinjection_tracks_the_tail_at_each_sigma():
    """The leading target rows follow (1 - sigma) * tail + sigma * initial noise
    after every step and equal the tail exactly at sigma 0; other target rows and
    the frozen condition rows are left alone."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _history_reinjection

    frame_rows, latent_t, history_t = 4, 6, 2
    cond_rows = frame_rows  # one still keyframe ahead of the target rows
    update_mask = torch.cat(
        [torch.zeros(cond_rows, dtype=torch.bool), torch.ones(latent_t * frame_rows, dtype=torch.bool)]
    )
    branch = SimpleNamespace(update_mask_dev=update_mask)
    initial = torch.randn(cond_rows + latent_t * frame_rows, 96)
    tail = torch.full((history_t * frame_rows, 96), 3.0)
    inputs: dict[str, Any] = {"branch": branch, "video_rows": initial}
    seen: list[int] = []
    sigmas = [1.0, 0.6, 0.25, 0.0]
    reinject = _history_reinjection(
        inputs, history_rows=tail, sigmas_video=sigmas, on_step=lambda step, v, a: seen.append(step), release_sigma=0.0
    )
    history = slice(cond_rows, cond_rows + history_t * frame_rows)
    rows = initial.clone()
    for step in range(len(sigmas) - 1):
        rows[update_mask] += 1.0  # what a denoise step would do to every target row
        reinject(step, rows, torch.zeros(0))
        sigma = sigmas[step + 1]
        torch.testing.assert_close(rows[history], (1 - sigma) * tail + sigma * initial[history])
    torch.testing.assert_close(rows[history], tail)
    # Condition rows and the generated rows past the history are untouched by the hook.
    torch.testing.assert_close(rows[:cond_rows], initial[:cond_rows])
    torch.testing.assert_close(rows[history.stop :], initial[history.stop :] + 3.0)
    assert seen == [0, 1, 2]
    # With a release sigma the hold stops once the schedule drops below it: the
    # step ending at 0.25 and the final step leave the rows to the denoiser.
    released = _history_reinjection(inputs, history_rows=tail, sigmas_video=sigmas, on_step=None, release_sigma=0.3)
    rows = initial.clone()
    for step in range(len(sigmas) - 1):
        rows[update_mask] += 1.0
        released(step, rows, torch.zeros(0))
    torch.testing.assert_close(rows[history], 0.4 * tail + 0.6 * initial[history] + 2.0)
    with pytest.raises(ValueError):
        _history_reinjection(
            inputs, history_rows=torch.zeros(latent_t * frame_rows, 96), sigmas_video=sigmas, on_step=None
        )
    # Audio history is held in the leading steps of BOTH channel blocks with the
    # audio schedule's sigma; audio rows are channel-major [ch0 t.., ch1 t..].
    wa, hold = 6, 2
    audio_branch = SimpleNamespace(
        update_mask_dev=update_mask, audio_update_mask_dev=torch.ones(2 * wa, dtype=torch.bool)
    )
    audio_initial = torch.randn(2 * wa, 32)
    audio_tail = torch.full((2 * hold, 32), 5.0)
    audio_inputs: dict[str, Any] = {"branch": audio_branch, "video_rows": initial, "audio_rows": audio_initial}
    sigmas_audio = [1.0, 0.5, 0.2, 0.0]
    hold_both = _history_reinjection(
        audio_inputs,
        history_rows=tail,
        sigmas_video=sigmas,
        on_step=None,
        release_sigma=0.0,
        audio_history_rows=audio_tail,
        sigmas_audio=sigmas_audio,
    )
    audio_rows = audio_initial.clone() + 1.0
    hold_both(0, initial.clone(), audio_rows)
    held_idx = torch.tensor([0, 1, wa, wa + 1])
    torch.testing.assert_close(audio_rows[held_idx], 0.5 * audio_tail + 0.5 * audio_initial[held_idx])
    free_idx = torch.tensor([2, 3, 4, 5, wa + 2, wa + 3, wa + 4, wa + 5])
    torch.testing.assert_close(audio_rows[free_idx], audio_initial[free_idx] + 1.0)
    # The audio hold releases on the AUDIO schedule: the step ending at audio
    # sigma 0.2 (< 0.3) leaves the audio rows alone even though the video
    # sigma (0.25) is still held with release_sigma 0.0 above; here both use 0.3.
    released_both = _history_reinjection(
        audio_inputs,
        history_rows=tail,
        sigmas_video=sigmas,
        on_step=None,
        release_sigma=0.3,
        audio_history_rows=audio_tail,
        sigmas_audio=sigmas_audio,
    )
    audio_rows = audio_initial.clone() + 1.0
    released_both(1, initial.clone(), audio_rows)
    torch.testing.assert_close(audio_rows[held_idx], audio_initial[held_idx] + 1.0)
    with pytest.raises(ValueError):
        _history_reinjection(
            audio_inputs, history_rows=tail, sigmas_video=sigmas, on_step=None, audio_history_rows=audio_tail
        )


def _smoothstep_alpha(span: int) -> torch.Tensor:
    ramp = torch.arange(1, span + 1, dtype=torch.float32) / (span + 1)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def test_match_audio_onset_lifts_a_quiet_start_towards_the_previous_level():
    """A new window whose audio fades in from silence is lifted towards the
    previous window's level at its onset, with a capped gain that decays to
    unity; a window that is already as loud is left alone."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _audio_level, _match_audio_onset

    sr = 32000
    torch.manual_seed(0)
    previous = torch.randn(1, 2, 3 * sr) * 0.1  # steady ambience at RMS ~0.1
    reference = _audio_level(previous, sample_rate=sr)
    assert 0.09 < reference < 0.11
    quiet = torch.randn(1, 2, 6 * sr) * 0.1
    ramp = torch.linspace(0.05, 1.0, 3 * sr)  # fades in over 3 s
    quiet[..., : 3 * sr] *= ramp
    lifted = _match_audio_onset(
        reference, quiet.clone(), sample_rate=sr, match_seconds=2.0, release_seconds=4.0, max_gain=8.0
    )
    rms = lambda x: float(x.pow(2).mean().sqrt())  # noqa: E731
    # The first half second is lifted (gain capped at 8x), the level over the
    # matched two seconds is close to the previous window's, and the tail past
    # the release is untouched.
    assert rms(lifted[..., : sr // 2]) > 5.0 * rms(quiet[..., : sr // 2])
    assert rms(lifted[..., : sr // 2]) <= 8.05 * rms(quiet[..., : sr // 2])
    assert 0.7 < rms(lifted[..., sr // 2 : 2 * sr]) / reference < 1.2
    torch.testing.assert_close(lifted[..., 4 * sr :], quiet[..., 4 * sr :])
    # Louder than the previous window, or a silent previous window: the gain
    # never attenuates, so nothing changes.
    loud = torch.randn(1, 2, 5 * sr) * 0.3
    torch.testing.assert_close(_match_audio_onset(reference, loud.clone(), sample_rate=sr), loud)
    torch.testing.assert_close(_match_audio_onset(0.0, quiet.clone(), sample_rate=sr), quiet)
    assert _audio_level(torch.zeros(1, 2, 0), sample_rate=sr) == 0.0


def test_splice_span_holds_fades_then_hands_over():
    """Inside the shared span the previous window is kept for ``hold`` entries,
    cross-faded into the next window's head for ``fade`` entries (smoothstep for
    video), and the remainder is the next window's rendering; everything before
    the span is untouched."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _splice_span

    span, hold, fade = 12, 3, 4
    previous = torch.zeros(1, 3, 20, 2, 2)
    head = torch.ones(1, 3, span, 2, 2)
    out = _splice_span(previous, head, dim=2, hold=hold, fade=fade)
    assert out is previous and out.shape == (1, 3, 20, 2, 2)
    start = 20 - span
    assert bool((out[:, :, : start + hold] == 0).all())
    torch.testing.assert_close(out[0, 0, start + hold : start + hold + fade, 0, 0], _smoothstep_alpha(fade))
    assert bool((out[:, :, start + hold + fade :] == 1.0).all())
    # Audio (B, C, samples) fades equal-power: two unit signals sum to sqrt(2) at the midpoint.
    prev_audio = torch.ones(1, 2, 20)
    out_audio = _splice_span(prev_audio, torch.ones(1, 2, 12), dim=-1, hold=2, fade=9)
    assert bool((out_audio[..., :10] == 1.0).all())
    assert abs(float(out_audio[0, 0, 10 + 4]) - 2**0.5) < 1e-5
    assert bool((out_audio[..., 19:] == 1.0).all())
    # hold/fade are clamped to the span; no span is a no-op; a span longer than
    # the previous window is an error.
    clamped = _splice_span(torch.zeros(1, 3, 20, 2, 2), head, dim=2, hold=50, fade=50)
    assert bool((clamped[:, :, start:] == 0).all())
    assert _splice_span(previous, torch.ones(1, 3, 0, 2, 2), dim=2, hold=1, fade=1) is previous
    with pytest.raises(ValueError):
        _splice_span(previous, torch.ones(1, 3, 21, 2, 2), dim=2, hold=1, fade=1)


# --------------------------------------------------------------------------- #
# _generate_windowed plumbing on a fake pipeline (no model, no GPU)
# --------------------------------------------------------------------------- #
# A 64x32 canvas: latent 4x2, one token row per latent frame after (1, 2, 2) patching.
_FAKE_HEIGHT, _FAKE_WIDTH = 32, 64
_FAKE_LATENT_H, _FAKE_LATENT_W = _FAKE_HEIGHT // 16, _FAKE_WIDTH // 16
_FAKE_FRAME_ROWS = (_FAKE_LATENT_H // 2) * (_FAKE_LATENT_W // 2)


def _fake_image(value: int):
    from PIL import Image

    return Image.new("RGB", (4, 4), (value, value, value))


def _run_fake_windowed(
    *,
    task: str,
    keyframes: list[int] | None,
    image_values: list[int],
    text_encoder=object(),
    num_segments: int | None = None,
    overlap_frames: int | None = None,
    audio_levels: tuple[float, ...] = (1.0, 2.0, 3.0),
):
    """Drive MiniMaxH3Pipeline._generate_windowed with stubbed model calls.

    Decoded frame ``t`` has constant pixel ``t / 1000`` so the handoff still is
    recognisable, decoded audio of the n-th decode is the constant ``n`` so the
    cross-fade is checkable, and every visual-condition row carries its source
    image's pixel value so condition-row order is checkable.
    """
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
        _resolve_minimax_h3_windowing,
    )
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER

    plan = _resolve_minimax_h3_windowing(
        duration=30.0 if num_segments is None else 15.0 * num_segments,
        fps=24,
        num_segments=num_segments,
        overlap_frames=overlap_frames,
        window_duration=None,
    )
    calls: dict[str, list] = {"encode_prompt": [], "build": [], "encode_image": [], "step_rows": [], "decode": []}

    def encode_prompt(*, task, prompt, images=None):
        calls["encode_prompt"].append((task, [img.getpixel((0, 0))[0] for img in (images or [])]))
        n = 24 + 100 * len(images or [])
        return torch.zeros(n, 8), torch.ones(n, dtype=torch.long)

    def build(**kw):
        target_rows = kw["latent_t"] * _FAKE_FRAME_ROWS
        branch = SimpleNamespace(
            update_mask_dev=torch.ones(target_rows, dtype=torch.bool),
            audio_update_mask_dev=torch.ones(2 * kw["audio_t"], dtype=torch.bool),
            audio_update_mask=torch.ones(2 * kw["audio_t"], dtype=torch.bool),
        )
        inputs: dict[str, Any] = {
            "branch": branch,
            "video_rows": torch.zeros(target_rows, 96),
            "audio_rows": torch.zeros(2 * kw["audio_t"], 32),
            "audio_anchor": None,
        }
        calls["build"].append((kw, inputs))
        return inputs

    def run_window_denoise(*, inputs, transformer, latent_t, latent_h, latent_w, audio_t, on_step=None):
        # Drive the step callback once on rows of 7.0 so the history re-injection
        # wiring is observable; the "denoised" video latent encodes its own
        # temporal index (latent t has the constant value t / 10) so the held
        # slice is identifiable.
        rows = torch.full((int(inputs["branch"].update_mask_dev.shape[0]), 96), 7.0)
        audio_rows = torch.full((2 * audio_t, 32), 7.0)
        if on_step is not None:
            on_step(0, rows, audio_rows)
        calls["step_rows"].append((rows, audio_rows))
        # Unpacked video latent is (B, 24, T, latent_h, latent_w); audio latent is (channels=2, 32, T).
        latent = (torch.arange(latent_t, dtype=torch.float32) / 10).view(1, 1, latent_t, 1, 1)
        return latent.expand(1, 24, latent_t, latent_h, latent_w).clone(), torch.zeros(2, 32, audio_t)

    def decode(video_latent, audio_latent, *, height, width):
        # Decoded video is (B, C, T, height, width) in [0, 1], cropped to the request canvas.
        frames = MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(int(video_latent.shape[2]))
        video = torch.arange(frames, dtype=torch.float32).div(1000).view(1, 1, frames, 1, 1)
        calls["decode"].append(frames)
        audio = torch.full((1, 2, int(audio_latent.shape[2]) * 800), float(audio_levels[len(calls["decode"]) - 1]))
        return video.expand(1, 3, frames, height, width).clone(), audio

    def encode_image(image):
        # The handoff still must be a canvas-sized picture (PIL size is (width, height)).
        calls["encode_image"].append((image.getpixel((0, 0))[0], image.size))
        return torch.full((_FAKE_FRAME_ROWS, 96), float(image.getpixel((0, 0))[0]))

    @contextmanager
    def ctx(*args, **kwargs):
        yield SimpleNamespace(update=lambda: None)

    transformer = object()
    fake = SimpleNamespace(
        text_encoder=text_encoder,
        transformer=transformer,
        _transformer_for_task=lambda task: transformer,
        progress_bar=ctx,
        _resident_dit_layers_on_device=ctx,
        _component_on_device=ctx,
        _build_denoise_inputs=build,
        _run_window_denoise=run_window_denoise,
        decode=decode,
        video_vae=SimpleNamespace(encode_image=encode_image),
        encode_prompt=encode_prompt,
        device=torch.device("cpu"),
    )
    images = [_fake_image(v) for v in image_values]
    visual_condition = (
        torch.cat([torch.full((_FAKE_FRAME_ROWS, 96), float(v)) for v in image_values]) if image_values else None
    )
    request_text = (
        torch.zeros(24 + 100 * len(image_values), 8),
        torch.ones(24 + 100 * len(image_values), dtype=torch.long),
    )
    video, audio = MiniMaxH3Pipeline._generate_windowed(
        fake,
        task=task,
        text_embeddings=request_text[0],
        text_tags=request_text[1],
        seed=7,
        latent_t=plan.window_latent_t,
        latent_h=_FAKE_LATENT_H,
        latent_w=_FAKE_LATENT_W,
        audio_t=plan.window_audio_t,
        num_frames=plan.window_num_frames,
        num_steps=4,
        video_shift=5.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=visual_condition,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=None,
        visual_condition_shapes=[(1, _FAKE_LATENT_H, _FAKE_LATENT_W)] * len(image_values) or None,
        audio_condition_lengths=None,
        keyframe_frame_indices=keyframes,
        windowing=plan,
        prompt="a coast",
        images=images,
        height=_FAKE_HEIGHT,
        width=_FAKE_WIDTH,
    )
    return plan, calls, video, audio


def _text_len(kw) -> int:
    assert kw["text_embeddings"].shape[0] == kw["text_tags"].shape[0]
    return int(kw["text_embeddings"].shape[0])


def test_generate_windowed_t2va_hands_off_frame_306_as_a_first_frame_request():
    plan, calls, video, audio = _run_fake_windowed(task="t2va", keyframes=None, image_values=[])
    # Window 0 keeps the request text (no re-encode); window 1 is a first-frame
    # fl2va request around the decoded handoff frame 306 (pixel 0.306 -> 78).
    assert calls["encode_prompt"] == [("fl2va", [78])]
    assert calls["encode_image"] == [(78, (_FAKE_WIDTH, _FAKE_HEIGHT))]
    kw0, _ = calls["build"][0]
    kw1, inputs1 = calls["build"][1]
    assert kw0["keyframe_frame_indices"] is None and kw0["visual_condition"] is None
    assert kw0["seed"] == 7 and kw1["seed"] == 8
    # The text each window denoises with is the text encoded for that window:
    # the request's 24 tokens for window 0, the one-picture fl2va encoding after.
    assert _text_len(kw0) == 24 and _text_len(kw1) == 124
    assert kw1["keyframe_frame_indices"] == [0]
    assert kw1["visual_condition_shapes"] == [(1, _FAKE_LATENT_H, _FAKE_LATENT_W)]
    assert bool((kw1["visual_condition"] == 78.0).all())
    assert kw1["num_frames"] == plan.window_num_frames and kw1["latent_t"] == plan.window_latent_t
    # No audio rows are pinned or conditioned: full audio_t for every window.
    assert bool(inputs1["branch"].audio_update_mask_dev.all()) and inputs1["audio_anchor"] is None
    assert kw1["audio_t"] == plan.window_audio_t
    # The first two latents of the shared span (window 0's latents 90 and 91 of
    # 107, valued 9.0 and 9.1) are held in window 1's leading target rows at the
    # step's sigma over the window's initial noise (zeros here); window 0 and
    # the rows past the history are left to the denoiser (7.0).
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import minimax_h3_time_shift_sigmas

    sigma_1 = minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=5.0, base_schedule=None)[1]
    span_start_t = plan.window_latent_t - plan.overlap_latent_t
    assert span_start_t == 90
    (rows0, audio0), (rows1, audio1) = calls["step_rows"]
    assert bool((rows0 == 7.0).all()) and bool((audio0 == 7.0).all())
    for k in range(2):  # MINIMAX_H3_HISTORY_HOLD_LATENTS
        held = rows1[k * _FAKE_FRAME_ROWS : (k + 1) * _FAKE_FRAME_ROWS]
        torch.testing.assert_close(held, torch.full_like(held, (1.0 - sigma_1) * (span_start_t + k) / 10))
    assert bool((rows1[2 * _FAKE_FRAME_ROWS :] == 7.0).all())
    # The first 0.5 s (20 latents) of the span's audio is held too, in BOTH
    # channel blocks (channel-major rows); the fake tail and noise are zeros.
    wa = plan.window_audio_t
    held_audio = torch.cat([audio1[:20], audio1[wa : wa + 20]])
    assert bool((held_audio == 0.0).all())
    assert bool((audio1[20:wa] == 7.0).all()) and bool((audio1[wa + 20 :] == 7.0).all())
    # Output: 362 + 306 frames; audio 15.075 s + (15.075 - 2.325) s.
    assert video.shape[2] == plan.total_num_frames == 668
    assert audio.shape[-1] == (2 * plan.window_audio_t - plan.overlap_audio_t) * 800
    # Audio: window 0 (1.0) through the held 5 frames (6667 samples of the
    # span), an equal-power fade to window 1 (2.0) over half a second, then
    # window 1's rendering.
    span_start = (plan.window_audio_t - plan.overlap_audio_t) * 800
    hold_samples, fade_samples = 6667, 16000
    assert bool((audio[0, 0, : span_start + hold_samples] == 1.0).all())
    assert bool((audio[0, 0, span_start + hold_samples + fade_samples :] == 2.0).all())
    fade = audio[0, 0, span_start + hold_samples : span_start + hold_samples + fade_samples]
    assert abs(float(fade[0]) - 1.0) < 1e-3 and abs(float(fade[-1]) - 2.0) < 1e-3
    # Equal-power weights (cos, sin) do not sum to 1: two correlated constants
    # peak at sqrt(1^2 + 2^2) where theta = atan(2), which is the intended
    # loudness behaviour for decorrelated takes.
    assert abs(float(fade.max()) - 5**0.5) < 1e-3 and float(fade.min()) >= 1.0 - 1e-4
    # Frames 0..310 are window 0 (306..310 are the held head of the span);
    # 311..322 cross-fade to window 1's frames 5..16 over half a second; from
    # 323 on the output is window 1's rendering (its frame 17 onwards).
    span = plan.window_num_frames - 306
    assert span == 56
    torch.testing.assert_close(video[0, 0, :311, 0, 0], torch.arange(311, dtype=torch.float32) / 1000)
    alpha = _smoothstep_alpha(12)
    expected = (1 - alpha) * (torch.arange(311, 323, dtype=torch.float32) / 1000) + alpha * (
        torch.arange(5, 17, dtype=torch.float32) / 1000
    )
    torch.testing.assert_close(video[0, 0, 311:323, 0, 0], expected)
    torch.testing.assert_close(video[0, 0, 323:363, 0, 0], torch.arange(17, 57, dtype=torch.float32) / 1000)


def test_generate_windowed_fl2va_keeps_user_keyframes_paired_with_their_text():
    # [0, -1]: window 0 anchors image 0 only (re-encoded with only that
    # picture); the final window anchors [handoff, image -1] in that order.
    _, calls, _, _ = _run_fake_windowed(task="fl2va", keyframes=[0, -1], image_values=[10, 20])
    assert calls["encode_prompt"] == [("fl2va", [10]), ("fl2va", [78, 20])]
    kw0, _ = calls["build"][0]
    kw1, _ = calls["build"][1]
    assert _text_len(kw0) == 124 and _text_len(kw1) == 224
    assert kw0["keyframe_frame_indices"] == [0]
    assert bool((kw0["visual_condition"] == 10.0).all()) and kw0["visual_condition"].shape[0] == _FAKE_FRAME_ROWS
    assert kw1["keyframe_frame_indices"] == [0, -1]
    assert kw1["visual_condition_shapes"] == [(1, _FAKE_LATENT_H, _FAKE_LATENT_W)] * 2
    assert bool((kw1["visual_condition"][:_FAKE_FRAME_ROWS] == 78.0).all())
    assert bool((kw1["visual_condition"][_FAKE_FRAME_ROWS:] == 20.0).all())

    # [-1] only: window 0 becomes a plain t2va window; the last frame moves to
    # the final window behind the handoff still.
    _, calls, _, _ = _run_fake_windowed(task="fl2va", keyframes=[-1], image_values=[20])
    assert calls["encode_prompt"] == [("t2va", []), ("fl2va", [78, 20])]
    kw0, _ = calls["build"][0]
    kw1, _ = calls["build"][1]
    assert kw0["keyframe_frame_indices"] is None and kw0["visual_condition"] is None
    assert _text_len(kw0) == 24 and _text_len(kw1) == 224

    # [0] only: window 0 is exactly the request; window 1 anchors the handoff.
    _, calls, _, _ = _run_fake_windowed(task="fl2va", keyframes=[0], image_values=[10])
    assert calls["encode_prompt"] == [("fl2va", [78])]
    kw0, _ = calls["build"][0]
    kw1, _ = calls["build"][1]
    assert kw0["keyframe_frame_indices"] == [0] and bool((kw0["visual_condition"] == 10.0).all())
    assert _text_len(kw0) == 124 and _text_len(kw1) == 124


def test_generate_windowed_three_windows_chain_handoffs_and_fades():
    plan, calls, video, audio = _run_fake_windowed(task="t2va", keyframes=None, image_values=[], num_segments=3)
    assert plan.num_windows == 3
    # Each continuation hands off frame 306 of its predecessor and is a
    # first-frame fl2va request; the window before it has already been
    # blended once, and the splice still fits (contribution >= overlap).
    assert calls["encode_prompt"] == [("fl2va", [78]), ("fl2va", [78])]
    assert [kw["seed"] for kw, _ in calls["build"]] == [7, 8, 9]
    assert video.shape[2] == plan.total_num_frames == 362 + 2 * 306 == 974
    assert audio.shape[-1] == (3 * plan.window_audio_t - 2 * plan.overlap_audio_t) * 800
    # Frames past both hand-overs come from the last window (its frames 17..).
    assert round(float(video[0, 0, -1, 0, 0]) * 1000) == plan.window_num_frames - 1
    assert bool((audio[0, 0, -100:] == 3.0).all())
    # Window 1 contributes its frames 56..361 as output frames 362..667, so its
    # last 56 frames (306..361) sit at output 612..667: that is where window 2's
    # span is spliced in. Output 616 is still window 1's frame 310 (held), and
    # from output 629 on the video is window 2's rendering (its frame 17..).
    second_span_start = 362 + (306 - 56)
    assert second_span_start == 612
    assert round(float(video[0, 0, second_span_start + 4, 0, 0]) * 1000) == 310
    assert round(float(video[0, 0, second_span_start + 17, 0, 0]) * 1000) == 17


def test_generate_windowed_lifts_a_quiet_continuation_without_a_step():
    """A continuation whose audio is much quieter than the previous window is
    lifted towards the previous level (measured before the splice), the gain is
    applied before splicing so there is no step at the hand-over, and the lift
    has released by four seconds into the new window."""
    plan, _, _, audio = _run_fake_windowed(task="t2va", keyframes=None, image_values=[], audio_levels=(1.0, 0.1))
    sr = 32000
    span_start = (plan.window_audio_t - plan.overlap_audio_t) * 800
    span_end = plan.window_audio_t * 800
    # Held frames keep the previous window; the fade blends 1.0 with the lifted
    # new audio (0.1 x 8 = 0.8 for the first two seconds, then the lift starts
    # releasing: 0.1 x 6.86 at the span end), so nothing in the span drops
    # below ~0.68.
    assert bool((audio[0, 0, : span_start + 6667] == 1.0).all())
    assert float(audio[0, 0, span_start + 6667 : span_start + 2 * sr].min()) >= 0.79
    assert float(audio[0, 0, span_start + 2 * sr : span_end].min()) >= 0.66
    # No step at the concatenation point: the same lifted rendering continues.
    assert abs(float(audio[0, 0, span_end]) - float(audio[0, 0, span_end - 1])) < 0.01
    # Lifted at the start of the new window's own contribution (2.325 s in, mid release)...
    assert 0.5 < float(audio[0, 0, span_end]) < 0.8
    # ...and back to the take's own level once the release is over (4 s into the new window).
    assert abs(float(audio[0, 0, span_start + 4 * sr + 100]) - 0.1) < 1e-4


def test_generate_windowed_smallest_overlap_hands_over_at_the_held_frames():
    """With a 5-frame overlap (2 latents) the whole span is held, the fade
    clamps to nothing, and the new window starts right after the held frames."""
    plan, calls, video, audio = _run_fake_windowed(task="t2va", keyframes=None, image_values=[], overlap_frames=5)
    assert plan.overlap_latent_t == 2 and plan.overlap_frames == 5 and plan.total_num_frames == 719
    # The handoff still is frame 357 (0.357 -> 91); the held slice is latents 105, 106.
    assert calls["encode_image"][0][0] == 91
    _, (rows1, audio1) = calls["step_rows"]
    assert bool((rows1[2 * _FAKE_FRAME_ROWS :] == 7.0).all()) and bool((rows1[: 2 * _FAKE_FRAME_ROWS] != 7.0).all())
    # The audio hold clamps to the 8-latent audio overlap, in both channel blocks.
    wa = plan.window_audio_t
    assert bool((audio1[:8] == 0.0).all()) and bool((audio1[8:wa] == 7.0).all())
    assert bool((audio1[wa : wa + 8] == 0.0).all()) and bool((audio1[wa + 8 :] == 7.0).all())
    torch.testing.assert_close(video[0, 0, :362, 0, 0], torch.arange(362, dtype=torch.float32) / 1000)
    assert round(float(video[0, 0, 362, 0, 0]) * 1000) == 5
    span_start = (plan.window_audio_t - plan.overlap_audio_t) * 800
    assert bool((audio[0, 0, : span_start + 6400] == 1.0).all()) and bool(
        (audio[0, 0, span_start + 6400 :] == 2.0).all()
    )


def test_history_release_frees_the_last_four_updates_of_the_default_schedule():
    """With the default 50-point schedule (49 updates) and video shift 5.0, the
    history is held through the update ending at sigma 0.3077 and released for
    the four updates below MINIMAX_H3_HISTORY_RELEASE_SIGMA."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_HISTORY_RELEASE_SIGMA,
        minimax_h3_time_shift_sigmas,
    )

    sigmas = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=5.0, base_schedule=None)
    assert len(sigmas) == 50 and sigmas[0] == 1.0 and sigmas[-1] == 0.0
    released = [s for s in sigmas[1:] if s < MINIMAX_H3_HISTORY_RELEASE_SIGMA]
    assert len(released) == 4


def test_windowing_overlap_is_capped_at_half_the_window():
    """A continuation window must contribute at least the shared span it
    splices into its predecessor, so the overlap never exceeds half the window
    (short windows or large overlap requests would otherwise fail the splice
    from the third window on)."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    short = _resolve_minimax_h3_windowing(
        duration=12.0, fps=24, num_segments=3, overlap_frames=None, window_duration=4.0
    )
    assert 2 * short.overlap_latent_t <= short.window_latent_t
    assert short.window_num_frames - short.overlap_frames >= short.overlap_frames
    wide = _resolve_minimax_h3_windowing(
        duration=30.0, fps=24, num_segments=2, overlap_frames=209, window_duration=None
    )
    assert wide.overlap_latent_t == 47  # largest value <= 107 // 2 on the 2 (mod 15) grid
    assert wide.window_num_frames - wide.overlap_frames >= wide.overlap_frames


def test_build_denoise_inputs_keyframe_segment_follows_the_indices_not_the_task():
    """A t2va request's continuation window carries a [0] keyframe (the handoff
    still); the packed layout must reserve its condition rows or the denoise
    loop rejects the anchor (``keyframe_cond_rows != layout cond rows``)."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    latent_t, latent_h, latent_w, audio_t, num_frames = 17, 2, 4, 93, 56
    frame_rows = (latent_h // 2) * (latent_w // 2)
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        _initial_noise=lambda **kw: MiniMaxH3Pipeline._initial_noise(None, **kw),
    )

    def build(task, indices):
        return MiniMaxH3Pipeline._build_denoise_inputs(
            fake,
            task=task,
            text_embeddings=torch.zeros(24, 8),
            text_tags=torch.ones(24, dtype=torch.long),
            seed=3,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            num_frames=num_frames,
            num_steps=4,
            video_shift=5.0,
            audio_shift=3.0,
            base_schedule=None,
            visual_condition=None if indices is None else torch.zeros(len(indices) * frame_rows, 96),
            visual_condition_shape=None,
            audio_condition=None,
            ref_audio_t=None,
            visual_condition_shapes=None if indices is None else [(1, latent_h, latent_w)] * len(indices),
            keyframe_frame_indices=indices,
        )

    continuation = build("t2va", [0])
    frozen = int((~continuation["branch"].update_mask).sum())
    assert continuation["cond_anchor"].shape[0] == frame_rows
    assert frozen == frame_rows, "the layout must reserve rows for the handoff still"
    # ...and it is the same layout a first-frame fl2va request gets.
    first_frame = build("fl2va", [0])
    torch.testing.assert_close(first_frame["branch"].update_mask, continuation["branch"].update_mask)
    # Single-window t2va (no indices) is unchanged: no condition rows at all.
    plain = build("t2va", None)
    assert plain["cond_anchor"] is None
    assert int((~plain["branch"].update_mask).sum()) == 0


def test_generate_windowed_requires_a_local_text_encoder():
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError):
        _run_fake_windowed(task="t2va", keyframes=None, image_values=[], text_encoder=None)
