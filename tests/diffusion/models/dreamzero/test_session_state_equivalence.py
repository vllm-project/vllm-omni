# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-equivalence tests: DreamZeroStateAdapter vs bespoke DreamZeroState.

These drive both state objects directly with tiny CPU tensors -- no model, no
GPU -- and assert the adapter (backed by the SessionStateManager) stores and
returns exactly what the bespoke DreamZeroState does. This gates routing
DreamZero through the shared session state manager (RFC #4480).

DreamZero's attention KV is engine-owned (AR-Diffusion paged pool, PR #4534)
and out of the adapter's scope; the state surface tested here is the model's
non-KV session state.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.state_dreamzero import (
    FRAMES_PER_CHUNK,
    DreamZeroState,
)
from vllm_omni.experimental.world_models.adapters.state_dreamzero_adapter import DreamZeroStateAdapter
from vllm_omni.experimental.world_models.session_state import SessionStateManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _create_both() -> tuple[DreamZeroState, DreamZeroStateAdapter]:
    return DreamZeroState(), DreamZeroStateAdapter("session-0", SessionStateManager())


def test_accumulate_frames_and_call_count_match() -> None:
    bespoke, adapter = _create_both()
    rng = np.random.default_rng(0)
    for _ in range(2 * FRAMES_PER_CHUNK + 1):
        frame = rng.integers(0, 255, size=(5, 6, 3), dtype=np.uint8)
        b_out = bespoke.accumulate_frames(frame)
        a_out = adapter.accumulate_frames(frame)
        np.testing.assert_array_equal(b_out, a_out)
        assert bespoke.call_count == adapter.call_count
        assert len(bespoke.stitched_buffer) == len(adapter.stitched_buffer)
    # Deque rollover capped at FRAMES_PER_CHUNK.
    assert len(adapter.stitched_buffer) == FRAMES_PER_CHUNK


def test_accumulate_frames_4d_extend_matches() -> None:
    bespoke, adapter = _create_both()
    clip = np.random.default_rng(1).integers(0, 255, size=(3, 5, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(bespoke.accumulate_frames(clip), adapter.accumulate_frames(clip))
    assert len(bespoke.stitched_buffer) == len(adapter.stitched_buffer)


def test_video_latents_roundtrip_matches() -> None:
    bespoke, adapter = _create_both()
    assert bespoke.get_concatenated_video_latents() is None
    assert adapter.get_concatenated_video_latents() is None

    rng = torch.Generator().manual_seed(0)
    for _ in range(3):
        # (B, T, C, H, W) as returned by the denoise loop.
        chunk = torch.randn(1, 2, 3, 4, 4, generator=rng)
        bespoke.append_video_latents(chunk)
        adapter.append_video_latents(chunk)
        torch.testing.assert_close(
            bespoke.get_concatenated_video_latents(),
            adapter.get_concatenated_video_latents(),
        )

    bespoke.clear_video_latents()
    adapter.clear_video_latents()
    assert bespoke.get_concatenated_video_latents() is None
    assert adapter.get_concatenated_video_latents() is None


def test_vae_stream_fields_match_bespoke() -> None:
    bespoke, adapter = _create_both()
    for state in (bespoke, adapter):
        assert state.vae_stream_initialized is False
        assert state.vae_enc_feat_map is None
        assert state.vae_encoder_out is None
        assert state.vae_pending_body_frames is None

    feat_map: list[torch.Tensor | None] = [torch.ones(1, 2), None]
    encoder_out = torch.randn(1, 4, 2, 3, 3)
    pending = torch.randn(1, 3, 2, 8, 8)
    for state in (bespoke, adapter):
        state.vae_enc_feat_map = feat_map
        state.vae_encoder_out = encoder_out
        state.vae_pending_body_frames = pending
        state.vae_stream_initialized = True

    assert adapter.vae_stream_initialized is bespoke.vae_stream_initialized is True
    torch.testing.assert_close(adapter.vae_encoder_out, bespoke.vae_encoder_out)
    torch.testing.assert_close(adapter.vae_pending_body_frames, bespoke.vae_pending_body_frames)
    assert adapter.vae_enc_feat_map is feat_map

    for state in (bespoke, adapter):
        state.reset_vae_encoder_stream()
        assert state.vae_stream_initialized is False
        assert state.vae_enc_feat_map is None
        assert state.vae_encoder_out is None
        assert state.vae_pending_body_frames is None


def test_reset_clears_state_and_should_reset_matches() -> None:
    bespoke, adapter = _create_both()
    tokens = torch.tensor([1, 2, 3])
    # language unset -> both want reset.
    assert bespoke.should_reset(tokens, 4, -1) is adapter.should_reset(tokens, 4, -1) is True

    for state in (bespoke, adapter):
        state.language = tokens
        state.current_start_frame = 2
    # same language -> no reset; both agree.
    assert bespoke.should_reset(tokens, 4, -1) == adapter.should_reset(tokens, 4, -1) is False
    # language change -> reset.
    other = torch.tensor([9, 9])
    assert bespoke.should_reset(other, 4, -1) == adapter.should_reset(other, 4, -1) is True
    # local_attn_size exceeded -> reset ("inference", not "session").
    assert bespoke.reset_reason(tokens, 4, 1) == adapter.reset_reason(tokens, 4, 1) == "inference"

    # A session reset clears everything, including the prompt-embed cache and
    # the VAE encoder stream (mirrors DreamZeroState.reset()).
    for state in (bespoke, adapter):
        state.prompt_embeds = torch.ones(1, 2)
        state.vae_encoder_out = torch.ones(1, 4, 1, 2, 2)
        state.vae_stream_initialized = True
        state.reset()
        assert state.language is None
        assert state.prompt_embeds is None
        assert state.current_start_frame == 0
        assert state.call_count == 0
        assert state.vae_stream_initialized is False
        assert state.vae_encoder_out is None


def test_window_reset_preserves_latents_prompt_and_vae_stream() -> None:
    # reset(clear_video_latents=False) is the window ("inference") reset: the
    # prompt is unchanged and the VAE feat_cache history is independent of the
    # DiT attention window, so both survive alongside the video latents.
    bespoke, adapter = _create_both()
    tokens = torch.tensor([1, 2, 3])
    embeds = torch.randn(1, 4)
    encoder_out = torch.randn(1, 4, 2, 3, 3)
    chunk = torch.randn(1, 2, 3, 4, 4)
    for state in (bespoke, adapter):
        state.language = tokens
        state.prompt_embeds = embeds
        state.vae_encoder_out = encoder_out
        state.vae_stream_initialized = True
        state.current_start_frame = 7
        state.append_video_latents(chunk)

        state.reset_inference_state()

        assert state.current_start_frame == 0
        assert state.call_count == 0
        assert state.clip_feas is None
        torch.testing.assert_close(state.language, tokens)
        torch.testing.assert_close(state.prompt_embeds, embeds)
        assert state.vae_stream_initialized is True
        torch.testing.assert_close(state.vae_encoder_out, encoder_out)
        assert state.get_concatenated_video_latents() is not None

    torch.testing.assert_close(
        bespoke.get_concatenated_video_latents(),
        adapter.get_concatenated_video_latents(),
    )


def test_metadata_fields_persist_across_adapter_instances() -> None:
    # A fresh adapter for the same session must see prior metadata (the manager
    # is the single source of truth).
    manager = SessionStateManager()
    first = DreamZeroStateAdapter("s", manager)
    first.current_start_frame = 5
    first.clip_feas = torch.ones(2, 2)
    first.append_video_latents(torch.ones(1, 2, 3, 4, 4))

    second = DreamZeroStateAdapter("s", manager)
    assert second.current_start_frame == 5
    torch.testing.assert_close(second.clip_feas, torch.ones(2, 2))
    # Video latents appended via `first` are visible via `second`.
    assert second.get_concatenated_video_latents() is not None


def test_session_lru_caps_retained_sessions() -> None:
    manager = SessionStateManager(max_sessions=3)
    for i in range(5):
        DreamZeroStateAdapter(f"s{i}", manager)
    assert len(manager) == 3
    # Oldest two evicted; newest three retained.
    assert "s0" not in manager
    assert "s1" not in manager
    for i in (2, 3, 4):
        assert f"s{i}" in manager


def test_adapter_keeps_state_when_its_session_is_evicted() -> None:
    # Regression: an adapter mid-generation must not lose its data when the
    # manager evicts its session to make room for newer ones. The adapter pins
    # the session, so its stored latents survive eviction from the lookup table.
    manager = SessionStateManager(max_sessions=2)
    adapter = DreamZeroStateAdapter("active", manager)
    chunk = torch.randn(1, 2, 3, 4, 4)
    adapter.append_video_latents(chunk)
    adapter.language = torch.tensor([1, 2, 3])

    # Push "active" out of the manager's table with newer sessions.
    for i in range(3):
        DreamZeroStateAdapter(f"other{i}", manager)
    assert "active" not in manager

    # The in-use adapter still returns its own data.
    assert adapter.get_concatenated_video_latents() is not None
    torch.testing.assert_close(adapter.language, torch.tensor([1, 2, 3]))


def test_session_reset_clears_metadata() -> None:
    # Regression: resetting a session must not leave stale metadata behind, so a
    # fresh adapter for the same session sees defaults, not the old values.
    manager = SessionStateManager()
    adapter = DreamZeroStateAdapter("s", manager)
    adapter.language = torch.tensor([1, 2, 3])
    adapter.current_start_frame = 5
    adapter.clip_feas = torch.ones(2, 2)
    adapter.prompt_embeds = torch.ones(1, 2)

    adapter.reset()

    fresh = DreamZeroStateAdapter("s", manager)
    assert fresh.language is None
    assert fresh.current_start_frame == 0
    assert fresh.clip_feas is None
    assert fresh.prompt_embeds is None


def test_bespoke_path_without_memory_manager() -> None:
    # Regression: a pipeline built via __new__ (lightweight fixtures) seeds only
    # the bespoke fields and never sets _memory_manager; _get_or_create_state
    # must fall through to the bespoke DreamZeroState path without AttributeError.
    from collections import OrderedDict

    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import (
        MAX_DREAMZERO_SESSIONS,
        DreamZeroPipeline,
    )

    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._max_session_states = MAX_DREAMZERO_SESSIONS

    state = DreamZeroPipeline._get_or_create_state(pipe, "x")
    assert isinstance(state, DreamZeroState)
