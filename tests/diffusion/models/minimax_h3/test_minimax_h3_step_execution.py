# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 step-wise execution (continuous batching) contract tests.

These run on CPU against a stand-in DiT, so they cover the packing, the
per-request scheduler math, and the runner-facing state wiring without needing
checkpoint weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.attention.test_attention_schedule_candidates import _make_metadata_candidate
from vllm_omni.diffusion.worker.input_batch import InputBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_HIDDEN = 8


class _SegmentMeanModel:
    """Stand-in DiT whose output depends on packed-document boundaries.

    Every row is mixed with the mean of its own ``cu_seqlens`` document and with
    its own timestep, so a wrong document boundary, row offset, or timestep
    assignment shows up as a numeric difference instead of passing silently.
    """

    def __init__(self):
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, **kwargs):
        x = kwargs["x"][0]
        audio_x = kwargs["audio_x"][0]
        bounds = kwargs["packed_seq_params"]["cu_seqlens_q"].tolist()
        if kwargs["packed_seq_params"].get("num_requests", 1) > 1:
            assert all(start < stop for start, stop in zip(bounds[:-1], bounds[1:]))
        img_pos = kwargs["img_pos_info"]["position_ids"]
        audio_pos = kwargs["audio_pos_info"]["position_ids"]

        pooled_video = torch.zeros_like(x)
        pooled_audio = torch.zeros_like(audio_x)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop <= start:
                continue
            pooled_video[start:stop] = x[start:stop].mean(dim=0, keepdim=True)
            pooled_audio[start:stop] = audio_x[start:stop].mean(dim=0, keepdim=True)

        row_timesteps = kwargs["unique_timesteps"][kwargs["inverse_indices"]].unsqueeze(-1)
        self.calls.append(
            {
                "video_rows": x[img_pos].clone(),
                "audio_rows": audio_x[audio_pos].clone(),
                "video_timesteps": row_timesteps[img_pos, 0].clone(),
                "audio_timesteps": row_timesteps[audio_pos, 0].clone(),
            }
        )
        video = (pooled_video + 0.5 * x + row_timesteps)[img_pos]
        audio = (pooled_audio + 0.5 * audio_x + row_timesteps)[audio_pos]
        if not kwargs.get("skip_mask_out_condition", False):
            video = video * kwargs["update_mask"].view(-1).unsqueeze(-1)
        return video, audio


def _make_branch(*, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int, seed: int):
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence

    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    generator = torch.Generator().manual_seed(seed)
    text_embeddings = torch.randn(text_len, _HIDDEN, generator=generator, dtype=torch.float32)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=text_embeddings,
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    video_rows = torch.randn(int(branch.img_pos.shape[0]), 96, generator=generator, dtype=torch.float32)
    audio_rows = torch.randn(int(branch.audio_pos.shape[0]), 32, generator=generator, dtype=torch.float32)
    return branch, video_rows, audio_rows


def _sigmas(num_steps: int, shift: float) -> list[float]:
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    return minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=shift)


def _make_state(request_id: str, model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    state = StepRequestState(
        request_id=request_id, sampling=OmniDiffusionSamplingParams(guidance_scale=1.0, true_cfg_scale=4.0)
    )
    state.latents = video_rows.clone()
    state.timesteps = torch.tensor([1.0 - sigma for sigma in sigmas_video[:-1]], dtype=torch.float32)
    state.step_index = 0
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    state.extra = {
        mod._STEP_VIDEO_SOLVER: create_h3_sample_solver("euler", sigmas_video),
        mod._STEP_AUDIO_SOLVER: create_h3_sample_solver("euler", sigmas_audio),
        mod._STEP_BRANCH: branch,
        # Co-batched requests must share one DiT instance, or denoise_step()
        # treats the batch as mixed-task and falls back to one forward each.
        mod._STEP_TRANSFORMER: model,
        mod._STEP_AUDIO_ROWS: audio_rows.clone(),
        mod._STEP_COND_ANCHOR: None,
        mod._STEP_AUDIO_ANCHOR: None,
        mod._STEP_SIGMAS_VIDEO: sigmas_video,
        mod._STEP_SIGMAS_AUDIO: sigmas_audio,
    }
    return state


def _step_pipeline(model, *, packed_batch_supported: bool = True):
    """A pipeline instance carrying only what the step methods touch."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    pipeline.transformer = model
    pipeline.device = torch.device("cpu")
    pipeline._transformer_for_task = lambda task: model
    pipeline._packed_batch_supported = lambda transformer: packed_batch_supported
    return pipeline


@pytest.mark.parametrize("num_steps", [1, 8, 50])
def test_step_execution_matches_request_mode_denoise_loop(num_steps, mocker):
    """Stepping through the contract must reproduce the request-mode loop."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    model = mocker.Mock(wraps=_SegmentMeanModel())
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=5)
    sigmas_video = _sigmas(num_steps, 12.0)
    sigmas_audio = _sigmas(num_steps, 3.0)

    reference_video, reference_audio = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video_rows,
        initial_audio_rows=audio_rows,
        keyframe_cond_rows=None,
        sigmas_video=sigmas_video,
        sigmas_audio=sigmas_audio,
        device=torch.device("cpu"),
    )

    assert model.call_count == num_steps
    model.reset_mock()
    pipeline = _step_pipeline(model)
    state = _make_state("req-0", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
    input_batch = InputBatch.make_batch([state])

    steps = 0
    while not state.denoise_completed:
        noise_pred = pipeline.denoise_step(input_batch, states=[state])
        pipeline.step_scheduler(state, noise_pred)
        steps += 1

    assert steps == num_steps
    assert model.call_count == num_steps
    assert state.total_steps == steps
    torch.testing.assert_close(state.latents, reference_video)
    torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], reference_audio)


def test_request_mode_cancellation_stops_before_next_denoise_step(monkeypatch):
    from vllm_omni.diffusion.cancellation import RequestCancellationRegistry, request_cancellation_scope
    from vllm_omni.diffusion.data import DiffusionRequestAbortedError
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
    from vllm_omni.platforms import current_omni_platform

    # This test runs real packing/Euler updates on CPU with the small DiT above.
    monkeypatch.setattr(current_omni_platform, "synchronize", lambda: None)
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=5)
    registry = RequestCancellationRegistry()
    signal = registry.create("request")
    steps = []

    def cancel_after_first_step(step, video, audio):
        steps.append(step)
        registry.cancel(["request"])

    try:
        with request_cancellation_scope([signal]), pytest.raises(DiffusionRequestAbortedError):
            minimax_h3_denoise_loop(
                model=_SegmentMeanModel(),
                positive=branch,
                initial_video_rows=video_rows,
                initial_audio_rows=audio_rows,
                keyframe_cond_rows=None,
                sigmas_video=_sigmas(6, 12.0),
                sigmas_audio=_sigmas(6, 3.0),
                device=torch.device("cpu"),
                on_step=cancel_after_first_step,
            )
    finally:
        registry.close()
    assert steps == [0]


def test_step_execution_matches_request_mode_with_latent_edits():
    """Masked model rows, row timesteps, and scheduler math match both paths."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import MiniMaxH3LatentEdit

    model = _SegmentMeanModel()
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=51)
    sigmas_video = _sigmas(5, 12.0)
    sigmas_audio = _sigmas(5, 3.0)
    video_clean = torch.linspace(-1.0, 1.0, video_rows.numel()).reshape_as(video_rows)
    audio_clean = torch.linspace(1.0, -1.0, audio_rows.numel()).reshape_as(audio_rows)
    video_mask = torch.linspace(0.0, 1.0, video_rows.shape[0])
    video_edit = MiniMaxH3LatentEdit.from_rows(
        video_clean,
        0.999 * video_clean + 0.001 * video_rows,
        video_mask,
        video_mask,
    )
    audio_mask = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    audio_edit = MiniMaxH3LatentEdit.from_rows(
        audio_clean,
        audio_clean,
        audio_mask,
        audio_mask,
    )

    reference_video, reference_audio = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video_rows,
        initial_audio_rows=audio_rows,
        keyframe_cond_rows=None,
        video_edit=video_edit,
        audio_edit=audio_edit,
        sigmas_video=sigmas_video,
        sigmas_audio=sigmas_audio,
        device=torch.device("cpu"),
    )
    request_first_call = model.calls[0]
    torch.testing.assert_close(request_first_call["video_rows"], video_edit.model_rows(video_rows))
    torch.testing.assert_close(request_first_call["audio_rows"], audio_edit.model_rows(audio_rows))
    torch.testing.assert_close(
        request_first_call["video_timesteps"],
        video_edit.target_timesteps(1.0 - sigmas_video[0], 0.999, sigma=sigmas_video[0]),
    )
    torch.testing.assert_close(
        request_first_call["audio_timesteps"],
        audio_edit.target_timesteps(1.0 - sigmas_audio[0], 1.0, sigma=sigmas_audio[0]),
    )
    model.calls.clear()

    state = _make_state("req-edit", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
    state.extra[mod._STEP_VIDEO_EDIT] = video_edit
    state.extra[mod._STEP_AUDIO_EDIT] = audio_edit
    pipeline = _step_pipeline(model)
    while not state.denoise_completed:
        velocity = pipeline.denoise_step(InputBatch.make_batch([state]), states=[state])
        pipeline.step_scheduler(state, velocity)

    for name, expected in request_first_call.items():
        torch.testing.assert_close(model.calls[0][name], expected)
    torch.testing.assert_close(state.latents, reference_video)
    torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], reference_audio)


@pytest.mark.parametrize("lock_audio", [False, True])
def test_batched_step_execution_matches_independent_requests(lock_audio):
    """Two co-batched requests must land where they would have landed alone."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import MiniMaxH3LatentEdit

    model = _SegmentMeanModel()
    specs = [
        # Exactly 64 packed rows exercises the no-padding-tail boundary case.
        dict(text_len=46, latent_t=7, latent_h=2, latent_w=4, audio_t=2, seed=7),
        dict(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=6),
    ]
    # Different step counts, so the batch composition changes mid-flight.
    schedules = [(_sigmas(6, 12.0), _sigmas(6, 3.0)), (_sigmas(4, 12.0), _sigmas(4, 3.0))]

    pipeline = _step_pipeline(model)

    def make_state(request_id: str, index: int):
        branch, video_rows, audio_rows = _make_branch(**specs[index])
        sigmas_video, sigmas_audio = schedules[index]
        state = _make_state(request_id, model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
        video_mask = torch.linspace(0.1 * index, 0.8 + 0.1 * index, video_rows.shape[0])
        video_edit = MiniMaxH3LatentEdit.from_rows(
            torch.full_like(video_rows, 10.0 + index),
            torch.full_like(video_rows, 20.0 + index),
            video_mask,
            video_mask,
        )
        audio_mask = torch.linspace(0.2 - 0.1 * index, 1.0 - 0.1 * index, audio_rows.shape[0])
        audio_edit = MiniMaxH3LatentEdit.from_rows(
            torch.full_like(audio_rows, 30.0 + index),
            torch.full_like(audio_rows, 40.0 + index),
            audio_mask,
            audio_mask,
        )
        state.extra[mod._STEP_VIDEO_EDIT] = video_edit
        if lock_audio and index == 0:
            branch.locked_audio_rows = audio_rows.clone()
        else:
            state.extra[mod._STEP_AUDIO_EDIT] = audio_edit
        return state

    alone: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(len(specs)):
        state = make_state("solo", index)
        while not state.denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(InputBatch.make_batch([state]), states=[state]))
        alone.append((state.latents, state.extra[mod._STEP_AUDIO_ROWS]))

    states = [make_state(f"req-{index}", index) for index in range(len(specs))]
    assert states[0].extra[mod._STEP_BRANCH].used_len == states[0].extra[mod._STEP_BRANCH].seq_len

    active = list(states)
    while active:
        noise_pred = pipeline.denoise_step(SimpleNamespace(states=tuple(active)), states=active)
        offset = 0
        for state in active:
            rows = state.latents.shape[0]
            pipeline.step_scheduler(state, noise_pred[offset : offset + rows])
            offset += rows
        assert offset == noise_pred.shape[0]
        # Finished requests leave the batch, exactly like the runner drops them.
        active = [state for state in active if not state.denoise_completed]

    for state, (expected_video, expected_audio) in zip(states, alone):
        torch.testing.assert_close(state.latents, expected_video)
        torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], expected_audio)


def test_both_modes_publish_denoise_progress_for_gated_attention():
    """TRTLLM's skip gate and RAINFUSION's warmup stay dense without these."""
    from vllm_omni.diffusion import forward_context as fc
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    model = _SegmentMeanModel()
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=14)
    sigmas_video = _sigmas(4, 12.0)
    sigmas_audio = _sigmas(4, 3.0)

    published: list[tuple[int | None, float | None]] = []

    class _Recorder:
        def __init__(self):
            self.denoise_step_idx = None
            self.denoise_timestep = None

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "denoise_timestep":
                published.append((self.denoise_step_idx, value))

    recorder = _Recorder()
    published.clear()  # drop the pair __init__ recorded
    original = fc._forward_context
    fc._forward_context = recorder
    try:
        minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=video_rows,
            initial_audio_rows=audio_rows,
            keyframe_cond_rows=None,
            sigmas_video=sigmas_video,
            sigmas_audio=sigmas_audio,
            device=torch.device("cpu"),
        )
        request_mode = list(published)

        published.clear()
        pipeline = _step_pipeline(model)
        state = _make_state("req-0", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
        while not state.denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(InputBatch.make_batch([state]), states=[state]))
        step_mode = list(published)
    finally:
        fc._forward_context = original

    num_steps = len(sigmas_video) - 1
    expected = [(step, sigmas_video[step]) for step in range(num_steps)]
    assert request_mode == expected + [(None, None)]
    # Step mode has no loop to close, so it publishes only the per-step pairs.
    assert step_mode == expected


def test_mixed_step_batch_leaves_gated_attention_dense():
    """A batch spanning different steps has no single progress point to publish."""
    from vllm_omni.diffusion import forward_context as fc

    model = _SegmentMeanModel()
    first, first_video, first_audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=15)
    second, second_video, second_audio = _make_branch(
        text_len=5, latent_t=3, latent_h=6, latent_w=4, audio_t=2, seed=16
    )
    sigmas_video, sigmas_audio = _sigmas(6, 12.0), _sigmas(6, 3.0)
    states = [
        _make_state("req-0", model, first, first_video, first_audio, sigmas_video, sigmas_audio),
        _make_state("req-1", model, second, second_video, second_audio, sigmas_video, sigmas_audio),
    ]
    states[1].step_index = 2  # admitted later, so it trails the batch

    recorder = SimpleNamespace(denoise_step_idx="unset", denoise_timestep="unset")
    original = fc._forward_context
    fc._forward_context = recorder
    try:
        _step_pipeline(model).denoise_step(InputBatch.make_batch(states), states=states)
    finally:
        fc._forward_context = original

    assert recorder.denoise_step_idx is None
    assert recorder.denoise_timestep is None


class _ProgressRecordingModel(_SegmentMeanModel):
    """Records the denoise progress in the forward context that each forward runs under."""

    def __init__(self, *, fail_on_call: int | None = None):
        super().__init__()
        self.progress: list[tuple[int | None, float | None, int | None, bool, int]] = []
        self.fail_on_call = fail_on_call

    def __call__(self, **kwargs):
        from vllm_omni.diffusion.forward_context import get_forward_context

        self.progress.append(
            (*_denoise_progress(get_forward_context()), kwargs["packed_seq_params"].get("num_requests", 1))
        )
        if len(self.progress) == self.fail_on_call:
            raise RuntimeError("request forward failed")
        return super().__call__(**kwargs)


def _denoise_progress(ctx) -> tuple[int | None, float | None, int | None, bool]:
    return ctx.denoise_step_idx, ctx.denoise_timestep, ctx.total_denoise_steps, ctx.attention_schedule_denoise_active


def _mixed_progress_states(model):
    """Two requests at different steps of sequences with different lengths (6 and 4 steps)."""
    first, first_video, first_audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=21)
    second, second_video, second_audio = _make_branch(
        text_len=5, latent_t=3, latent_h=6, latent_w=4, audio_t=2, seed=22
    )
    states = [
        _make_state("req-0", model, first, first_video, first_audio, _sigmas(6, 12.0), _sigmas(6, 3.0)),
        _make_state("req-1", model, second, second_video, second_audio, _sigmas(4, 12.0), _sigmas(4, 3.0)),
    ]
    states[1].step_index = 2
    return states


def _sparse_schedule():
    from vllm_omni.diffusion.attention.schedule import AttentionScheduleRange

    return (AttentionScheduleRange(0, None, "sparse"),)


def test_scheduled_mixed_step_batch_runs_each_request_under_its_own_progress():
    """Under a bound schedule, each request selects attention at its own step and total."""
    from vllm_omni.diffusion.forward_context import bind_attention_schedule, get_forward_context, set_forward_context
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    model = _ProgressRecordingModel()
    states = _mixed_progress_states(model)
    pipeline = _step_pipeline(model)
    with set_forward_context():
        packed_velocity = pipeline.denoise_step(InputBatch.make_batch(states), states=states)
        packed_progress = list(model.progress)
        model.progress.clear()
        with bind_attention_schedule(_sparse_schedule()):
            velocity = pipeline.denoise_step(InputBatch.make_batch(states), states=states)
            after = _denoise_progress(get_forward_context())

    first_sigmas = states[0].extra[mod._STEP_SIGMAS_VIDEO]
    second_sigmas = states[1].extra[mod._STEP_SIGMAS_VIDEO]
    assert packed_progress == [(None, None, None, False, 2)]
    assert model.progress == [
        (0, float(first_sigmas[0]), 6, True, 1),
        (2, float(second_sigmas[2]), 4, True, 1),
    ]
    # The batch-level progress (no single point for this batch) is back after the requests.
    assert after == (None, None, None, False)
    # One forward per request computes the same velocities as the packed forward.
    torch.testing.assert_close(velocity, packed_velocity)


def test_sigma_step_batch_isolates_equal_step_requests_with_different_noise():
    from vllm_omni.diffusion.attention.schedule import parse_attention_sigma_schedule, select_attention_profile_by_sigma
    from vllm_omni.diffusion.forward_context import (
        bind_attention_sigma_schedule,
        get_forward_context,
        set_forward_context,
    )
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    seen = []

    class _SigmaModel(_ProgressRecordingModel):
        def __call__(self, **kwargs):
            ctx = get_forward_context()
            seen.append(
                (
                    ctx.denoise_step_idx,
                    ctx.denoise_sigma,
                    select_attention_profile_by_sigma(ctx.attention_sigma_schedule, ctx.denoise_sigma),
                    kwargs["packed_seq_params"].get("num_requests", 1),
                )
            )
            return super().__call__(**kwargs)

    model = _SigmaModel()
    states = _mixed_progress_states(model)
    for state, sigma in zip(states, [0.8, 0.2]):
        state.step_index = 1
        state.extra[mod._STEP_SIGMAS_VIDEO][1] = sigma
    windows = parse_attention_sigma_schedule(
        [
            {"low": 0.0, "high": 0.3, "profile": "low"},
            {"low": 0.3, "high": 1.0, "profile": "high"},
        ]
    )
    with set_forward_context(), bind_attention_sigma_schedule(windows):
        _step_pipeline(model).denoise_step(InputBatch.make_batch(states), states=states)
        assert get_forward_context().denoise_sigma is None
        assert not get_forward_context().attention_sigma_schedule_active
    assert seen == [(1, 0.8, "high", 1), (1, 0.2, "low", 1)]


def test_scheduled_request_loop_restarts_progress_for_each_denoise_sequence():
    """forward() runs one request-mode loop per seed and per window; each selects from step 0 of its own total."""
    from vllm_omni.diffusion.forward_context import bind_attention_schedule, get_forward_context, set_forward_context
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    model = _ProgressRecordingModel()
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=23)
    sigmas_video, sigmas_audio = _sigmas(4, 12.0), _sigmas(4, 3.0)
    total = len(sigmas_video) - 1
    with set_forward_context(), bind_attention_schedule(_sparse_schedule()):
        for _ in range(2):
            model.progress.clear()
            minimax_h3_denoise_loop(
                model=model,
                positive=branch,
                initial_video_rows=video_rows,
                initial_audio_rows=audio_rows,
                keyframe_cond_rows=None,
                sigmas_video=sigmas_video,
                sigmas_audio=sigmas_audio,
                device=torch.device("cpu"),
            )
            assert model.progress == [(step, sigmas_video[step], total, True, 1) for step in range(total)]
            assert _denoise_progress(get_forward_context()) == (None, None, None, False)


@pytest.mark.parametrize("batch_frames", [1, 33])
def test_prepare_encode_seeds_runner_visible_state(monkeypatch, batch_frames):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=8)
    video_edit = object()
    audio_edit = object()
    sigmas_video = _sigmas(6, 12.0)
    sigmas_audio = _sigmas(6, 3.0)
    context = {
        "height": 96,
        "width": 64,
        "preencode_mp4": True,
        "preencode_batch_frames": batch_frames,
        "latent_t": 2,
        "latent_h": 4,
        "latent_w": 6,
        "audio_t": 3,
        **{key: None for key in mod._MINIMAX_H3_DENOISE_INPUT_KEYS},
    }

    pipeline = _step_pipeline(_SegmentMeanModel())
    conditioning = object()
    monkeypatch.setattr(
        mod.MiniMaxH3Pipeline,
        "_extract_encoder_conditioning",
        staticmethod(lambda _: conditioning),
    )
    monkeypatch.setattr(
        mod.MiniMaxH3Pipeline,
        "_prepare_encoder_conditioning_inputs",
        lambda self, value, sampling: context if value is conditioning else pytest.fail("wrong encoder handoff"),
    )
    monkeypatch.setattr(
        mod.MiniMaxH3Pipeline,
        "_build_denoise_inputs",
        lambda self, **_: {
            "branch": branch,
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "cond_anchor": None,
            "audio_anchor": None,
            "video_edit": video_edit,
            "audio_edit": audio_edit,
            "sigmas_video": sigmas_video,
            "sigmas_audio": sigmas_audio,
        },
    )

    from vllm_omni.diffusion.worker.utils import StepRequestState

    state = StepRequestState(
        request_id="req-0",
        sampling=SimpleNamespace(num_outputs_per_prompt=1),
        prompt="a prompt",
    )
    pipeline.prepare_encode(state)

    # The runner slices the batched velocity by this row count.
    assert state.latents.shape == (int(branch.img_pos.shape[0]), 96)
    assert state.total_steps == len(sigmas_video) - 1
    assert state.step_index == 0
    assert state.do_true_cfg is False
    torch.testing.assert_close(state.current_timestep, torch.tensor(1.0 - sigmas_video[0]))
    assert state.extra[mod._STEP_BRANCH] is branch
    assert state.extra[mod._STEP_VIDEO_EDIT] is video_edit
    assert state.extra[mod._STEP_AUDIO_EDIT] is audio_edit
    assert state.extra[mod._STEP_SHAPE]["height"] == 96

    pipeline.od_config = SimpleNamespace()
    monkeypatch.setattr(pipeline, "_unpack_denoised_rows", lambda *args, **kwargs: (torch.zeros(1), torch.zeros(1)))
    calls = []

    def decode_to_mp4(*args, **kwargs):
        calls.append(kwargs)
        return b"mp4"

    monkeypatch.setattr(pipeline, "decode_to_mp4", decode_to_mp4)
    assert pipeline.post_decode(state).output == (b"mp4", None)
    assert calls[0]["batch_frames"] == batch_frames


def test_prepare_encode_rejects_request_mode_only_features():
    """Multi-output and DLO have no representation in the step contract."""
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.errors import OmniClientError

    # A request state carries exactly one latent tensor.
    multi_output = StepRequestState(
        request_id="req-0",
        sampling=SimpleNamespace(num_outputs_per_prompt=2),
        prompt="a prompt",
    )
    with pytest.raises(OmniClientError, match="one output per request"):
        _step_pipeline(_SegmentMeanModel()).prepare_encode(multi_output)

    # Distributed layerwise offload streams the DiT around a whole denoise loop.
    single_output = StepRequestState(
        request_id="req-1",
        sampling=SimpleNamespace(num_outputs_per_prompt=1),
        prompt="a prompt",
    )
    dlo_pipeline = _step_pipeline(_SegmentMeanModel())
    dlo_pipeline._dlo_residency_controller = object()
    with pytest.raises(ValueError, match="distributed layerwise offload"):
        dlo_pipeline.prepare_encode(single_output)


def test_prepare_encode_rejects_high_quality_cache_dit():
    """quality=high installs a transformer-scoped Cache-DiT profile that
    would leak across interleaved or co-batched step-mode requests."""
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.errors import OmniClientError

    state = StepRequestState(
        request_id="req-hi",
        sampling=SimpleNamespace(num_outputs_per_prompt=1, quality="high"),
        prompt="a prompt",
    )
    with pytest.raises(OmniClientError, match="quality=high"):
        _step_pipeline(_SegmentMeanModel()).prepare_encode(state)


def _fake_attention_module(
    *,
    use_ring: bool,
    backend: str = "FLASH_ATTN",
    supports_multi_doc: bool = True,
):
    """Build a bare MiniMaxH3Attention with the attributes ``_packed_batch_supported`` reads.

    ``supports_multi_doc`` models the platform-dependent capability probe on
    ``AttentionBackend`` (e.g. FLASH_ATTN returns True on CUDA/ROCm/MUSA and
    False on NPU/XPU); ``backend`` remains only for logging/back-compat.
    """
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    attn = object.__new__(MiniMaxH3Attention)
    attn.attention = SimpleNamespace(
        attn_backend=SimpleNamespace(
            get_name=lambda: backend,
            supports_multi_doc_packed_varlen=lambda: supports_multi_doc,
        ),
        use_ring=use_ring,
    )
    return attn


class _FakeTransformer:
    """The minimal ``modules()`` protocol ``_packed_batch_supported`` walks."""

    def __init__(self, attentions):
        self._attentions = list(attentions)

    def modules(self):
        return iter([self, *self._attentions])


@pytest.mark.parametrize(
    "attention",
    [
        _fake_attention_module(use_ring=True),
        _fake_attention_module(use_ring=False, backend="XFORMERS", supports_multi_doc=False),
        _fake_attention_module(use_ring=False, backend="FLASH_ATTN", supports_multi_doc=False),
    ],
)
def test_packed_batch_rejects_backends_that_cannot_isolate_requests(attention):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    assert MiniMaxH3Pipeline._packed_batch_supported(_FakeTransformer([attention])) is False


def test_locked_driving_audio_is_clean_and_unchanged_during_denoising():
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    branch, video, audio = _make_branch(text_len=3, latent_t=2, latent_h=2, latent_w=2, audio_t=3, seed=7)
    branch.locked_audio_rows = audio.clone()
    seen = []

    def model(**kwargs):
        positions = kwargs["audio_pos_info"]["position_ids"]
        times = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
        torch.testing.assert_close(times[positions], torch.ones_like(times[positions]))
        torch.testing.assert_close(kwargs["audio_x"][0, positions], audio)
        seen.append(True)
        return torch.ones_like(video), torch.ones_like(audio)

    result_video, result_audio = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video,
        initial_audio_rows=audio,
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.25, 0.0],
        device=torch.device("cpu"),
    )
    assert len(seen) == 2
    torch.testing.assert_close(result_audio, audio)
    assert not torch.equal(result_video, video)


def _schedule_candidate(name: str, *, mask_free=False, prefix_kv_slicing=False, attention_mask=False):
    """A prepared schedule candidate carrying only what the MiniMax H3 checks read."""

    class _Backend:
        supports_prefix_kv_slicing = prefix_kv_slicing

        @staticmethod
        def get_name() -> str:
            return name

        @classmethod
        def supports_packed_mask_free(cls) -> bool:
            return mask_free

        @classmethod
        def supports_attention_mask(cls, spec) -> bool:
            del spec
            return attention_mask

    return _make_metadata_candidate(_Backend)


def test_vdn_attention_accepts_only_vdnh3_candidates(monkeypatch):
    """VDN-H3 checkpoints read the frame window only in VDNH3_ATTN, so other profiles are rejected at startup."""
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    checks: list = []
    attention = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "VDNH3_ATTN"),
        add_schedule_candidate_check=checks.append,
    )
    owner = SimpleNamespace(
        attention=attention, _gate_hidden_size=8, total_num_heads=2, head_dim=4, _gate_quant_config=None, _prefix="b0"
    )
    owner.add_schedule_candidate_check = lambda check: h3.MiniMaxH3Attention.add_schedule_candidate_check(owner, check)
    monkeypatch.setattr(h3, "VDNH3HybridAttention", lambda *args, **kwargs: object())

    h3.MiniMaxH3Attention.enable_vdn(owner, config=None)

    assert [check(_schedule_candidate("VDNH3_ATTN")) for check in checks] == [None]
    assert "VDNH3_ATTN" in checks[0](_schedule_candidate("FLASH_ATTN"))


def test_gateless_dit_accepts_a_vsa_candidate_when_its_baseline_is_vsa(monkeypatch):
    """FastH3 VSA gates only self.transformer (the FL2VA DiT); the combined partition's transformers_ref has none.

    Its self role still resolves to FASTVIDEO_VSA, which the FastH3 VSA contract requires, so one self-role
    profile is checked on both DiTs. The DiTs use real Attention layers, whose baseline the check reads.
    """
    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import _FakeLinear, _small_od_config
    from vllm_omni.diffusion.attention import layer
    from vllm_omni.diffusion.attention.backends.fastvideo_vsa import FastVideoVSABackend
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    for name in ("ColumnParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(h3, name, _FakeLinear)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(layer, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())

    def dit_attention(baseline, *, gated: bool):
        monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (baseline, None))
        model = h3.MiniMaxH3DiTModel(_small_od_config())
        if gated:
            model.enable_vsa_gates()
        attention = [block.attn.attention for block in model.blocks]
        assert all(isinstance(a, layer.Attention) and a.attn_backend is baseline for a in attention)
        return attention

    def rejections(attention, record) -> list[str]:
        return [reason for check in attention._schedule_candidate_checks if (reason := check(record)) is not None]

    gated = dit_attention(FastVideoVSABackend, gated=True)
    gateless = dit_attention(FastVideoVSABackend, gated=False)

    # One self-role profile covers both DiTs, so it has to pass on each of them.
    vsa = _schedule_candidate("FASTVIDEO_VSA", mask_free=True)
    assert all(rejections(attention, vsa) == [] for attention in gated + gateless)
    # The gated FastH3 DiT still accepts only FASTVIDEO_VSA.
    sdpa = _schedule_candidate("SDPA", attention_mask=True)
    assert all(any("FastH3" in reason for reason in rejections(a, sdpa)) for a in gated)
    # A gateless DiT on another baseline still rejects the VSA candidate.
    dense = dit_attention(SDPABackend, gated=False)
    assert all(any("VSA compression gate" in reason for reason in rejections(a, vsa)) for a in dense)
