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


def _make_state(request_id: str, model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio, sampler="euler"):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.worker.utils import StepRequestState

    state = StepRequestState(request_id=request_id, sampling=SimpleNamespace())
    state.latents = video_rows.clone()
    state.timesteps = torch.tensor([1.0 - sigma for sigma in sigmas_video[:-1]], dtype=torch.float32)
    state.step_index = 0
    from vllm_omni.diffusion.sampler import SamplingState

    state.extra = {
        mod._STEP_SAMPLER: sampler,
        mod._STEP_SAMPLING_STATE: SamplingState(
            (video_rows[branch.update_mask_dev], audio_rows[branch.audio_update_mask_dev]),
            (sigmas_video, sigmas_audio),
        ),
        mod._STEP_BRANCH: branch,
        # Co-batched requests must share one DiT instance, or the model adapter
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


@pytest.mark.parametrize("sampler", ["euler", "res_multistep", "heun"])
@pytest.mark.parametrize("num_steps", [1, 8, 50])
def test_step_execution_matches_request_mode_denoise_loop(num_steps, sampler, mocker):
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
        sampler=sampler,
        device=torch.device("cpu"),
    )

    assert model.call_count == (2 * num_steps - 1 if sampler == "heun" else num_steps)
    model.reset_mock()
    pipeline = _step_pipeline(model)
    state = _make_state("req-0", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio, sampler)
    input_batch = SimpleNamespace(states=(state,))

    steps = 0
    while not state.denoise_completed:
        assert pipeline.denoise_step(input_batch, states=[state]) == {state.request_id: None}
        steps += 1

    assert steps == num_steps
    assert model.call_count == (2 * num_steps - 1 if sampler == "heun" else num_steps)
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
                update_progress=cancel_after_first_step,
            )
    finally:
        registry.close()
    assert steps == [0]


@pytest.mark.parametrize("mixed_unguided", [False, True])
def test_timeline_guides_request_step_batch_parity_and_target_only_unpack(mixed_unguided):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        minimax_h3_denoise_loop,
        minimax_h3_prepare_denoise_rows,
    )

    batch_sizes = []

    class RecordingModel(_SegmentMeanModel):
        def __call__(self, **kwargs):
            batch_sizes.append(kwargs["packed_seq_params"].get("num_requests", 1))
            return super().__call__(**kwargs)

    model = RecordingModel()
    pipeline = _step_pipeline(model)
    states, expected, target_shapes = [], [], []
    for index in range(2):
        inputs_kwargs = dict(
            task="ref2va",
            text_embeddings=torch.ones(3 + index, _HIDDEN),
            text_tags=torch.ones(3 + index, dtype=torch.long),
            seed=17 + index,
            latent_t=7,
            latent_h=4,
            latent_w=4,
            audio_t=37,
            num_frames=22,
            num_steps=4 + index,
            video_shift=12.0,
            audio_shift=3.0,
            base_schedule=None,
            visual_condition=torch.arange(16 * 96, dtype=torch.float32).reshape(16, 96) / 100,
            visual_condition_shape=None,
            visual_condition_shapes=[(2, 4, 4), (1, 4, 4), (1, 4, 4)],
            audio_condition=torch.arange(18 * 32, dtype=torch.float32).reshape(18, 32) / 100,
            ref_audio_t=None,
            audio_condition_lengths=[6, 3],
            guide_blocks=[
                {
                    "kind": "video_audio",
                    "frame_index": 7,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                    "ref_audio_t": 6,
                },
                {"kind": "image", "frame_index": 1, "latent_h": 4, "latent_w": 4},
            ],
            ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "audio", "ref_audio_t": 3}],
        )
        if mixed_unguided and index == 1:
            inputs_kwargs.update(
                latent_t=4,
                latent_w=6,
                audio_t=19,
                num_frames=11,
                visual_condition=torch.full((4, 96), 0.25),
                visual_condition_shapes=[(1, 4, 4)],
                audio_condition=None,
                audio_condition_lengths=None,
                guide_blocks=None,
                ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 4}],
            )
        target_shapes.append(tuple(inputs_kwargs[key] for key in ("latent_t", "latent_h", "latent_w", "audio_t")))
        inputs = pipeline._build_denoise_inputs(**inputs_kwargs)
        branch = inputs["branch"]
        video, audio = minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=inputs["video_rows"],
            initial_audio_rows=inputs["audio_rows"],
            keyframe_cond_rows=inputs["cond_anchor"],
            audio_ref_rows=inputs["audio_anchor"],
            sigmas_video=inputs["sigmas_video"],
            sigmas_audio=inputs["sigmas_audio"],
            device=torch.device("cpu"),
        )
        expected.append((video, audio))
        video_rows, audio_rows, visual_anchor, audio_anchor = minimax_h3_prepare_denoise_rows(
            positive=branch,
            initial_video_rows=inputs["video_rows"],
            initial_audio_rows=inputs["audio_rows"],
            keyframe_cond_rows=inputs["cond_anchor"],
            audio_ref_rows=inputs["audio_anchor"],
            device=torch.device("cpu"),
        )
        state = _make_state(
            str(index), model, branch, video_rows, audio_rows, inputs["sigmas_video"], inputs["sigmas_audio"]
        )
        state.extra[mod._STEP_COND_ANCHOR] = visual_anchor
        state.extra[mod._STEP_AUDIO_ANCHOR] = audio_anchor
        states.append(state)
    batch_sizes.clear()
    active = list(states)
    while active:
        assert pipeline.denoise_step(SimpleNamespace(states=tuple(active)), states=active) == {
            state.request_id: None for state in active
        }
        for state in active:
            branch = state.extra[mod._STEP_BRANCH]
            torch.testing.assert_close(state.latents[~branch.update_mask], state.extra[mod._STEP_COND_ANCHOR])
            if state.extra[mod._STEP_AUDIO_ANCHOR] is None:
                assert branch.audio_update_mask.all()
            else:
                torch.testing.assert_close(
                    state.extra[mod._STEP_AUDIO_ROWS][~branch.audio_update_mask],
                    state.extra[mod._STEP_AUDIO_ANCHOR],
                )
        active = [state for state in active if not state.denoise_completed]
    assert 2 in batch_sizes and batch_sizes[-1] == 1
    for state, (video, audio), (latent_t, latent_h, latent_w, audio_t) in zip(
        states, expected, target_shapes, strict=True
    ):
        torch.testing.assert_close(state.latents, video)
        torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], audio)
        branch = state.extra[mod._STEP_BRANCH]
        target_video, target_audio = pipeline._unpack_denoised_rows(
            branch,
            state.latents,
            state.extra[mod._STEP_AUDIO_ROWS],
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        assert target_video.shape == (1, 24, latent_t, latent_h, latent_w)
        assert target_audio.shape == (2, 32, audio_t)


@pytest.mark.parametrize("sampler", ["euler", "res_multistep", "heun"])
def test_step_execution_matches_request_mode_with_latent_edits(sampler):
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
        sampler=sampler,
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

    state = _make_state("req-edit", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio, sampler)
    state.extra[mod._STEP_VIDEO_EDIT] = video_edit
    state.extra[mod._STEP_AUDIO_EDIT] = audio_edit
    pipeline = _step_pipeline(model)
    while not state.denoise_completed:
        assert pipeline.denoise_step(SimpleNamespace(states=(state,))) == {state.request_id: None}

    for name, expected in request_first_call.items():
        torch.testing.assert_close(model.calls[0][name], expected)
    torch.testing.assert_close(state.latents, reference_video)
    torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], reference_audio)


@pytest.mark.parametrize(
    "samplers",
    [
        ("euler", "euler"),
        ("res_multistep", "res_multistep"),
        ("euler", "res_multistep"),
        ("heun", "heun"),
        ("euler", "heun"),
        ("res_multistep", "heun"),
    ],
)
@pytest.mark.parametrize("lock_audio", [False, True])
def test_batched_step_execution_matches_independent_requests(lock_audio, samplers):
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
        state = _make_state(
            request_id, model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio, samplers[index]
        )
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
            assert pipeline.denoise_step(SimpleNamespace(states=(state,))) == {state.request_id: None}
        alone.append((state.latents, state.extra[mod._STEP_AUDIO_ROWS]))

    states = [make_state(f"req-{index}", index) for index in range(len(specs))]
    assert states[0].extra[mod._STEP_BRANCH].used_len == states[0].extra[mod._STEP_BRANCH].seq_len

    active = list(states)
    while active:
        assert pipeline.denoise_step(SimpleNamespace(states=tuple(active))) == {s.request_id: None for s in active}
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
            self.denoise_progress = {}

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
            assert pipeline.denoise_step(SimpleNamespace(states=(state,))) == {state.request_id: None}
        step_mode = list(published)
    finally:
        fc._forward_context = original

    num_steps = len(sigmas_video) - 1
    expected = [(step, sigmas_video[step]) for step in range(num_steps)]
    assert [p for p in request_mode if p[0] is not None] == expected
    assert [p for p in step_mode if p[0] is not None] == expected
    assert request_mode[-1] == step_mode[-1] == (None, None)


@pytest.mark.parametrize("sampler", ["euler", "res_multistep", "heun"])
@pytest.mark.parametrize("batch_frames", [1, 33])
def test_prepare_encode_seeds_runner_visible_state(monkeypatch, batch_frames, sampler):
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
            "sampler": sampler,
        },
    )

    from vllm_omni.diffusion.worker.utils import StepRequestState

    state = StepRequestState(
        request_id="req-0",
        sampling=SimpleNamespace(num_outputs_per_prompt=1),
        prompt="a prompt",
    )
    pipeline.prepare_encode(state)

    # Video rows remain the runner-visible latent state.
    assert state.latents.shape == (int(branch.img_pos.shape[0]), 96)
    assert state.total_steps == len(sigmas_video) - 1
    assert state.step_index == 0
    assert state.do_true_cfg is False
    torch.testing.assert_close(state.current_timestep, torch.tensor(1.0 - sigmas_video[0]))
    assert state.extra[mod._STEP_BRANCH] is branch
    assert state.extra[mod._STEP_SAMPLER] == sampler
    sample = state.extra[mod._STEP_SAMPLING_STATE]
    assert sample.step_index == 0
    assert sample.sigmas == (tuple(sigmas_video), tuple(sigmas_audio))
    torch.testing.assert_close(sample.latents[0], video_rows[branch.update_mask_dev])
    torch.testing.assert_close(sample.latents[1], audio_rows[branch.audio_update_mask_dev])
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


@pytest.mark.parametrize("packed", [True, False])
def test_heun_batches_correction_and_skips_terminal_requests(packed):
    model = _SegmentMeanModel()
    states = []
    for index, schedule in enumerate(([1.0, 0.5, 0.0], [1.0, 0.7, 0.0], [1.0, 0.0])):
        branch, video, audio = _make_branch(text_len=5, latent_t=2, latent_h=4, latent_w=4, audio_t=3, seed=index)
        states.append(_make_state(str(index), model, branch, video, audio, schedule, schedule, "heun"))
    pipeline = _step_pipeline(model, packed_batch_supported=packed)
    assert pipeline.denoise_step(SimpleNamespace(states=states)) == {s.request_id: None for s in states}
    assert len(model.calls) == (2 if packed else 5)
    if packed:
        assert len(model.calls[0]["video_rows"]) == sum(len(s.latents) for s in states)
        assert len(model.calls[1]["video_rows"]) == sum(len(s.latents) for s in states[:2])
    assert all(s.step_index == 1 for s in states)


def test_mixed_progress_reaches_model_for_each_heun_evaluation(monkeypatch):
    from vllm_omni.diffusion import forward_context as fc
    from vllm_omni.diffusion.attention.backends.abstract import DenoiseProgress

    context = fc.ForwardContext()
    monkeypatch.setattr(fc, "_forward_context", context)
    recorded = []

    class RecordingModel(_SegmentMeanModel):
        def __call__(self, **kwargs):
            recorded.append(context.denoise_progress)
            return super().__call__(**kwargs)

    model = RecordingModel()
    states, branches = [], []
    schedule = [1.0, 0.5, 0.2, 0.0]
    for index in range(2):
        branch, video, audio = _make_branch(
            text_len=5 + index, latent_t=2, latent_h=4, latent_w=4, audio_t=3, seed=index
        )
        branches.append(branch)
        states.append(_make_state(str(index), model, branch, video, audio, schedule, schedule, "heun"))
    pipeline = _step_pipeline(model)
    for _ in range(2):
        result = pipeline.denoise_step(SimpleNamespace(states=[states[1]]))
        assert result == {"1": None}
    recorded.clear()
    context.denoise_progress = {"self": (DenoiseProgress(1, 99, 0.123, 100),)}
    result = pipeline.denoise_step(SimpleNamespace(states=states))
    assert result == {"0": None, "1": None}
    assert len(recorded) == 2
    for actual, indices, times, steps in (
        (recorded[0], (0, 1), (1.0, 0.2), (0, 2)),
        (recorded[1], (0,), (0.5,), (0,)),
    ):
        assert actual["minimax_h3.token_refiner"] == tuple(
            DenoiseProgress(branches[i].text_len, step, time, 3)
            for i, time, step in zip(indices, times, steps, strict=True)
        )
        assert actual["self"] == tuple(
            DenoiseProgress(length, step, time, 3)
            for i, time, step in zip(indices, times, steps, strict=True)
            for length in (branches[i].used_len, branches[i].seq_len - branches[i].used_len)
            if length
        )
    assert context.denoise_progress == {}


def test_heun_failed_correction_keeps_both_streams_and_history_retryable():
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    model = _SegmentMeanModel()

    def failing(**kwargs):
        video, audio = model(**kwargs)
        if len(model.calls) == 2:
            raise RuntimeError("correction failed")
        return video, audio

    branch, video, audio = _make_branch(text_len=5, latent_t=2, latent_h=4, latent_w=4, audio_t=3, seed=7)
    state = _make_state("req", failing, branch, video, audio, [1.0, 0.5, 0.0], [1.0, 0.2, 0.0], "heun")
    original = state.extra[mod._STEP_SAMPLING_STATE]
    original_rows = (state.latents, state.extra[mod._STEP_AUDIO_ROWS])
    pipeline = _step_pipeline(failing)
    result = pipeline.denoise_step(SimpleNamespace(states=[state]))
    assert isinstance(result["req"], RuntimeError)
    assert state.extra[mod._STEP_SAMPLING_STATE] is original
    assert state.step_index == original.step_index == 0
    assert torch.equal(state.latents, video)
    assert torch.equal(state.extra[mod._STEP_AUDIO_ROWS], audio)
    buffers = state.extra[mod._STEP_MODEL_ROWS]
    assert pipeline.denoise_step(SimpleNamespace(states=[state])) == {"req": None}
    assert state.step_index == 1
    assert state.latents is original_rows[0]
    assert state.extra[mod._STEP_AUDIO_ROWS] is original_rows[1]
    assert state.extra[mod._STEP_MODEL_ROWS] is buffers


@pytest.mark.parametrize("sampler", ["euler", "res_multistep", "heun"])
def test_steps_reuse_request_buffers_and_write_into_original_rows(sampler, monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.sampling import MiniMaxH3Model

    model = _SegmentMeanModel()
    states, originals = [], {}
    buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index in range(2):
        branch, video, audio = _make_branch(text_len=3, latent_t=2, latent_h=2, latent_w=2, audio_t=3, seed=index)
        state = _make_state(
            str(index), model, branch, video, audio, [1.0, 0.6, 0.2, 0.0], [1.0, 0.6, 0.2, 0.0], sampler
        )
        states.append(state)
        originals[id(branch)] = (state.latents, state.extra[mod._STEP_AUDIO_ROWS])

    prepare = MiniMaxH3Model.prepare
    before_step: dict[int, tuple[torch.Tensor, ...]] = {}

    def checked_prepare(adapter, evaluation):
        result = prepare(adapter, evaluation)
        key = id(adapter.branch)
        first_buffers = buffers.setdefault(key, adapter.model_rows)
        for row, original, saved, buffer, first in zip(
            result[0], originals[key], before_step[key], adapter.model_rows, first_buffers, strict=True
        ):
            assert row is buffer is first
            assert buffer.data_ptr() != original.data_ptr()
            torch.testing.assert_close(original, saved)
        return result

    monkeypatch.setattr(MiniMaxH3Model, "prepare", checked_prepare)
    pipeline = _step_pipeline(model)
    for _ in range(3):
        before_step = {key: tuple(row.clone() for row in rows) for key, rows in originals.items()}
        assert pipeline.denoise_step(SimpleNamespace(states=states)) == {s.request_id: None for s in states}
        for state in states:
            rows = originals[id(state.extra[mod._STEP_BRANCH])]
            assert state.latents is rows[0]
            assert state.extra[mod._STEP_AUDIO_ROWS] is rows[1]
    assert len({row.data_ptr() for pair in buffers.values() for row in pair}) == 4
    assert all(s.step_index == 3 for s in states)
    assert len(model.calls) == (5 if sampler == "heun" else 3)


@pytest.mark.parametrize("edited", [False, True])
def test_reused_model_rows_preserve_conditions_and_previous_predictions(edited):
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import MiniMaxH3LatentEdit, minimax_h3_prepare_edit_rows
    from vllm_omni.diffusion.models.minimax_h3.sampling import MiniMaxH3Model
    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0
    from vllm_omni.diffusion.sampler import ModelEvaluation, SamplingState

    rows = (torch.arange(12, dtype=torch.float32).reshape(4, 3), torch.arange(8, dtype=torch.float32).reshape(4, 2))
    masks = (torch.tensor([True, False, True, False]), torch.tensor([False, True, False, True]))
    branch = SimpleNamespace(update_mask_dev=masks[0], audio_update_mask_dev=masks[1], locked_audio_rows=None)
    edits = tuple(
        MiniMaxH3LatentEdit.from_rows(
            row[mask] * 0.1, row[mask] * 0.2, torch.tensor([0.3, 0.7]), torch.tensor([0.4, 0.8])
        )
        if edited
        else None
        for row, mask in zip(rows, masks, strict=True)
    )
    adapter = MiniMaxH3Model(None, branch, rows, edits)
    buffers = adapter.model_rows
    saved_rows = tuple(row.clone() for row in rows)
    state = SamplingState(tuple(row[mask] for row, mask in zip(rows, masks, strict=True)), ([1.0, 0.0], [1.0, 0.0]))
    previous = []
    for offset in range(4):
        latents = tuple(x + offset for x in state.latents)
        prepared = adapter.prepare(ModelEvaluation(state, latents, (0.5, 0.5)))
        expected = []
        for index, (row, mask, latent, edit) in enumerate(zip(rows, masks, latents, edits, strict=True)):
            reference = row.clone()
            reference[mask] = latent
            model_rows, times = minimax_h3_prepare_edit_rows(
                reference, mask, edit, 0.5, adapter.cond_timesteps[index], sigma=0.5
            )
            assert prepared[0][index] is buffers[index]
            assert prepared[1][index][0] is buffers[index]
            torch.testing.assert_close(buffers[index], model_rows)
            torch.testing.assert_close(prepared[1][index][1], times)
            expected.append(
                minimax_h3_rf_v_to_x0(latent, torch.ones_like(latent), row.new_tensor(0.5))
                if edit is None
                else edit.x0(model_rows[mask], torch.ones_like(latent), 0.5)
            )
        predictions = adapter.to_denoised(prepared, tuple(torch.ones_like(row) for row in rows))
        for actual, reference in zip(predictions, expected, strict=True):
            torch.testing.assert_close(actual, reference)
        previous.append((predictions, tuple(x.clone() for x in predictions)))
        for actual, reference in previous:
            torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(rows, saved_rows)

    state.latents = latents
    adapter.validate_write_back(state)
    result = adapter.write_back(state)
    for actual, original, mask, latent, saved in zip(result, rows, masks, latents, saved_rows, strict=True):
        assert actual is original
        torch.testing.assert_close(actual[mask], latent)
        torch.testing.assert_close(actual[~mask], saved[~mask])


def test_invalid_final_audio_prevents_write_back_for_entire_group(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    model = _SegmentMeanModel()
    states = []
    for index in range(2):
        branch, video, audio = _make_branch(text_len=3, latent_t=2, latent_h=2, latent_w=2, audio_t=3, seed=index)
        states.append(_make_state(str(index), model, branch, video, audio, [1.0, 0.0], [1.0, 0.0]))
    originals = [
        (s.latents.clone(), s.extra[mod._STEP_AUDIO_ROWS].clone(), s.extra[mod._STEP_SAMPLING_STATE]) for s in states
    ]
    sampler = mod.sampler_object("euler")

    def invalid_sample(model, pending, **kwargs):
        sampler.sample(model, pending, **kwargs)
        video, audio = pending[-1].latents
        pending[-1].latents = (video, audio[:-1])

    monkeypatch.setattr(mod, "sampler_object", lambda _: SimpleNamespace(sample=invalid_sample))
    result = _step_pipeline(model).denoise_step(SimpleNamespace(states=states))
    assert all(isinstance(error, ValueError) for error in result.values())
    for state, (video, audio, sample) in zip(states, originals, strict=True):
        torch.testing.assert_close(state.latents, video)
        torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], audio)
        assert state.extra[mod._STEP_SAMPLING_STATE] is sample
        assert state.step_index == sample.step_index == 0


def test_heun_rejects_prediction_cache_before_model_evaluation():
    model = _SegmentMeanModel()
    branch, video, audio = _make_branch(text_len=5, latent_t=2, latent_h=4, latent_w=4, audio_t=3, seed=7)
    state = _make_state("req", model, branch, video, audio, [1.0, 0.5, 0.0], [1.0, 0.2, 0.0], "heun")
    pipeline = _step_pipeline(model)
    pipeline.od_config = SimpleNamespace(cache_backend="cache_dit")
    error = pipeline.denoise_step(SimpleNamespace(states=[state]))["req"]
    assert "uncached" in str(error)
    assert state.step_index == 0
    assert not model.calls
