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
from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import _FakeLinear, _small_od_config
from vllm_omni.diffusion.attention import layer
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import FastVideoVSABackend
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNAttentionBackend
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    parse_attention_sigma_schedule,
    resolve_request_attention_schedule,
)
from vllm_omni.diffusion.data import AttentionConfig, AttentionScheduleConfig
from vllm_omni.diffusion.forward_context import (
    bind_attention_schedule,
    get_attention_schedule_profile,
    get_forward_context,
    set_forward_context,
)
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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

    def modules(self):
        """The nn.Module walk the serial step branch uses to look for ring attention; this DiT has none."""
        return iter([self])


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

    state = StepRequestState(
        request_id=request_id, sampling=OmniDiffusionSamplingParams(guidance_scale=1.0, true_cfg_scale=4.0)
    )
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
    }
    return state


def _set_step_index(state, step: int) -> None:
    """Place a request mid-sequence; the runner and its sampling state must agree."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    state.step_index = step
    state.extra[mod._STEP_SAMPLING_STATE].step_index = step


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


@pytest.mark.parametrize("steps", [(0,), (0, 7)])
def test_pdd_step_execution_keeps_request_steps_without_backend_warning(steps, mocker):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import FastH3OmniCheckpointSpec
    from vllm_omni.diffusion.models.minimax_h3.pdd import pdd_sigmas

    model = mocker.Mock(wraps=_SegmentMeanModel())
    # The sampler adapter detects PDD heads from the DiT, as request mode does.
    model.arch = SimpleNamespace(pdd_steps=32)
    pipeline = _step_pipeline(model)
    pipeline._fasth3_checkpoint = FastH3OmniCheckpointSpec()
    warning = mocker.patch.object(mod.logger, "warning_once")
    states = []
    for index, step in enumerate(steps):
        branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=index)
        state = _make_state(str(index), model, branch, video, audio, pdd_sigmas(12), pdd_sigmas(3))
        _set_step_index(state, step)
        states.append(state)
    pipeline.denoise_step(None, states=states)
    assert [call.kwargs["pdd_step"] for call in model.call_args_list] == list(steps)
    warning.assert_not_called()


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
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

    assert model.call_count == num_steps
    model.reset_mock()
    pipeline = _step_pipeline(model)
    state = _make_state("req-0", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio, sampler)
    input_batch = SimpleNamespace(states=(state,))

    steps = 0
    while not state.denoise_completed:
        assert pipeline.denoise_step(input_batch, states=[state]) == {state.request_id: None}
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
    states, expected, target_shapes, anchors = [], [], [], {}
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
        video_rows, audio_rows = minimax_h3_prepare_denoise_rows(
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
        anchors[state.request_id] = (inputs["cond_anchor"], inputs["audio_anchor"])
        states.append(state)
    batch_sizes.clear()
    active = list(states)
    while active:
        assert pipeline.denoise_step(SimpleNamespace(states=tuple(active)), states=active) == {
            state.request_id: None for state in active
        }
        for state in active:
            branch = state.extra[mod._STEP_BRANCH]
            cond_anchor, audio_anchor = anchors[state.request_id]
            # Condition rows stay pinned to their anchors through every step.
            if cond_anchor is None:
                assert branch.update_mask.all()
            else:
                torch.testing.assert_close(state.latents[~branch.update_mask], cond_anchor.float())
            if audio_anchor is None:
                assert branch.audio_update_mask.all()
            else:
                torch.testing.assert_close(
                    state.extra[mod._STEP_AUDIO_ROWS][~branch.audio_update_mask],
                    audio_anchor.float(),
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


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
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


@pytest.mark.parametrize(
    ("mixed_dits", "packed_batch_supported", "reason"),
    [
        (True, True, "different task-specific DiTs"),
        (False, False, "isolates packed multi-document cu_seqlens"),
    ],
)
def test_unpacked_step_batch_logs_why_it_runs_one_forward_per_request(
    mixed_dits, packed_batch_supported, reason, monkeypatch
):
    """An unpacked batch must say why, or a throughput drop has no explanation."""
    from vllm_omni.diffusion.models.minimax_h3 import sampling

    warnings = []
    monkeypatch.setattr(sampling.logger, "warning_once", lambda message, *args: warnings.append(message % args))
    models = [_SegmentMeanModel(), _SegmentMeanModel()] if mixed_dits else [_SegmentMeanModel()] * 2
    states = []
    for index, model in enumerate(models):
        branch, video, audio = _make_branch(text_len=5, latent_t=2, latent_h=4, latent_w=4, audio_t=3, seed=index)
        states.append(_make_state(str(index), model, branch, video, audio, _sigmas(4, 12.0), _sigmas(4, 3.0)))
    pipeline = _step_pipeline(models[0], packed_batch_supported=packed_batch_supported)

    assert pipeline.denoise_step(SimpleNamespace(states=states)) == {s.request_id: None for s in states}

    assert [len(model.calls) for model in dict.fromkeys(models)] == ([1, 1] if mixed_dits else [2])
    assert len(warnings) == 1
    assert reason in warnings[0]
    assert "Running 2 requests one forward at a time." in warnings[0]


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
            self.attention_sigma_schedule = None

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
    _set_step_index(states[1], 2)  # admitted later, so it trails the batch

    recorder = SimpleNamespace(denoise_step_idx="unset", denoise_timestep="unset", attention_sigma_schedule=None)
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
        self.progress: list[tuple[int | None, float | None, int | None, str | None, int]] = []
        self.fail_on_call = fail_on_call

    def __call__(self, **kwargs):
        self.progress.append(
            (*_denoise_progress(get_forward_context()), kwargs["packed_seq_params"].get("num_requests", 1))
        )
        if len(self.progress) == self.fail_on_call:
            raise RuntimeError("request forward failed")
        return super().__call__(**kwargs)


def _denoise_progress(ctx) -> tuple[int | None, float | None, int | None, str | None]:
    return ctx.denoise_step_idx, ctx.denoise_timestep, ctx.total_denoise_steps, get_attention_schedule_profile()


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
    _set_step_index(states[1], 2)
    return states


def _assert_same_committed_rows(actual_states, expected_states):
    for actual, expected in zip(actual_states, expected_states, strict=True):
        assert actual.step_index == expected.step_index
        torch.testing.assert_close(actual.latents, expected.latents)
        torch.testing.assert_close(
            actual.extra[pipeline_minimax_h3._STEP_AUDIO_ROWS], expected.extra[pipeline_minimax_h3._STEP_AUDIO_ROWS]
        )


def _sparse_schedule():
    return (AttentionScheduleRange(0, None, "sparse"),)


def test_scheduled_mixed_step_batch_runs_each_request_under_its_own_progress():
    """Under a bound schedule, each request selects attention at its own step and total."""
    model = _ProgressRecordingModel()
    packed_states = _mixed_progress_states(model)
    states = _mixed_progress_states(model)
    pipeline = _step_pipeline(model)
    with set_forward_context():
        result = pipeline.denoise_step(InputBatch.make_batch(packed_states), states=packed_states)
        assert result == {state.request_id: None for state in packed_states}
        packed_progress = list(model.progress)
        model.progress.clear()
        with bind_attention_schedule(_sparse_schedule()):
            result = pipeline.denoise_step(InputBatch.make_batch(states), states=states)
            assert result == {state.request_id: None for state in states}
            after = _denoise_progress(get_forward_context())

    first_sigmas = states[0].extra[pipeline_minimax_h3._STEP_SAMPLING_STATE].sigmas[0]
    second_sigmas = states[1].extra[pipeline_minimax_h3._STEP_SAMPLING_STATE].sigmas[0]
    assert packed_progress == [(None, None, None, None, 2)]
    assert model.progress == [
        (0, float(first_sigmas[0]), 6, "sparse", 1),
        (2, float(second_sigmas[2]), 4, "sparse", 1),
    ]
    # The batch-level progress (no single point for this batch) is back after the requests.
    assert after == (None, None, None, None)
    # One forward per request commits the same latents as the packed forward.
    _assert_same_committed_rows(states, packed_states)


def test_sigma_step_batch_isolates_equal_step_requests_with_different_noise():
    seen = []

    class _SigmaModel(_ProgressRecordingModel):
        def __call__(self, **kwargs):
            ctx = get_forward_context()
            seen.append(
                (
                    ctx.denoise_step_idx,
                    ctx.denoise_sigma,
                    get_attention_schedule_profile(),
                    kwargs["packed_seq_params"].get("num_requests", 1),
                )
            )
            return super().__call__(**kwargs)

    model = _SigmaModel()
    states = _mixed_progress_states(model)
    for state, sigma in zip(states, [0.8, 0.2]):
        state.step_index = 1
        state.extra[pipeline_minimax_h3._STEP_SIGMAS_VIDEO][1] = sigma
    windows = parse_attention_sigma_schedule(
        [
            {"low": 0.0, "high": 0.3, "profile": "low"},
            {"low": 0.3, "high": 1.0, "profile": "high"},
        ]
    )
    with set_forward_context(), bind_attention_schedule(None, windows):
        _step_pipeline(model).denoise_step(InputBatch.make_batch(states), states=states)
        assert get_forward_context().denoise_sigma is None
        assert get_attention_schedule_profile() is None
    assert seen == [(1, 0.8, "high", 1), (1, 0.2, "low", 1)]


def test_scheduled_request_loop_restarts_progress_for_each_denoise_sequence():
    """forward() runs one request-mode loop per seed and per window; each selects from step 0 of its own total."""
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
            assert model.progress == [(step, sigmas_video[step], total, "sparse", 1) for step in range(total)]
            assert _denoise_progress(get_forward_context()) == (None, None, None, None)


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
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
        lambda self, value, sampling, **_: context if value is conditioning else pytest.fail("wrong encoder handoff"),
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
    candidate: SimpleNamespace | None = None,
):
    """Build a bare MiniMaxH3Attention with the attributes ``_packed_batch_supported`` reads.

    ``candidate`` stands for the backend a scheduled step selects; without it the
    layer runs its baseline.

    ``supports_multi_doc`` models the platform-dependent capability probe on
    ``AttentionBackend`` (e.g. FLASH_ATTN returns True on CUDA/ROCm/MUSA and
    False on NPU/XPU); ``backend`` remains only for logging/back-compat.
    """
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    attn = object.__new__(MiniMaxH3Attention)
    attn_backend = _fake_packed_backend(backend, supports_multi_doc)
    selected = attn_backend if candidate is None else candidate
    attn.attention = SimpleNamespace(
        attn_backend=attn_backend,
        effective_attention=lambda: (None, selected, None),
        use_ring=use_ring,
    )
    return attn


def _fake_packed_backend(name: str, supports_multi_doc: bool) -> SimpleNamespace:
    return SimpleNamespace(get_name=lambda: name, supports_multi_doc_packed_varlen=lambda: supports_multi_doc)


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


def test_packed_batch_checks_the_backend_the_step_selects():
    """A scheduled step is packed only when the candidate it runs isolates requests, whatever the baseline."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    isolating = _fake_packed_backend("FLASH_ATTN", True)
    leaking = _fake_packed_backend("XFORMERS", False)
    supported = MiniMaxH3Pipeline._packed_batch_supported

    leaking_step = _fake_attention_module(use_ring=False, candidate=leaking)
    isolating_step = _fake_attention_module(
        use_ring=False, backend="XFORMERS", supports_multi_doc=False, candidate=isolating
    )

    assert supported(_FakeTransformer([leaking_step])) is False
    assert supported(_FakeTransformer([isolating_step])) is True


def _same_progress_states(model, schedules):
    """Two requests at the same step of equal-length sequences, each carrying its own resolved schedule."""
    states = []
    for index, schedule in enumerate(schedules):
        branch, video, audio = _make_branch(
            text_len=9 - 4 * index, latent_t=2 + index, latent_h=4, latent_w=6, audio_t=3, seed=31 + index
        )
        state = _make_state(f"req-{index}", model, branch, video, audio, _sigmas(4, 12.0), _sigmas(4, 3.0))
        _set_step_index(state, 1)
        state.attention_schedule = schedule
        states.append(state)
    return states


def test_scheduled_step_batch_at_one_progress_is_packed():
    """Requests with one schedule at one progress select one profile, so they share a packed forward."""
    model = _ProgressRecordingModel()
    states = _same_progress_states(model, [_sparse_schedule(), _sparse_schedule()])
    serial_states = _same_progress_states(model, [_sparse_schedule(), _sparse_schedule()])
    pipeline = _step_pipeline(model)
    with set_forward_context(), bind_attention_schedule(_sparse_schedule()):
        result = pipeline.denoise_step(InputBatch.make_batch(states), states=states)
    assert result == {state.request_id: None for state in states}

    sigmas = states[0].extra[pipeline_minimax_h3._STEP_SAMPLING_STATE].sigmas[0]
    assert model.progress == [(1, float(sigmas[1]), 4, "sparse", 2)]

    model.progress.clear()
    with set_forward_context(), bind_attention_schedule(_sparse_schedule()):
        result = _step_pipeline(model, packed_batch_supported=False).denoise_step(
            InputBatch.make_batch(serial_states), states=serial_states
        )
    assert result == {state.request_id: None for state in serial_states}
    assert [entry[-1] for entry in model.progress] == [1, 1]
    _assert_same_committed_rows(serial_states, states)


def test_inherited_and_explicit_equal_schedules_resolve_to_one_hashable_value():
    """An omitted schedule and the same ranges sent explicitly resolve to equal, hashable tuples.

    So the pipeline gate packs them: ``test_scheduled_step_batch_at_one_progress_is_packed`` drives
    the gate with two equal resolved schedules. The runner storing the resolved value is pinned by
    ``test_cached_step_reuses_the_schedule_resolved_at_admission``. The scheduler keeps these two
    requests in different batches (the key also records whether the request sent a schedule).
    """
    service = AttentionScheduleConfig(
        profiles={"sparse": AttentionConfig()}, default=[{"start": 0, "end": None, "profile": "sparse"}]
    )
    inherited = resolve_request_attention_schedule(OmniDiffusionSamplingParams(), service)
    explicit = resolve_request_attention_schedule(
        OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": None, "profile": "sparse"}]), service
    )
    assert inherited == explicit == _sparse_schedule()
    assert type(inherited) is tuple and type(explicit) is tuple
    assert hash(inherited) == hash(explicit)
    assert len({inherited, explicit}) == 1


@pytest.mark.parametrize(
    ("schedules", "packed_batch_supported"),
    [
        # Same progress, but the requests carry different schedules, so they can select different profiles.
        ([_sparse_schedule(), (AttentionScheduleRange(0, None, "dense"),)], True),
        # One profile, but its backend does not isolate packed requests.
        ([_sparse_schedule(), _sparse_schedule()], False),
    ],
)
def test_scheduled_step_batch_runs_serially_unless_one_packable_profile(schedules, packed_batch_supported):
    model = _ProgressRecordingModel()
    states = _same_progress_states(model, schedules)
    with set_forward_context(), bind_attention_schedule(_sparse_schedule()):
        _step_pipeline(model, packed_batch_supported=packed_batch_supported).denoise_step(
            InputBatch.make_batch(states), states=states
        )

    assert [entry[-1] for entry in model.progress] == [1, 1]


@pytest.mark.parametrize(
    ("schedule", "packed"),
    [
        ((AttentionScheduleRange(0, None, "sparse"),), True),
        ((AttentionScheduleRange(0, 1, "sparse"), AttentionScheduleRange(1, None, "dense")), False),
    ],
)
def test_packed_gate_sees_the_profile_selected_for_the_published_step(schedule, packed):
    """The gate runs after the step's progress is published, so it reads the step's profile, not the baseline."""
    model = _ProgressRecordingModel()
    states = _same_progress_states(model, [schedule, schedule])
    pipeline = _step_pipeline(model)
    pipeline._packed_batch_supported = lambda transformer: get_attention_schedule_profile() == "sparse"
    with set_forward_context(), bind_attention_schedule(schedule):
        pipeline.denoise_step(InputBatch.make_batch(states), states=states)

    profile = "sparse" if packed else "dense"
    assert [(entry[3], entry[-1]) for entry in model.progress] == ([(profile, 2)] if packed else [(profile, 1)] * 2)


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


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
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
    assert len(model.calls) == 3


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


def _small_dit(monkeypatch, baseline) -> h3.MiniMaxH3DiTModel:
    """A small CPU MiniMax H3 DiT whose blocks use real Attention layers resolved to ``baseline``."""
    # The DiT blocks live in minimax_h3_blocks; the model's own projections stay in minimax_h3_transformer.
    for name in ("ColumnParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(blocks, name, _FakeLinear)
    for name in ("ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(h3, name, _FakeLinear)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(layer, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (baseline, None))
    config = _small_od_config()
    # With the AdaLN cache on, the constructor probes the current CUDA device for the SM120 offload
    # default, which fails when CUDA is built but no device is visible. These checks need no cache.
    config.cache_config = {"minimax_h3_adaln_cache": False}
    model = h3.MiniMaxH3DiTModel(config)
    attention = [block.attn.attention for block in model.blocks]
    assert all(isinstance(a, layer.Attention) and a.attn_backend is baseline for a in attention)
    return model


def _candidate_rejections(attention: layer.Attention, record) -> list[str]:
    return [reason for check in attention._schedule_candidate_checks if (reason := check(record)) is not None]


def test_vdn_attention_accepts_only_vdnh3_candidates(monkeypatch):
    """VDN-H3 checkpoints read the frame window only in VDNH3_ATTN, so other profiles are rejected at startup."""
    monkeypatch.setattr(blocks, "VDNH3HybridAttention", lambda *args, **kwargs: object())
    model = _small_dit(monkeypatch, VDNAttentionBackend)

    model.enable_vdn(config=None)

    vdn = _schedule_candidate("VDNH3_ATTN", mask_free=True)
    flash = _schedule_candidate("FLASH_ATTN", mask_free=True)
    for attention in (block.attn.attention for block in model.blocks):
        assert _candidate_rejections(attention, vdn) == []
        [reason] = _candidate_rejections(attention, flash)
        assert "VDNH3_ATTN" in reason


def test_gateless_dit_accepts_a_vsa_candidate_when_its_baseline_is_vsa(monkeypatch):
    """FastH3 VSA gates only self.transformer (the FL2VA DiT); the combined partition's transformers_ref has none.

    Its self role still resolves to FASTVIDEO_VSA, which the FastH3 VSA contract requires, so one self-role
    profile is checked on both DiTs. The DiTs use real Attention layers, whose baseline the check reads.
    """

    def dit_attention(baseline, *, gated: bool):
        model = _small_dit(monkeypatch, baseline)
        if gated:
            model.enable_vsa_gates()
        return [block.attn.attention for block in model.blocks]

    gated = dit_attention(FastVideoVSABackend, gated=True)
    gateless = dit_attention(FastVideoVSABackend, gated=False)

    # One self-role profile covers both DiTs, so it has to pass on each of them.
    vsa = _schedule_candidate("FASTVIDEO_VSA", mask_free=True)
    assert all(_candidate_rejections(attention, vsa) == [] for attention in gated + gateless)
    # The gated FastH3 DiT still accepts only FASTVIDEO_VSA.
    sdpa = _schedule_candidate("SDPA", attention_mask=True)
    assert all(any("FastH3" in reason for reason in _candidate_rejections(a, sdpa)) for a in gated)
    # A gateless DiT on another baseline still rejects the VSA candidate.
    dense = dit_attention(SDPABackend, gated=False)
    assert all(any("VSA compression gate" in reason for reason in _candidate_rejections(a, vsa)) for a in dense)
