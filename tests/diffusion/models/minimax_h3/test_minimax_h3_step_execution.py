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


def _make_state(request_id: str, model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.worker.utils import StepRequestState

    state = StepRequestState(request_id=request_id, sampling=SimpleNamespace())
    state.latents = video_rows.clone()
    state.timesteps = torch.tensor([1.0 - sigma for sigma in sigmas_video[:-1]], dtype=torch.float32)
    state.step_index = 0
    state.extra = {
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
    input_batch = SimpleNamespace(states=(state,))

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
        velocity = pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state])
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
            pipeline.step_scheduler(state, pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state]))
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
            pipeline.step_scheduler(state, pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state]))
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
        _step_pipeline(model).denoise_step(SimpleNamespace(states=tuple(states)), states=states)
    finally:
        fc._forward_context = original

    assert recorder.denoise_step_idx is None
    assert recorder.denoise_timestep is None


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


@pytest.mark.parametrize("execution", ["request", "step"])
def test_pdd_base_pdd_switch_at_execution_boundary(execution):
    """A base request after PDD must restore projections without manual disarm."""
    from contextlib import nullcontext

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.pdd import PDDAdapter, PDDConfig

    class HeadModel(_SegmentMeanModel):
        def __init__(self):
            super().__init__()
            self.final_layer = torch.nn.Module()
            self.final_layer.video_out = torch.nn.Linear(96, 96)
            self.final_layer.audio_out = torch.nn.Linear(32, 32)

        def __call__(self, **kwargs):
            video, audio = super().__call__(**kwargs)
            return self.final_layer.video_out(video)[0], self.final_layer.audio_out(audio)[0]

    torch.manual_seed(42)
    model = HeadModel()
    cfg = PDDConfig()
    vp, ap = cfg.plans()
    adapter = PDDAdapter(config=cfg, lora_id=1, video_plans=vp, audio_plans=ap)
    adapter.install_heads(model)
    with torch.no_grad():
        for head in (model.final_layer.video_out, model.final_layer.audio_out):
            head.weight.add_(0.01)
            head.bias.add_(0.1)
    branch, video, audio = _make_branch(text_len=3, latent_t=1, latent_h=2, latent_w=2, audio_t=2, seed=5)
    sv, sa = _sigmas(9, 12.0), _sigmas(9, 3.0)
    pipeline = _step_pipeline(model)
    pipeline._resident_dit_layers_on_device = lambda **kwargs: nullcontext()
    pipeline.progress_bar = lambda **kwargs: nullcontext(SimpleNamespace(update=lambda: None))
    pipeline._unpack_denoised_rows = lambda branch, v, a, **kwargs: (v, a)
    pipeline._build_denoise_inputs = lambda **kwargs: {
        "branch": branch,
        "video_rows": video.clone(),
        "audio_rows": audio.clone(),
        "cond_anchor": None,
        "audio_anchor": None,
        "video_edit": None,
        "audio_edit": None,
        "sigmas_video": sv,
        "sigmas_audio": sa,
    }

    def run(active):
        if execution == "request":
            return pipeline.diffuse(
                task="t2va",
                text_embeddings=torch.empty(0),
                text_tags=torch.empty(0),
                seed=5,
                latent_t=1,
                latent_h=2,
                latent_w=2,
                audio_t=2,
                num_frames=5,
                num_steps=9,
                video_shift=12.0,
                audio_shift=3.0,
                base_schedule=None,
                visual_condition=None,
                visual_condition_shape=None,
                audio_condition=None,
                ref_audio_t=None,
                pdd_adapter=active,
            )
        state = _make_state("switch", model, branch, video, audio, sv, sa)
        state.extra[mod._STEP_PDD_ADAPTER] = active
        while not state.denoise_completed:
            pred = pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state])
            pipeline.step_scheduler(state, pred)
        return state.latents, state.extra[mod._STEP_AUDIO_ROWS]

    base_before = run(None)
    pdd_before = run(adapter)
    base_after = run(None)
    pdd_after = run(adapter)
    for expected, restored in zip(base_before, base_after):
        torch.testing.assert_close(expected, restored)
    for expected, restored in zip(pdd_before, pdd_after):
        torch.testing.assert_close(expected, restored)
    assert not torch.allclose(base_before[0], pdd_before[0])
