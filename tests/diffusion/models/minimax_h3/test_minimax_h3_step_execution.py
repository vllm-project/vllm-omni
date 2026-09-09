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


def _sigmas(num_points: int, shift: float) -> list[float]:
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    return minimax_h3_time_shift_sigmas(num_steps=num_points, shift_scale=shift)


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
    pipeline.transformer = model
    pipeline.device = torch.device("cpu")
    pipeline._transformer_for_task = lambda task: model
    pipeline._packed_batch_supported = lambda transformer: packed_batch_supported
    return pipeline


def test_step_execution_matches_request_mode_denoise_loop():
    """Stepping through the contract must reproduce the request-mode loop."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    model = _SegmentMeanModel()
    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=5)
    sigmas_video = _sigmas(6, 12.0)
    sigmas_audio = _sigmas(6, 3.0)

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

    pipeline = _step_pipeline(model)
    state = _make_state("req-0", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
    input_batch = SimpleNamespace(states=(state,))

    steps = 0
    while not state.denoise_completed:
        noise_pred = pipeline.denoise_step(input_batch, states=[state])
        pipeline.step_scheduler(state, noise_pred)
        steps += 1

    assert steps == len(sigmas_video) - 1
    assert state.total_steps == steps
    torch.testing.assert_close(state.latents, reference_video)
    torch.testing.assert_close(state.extra[mod._STEP_AUDIO_ROWS], reference_audio)


def test_batched_step_execution_matches_independent_requests():
    """Two co-batched requests must land where they would have landed alone."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    model = _SegmentMeanModel()
    specs = [
        # Exactly 64 packed rows exercises the no-padding-tail boundary case.
        dict(text_len=46, latent_t=7, latent_h=2, latent_w=4, audio_t=2, seed=7),
        dict(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=6),
    ]
    # Different step counts, so the batch composition changes mid-flight.
    schedules = [(_sigmas(6, 12.0), _sigmas(6, 3.0)), (_sigmas(4, 12.0), _sigmas(4, 3.0))]

    pipeline = _step_pipeline(model)

    alone: list[tuple[torch.Tensor, torch.Tensor]] = []
    for spec, (sigmas_video, sigmas_audio) in zip(specs, schedules):
        branch, video_rows, audio_rows = _make_branch(**spec)
        state = _make_state("solo", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio)
        while not state.denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state]))
        alone.append((state.latents, state.extra[mod._STEP_AUDIO_ROWS]))

    states = []
    for index, (spec, (sigmas_video, sigmas_audio)) in enumerate(zip(specs, schedules)):
        branch, video_rows, audio_rows = _make_branch(**spec)
        if index == 0:
            assert branch.used_len == branch.seq_len
        states.append(_make_state(f"req-{index}", model, branch, video_rows, audio_rows, sigmas_video, sigmas_audio))

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


def test_prepare_encode_seeds_runner_visible_state(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    branch, video_rows, audio_rows = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=8)
    sigmas_video = _sigmas(6, 12.0)
    sigmas_audio = _sigmas(6, 3.0)
    context = {
        "height": 96,
        "width": 64,
        "latent_t": 2,
        "latent_h": 4,
        "latent_w": 6,
        "audio_t": 3,
        **{key: None for key in mod._MINIMAX_H3_DENOISE_INPUT_KEYS},
    }

    pipeline = _step_pipeline(_SegmentMeanModel())
    monkeypatch.setattr(mod.MiniMaxH3Pipeline, "_extract_prompt", staticmethod(lambda _: ("a prompt", {})))
    monkeypatch.setattr(mod.MiniMaxH3Pipeline, "_prepare_request_inputs", lambda self, **_: context)
    monkeypatch.setattr(
        mod.MiniMaxH3Pipeline,
        "_build_denoise_inputs",
        lambda self, **_: {
            "branch": branch,
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "cond_anchor": None,
            "audio_anchor": None,
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
    assert state.extra[mod._STEP_SHAPE]["height"] == 96


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


def test_broadcast_rank0_exception_single_rank_reraises():
    """Single-rank execution has no group; the helper just reraises."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _broadcast_rank0_exception,
    )
    from vllm_omni.errors import OmniClientError

    _broadcast_rank0_exception(None)
    with pytest.raises(OmniClientError, match="bad ref"):
        _broadcast_rank0_exception(OmniClientError("bad ref"))


def test_broadcast_rank0_exception_propagates_to_non_zero_ranks(monkeypatch):
    """A rank-0 error becomes a matching client error on every other DiT rank."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.errors import OmniClientError

    def fake_rank_world(rank):
        return lambda: (object(), rank, 4)

    def make_broadcast(rank0_payload):
        def fake_broadcast(payload_list, *, src, group):
            payload_list[0] = rank0_payload

        return fake_broadcast

    monkeypatch.setattr(mod, "_dit_rank_world", fake_rank_world(0))
    err = OmniClientError("invalid reference-video file", status_code=422, error_type="UnprocessableEntityError")
    rank0_payload = {
        "type": type(err).__name__,
        "message": str(err),
        "status_code": err.status_code,
        "error_type": err.error_type,
    }
    monkeypatch.setattr(mod.dist, "broadcast_object_list", make_broadcast(rank0_payload))
    with pytest.raises(OmniClientError) as rank0_info:
        mod._broadcast_rank0_exception(err)
    assert rank0_info.value is err

    monkeypatch.setattr(mod, "_dit_rank_world", fake_rank_world(2))
    with pytest.raises(OmniClientError) as rank2_info:
        mod._broadcast_rank0_exception(None)
    assert rank2_info.value.status_code == 422
    assert rank2_info.value.error_type == "UnprocessableEntityError"
    assert "invalid reference-video file" in str(rank2_info.value)
