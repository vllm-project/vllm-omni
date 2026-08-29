# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU test: drive a real registered pipeline's step-execution contract
through ``run_step_execution_to_completion`` and check it reproduces the
per-request-alone result, on CUDA.

Reuses the MiniMax H3 pipeline (a real ``SupportsStepExecution`` model) and
the same stand-in-DiT technique as
``tests/diffusion/models/minimax_h3/test_minimax_h3_step_execution.py``: the
transformer and VAE are the only weight-bearing components, so they are
swapped for tiny CUDA stand-ins instead of downloading real checkpoints.
Everything else -- packing, the real ``denoise_step``/``step_scheduler``/
``post_decode`` methods, and the bridge's own batching/pruning logic -- runs
for real on the GPU.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_HIDDEN = 8


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


class _SegmentMeanModel:
    """Stand-in DiT identical in spirit to the CPU contract test's: cheap,
    deterministic, and sensitive to packed-batch boundaries so a bridge bug
    (wrong offsets, cross-request bleed) shows up as a numeric mismatch."""

    def __call__(self, **kwargs):
        x = kwargs["x"][0]
        audio_x = kwargs["audio_x"][0]
        bounds = kwargs["packed_seq_params"]["cu_seqlens_q"].tolist()
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


def _make_branch(*, device, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int, seed: int):
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
    generator = torch.Generator(device=device).manual_seed(seed)
    text_embeddings = torch.randn(text_len, _HIDDEN, generator=generator, device=device, dtype=torch.float32)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=text_embeddings,
        token_tags=packed["token_tags"],
        device=device,
    )
    video_rows = torch.randn(int(branch.img_pos.shape[0]), 96, generator=generator, device=device, dtype=torch.float32)
    audio_rows = torch.randn(
        int(branch.audio_pos.shape[0]), 32, generator=generator, device=device, dtype=torch.float32
    )
    return branch, video_rows, audio_rows


def _sigmas(num_points: int, shift: float) -> list[float]:
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    return minimax_h3_time_shift_sigmas(num_steps=num_points, shift_scale=shift)


def _step_pipeline(model, device):
    """Real ``MiniMaxH3Pipeline`` instance carrying only what the step
    methods and ``post_decode`` touch -- same pattern as the CPU contract
    test's ``_step_pipeline``, but with the VAE decode stubbed too (that's
    weight-bearing and orthogonal to what the bridge orchestrates)."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.transformer = model
    pipeline.device = device
    pipeline._transformer_for_task = lambda task: model
    pipeline._packed_batch_supported = lambda transformer: True
    pipeline.od_config = types.SimpleNamespace()
    # Stub only the weight-bearing VAE decode; _unpack_denoised_rows (real,
    # pure-tensor bookkeeping) still runs.
    pipeline.decode = lambda video_latent, audio_latent, **kwargs: (video_latent, audio_latent)
    return pipeline


@dataclass
class _FakeSamplingParams:
    seed: int | None = None
    generator: Any = None
    generator_device: str | None = None


@dataclass
class _FakeRequest:
    prompt: str
    sampling_params: _FakeSamplingParams
    request_id: str
    kv_sender_info: dict | None = None
    prepared_layout: Any | None = None


def _install_fake_prepare_encode(pipeline, precomputed: dict[str, dict[str, Any]], model):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    def _prepare_encode(self, state, **kwargs):
        del kwargs
        spec = precomputed[state.request_id]
        state.latents = spec["video_rows"].clone()
        state.timesteps = torch.tensor(
            [1.0 - sigma for sigma in spec["sigmas_video"][:-1]],
            dtype=torch.float32,
            device=spec["video_rows"].device,
        )
        state.step_index = 0
        state.extra = {
            mod._STEP_BRANCH: spec["branch"],
            mod._STEP_TRANSFORMER: model,
            mod._STEP_AUDIO_ROWS: spec["audio_rows"].clone(),
            mod._STEP_COND_ANCHOR: None,
            mod._STEP_AUDIO_ANCHOR: None,
            mod._STEP_SIGMAS_VIDEO: spec["sigmas_video"],
            mod._STEP_SIGMAS_AUDIO: spec["sigmas_audio"],
            mod._STEP_SHAPE: spec["shape"],
        }
        return state

    pipeline.prepare_encode = types.MethodType(_prepare_encode, pipeline)


def test_step_execution_bridge_matches_solo_requests_on_gpu():
    """Two co-batched, differently-sized, differently-scheduled requests must
    land where they would have landed run alone -- through the real pipeline
    contract, driven by the generic bridge, on CUDA."""
    device = _require_cuda()
    from vllm_omni.diffusion.worker.step_execution_batch import run_step_execution_to_completion

    model = _SegmentMeanModel()
    specs = [
        dict(
            text_len=46,
            latent_t=7,
            latent_h=2,
            latent_w=4,
            audio_t=2,
            seed=7,
            shape=dict(height=112, width=64, latent_t=7, latent_h=2, latent_w=4, audio_t=2),
        ),
        dict(
            text_len=9,
            latent_t=2,
            latent_h=4,
            latent_w=6,
            audio_t=3,
            seed=6,
            shape=dict(height=64, width=96, latent_t=2, latent_h=4, latent_w=6, audio_t=3),
        ),
    ]
    schedules = [(_sigmas(6, 12.0), _sigmas(6, 3.0)), (_sigmas(4, 12.0), _sigmas(4, 3.0))]

    def build_spec(spec, sigmas_video, sigmas_audio):
        branch, video_rows, audio_rows = _make_branch(
            device=device,
            text_len=spec["text_len"],
            latent_t=spec["latent_t"],
            latent_h=spec["latent_h"],
            latent_w=spec["latent_w"],
            audio_t=spec["audio_t"],
            seed=spec["seed"],
        )
        return {
            "branch": branch,
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "sigmas_video": sigmas_video,
            "sigmas_audio": sigmas_audio,
            "shape": spec["shape"],
        }

    # Reference: run each request alone, driving the *real* pipeline methods
    # by hand (no bridge involved), exactly like the CPU contract test does.
    from vllm_omni.diffusion.worker.utils import StepRequestState

    reference_outputs = []
    for spec, (sigmas_video, sigmas_audio) in zip(specs, schedules):
        solo_spec = build_spec(spec, sigmas_video, sigmas_audio)
        pipeline = _step_pipeline(model, device)
        _install_fake_prepare_encode(pipeline, {"solo": solo_spec}, model)
        state = pipeline.prepare_encode(StepRequestState(request_id="solo", sampling=types.SimpleNamespace()))
        while not state.denoise_completed:
            noise_pred = pipeline.denoise_step(types.SimpleNamespace(states=(state,)), states=[state])
            pipeline.step_scheduler(state, noise_pred)
        result = pipeline.post_decode(state)
        reference_outputs.append(result.output)

    # Now run both requests together through the generic bridge.
    batch_specs = {
        f"req-{i}": build_spec(spec, sigmas_video, sigmas_audio)
        for i, (spec, (sigmas_video, sigmas_audio)) in enumerate(zip(specs, schedules))
    }
    pipeline = _step_pipeline(model, device)
    _install_fake_prepare_encode(pipeline, batch_specs, model)
    requests = [
        _FakeRequest(prompt=f"prompt-{i}", sampling_params=_FakeSamplingParams(), request_id=f"req-{i}")
        for i in range(len(specs))
    ]

    outputs = run_step_execution_to_completion(pipeline, requests)

    assert len(outputs) == 2
    for i, output in enumerate(outputs):
        video, audio = output.output
        ref_video, ref_audio = reference_outputs[i]
        assert video.is_cuda and audio.is_cuda
        torch.testing.assert_close(video, ref_video)
        torch.testing.assert_close(audio, ref_audio)
