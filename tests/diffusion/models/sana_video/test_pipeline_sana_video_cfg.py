# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CFG dispatch tests for the native SANA-Video pipelines.

These exercise the CFG world-size-1 batch path, the CFG world-size-2 dispatch,
and the fp32 combine semantics on CPU / mocked distributed. The real all-gather
path is covered by the multi-process suite, not here.
"""

import pytest
import torch

from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import SanaVideoPipeline
from vllm_omni.diffusion.models.sana_video.pipeline_sana_video_i2v import SanaImageToVideoPipeline
from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
    SanaVideoTransformerOutput,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_predict_noise_calls_transformer_with_return_dict_false():
    """predict_noise must call the transformer with return_dict=False.

    The transformer returns a SanaVideoTransformerOutput dataclass when
    return_dict is True (its default), which is not subscriptable. The CFG
    mixin's default predict_noise does ``result[0]`` and would therefore
    raise TypeError on SANA; the pipeline override must force return_dict=False
    so the tuple form is returned and indexed.
    """
    captured = {}
    expected = torch.full((1, 2, 3), 7.0)

    class _SpyTransformer:
        def __call__(self, *args, return_dict=True, **kwargs):
            captured["return_dict"] = return_dict
            # Mirror the real transformer: tuple on False, dataclass on True.
            return (expected,) if not return_dict else SanaVideoTransformerOutput(sample=expected)

    pipe = SanaVideoPipeline.__new__(SanaVideoPipeline)  # bypass heavy __init__
    pipe.transformer = _SpyTransformer()

    out = pipe.predict_noise(
        torch.zeros(1, 2, 3),
        encoder_hidden_states=torch.zeros(1, 1, 4),
        encoder_attention_mask=None,
        timestep=torch.zeros(1),
    )

    assert captured["return_dict"] is False
    assert torch.equal(out, expected)


class _IdentityScheduler:
    """Scheduler stub whose step is a no-op, so diffuse tests isolate the CFG
    dispatch from real scheduler math."""

    order = 1

    def step(self, noise_pred, t, latents, **kwargs):
        return (latents,)


def _set_cfg_world_size(monkeypatch, n):
    """Pipelines always run inside an initialized CFG group; unit tests stub its
    world size instead of spinning up torch.distributed. Patch both pipeline
    modules since each resolves the symbol in its own namespace."""
    import vllm_omni.diffusion.distributed.cfg_parallel as cfg
    import vllm_omni.diffusion.models.sana_video.pipeline_sana_video as t2v
    import vllm_omni.diffusion.models.sana_video.pipeline_sana_video_i2v as i2v

    monkeypatch.setattr(t2v, "get_classifier_free_guidance_world_size", lambda: n, raising=False)
    monkeypatch.setattr(i2v, "get_classifier_free_guidance_world_size", lambda: n, raising=False)
    monkeypatch.setattr(cfg, "get_classifier_free_guidance_world_size", lambda: n, raising=False)
    monkeypatch.setattr(t2v, "model_parallel_is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(i2v, "model_parallel_is_initialized", lambda: True, raising=False)


def _bare_t2v_pipeline(transformer, guidance_scale):
    pipe = SanaVideoPipeline.__new__(SanaVideoPipeline)  # bypass heavy __init__
    pipe.transformer = transformer
    pipe.scheduler = _IdentityScheduler()
    pipe._interrupt = False
    pipe._guidance_scale = guidance_scale
    return pipe


def test_cfg_world_size_1_does_single_batch2_forward_per_step(monkeypatch):
    """CFG world size 1 with guidance must do exactly one batch-2 transformer
    forward per step (the batch path), not the mixin's sequential two-forward
    path."""
    _set_cfg_world_size(monkeypatch, 1)
    batch_dims = []

    class _SpyTransformer:
        dtype = torch.float32

        def __call__(self, hidden_states, *, encoder_hidden_states, encoder_attention_mask, timestep, return_dict):
            batch_dims.append(hidden_states.shape[0])
            # out_channels == in_channels here (no learned-sigma split).
            return (torch.zeros_like(hidden_states),)

    b = 1
    pipe = _bare_t2v_pipeline(_SpyTransformer(), guidance_scale=6.0)

    out = pipe.diffuse(
        latents=torch.randn(b, 4, 2, 4, 4),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=torch.randn(b, 3, 8),
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=torch.randn(b, 3, 8),
        negative_prompt_attention_mask=torch.ones(b, 3),
        guidance_scale=6.0,
        extra_step_kwargs={},
        dtype=torch.float32,
        output_slice=None,
    )

    assert len(batch_dims) == 1, f"expected 1 forward this step, got {len(batch_dims)}"
    assert batch_dims[0] == 2 * b, f"expected batch-2 (neg+pos) forward, got batch {batch_dims[0]}"
    assert out.shape == (b, 4, 2, 4, 4)


class _DeterministicTransformer:
    """Learned-sigma transformer whose output depends on the row's encoder
    embeds and timestep, so uncond != text and the CFG combine is non-trivial."""

    dtype = torch.float32

    def __init__(self, latent_channels):
        self.out_channels = 2 * latent_channels  # learned sigma

    def __call__(self, hidden_states, *, encoder_hidden_states, encoder_attention_mask, timestep, return_dict):
        b, c = hidden_states.shape[0], hidden_states.shape[1]
        sig = encoder_hidden_states.float().mean(dim=(1, 2)).view(b, 1, 1, 1, 1)
        base = hidden_states.float() + sig + timestep.float().view(b, 1, 1, 1, 1)
        return (base.repeat(1, self.out_channels // c, 1, 1, 1),)


class _LinearScheduler:
    """Deterministic stepping scheduler so multi-step divergence compounds."""

    order = 1

    def step(self, noise_pred, t, latents, return_dict=False, **kwargs):
        return (latents - 0.1 * noise_pred,)


def _pr1_reference_loop(
    transformer,
    scheduler,
    latents,
    timesteps,
    prompt_embeds,
    prompt_mask,
    negative_embeds,
    negative_mask,
    guidance_scale,
    latent_channels,
    dtype,
):
    """The original CFG1 denoise math, verbatim, as the parity oracle."""
    prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
    prompt_mask = torch.cat([negative_mask, prompt_mask], dim=0)
    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2)
        timestep = t.expand(latent_model_input.shape[0])
        noise_pred = transformer(
            latent_model_input.to(dtype=dtype),
            encoder_hidden_states=prompt_embeds.to(dtype=dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timestep,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if transformer.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents


def test_cfg1_refactored_diffuse_is_bit_identical_to_pr1_loop(monkeypatch):
    """The refactored diffuse must reproduce the original CFG1 latents bit-for-bit
    across multiple steps, preserving the TP1/CFG1 golden output."""
    _set_cfg_world_size(monkeypatch, 1)
    torch.manual_seed(0)
    b, c = 2, 4
    latents0 = torch.randn(b, c, 2, 4, 4)
    timesteps = torch.tensor([900.0, 600.0, 300.0])
    prompt_embeds = torch.randn(b, 3, 8)
    negative_embeds = torch.randn(b, 3, 8)
    prompt_mask = torch.ones(b, 3)
    negative_mask = torch.ones(b, 3)
    guidance_scale = 6.0

    transformer = _DeterministicTransformer(latent_channels=c)
    scheduler = _LinearScheduler()

    ref = _pr1_reference_loop(
        transformer,
        scheduler,
        latents0.clone(),
        timesteps,
        prompt_embeds,
        prompt_mask,
        negative_embeds,
        negative_mask,
        guidance_scale,
        latent_channels=c,
        dtype=torch.float32,
    )

    pipe = _bare_t2v_pipeline(transformer, guidance_scale=guidance_scale)
    pipe.scheduler = scheduler
    got = pipe.diffuse(
        latents=latents0.clone(),
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        negative_prompt_embeds=negative_embeds,
        negative_prompt_attention_mask=negative_mask,
        guidance_scale=guidance_scale,
        extra_step_kwargs={},
        dtype=torch.float32,
        output_slice=c,
    )

    assert torch.equal(got, ref)


def test_combine_cfg_noise_upcasts_to_fp32_before_combine():
    """combine_cfg_noise must upcast positive/negative to fp32 before the CFG
    formula, so the CFG2 (bf16 all-gather) combine is bit-identical to CFG1's
    fp32 combine."""
    pipe = SanaVideoPipeline.__new__(SanaVideoPipeline)
    torch.manual_seed(0)
    pos = torch.randn(1, 4, 2, 4, 4).bfloat16()
    neg = torch.randn(1, 4, 2, 4, 4).bfloat16()
    scale = 6.0

    out = pipe.combine_cfg_noise(pos, neg, scale, cfg_normalize=False)

    expected = neg.float() + scale * (pos.float() - neg.float())
    assert out.dtype == torch.float32
    assert torch.equal(out, expected)


def test_combine_cfg_noise_accepts_tuple_predictions_from_gather():
    """The CFG-parallel gather path hands combine_cfg_noise single-element
    tuples, not bare tensors; the override must upcast and combine them."""
    pipe = SanaVideoPipeline.__new__(SanaVideoPipeline)
    torch.manual_seed(0)
    pos = torch.randn(1, 4, 2, 4, 4).bfloat16()
    neg = torch.randn(1, 4, 2, 4, 4).bfloat16()
    scale = 6.0

    out = pipe.combine_cfg_noise((pos,), (neg,), scale, cfg_normalize=False)

    expected = neg.float() + scale * (pos.float() - neg.float())
    assert out.dtype == torch.float32
    assert torch.equal(out, expected)


def test_cfg2_dispatch_builds_independent_branch_kwargs(monkeypatch):
    """In CFG world size 2, diffuse must build separate batch-B positive/negative
    kwargs (no pre-cat), and call the mixin with cfg_normalize=False and the
    learned-sigma output_slice."""
    _set_cfg_world_size(monkeypatch, 2)

    b, c = 1, 4
    captured = {}

    def _spy_cfg(**kwargs):
        captured.update(kwargs)
        return torch.zeros(b, c, 2, 4, 4)

    prompt_embeds = torch.randn(b, 3, 8)
    negative_embeds = torch.randn(b, 3, 8)
    dtype = torch.float32

    pipe = _bare_t2v_pipeline(transformer=None, guidance_scale=6.0)
    pipe.predict_noise_maybe_with_cfg = _spy_cfg

    pipe.diffuse(
        latents=torch.randn(b, c, 2, 4, 4),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=negative_embeds,
        negative_prompt_attention_mask=torch.zeros(b, 3),
        guidance_scale=6.0,
        extra_step_kwargs={},
        dtype=dtype,
        output_slice=c,
    )

    assert captured, "predict_noise_maybe_with_cfg was not called in CFG world size 2"
    assert captured["do_true_cfg"] is True
    assert captured["true_cfg_scale"] == 6.0
    assert captured["cfg_normalize"] is False
    assert captured["output_slice"] == c

    pos = captured["positive_kwargs"]
    neg = captured["negative_kwargs"]
    assert pos["hidden_states"].shape[0] == b, "positive branch must be batch-B, not pre-concatenated"
    assert neg["hidden_states"].shape[0] == b, "negative branch must be batch-B, not pre-concatenated"
    assert pos["timestep"].shape[0] == b
    assert torch.equal(pos["encoder_hidden_states"], prompt_embeds.to(dtype))
    assert torch.equal(neg["encoder_hidden_states"], negative_embeds.to(dtype))


# ── I2V CFG dispatch ──


class _I2VSpyScheduler:
    """Records the frame count it steps on and mutates non-first frames so
    first-frame preservation is observable."""

    order = 1

    def __init__(self):
        self.stepped_frame_counts = []

    def step(self, noise_pred, t, latents, return_dict=False, **kwargs):
        self.stepped_frame_counts.append(latents.shape[2])
        return (latents + 1.0,)


def _bare_i2v_pipeline(transformer, scheduler, guidance_scale):
    pipe = SanaImageToVideoPipeline.__new__(SanaImageToVideoPipeline)
    pipe.transformer = transformer
    pipe.scheduler = scheduler
    pipe._interrupt = False
    pipe._guidance_scale = guidance_scale
    return pipe


def _conditioning_mask(b, frames):
    mask = torch.zeros(b, 1, frames, 1, 1)
    mask[:, :, 0] = 1
    return mask


def test_i2v_cfg1_preserves_first_frame_and_steps_only_tail(monkeypatch):
    """I2V CFG1 must keep the conditioning first frame bit-exact and only run the
    scheduler on frames [1:]."""
    _set_cfg_world_size(monkeypatch, 1)
    b, c, frames = 1, 4, 3

    class _SpyTransformer:
        dtype = torch.float32
        out_channels = 2 * c

        def __call__(self, hidden_states, *, encoder_hidden_states, encoder_attention_mask, timestep, return_dict):
            return (torch.zeros(hidden_states.shape[0], 2 * c, frames, 4, 4),)

    scheduler = _I2VSpyScheduler()
    pipe = _bare_i2v_pipeline(_SpyTransformer(), scheduler, guidance_scale=6.0)

    latents0 = torch.randn(b, c, frames, 4, 4)
    out = pipe.diffuse(
        latents=latents0.clone(),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=torch.randn(b, 3, 8),
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=torch.randn(b, 3, 8),
        negative_prompt_attention_mask=torch.ones(b, 3),
        guidance_scale=6.0,
        conditioning_mask=_conditioning_mask(b, frames),
        extra_step_kwargs={},
        dtype=torch.float32,
        output_slice=c,
    )

    assert torch.equal(out[:, :, :1], latents0[:, :, :1]), "first (conditioning) frame must be unchanged"
    assert torch.equal(out[:, :, 1:], latents0[:, :, 1:] + 1.0), "tail frames must be the scheduler output"
    assert scheduler.stepped_frame_counts == [frames - 1], "scheduler must step only on frames [1:]"


def test_i2v_cfg2_dispatch_builds_masked_branch_kwargs(monkeypatch):
    """I2V CFG world size 2 must build batch-B per-branch kwargs with the
    conditioning-frame timestep zeroed (mask not concatenated) and preserve the
    first frame."""
    _set_cfg_world_size(monkeypatch, 2)
    b, c, frames = 1, 4, 3
    captured = {}

    def _spy_cfg(**kwargs):
        captured.update(kwargs)
        return torch.zeros(b, c, frames, 4, 4)

    scheduler = _I2VSpyScheduler()
    pipe = _bare_i2v_pipeline(transformer=None, scheduler=scheduler, guidance_scale=6.0)
    pipe.predict_noise_maybe_with_cfg = _spy_cfg

    prompt_embeds = torch.randn(b, 3, 8)
    negative_embeds = torch.randn(b, 3, 8)
    latents0 = torch.randn(b, c, frames, 4, 4)
    mask = _conditioning_mask(b, frames)

    out = pipe.diffuse(
        latents=latents0.clone(),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=negative_embeds,
        negative_prompt_attention_mask=torch.zeros(b, 3),
        guidance_scale=6.0,
        conditioning_mask=mask,
        extra_step_kwargs={},
        dtype=torch.float32,
        output_slice=c,
    )

    assert captured["do_true_cfg"] is True
    assert captured["cfg_normalize"] is False
    assert captured["output_slice"] == c
    pos = captured["positive_kwargs"]
    neg = captured["negative_kwargs"]
    assert pos["hidden_states"].shape[0] == b, "positive branch must be batch-B"
    assert neg["hidden_states"].shape[0] == b, "negative branch must be batch-B"
    assert tuple(pos["timestep"].shape) == tuple(mask.shape), "mask must not be concatenated in CFG2"
    assert torch.equal(pos["timestep"][:, :, 0], torch.zeros(b, 1, 1, 1)), "conditioning frame timestep must be 0"
    assert torch.equal(pos["encoder_hidden_states"], prompt_embeds)
    assert torch.equal(neg["encoder_hidden_states"], negative_embeds)
    assert torch.equal(out[:, :, :1], latents0[:, :, :1]), "first frame must be preserved"


def test_diffuse_runs_cfg_parallel_validity_check(monkeypatch, mocker):
    """diffuse must run the CFG-parallel validity check so a scale<=1 or missing
    negative prompt warns the user instead of silently wasting a rank."""
    _set_cfg_world_size(monkeypatch, 2)

    b = 1
    pipe = _bare_t2v_pipeline(transformer=None, guidance_scale=6.0)
    pipe.predict_noise_maybe_with_cfg = lambda **kwargs: torch.zeros(b, 4, 2, 4, 4)
    check = mocker.patch.object(SanaVideoPipeline, "check_cfg_parallel_validity", return_value=True)

    pipe.diffuse(
        latents=torch.randn(b, 4, 2, 4, 4),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=torch.randn(b, 3, 8),
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=torch.randn(b, 3, 8),
        negative_prompt_attention_mask=torch.ones(b, 3),
        guidance_scale=6.0,
        extra_step_kwargs={},
        dtype=torch.float32,
        output_slice=None,
    )

    check.assert_called_once_with(6.0, True)


def test_i2v_diffuse_runs_cfg_parallel_validity_check(monkeypatch, mocker):
    """I2V has its own diffuse; it must also run the CFG-parallel validity check."""
    _set_cfg_world_size(monkeypatch, 2)
    b, c, frames = 1, 4, 3

    pipe = _bare_i2v_pipeline(None, _I2VSpyScheduler(), guidance_scale=6.0)
    pipe.predict_noise_maybe_with_cfg = lambda **kwargs: torch.zeros(b, c, frames, 4, 4)
    check = mocker.patch.object(SanaVideoPipeline, "check_cfg_parallel_validity", return_value=True)

    pipe.diffuse(
        latents=torch.randn(b, c, frames, 4, 4),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=torch.randn(b, 3, 8),
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=torch.randn(b, 3, 8),
        negative_prompt_attention_mask=torch.ones(b, 3),
        guidance_scale=6.0,
        conditioning_mask=_conditioning_mask(b, frames),
        extra_step_kwargs={},
        dtype=torch.float32,
        output_slice=c,
    )

    check.assert_called_once_with(6.0, True)


def test_diffuse_runs_without_initialized_parallel_groups():
    """Single-process runs (unit tests, offline scripts) never initialize the
    distributed groups; diffuse must fall back to the serial path instead of
    tripping the parallel-state assertion. No world-size patching on purpose."""

    class _SpyTransformer:
        dtype = torch.float32

        def __call__(self, hidden_states, *, encoder_hidden_states, encoder_attention_mask, timestep, return_dict):
            return (torch.zeros_like(hidden_states),)

    b = 1
    pipe = _bare_t2v_pipeline(_SpyTransformer(), guidance_scale=6.0)

    out = pipe.diffuse(
        latents=torch.randn(b, 4, 2, 4, 4),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=torch.randn(b, 3, 8),
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=torch.randn(b, 3, 8),
        negative_prompt_attention_mask=torch.ones(b, 3),
        guidance_scale=6.0,
        extra_step_kwargs={},
        dtype=torch.float32,
        output_slice=None,
    )

    assert out.shape == (b, 4, 2, 4, 4)


def test_diffuse_propagates_group_error_when_parallel_initialized(monkeypatch):
    """Inside an initialized parallel run a failing CFG-group lookup is a real
    error and must propagate instead of silently flipping to the serial path."""
    import vllm_omni.diffusion.models.sana_video.pipeline_sana_video as t2v

    monkeypatch.setattr(t2v, "model_parallel_is_initialized", lambda: True, raising=False)

    def _broken_group_lookup():
        raise AssertionError("classifier_free_guidance group is not initialized")

    monkeypatch.setattr(t2v, "get_classifier_free_guidance_world_size", _broken_group_lookup)

    b = 1
    pipe = _bare_t2v_pipeline(None, guidance_scale=6.0)

    with pytest.raises(AssertionError, match="classifier_free_guidance"):
        pipe.diffuse(
            latents=torch.randn(b, 4, 2, 4, 4),
            timesteps=torch.tensor([1000.0]),
            prompt_embeds=torch.randn(b, 3, 8),
            prompt_attention_mask=torch.ones(b, 3),
            negative_prompt_embeds=torch.randn(b, 3, 8),
            negative_prompt_attention_mask=torch.ones(b, 3),
            guidance_scale=6.0,
            extra_step_kwargs={},
            dtype=torch.float32,
            output_slice=None,
        )


def test_t2v_forwards_eta_and_generator_into_scheduler(monkeypatch):
    """T2V must keep passing eta/generator (extra_step_kwargs) into scheduler.step."""
    _set_cfg_world_size(monkeypatch, 1)
    captured = {}

    class _KwargScheduler:
        order = 1

        def step(self, noise_pred, t, latents, **kwargs):
            captured.update(kwargs)
            return (latents,)

    class _SpyTransformer:
        dtype = torch.float32

        def __call__(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states),)

    b = 1
    pipe = _bare_t2v_pipeline(_SpyTransformer(), guidance_scale=6.0)
    pipe.scheduler = _KwargScheduler()
    generator = torch.Generator().manual_seed(0)

    pipe.diffuse(
        latents=torch.randn(b, 4, 2, 4, 4),
        timesteps=torch.tensor([1000.0]),
        prompt_embeds=torch.randn(b, 3, 8),
        prompt_attention_mask=torch.ones(b, 3),
        negative_prompt_embeds=torch.randn(b, 3, 8),
        negative_prompt_attention_mask=torch.ones(b, 3),
        guidance_scale=6.0,
        extra_step_kwargs={"eta": 0.5, "generator": generator},
        dtype=torch.float32,
        output_slice=None,
    )

    assert captured.get("eta") == 0.5
    assert captured.get("generator") is generator
    assert captured.get("return_dict") is False
