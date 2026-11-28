# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exactness of the reusable VibeVoice DPM-Solver++ loop.

``sample_audio_latent`` used to build a fresh ``DPMSolverMultistepScheduler``
per audio token; it now resets a per-(steps, device) template
(``VibeVoiceDPMSolverLoop``). These tests pin the equivalence: the hoisted
loop must reproduce a fresh scheduler token-for-token, bit-exactly, in fp32
and bf16, and across sequential tokens (the reset path).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.vibevoice.diffusion import (
    VIBEVOICE_DEFAULT_GUIDANCE_SCALE,
    VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS,
    VibeVoiceDiffusionHead,
    VibeVoiceDPMSolverLoop,
    create_dpm_scheduler,
)
from vllm_omni.model_executor.models.vibevoice.vibevoice import VibeVoiceModel
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceDiffusionHeadConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.usefixtures("single_threaded_cpu")]

LATENT = 32
HIDDEN = 64
GUIDANCE = VIBEVOICE_DEFAULT_GUIDANCE_SCALE


def _make_head(dtype: torch.dtype) -> VibeVoiceDiffusionHead:
    torch.manual_seed(0)
    config = VibeVoiceDiffusionHeadConfig(
        hidden_size=HIDDEN,
        latent_size=LATENT,
        num_hidden_layers=2,
        intermediate_size=128,
        frequency_embedding_size=32,
    )
    return VibeVoiceDiffusionHead(config).to(dtype=dtype).eval()


def _make_model(dtype: torch.dtype) -> VibeVoiceModel:
    """VibeVoiceModel shell with a real diffusion head and solver loop cache."""
    model = object.__new__(VibeVoiceModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(audio_config=SimpleNamespace(hidden_size=LATENT))
    model.diffusion_head = _make_head(dtype)
    model._dpm_loops = {}
    return model


def _reference_sample(head, positive, negative, noise, *, num_inference_steps, dtype):
    """The previous implementation, verbatim: fresh scheduler per token."""
    batch_size = positive.shape[0]
    condition = torch.cat([positive, negative], dim=0)
    latent = noise.to(condition)
    scheduler = create_dpm_scheduler()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    for timestep in scheduler.timesteps:
        shared = latent[:batch_size]
        combined = torch.cat([shared, shared], dim=0)
        prediction = head(combined, timestep.repeat(2 * batch_size).to(combined), condition)
        cond, uncond = prediction[:batch_size], prediction[batch_size:]
        guided = uncond + GUIDANCE * (cond - uncond)
        latent = scheduler.step(torch.cat([guided, guided], dim=0), timestep, latent).prev_sample
    return latent[:batch_size].unsqueeze(1)


def _conditions(batch_size: int, dtype: torch.dtype, seed: int):
    gen = torch.Generator().manual_seed(seed)
    positive = torch.randn(batch_size, HIDDEN, generator=gen).to(dtype)
    negative = torch.randn(batch_size, HIDDEN, generator=gen).to(dtype)
    noise = torch.randn(2 * batch_size, LATENT, generator=gen).to(dtype)
    return positive, negative, noise


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_sample_audio_latent_matches_fresh_scheduler_bit_exactly(dtype: torch.dtype, batch_size: int) -> None:
    """The hoisted loop must equal the previous per-token construction."""
    model = _make_model(dtype)
    head = model.diffusion_head
    with torch.inference_mode():
        # Several sequential tokens: the second and third exercise begin()'s
        # reset of state left behind by the previous token.
        for token, seed in enumerate((7, 8, 9)):
            positive, negative, noise = _conditions(batch_size, dtype, seed)
            expected = _reference_sample(
                head, positive, negative, noise, num_inference_steps=VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS, dtype=dtype
            )
            actual = model.sample_audio_latent(
                positive, negative, noise, num_inference_steps=VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS
            )
            assert actual.dtype == expected.dtype == dtype
            assert torch.equal(actual, expected), f"token {token} diverged"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sample_audio_latent_matches_reference_at_other_step_counts(dtype: torch.dtype) -> None:
    model = _make_model(dtype)
    head = model.diffusion_head
    seen_steps: set[int] = set()
    with torch.inference_mode():
        for steps in (4, 6, 25):
            positive, negative, noise = _conditions(2, dtype, 11)
            expected = _reference_sample(head, positive, negative, noise, num_inference_steps=steps, dtype=dtype)
            actual = model.sample_audio_latent(positive, negative, noise, num_inference_steps=steps)
            assert torch.equal(actual, expected), f"steps={steps} diverged"
            # Distinct step counts must keep distinct templates.
            seen_steps.add(steps)
            assert len(model._dpm_loops) == len(seen_steps)


def test_sample_audio_latent_rejects_bad_noise_shape() -> None:
    model = _make_model(torch.float32)
    positive, negative, _ = _conditions(2, torch.float32, 3)
    with pytest.raises(ValueError, match="noise must have shape"):
        model.sample_audio_latent(positive, negative, torch.randn(3, LATENT))


@pytest.mark.parametrize("steps", [4, 10, 25])
def test_solver_loop_matches_fresh_scheduler_step_math(steps: int) -> None:
    """Scheduler-level equivalence, independent of the head and CFG."""
    gen = torch.Generator().manual_seed(5)
    sample = torch.randn(2, LATENT, generator=gen)
    model_output = torch.randn(2, LATENT, generator=gen)

    fresh = create_dpm_scheduler()
    fresh.set_timesteps(num_inference_steps=steps)
    loop = VibeVoiceDPMSolverLoop(steps, torch.device("cpu"))
    assert torch.equal(loop.timesteps, fresh.timesteps)

    loop.begin()
    latent_fresh, latent_loop = sample.clone(), sample.clone()
    for step_index, timestep in enumerate(fresh.timesteps):
        latent_fresh = fresh.step(model_output, timestep, latent_fresh).prev_sample
        latent_loop = loop.step(model_output, step_index, latent_loop)
    assert torch.equal(latent_fresh, latent_loop)
    # The multistep history the template accumulated is the fresh one's too.
    assert len(loop._template.model_outputs) == len(fresh.model_outputs)
    for a, b in zip(loop._template.model_outputs, fresh.model_outputs, strict=True):
        assert torch.equal(a, b)


def test_solver_loop_begin_resets_token_state() -> None:
    """A token's loop must not see the previous token's multistep history."""
    loop = VibeVoiceDPMSolverLoop(10, torch.device("cpu"))
    loop.begin()
    sample = torch.zeros(2, LATENT)
    model_output = torch.ones(2, LATENT)
    for step_index in range(5):
        loop.step(model_output, step_index, sample)
    assert loop._template.model_outputs, "multistep history should have accumulated"
    loop.begin()
    # set_timesteps pre-sizes the history (solver_order slots of None).
    assert loop._template.model_outputs == [None] * len(loop._model_outputs_reset)
    assert loop._template._step_index is None
    assert loop._template._begin_index is None


def test_solver_loop_deterministic_across_reuse() -> None:
    loop = VibeVoiceDPMSolverLoop(10, torch.device("cpu"))
    model_output = torch.randn(2, LATENT, generator=torch.Generator().manual_seed(1))
    sample = torch.randn(2, LATENT, generator=torch.Generator().manual_seed(2))

    def run() -> torch.Tensor:
        loop.begin()
        latent = sample.clone()
        for step_index in range(10):
            latent = loop.step(model_output, step_index, latent)
        return latent

    first, second = run(), run()
    assert torch.equal(first, second)
