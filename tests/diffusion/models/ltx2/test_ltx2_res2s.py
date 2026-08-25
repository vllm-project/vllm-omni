# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Golden tests for the official LTX-2 res_2s sampler."""

import math

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.ltx2_res2s import (
    LTX_RES2S_STEP_NOISE_SEED,
    LTX_RES2S_SUBSTEP_NOISE_SEED,
    LTX_RES2S_TERMINAL_SIGMA,
    LTXRes2sExecutor,
    build_ltx2_res2s_sigmas,
    get_res2s_coefficients,
    normalized_res2s_audio_noise,
    refine_res2s_anchor,
    res2s_sde_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_hq_schedule_matches_official_token_dependent_golden():
    sigmas = build_ltx2_res2s_sigmas(15, 16 * 17 * 30)
    expected = torch.tensor(
        [
            1.0,
            0.9934906959533691,
            0.9860149025917053,
            0.9773396253585815,
            0.9671507477760315,
            0.9550145864486694,
            0.9403136372566223,
            0.9221396446228027,
            0.8990960121154785,
            0.8689230680465698,
            0.8277072310447693,
            0.7680275440216064,
            0.6738965511322021,
            0.5033778548240662,
            0.10000002384185791,
            0.0,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(sigmas, expected, rtol=0, atol=0)


def test_token_dependent_schedule_rejects_single_step_terminal_stretch():
    with pytest.raises(ValueError, match="at least 2"):
        build_ltx2_res2s_sigmas(1, 16 * 17 * 30)


def test_token_dependent_schedule_rejects_zero_stretch_terminal():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        build_ltx2_res2s_sigmas(2, 16 * 17 * 30, terminal=0.0)


def test_res2s_coefficients_match_log_two_golden():
    actual = get_res2s_coefficients(math.log(2))
    expected = (0.4225555942921739, -0.08267358032783734, 0.804021100772319)
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_legacy_sde_step_matches_official_golden():
    result = res2s_sde_step(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.25, -0.5], dtype=torch.float64),
        0.8,
        0.4,
        torch.tensor([0.1, -0.2], dtype=torch.float64),
    )
    expected = torch.tensor([0.1963139720814414, -0.7141669750802295], dtype=torch.float64)
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)


def test_bongmath_runs_the_exact_fixed_recurrence():
    midpoint = torch.tensor([0.3, -0.7], dtype=torch.float64)
    denoised = torch.tensor([0.2, 0.4], dtype=torch.float64)
    epsilon = torch.tensor([-0.5, 0.25], dtype=torch.float64)
    h = 0.2
    a21 = 0.47

    expected_anchor = midpoint
    expected_epsilon = epsilon
    for _ in range(100):
        expected_anchor = midpoint - h * a21 * expected_epsilon
        expected_epsilon = denoised - expected_anchor

    anchor, refined_epsilon = refine_res2s_anchor(midpoint, denoised, epsilon, h, a21)
    torch.testing.assert_close(anchor, expected_anchor, rtol=0, atol=0)
    torch.testing.assert_close(refined_epsilon, expected_epsilon, rtol=0, atol=0)


def test_executor_matches_full_interval_golden_and_passes_explicit_sigma():
    calls = []

    def predict_x0(state, sigma):
        calls.append((float(sigma), sigma.dtype))
        return LTXAVState(video=0.25 * state.video + 0.1 * sigma, audio=0.25 * state.audio + 0.1 * sigma)

    sub_noise = torch.tensor([[[0.2, -0.1]]], dtype=torch.float64)
    main_noise = torch.tensor([[[-0.3, 0.4]]], dtype=torch.float64)

    def fixed_noise(reference, generator):
        noise = sub_noise if generator.initial_seed() == LTX_RES2S_SUBSTEP_NOISE_SEED else main_noise
        return noise.expand_as(reference)

    initial = torch.tensor([[[1.0, -2.0]]], dtype=torch.float64)
    result = LTXRes2sExecutor.run(
        LTXAVState(video=initial, audio=initial.clone()),
        torch.tensor([0.8, 0.4]),
        predict_x0,
        bongmath=False,
        model_dtype=torch.float64,
        noise_fn=fixed_noise,
    )

    # Current/midpoint integration is fp64, but the official full-step SDE
    # deliberately retains the original fp32 schedule boundaries.
    expected = torch.tensor([[[0.49638621085862233, -0.9269363256806813]]], dtype=torch.float64)
    torch.testing.assert_close(result.video, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(result.audio, expected, rtol=1e-13, atol=1e-13)
    assert calls[0][0] == pytest.approx(0.8)
    assert calls[0][1] is torch.float32
    assert calls[1][0] == pytest.approx(math.sqrt(0.8 * 0.4))
    assert calls[1][1] is torch.float64


def test_executor_terminal_prediction_and_i2v_restore_cover_every_intermediate_state():
    clean = torch.tensor([[[7.0, 0.0]]], dtype=torch.float64)
    mask = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    model_sigmas = []
    model_inputs = []
    restore_calls = 0

    def restore(video):
        nonlocal restore_calls
        restore_calls += 1
        return video * mask + clean * (1 - mask)

    def predict_x0(state, sigma):
        model_sigmas.append(float(sigma))
        model_inputs.append(state.video.clone())
        return LTXAVState(video=torch.zeros_like(state.video), audio=torch.zeros_like(state.audio))

    def zero_noise(reference, generator):
        del generator
        return torch.zeros_like(reference, dtype=torch.float64)

    result = LTXRes2sExecutor.run(
        LTXAVState(video=clean.clone(), audio=torch.zeros_like(clean)),
        torch.tensor([0.8, 0.4, 0.0]),
        predict_x0,
        bongmath=False,
        model_dtype=torch.float64,
        noise_fn=zero_noise,
        restore_video=restore,
    )

    assert model_sigmas == pytest.approx([0.8, math.sqrt(0.8 * 0.4), 0.4, math.sqrt(0.4 * 0.0011), 0.0011])
    assert 0.0 not in model_sigmas
    assert restore_calls == 9
    for model_input in model_inputs:
        assert model_input[0, 0, 0] == 7.0
    assert result.video[0, 0, 0] == 7.0


@pytest.mark.parametrize("penultimate_sigma", [LTX_RES2S_TERMINAL_SIGMA, 0.001])
def test_executor_rejects_zero_terminal_after_too_small_sigma(penultimate_sigma):
    state = LTXAVState(
        video=torch.zeros(1, 1, 2),
        audio=torch.zeros(1, 1, 2),
    )

    with pytest.raises(ValueError, match="preceding sigma must be greater than"):
        LTXRes2sExecutor.run(
            state,
            torch.tensor([0.8, penultimate_sigma, 0.0]),
            lambda current, sigma: current,
        )


def test_executor_uses_fresh_fixed_rngs_and_video_then_audio_order_per_phase():
    records = []

    def recording_noise(reference, generator):
        records.append((generator.initial_seed(), float(reference.flatten()[0])))
        return torch.zeros_like(reference, dtype=torch.float64)

    def predict_x0(state, sigma):
        del sigma
        return state

    state = LTXAVState(
        video=torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        audio=torch.tensor([[[2.0, 0.0]]], dtype=torch.float64),
    )
    for _ in range(2):
        LTXRes2sExecutor.run(
            state,
            torch.tensor([0.8, 0.4]),
            predict_x0,
            bongmath=False,
            model_dtype=torch.float64,
            noise_fn=recording_noise,
        )

    main_seed = torch.Generator().manual_seed(LTX_RES2S_STEP_NOISE_SEED).initial_seed()
    one_phase = [
        (LTX_RES2S_SUBSTEP_NOISE_SEED, 1.0),
        (LTX_RES2S_SUBSTEP_NOISE_SEED, 2.0),
        (main_seed, 1.0),
        (main_seed, 2.0),
    ]
    assert records == one_phase + one_phase


def test_audio_noise_excludes_sequence_parallel_padding_from_normalization():
    logical = torch.zeros(1, 3, 4)
    padded = torch.zeros(1, 4, 4)
    logical_generator = torch.Generator().manual_seed(123)
    padded_generator = torch.Generator().manual_seed(123)

    expected = normalized_res2s_audio_noise(logical, logical_generator, logical_token_count=3)
    actual = normalized_res2s_audio_noise(padded, padded_generator, logical_token_count=3)

    torch.testing.assert_close(actual[:, :3], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 3:], torch.zeros_like(actual[:, 3:]), rtol=0, atol=0)
