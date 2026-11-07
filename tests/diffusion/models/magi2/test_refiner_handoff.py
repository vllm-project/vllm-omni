# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.diffusion.models.magi2.sampler_magi2 import build_flow_unipc_scheduler, build_magi2_preview_schedulers
from vllm_omni.diffusion.models.magi2.sampler_magi2_refiner import (
    _zero_snr_sigmas,
    build_magi2_refiner_scheduler,
    refiner_latent_size,
    upsample_and_renoise,
    zero_snr_sigmas,
    zero_snr_signal_coefficients,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

# Reference values from SandAI's ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
# and FlowUniPCMultistepScheduler().set_timesteps(5, shift=5.0).
_ZERO_SNR_SHA256_16 = "4cf78ee690ad34ba"
_SIGNAL_220 = 0.8389104008674622
_NOISE_220 = 0.5442695617675781
_REFINER_SIGMAS = [
    0.9997998476028442,
    0.9521540403366089,
    0.8820932507514954,
    0.7689347267150879,
    0.5552467703819275,
    0.0,
]
_REFINER_TIMESTEPS = [999, 952, 882, 768, 555]
# Final sample of the fake 5-step loop in ``_run_fake_refiner_loop`` under the reference scheduler.
_REFERENCE_LOOP_SAMPLE = [
    -0.670550525,
    -0.460610688,
    -0.250670612,
    -0.0407305956,
    0.16920945,
    0.379149467,
    0.589089394,
    0.79902935,
    1.00896943,
    1.21890926,
    1.42884934,
    1.6387893,
]


def _sha16(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


class _DtypeRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.dtypes: set[torch.dtype] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for item in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(item, torch.Tensor):
                self.dtypes.add(item.dtype)
        return out


def test_zero_snr_table_matches_reference() -> None:
    sigmas = zero_snr_sigmas()

    assert sigmas.dtype == torch.float32
    assert sigmas.device.type == "cpu"
    assert sigmas.shape == (1000,)
    assert _sha16(sigmas) == _ZERO_SNR_SHA256_16
    assert sigmas[0].item() == 0.9995748996734619
    assert sigmas[-1].item() == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all())
    # The rescale evaluates s0 / (s0 - sT) first; left to right gives 0.998655378818512 here.
    assert sigmas[2].item() == 0.9986554384231567


def test_zero_snr_signal_coefficients_at_the_refiner_noise_index() -> None:
    assert zero_snr_signal_coefficients() == (_SIGNAL_220, _NOISE_220)
    assert zero_snr_signal_coefficients(220) == (_SIGNAL_220, _NOISE_220)
    signal, noise = zero_snr_signal_coefficients(999)
    assert (signal, noise) == (0.0, 1.0)
    for index in (-1, 1000):
        with pytest.raises(ValueError, match="ZeroSNR index"):
            zero_snr_signal_coefficients(index)


def test_zero_snr_table_is_built_on_the_host_under_any_default_device() -> None:
    # Pipelines are constructed under ``with target_device:``; the table must still be built on the CPU.
    _zero_snr_sigmas.cache_clear()
    with torch.device("meta"):
        coefficients = zero_snr_signal_coefficients()
        sigmas = zero_snr_sigmas()

    assert coefficients == (_SIGNAL_220, _NOISE_220)
    assert sigmas.device.type == "cpu"
    assert sigmas.dtype == torch.float32
    assert _sha16(sigmas) == _ZERO_SNR_SHA256_16


def test_zero_snr_table_copies_do_not_alias_the_cache() -> None:
    sigmas = zero_snr_sigmas()
    sigmas.zero_()

    assert zero_snr_signal_coefficients() == (_SIGNAL_220, _NOISE_220)
    assert _sha16(zero_snr_sigmas()) == _ZERO_SNR_SHA256_16


def test_refiner_latent_size() -> None:
    assert refiner_latent_size(32, 1088, 1920) == (63, 68, 120)
    assert refiner_latent_size(1, 16, 32) == (1, 1, 2)
    with pytest.raises(ValueError, match="multiple"):
        refiner_latent_size(32, 1080, 1920)
    with pytest.raises(ValueError, match="positive"):
        refiner_latent_size(0, 1088, 1920)


def test_temporal_upsampling_keeps_preview_frames_and_inserts_midpoints() -> None:
    latent = torch.randn(1, 3, 4, 2, 3, generator=torch.Generator().manual_seed(0))
    out = upsample_and_renoise(latent, (7, 2, 3), noise=torch.zeros(1, 3, 7, 2, 3))
    upsampled = out / _SIGNAL_220

    assert torch.equal(out[:, :, ::2], latent * _SIGNAL_220)
    torch.testing.assert_close(upsampled[:, :, 1::2], (latent[:, :, :-1] + latent[:, :, 1:]) / 2, rtol=0, atol=1e-6)


def test_spatial_upsampling_aligns_corners() -> None:
    # A linear ramp stays an exact ramp only with align_corners=True.
    ramp_h = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4, 1).expand(1, 1, 2, 4, 3)
    ramp_w = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 1, 3).expand(1, 1, 2, 4, 3)
    latent = 2.0 * ramp_h - 3.0 * ramp_w
    out = upsample_and_renoise(latent, (3, 10, 7), noise=torch.zeros(1, 1, 3, 10, 7)) / _SIGNAL_220
    expected_h = torch.linspace(0, 3, 10).view(1, 1, 1, 10, 1)
    expected_w = torch.linspace(0, 2, 7).view(1, 1, 1, 1, 7)
    expected = (2.0 * expected_h - 3.0 * expected_w).expand(1, 1, 3, 10, 7)

    torch.testing.assert_close(out, expected, rtol=0, atol=1e-5)


def _interpolation_matrix(n_in: int, n_out: int) -> torch.Tensor:
    matrix = torch.zeros(n_out, n_in, dtype=torch.float64)
    for i in range(n_out):
        src = i * (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
        lo = min(int(src), n_in - 1)
        hi = min(lo + 1, n_in - 1)
        frac = src - lo
        matrix[i, lo] += 1.0 - frac
        matrix[i, hi] += frac
    return matrix


def test_upsampling_matches_closed_form_trilinear() -> None:
    latent = torch.randn(1, 48, 4, 4, 6, generator=torch.Generator().manual_seed(1))
    out = upsample_and_renoise(latent, (7, 8, 12), noise=torch.zeros(1, 48, 7, 8, 12)) / _SIGNAL_220
    expected = torch.einsum(
        "ti,hj,wk,bcijk->bcthw",
        _interpolation_matrix(4, 7),
        _interpolation_matrix(4, 8),
        _interpolation_matrix(6, 12),
        latent.double(),
    )

    torch.testing.assert_close(out.double(), expected, rtol=0, atol=1e-5)


def test_renoise_mixes_injected_noise_with_reference_coefficients() -> None:
    gen = torch.Generator().manual_seed(2)
    latent = torch.randn(1, 4, 3, 4, 4, generator=gen)
    noise = torch.randn(1, 4, 5, 8, 8, generator=gen)
    upsampled = F.interpolate(latent, size=(5, 8, 8), mode="trilinear", align_corners=True)
    signal = torch.tensor(_SIGNAL_220, dtype=torch.float32)
    expected = upsampled * signal + noise * (1 - signal**2) ** 0.5

    out = upsample_and_renoise(latent, (5, 8, 8), noise=noise)

    assert out.dtype == torch.float32
    assert torch.equal(out, expected)


def test_renoise_draws_noise_after_upsampling() -> None:
    latent = torch.randn(1, 4, 3, 4, 4, generator=torch.Generator().manual_seed(3))
    size = (5, 8, 8)

    torch.manual_seed(1234)
    out = upsample_and_renoise(latent, size)
    torch.manual_seed(1234)
    upsampled = F.interpolate(latent, size=size, mode="trilinear", align_corners=True)
    expected = upsampled * _SIGNAL_220 + torch.randn_like(upsampled) * _NOISE_220
    assert torch.equal(out, expected)

    state = torch.random.get_rng_state()
    out = upsample_and_renoise(latent, size, generator=torch.Generator().manual_seed(7))
    assert torch.equal(torch.random.get_rng_state(), state)
    eps = torch.randn(1, 4, *size, generator=torch.Generator().manual_seed(7))
    assert torch.equal(out, upsampled * _SIGNAL_220 + eps * _NOISE_220)


def test_hand_off_runs_in_fp32_without_float64() -> None:
    latent = torch.randn(1, 4, 3, 4, 4, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
    noise = torch.randn(1, 4, 5, 8, 8, generator=torch.Generator().manual_seed(5))
    zero_snr_signal_coefficients()  # build the host-side table outside the recorder

    with _DtypeRecorder() as recorder:
        out = upsample_and_renoise(latent, (5, 8, 8), noise=noise)

    assert out.dtype == torch.float32
    assert torch.float64 not in recorder.dtypes
    assert torch.equal(out, upsample_and_renoise(latent.float(), (5, 8, 8), noise=noise))


def test_hand_off_rejects_bad_inputs() -> None:
    latent = torch.zeros(1, 4, 3, 4, 4)
    with pytest.raises(ValueError, match="B, C, T, H, W"):
        upsample_and_renoise(latent[0], (5, 8, 8))
    with pytest.raises(ValueError, match="three positive"):
        upsample_and_renoise(latent, (8, 8))
    with pytest.raises(ValueError, match="three positive"):
        upsample_and_renoise(latent, (0, 8, 8))
    with pytest.raises(ValueError, match="noise shape"):
        upsample_and_renoise(latent, (5, 8, 8), noise=torch.zeros(1, 4, 3, 4, 4))


def test_refiner_scheduler_matches_reference_schedule() -> None:
    scheduler = build_magi2_refiner_scheduler()

    assert scheduler.sigmas.dtype == torch.float32
    assert torch.equal(scheduler.sigmas, torch.tensor(_REFINER_SIGMAS, dtype=torch.float32))
    assert scheduler.timesteps.dtype == torch.int64
    assert scheduler.timesteps.tolist() == _REFINER_TIMESTEPS
    config = scheduler.config
    assert (config.solver_order, config.solver_type, config.predict_x0) == (2, "bh2", True)
    assert (config.lower_order_final, config.final_sigmas_type) == (True, "zero")
    # The shift is applied once by set_timesteps, not at construction.
    assert config.shift == 1.0


def _run_fake_refiner_loop(scheduler) -> tuple[torch.Tensor, list[int]]:
    sample = torch.linspace(-2, 2, 12, dtype=torch.float32).view(1, 1, 3, 2, 2)
    orders = []
    for k, t in enumerate(scheduler.timesteps):
        velocity = 0.5 * sample - 0.125 * (k + 1)
        sample = scheduler.step(velocity, t, sample, return_dict=False)[0]
        orders.append(scheduler.this_order)
    return sample, orders


def test_refiner_scheduler_reproduces_reference_unipc_loop() -> None:
    sample, orders = _run_fake_refiner_loop(build_magi2_refiner_scheduler())

    assert orders == [1, 2, 2, 2, 1]
    torch.testing.assert_close(
        sample.flatten(), torch.tensor(_REFERENCE_LOOP_SAMPLE, dtype=torch.float32), rtol=1e-5, atol=1e-6
    )


def test_refiner_scheduler_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="num_inference_steps"):
        build_magi2_refiner_scheduler(0)
    for shift in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="shift"):
            build_magi2_refiner_scheduler(5, shift)


def test_preview_and_refiner_builders_share_the_flow_unipc_builder() -> None:
    shared = build_flow_unipc_scheduler(4, shift=7.0, device="cpu")
    video, audio = build_magi2_preview_schedulers(4, device="cpu", shift=7.0)

    assert video is not audio
    for scheduler in (video, audio):
        assert torch.equal(scheduler.sigmas, shared.sigmas)
        assert torch.equal(scheduler.timesteps, shared.timesteps)
        assert scheduler.config.shift == 1.0
    # The video and audio schedulers keep separate multistep state.
    sample = torch.zeros(1, 1, 1, 2, 2)
    video.step(sample, video.timesteps[0], sample, return_dict=False)
    assert (video.step_index, audio.step_index) == (1, None)

    refiner = build_magi2_refiner_scheduler(device="cpu")
    assert torch.equal(refiner.sigmas, build_flow_unipc_scheduler(5, shift=5.0, device="cpu").sigmas)
    with pytest.raises(ValueError, match="num_inference_steps"):
        build_magi2_preview_schedulers(0, device="cpu")
    with pytest.raises(ValueError, match="shift"):
        build_magi2_preview_schedulers(4, device="cpu", shift=float("nan"))
