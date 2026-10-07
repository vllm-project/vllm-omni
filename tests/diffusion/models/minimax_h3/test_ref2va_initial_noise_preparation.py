# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_rows
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _MINIMAX_H3_DENOISE_INPUT_KEYS,
    MiniMaxH3Pipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def pipeline():
    # Exercise real CPU tensor methods without loading model weights.
    instance = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(instance)
    instance.device = torch.device("cpu")
    return instance


@pytest.mark.parametrize("seed", [0, 42, 2101])
@pytest.mark.parametrize("shape", [(2, 4, 6, 3), (7, 8, 10, 12)])
def test_background_initial_noise_matches_direct_sampling(pipeline, seed, shape):
    latent_t, latent_h, latent_w, audio_t = shape
    params = dict(seed=seed, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=audio_t)
    rng_state = torch.get_rng_state().clone()
    expected = pipeline._initial_noise(**params)
    with ThreadPoolExecutor(max_workers=1) as executor:
        actual = executor.submit(pipeline._initial_noise, **params).result()

    for direct, prepared in zip(expected, actual, strict=True):
        assert torch.equal(prepared.view(torch.int32), direct.view(torch.int32))
        assert prepared.device.type == "cpu"
        assert prepared.dtype == torch.float32
        assert prepared.is_contiguous()
    assert torch.equal(torch.get_rng_state(), rng_state)


def _ref2va_kwargs(seed):
    return dict(
        task="ref2va",
        text_embeddings=torch.zeros(2, 4),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=seed,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        num_frames=22,
        num_steps=3,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=torch.arange(6 * 96, dtype=torch.float32).reshape(6, 96) / 100,
        visual_condition_shape=(1, 4, 6),
        audio_condition=torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32) / 100,
        ref_audio_t=2,
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 6}, {"kind": "audio", "ref_audio_t": 2}],
    )


@pytest.mark.parametrize("seed", [42, 2101])
def test_ref2va_precomputed_noise_preserves_denoise_inputs(pipeline, seed):
    kwargs = _ref2va_kwargs(seed)
    expected = pipeline._build_denoise_inputs(**kwargs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        noise = executor.submit(
            pipeline._initial_noise, seed=seed, latent_t=2, latent_h=4, latent_w=6, audio_t=3
        ).result()
        visual_noise = executor.submit(
            minimax_h3_imgvid_cond_noise_rows,
            condition_shapes=[kwargs["visual_condition_shape"]],
            target_latent_t=kwargs["latent_t"],
            imgvid_cond_num_frames=1,
            seed=seed,
        ).result()
    actual = pipeline._build_denoise_inputs(
        **kwargs, precomputed_initial_noise=noise, precomputed_visual_condition_noise=visual_noise
    )

    for key in ("video_rows", "audio_rows", "cond_anchor", "audio_anchor"):
        assert torch.equal(actual[key].view(torch.int32), expected[key].view(torch.int32))
    assert actual["sigmas_video"] == expected["sigmas_video"]
    assert actual["sigmas_audio"] == expected["sigmas_audio"]
    assert torch.equal(actual["branch"].token_tags_dev, expected["branch"].token_tags_dev)


def test_shared_denoise_kwargs_exclude_first_seed_noise(pipeline):
    context = {key: None for key in _MINIMAX_H3_DENOISE_INPUT_KEYS}
    context.update(_ref2va_kwargs(42))
    context["precomputed_initial_noise"] = pipeline._initial_noise(
        seed=42, latent_t=2, latent_h=4, latent_w=6, audio_t=3
    )
    context["precomputed_visual_condition_noise"] = torch.zeros(4, 96)

    kwargs = pipeline._denoise_kwargs(context)
    assert "precomputed_initial_noise" not in kwargs
    assert "precomputed_visual_condition_noise" not in kwargs
    assert kwargs["seed"] == 42
    assert kwargs["latent_t"] == 2
    assert kwargs["latent_h"] == 4
    assert kwargs["latent_w"] == 6
    assert kwargs["audio_t"] == 3
