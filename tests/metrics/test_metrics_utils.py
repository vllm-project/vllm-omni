# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.metrics.utils import (
    count_audio_chunk_frames,
    count_audio_frames,
    count_image_pixels,
    count_tokens_from_outputs,
    count_video_frames,
    extract_diffusion_vae_decode_ms,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("shape", "expected_frames"),
    [
        ((1, 64, 64, 3), 1),
        ((3, 64, 64, 3), 3),
        ((4, 64, 64, 3), 4),
        ((8, 64, 64, 3), 8),
        ((3, 8, 64, 64), 8),
        ((1, 1, 64, 64, 3), 1),
        ((1, 3, 64, 64, 3), 3),
        ((1, 4, 64, 64, 3), 4),
        ((1, 3, 8, 64, 64), 8),
    ],
)
def test_count_video_frames_handles_short_channel_last_outputs(shape, expected_frames):
    assert count_video_frames(SimpleNamespace(shape=shape)) == expected_frames


@pytest.mark.parametrize(
    ("audio_chunk", "expected"),
    [
        (SimpleNamespace(shape=(2, 320)), 320),
        ([0.0, 0.0, 0.0], 3),
        (1.0, 1),
    ],
)
def test_count_audio_chunk_frames(audio_chunk: object, expected: int) -> None:
    assert count_audio_chunk_frames(audio_chunk) == expected


def test_count_audio_frames_sums_audio_chunks() -> None:
    mm_out = {
        "audio": [
            SimpleNamespace(shape=(1, 120)),
            SimpleNamespace(shape=(2, 80)),
        ]
    }

    assert count_audio_frames(mm_out) == 200


def test_count_audio_frames_supports_model_outputs() -> None:
    mm_out = {
        "model_outputs": [
            SimpleNamespace(shape=(1, 120)),
            SimpleNamespace(shape=(2, 80)),
        ]
    }

    assert count_audio_frames(mm_out) == 200


@pytest.mark.parametrize("mm_out", [{}, {"audio": None}, {"model_outputs": None}])
def test_count_audio_frames_returns_zero_without_audio(mm_out: dict[str, object]) -> None:
    assert count_audio_frames(mm_out) == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        (object(), 0),
        (SimpleNamespace(size=(10, 20)), 200),
        (SimpleNamespace(shape=(2, 3, 4, 5)), 40),
        (SimpleNamespace(shape=(3, 4, 5)), 20),
        (SimpleNamespace(shape=(6, 7, 3)), 42),
        (SimpleNamespace(shape=(6, 7)), 42),
    ],
)
def test_count_image_pixels(value: object, expected: int) -> None:
    assert count_image_pixels(value) == expected


def test_count_image_pixels_sums_nested_values() -> None:
    images = [
        SimpleNamespace(size=(10, 20)),
        (
            SimpleNamespace(shape=(3, 4, 5)),
            SimpleNamespace(shape=(6, 7, 3)),
        ),
    ]

    assert count_image_pixels(images) == 262


def test_count_tokens_from_outputs() -> None:
    engine_outputs = [
        SimpleNamespace(
            outputs=[
                SimpleNamespace(cumulative_token_ids=[1, 2, 3], token_ids=[3]),
                SimpleNamespace(cumulative_token_ids=None, token_ids=[4, 5]),
                SimpleNamespace(cumulative_token_ids=None, token_ids=None),
            ]
        ),
        SimpleNamespace(outputs=None),
        SimpleNamespace(
            outputs=[
                SimpleNamespace(cumulative_token_ids=[6], token_ids=[6]),
            ]
        ),
    ]

    assert count_tokens_from_outputs(engine_outputs) == 6


@pytest.mark.parametrize(
    ("durations", "expected_ms"),
    [
        # Default target.
        ({"QwenImagePipeline.vae.decode": 0.25, "QwenImagePipeline.diffuse": 2.0}, 250.0),
        # MammothModa2: prefixed VAE module.
        (
            {"MammothModa2DiTPipeline.gen_vae.decode": 0.5, "MammothModa2DiTPipeline.gen_transformer.forward": 3.0},
            500.0,
        ),
        # MiniMax-H3: latent decoders for video and audio are summed.
        (
            {
                "MiniMaxH3Pipeline.video_vae.decode_latent": 1.5,
                "MiniMaxH3Pipeline.audio_vae.decode_latent": 0.25,
                "MiniMaxH3Pipeline.decode": 2.0,
                "MiniMaxH3Pipeline.post_decode": 0.1,
            },
            1750.0,
        ),
    ],
)
def test_extract_vae_decode_ms_matches_vae_decode_targets(durations: dict[str, float], expected_ms: float) -> None:
    output = DiffusionOutput(stage_durations=durations)

    assert extract_diffusion_vae_decode_ms(output) == pytest.approx(expected_ms)


@pytest.mark.parametrize(
    "durations",
    [
        {"MiniMaxH3Pipeline.decode": 2.0, "MiniMaxH3Pipeline.post_decode": 0.1},
        {"Magi2Pipeline._decode_video": 1.0, "Magi2Pipeline._decode_audio": 0.5},
        {"QwenImagePipeline.vae.encode": 0.3, "QwenImagePipeline.diffuse": 2.0},
        {"SomePipeline.vae": 0.3, "SomePipeline.myvae.decode": 0.4},
        {},
    ],
)
def test_extract_vae_decode_ms_ignores_other_targets(durations: dict[str, float]) -> None:
    assert extract_diffusion_vae_decode_ms(DiffusionOutput(stage_durations=durations)) is None


def test_extract_vae_decode_ms_without_stage_durations() -> None:
    # Pipelines pass stage_durations=None when the profiler is off.
    assert extract_diffusion_vae_decode_ms(DiffusionOutput(stage_durations=None)) is None
