# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.metrics.utils import (
    count_audio_chunk_frames,
    count_audio_frames,
    count_image_pixels,
    count_tokens_from_outputs,
    count_video_frames,
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
