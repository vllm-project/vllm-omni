# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""What a duplex append reserves in the scheduler, and why that number.

The reservation is not advisory: ``build_duplex_data_plane_prompt`` turns it
into ``[scheduler_token_id] * budget``, and a unit whose embeddings outnumber
its slots has its tail dropped with a warning rather than failing
(``MiniCPMO45OmniModel.get_input_embeddings``). Over-reserving only wastes
slots, so every uncertainty here has to resolve upwards.

The camera side depends on HD slicing, which depends on the frame size
*relative to the checkpoint's normalization tile*. The tile is configuration,
not a constant, so the cases below drive the arithmetic at more than one tile
size and check it against ``MiniCPMVImageProcessor.get_sliced_grid`` -- the
same grid search the checkpoint's own processor runs.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import pytest
from PIL import Image

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    PRIVATE_RUNTIME_CONFIG_KEYS,
    _apply_default_scheduler_policy,
    _duplex_vision_tile_pixels,
    _duplex_vision_tokens,
    _model_vision_tile_pixels,
    duplex_scheduler_token_budget,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMVImageProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TOKENS_PER_BLOCK = 66
#: ``_official_max_slice_nums`` gives the first frame of a stacked pair 2 and
#: every other frame 1.
BASE_FRAME_MAX_SLICES = 2
#: What the released MiniCPM-o 4.5 checkpoint configures.
TILE_448 = 448 * 448

SIZES = [(224, 224), (320, 180), (448, 448), (448, 449), (640, 480), (960, 540), (1280, 720), (540, 960)]


def _frame(size: tuple[int, int], fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode()


def _blocks_the_model_makes(size: tuple[int, int], max_slice_nums: int, scale_resolution: int) -> int:
    """Source tile plus the HD patches, straight from the model's grid search."""
    processor = MiniCPMVImageProcessor(scale_resolution=scale_resolution)
    grid = processor.get_sliced_grid(image_size=size, max_slice_nums=max_slice_nums)
    return 1 + grid[0] * grid[1] if grid else 1


@pytest.mark.parametrize("scale_resolution", [336, 448, 560])
@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_a_stacked_pair_reserves_what_the_model_slices_it_into(
    size: tuple[int, int], fmt: str, scale_resolution: int
) -> None:
    """The tile comes from the checkpoint, so the arithmetic has to follow it."""
    frame = _frame(size, fmt)
    expected_blocks = _blocks_the_model_makes(size, BASE_FRAME_MAX_SLICES, scale_resolution) + _blocks_the_model_makes(
        size, 1, scale_resolution
    )

    reserved = _duplex_vision_tokens({"video_frames": [frame, frame]}, tile_pixels=scale_resolution**2)

    assert reserved == expected_blocks * TOKENS_PER_BLOCK


def test_a_small_frame_no_longer_reserves_three_hd_blocks() -> None:
    """A 448x448 frame is one tile, so a stacked pair is two blocks, not four.

    The size-independent count reserved 264 tokens for this pair. 132 of them
    were for patches the model never produces.
    """
    frame = _frame((448, 448))

    assert _duplex_vision_tokens({"video_frames": [frame, frame]}, tile_pixels=TILE_448) == 2 * TOKENS_PER_BLOCK


def test_one_tile_is_the_whole_slicing_decision() -> None:
    """``ceil(w * h / tile)`` capped at 2: a single pixel over the tile slices."""

    def pair(size: tuple[int, int]) -> int:
        frame = _frame(size)
        return _duplex_vision_tokens({"video_frames": [frame, frame]}, tile_pixels=TILE_448)

    assert pair((448, 448)) == 2 * TOKENS_PER_BLOCK
    assert pair((448, 449)) == 4 * TOKENS_PER_BLOCK


def test_only_the_first_frame_of_a_pair_is_hd_sliced() -> None:
    """``_official_max_slice_nums`` is ``[2, 1]``: the composite is never sliced.

    Sizing the wrong half of the pair would pass every same-size case, so the
    two orders have to disagree.
    """
    small, large = _frame((448, 448)), _frame((1280, 720))

    assert _duplex_vision_tokens({"video_frames": [small, large]}, tile_pixels=TILE_448) == 2 * TOKENS_PER_BLOCK
    assert _duplex_vision_tokens({"video_frames": [large, small]}, tile_pixels=TILE_448) == 4 * TOKENS_PER_BLOCK


def test_a_lone_frame_is_never_hd_sliced() -> None:
    """``_official_max_slice_nums(1)`` is ``[1]``: one frame is one tile at any size."""
    assert _duplex_vision_tokens({"video_frames": [_frame((1280, 720))]}, tile_pixels=TILE_448) == TOKENS_PER_BLOCK


def test_extra_frames_past_the_pair_each_cost_one_block() -> None:
    """The wire caps a frame list at two; the in-process API does not."""
    small, large = _frame((448, 448)), _frame((1280, 720))

    assert _duplex_vision_tokens({"video_frames": [small] * 3}, tile_pixels=TILE_448) == 3 * TOKENS_PER_BLOCK
    assert _duplex_vision_tokens({"video_frames": [large] * 3}, tile_pixels=TILE_448) == 5 * TOKENS_PER_BLOCK


def test_an_unknown_tile_size_keeps_the_sliced_reservation() -> None:
    """No tile, no shrinking. A checkpoint whose config cannot be read keeps the old number."""
    frame = _frame((224, 224))

    assert _duplex_vision_tokens({"video_frames": [frame, frame]}) == 4 * TOKENS_PER_BLOCK
    assert _duplex_vision_tokens({"video_frames": [frame, frame]}, tile_pixels=None) == 4 * TOKENS_PER_BLOCK


@pytest.mark.parametrize(
    "first_frame",
    [
        pytest.param("data:image/jpeg;base64," + _frame((224, 224)), id="data url"),
        pytest.param("!!! not base64 !!!", id="corrupt base64"),
        pytest.param(base64.b64encode(b"\x00" * 64).decode(), id="base64 of something else"),
        pytest.param(base64.b64encode(base64.b64decode(_frame((224, 224)))[:8]).decode(), id="truncated jpeg"),
    ],
)
def test_a_frame_whose_header_will_not_parse_keeps_the_sliced_reservation(first_frame: str) -> None:
    """The Realtime wire screens most of these; the in-process API does not.

    ``validate_realtime_video_frames`` rejects bad base64 and anything whose
    magic bytes are not JPEG or PNG, so on a websocket session only a truncated
    but well-headed frame gets this far. ``DuplexOmni.append_audio`` submits
    without that screen, so the reservation still has to hold on its own.
    """
    good = _frame((224, 224))

    assert _duplex_vision_tokens({"video_frames": [first_frame, good]}, tile_pixels=TILE_448) == 4 * TOKENS_PER_BLOCK


def test_no_camera_track_reserves_nothing() -> None:
    for payload in ({}, {"video_frames": []}, {"video_frames": "not a list"}, {"video_frames": [None, ""]}, None):
        assert _duplex_vision_tokens(payload, tile_pixels=TILE_448) == 0


def test_the_audio_budget_is_untouched_by_the_camera_track() -> None:
    """One second of pcm_f32le is 12 slots; the frames are added on top."""
    one_second = base64.b64encode(b"\x00" * 4 * 16000).decode()
    audio_only = {"audio": one_second, "format": "pcm_f32le"}
    with_frames = {**audio_only, "video_frames": [_frame((448, 448)), _frame((448, 448))]}

    assert duplex_scheduler_token_budget(audio_only, tile_pixels=TILE_448) == 12
    assert duplex_scheduler_token_budget(with_frames, tile_pixels=TILE_448) == 12 + 2 * TOKENS_PER_BLOCK


# ---- where the tile comes from ----


@dataclass(frozen=True)
class _SliceConfig:
    scale_resolution: int


@dataclass(frozen=True)
class _HFConfig:
    slice_config: object = None
    image_size: object = None


@dataclass(frozen=True)
class _ModelConfig:
    hf_config: object = None


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (_HFConfig(slice_config={"max_slice_nums": 1, "scale_resolution": 448}, image_size=448), TILE_448),
        (_HFConfig(slice_config=_SliceConfig(scale_resolution=336), image_size=448), 336 * 336),
        (_HFConfig(image_size=560), 560 * 560),
        (_HFConfig(), None),
        (_HFConfig(slice_config={"scale_resolution": 0}, image_size=0), None),
    ],
    ids=["released-checkpoint", "attribute", "image-size-fallback", "nothing-to-read", "nonsense"],
)
def test_the_tile_is_read_from_the_checkpoint(hf_config: object, expected: int | None) -> None:
    """First row is the released MiniCPM-o 4.5 ``config.json``, where ``slice_config`` is a dict."""
    assert _model_vision_tile_pixels(_ModelConfig(hf_config=hf_config)) == expected


def test_no_model_config_means_no_tile() -> None:
    assert _model_vision_tile_pixels(None) is None
    assert _model_vision_tile_pixels(_ModelConfig()) is None


def test_the_tile_reaches_the_budget_through_the_runtime_config() -> None:
    """``_apply_default_scheduler_policy`` writes it; ``build_duplex_data_plane_prompt`` reads it back."""
    runtime_config: dict[str, object] = {}

    _apply_default_scheduler_policy(
        runtime_config,
        config=DuplexSessionConfig(),
        tokenizer=None,
        model_config=_ModelConfig(hf_config=_HFConfig(slice_config={"scale_resolution": 448})),
    )

    assert runtime_config["duplex_vision_tile_pixels"] == TILE_448
    assert _duplex_vision_tile_pixels(runtime_config) == TILE_448


def test_a_checkpoint_that_cannot_be_read_leaves_the_key_out() -> None:
    """No key, no tile, and no tile is the sliced reservation."""
    runtime_config: dict[str, object] = {}

    _apply_default_scheduler_policy(runtime_config, config=DuplexSessionConfig(), tokenizer=None)

    assert "duplex_vision_tile_pixels" not in runtime_config
    assert _duplex_vision_tile_pixels(runtime_config) is None


def test_a_junk_tile_in_the_runtime_config_is_ignored() -> None:
    for value in (0, -1, "448", 448.0, None):
        assert _duplex_vision_tile_pixels({"duplex_vision_tile_pixels": value}) is None


def test_the_tile_is_server_owned() -> None:
    """A client that could set the tile could shrink the reservation under the worker."""
    assert "duplex_vision_tile_pixels" in PRIVATE_RUNTIME_CONFIG_KEYS
