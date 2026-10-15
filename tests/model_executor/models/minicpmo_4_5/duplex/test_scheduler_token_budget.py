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
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    PRIVATE_RUNTIME_CONFIG_KEYS,
    _apply_default_scheduler_policy,
    _duplex_vision_tile_pixels,
    _duplex_vision_tokens,
    _model_vision_tile_pixels,
    build_duplex_data_plane_prompt,
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
    payloads: tuple[object, ...] = (
        {},
        {"video_frames": []},
        {"video_frames": "not a list"},
        {"video_frames": [None, ""]},
        None,
    )
    for payload in payloads:
        assert _duplex_vision_tokens(payload, tile_pixels=TILE_448) == 0


def test_the_audio_budget_is_untouched_by_the_camera_track() -> None:
    """One second of pcm_f32le is 12 slots; the frames are added on top."""
    one_second = base64.b64encode(b"\x00" * 4 * 16000).decode()
    audio_only = {"audio": one_second, "format": "pcm_f32le"}
    with_frames = {**audio_only, "video_frames": [_frame((448, 448)), _frame((448, 448))]}

    assert duplex_scheduler_token_budget(audio_only, tile_pixels=TILE_448) == 12
    assert duplex_scheduler_token_budget(with_frames, tile_pixels=TILE_448) == 12 + 2 * TOKENS_PER_BLOCK


# ---- the first append decodes its frame once ----


def test_a_first_append_decodes_the_base_frame_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first-units branch reuses the vision count instead of decoding again."""
    calls: list[str] = []
    real = plugin._duplex_base_frame_blocks

    def counting(frame: str, tile_pixels: int | None) -> int:
        calls.append(frame)
        return real(frame, tile_pixels)

    monkeypatch.setattr(plugin, "_duplex_base_frame_blocks", counting)
    two_seconds = base64.b64encode(b"\x00" * 4 * 32000).decode()
    frame = _frame((960, 540))

    prompt = build_duplex_data_plane_prompt(
        request_id="tile-request",
        fence=DuplexFence("sid", turn_id=1),
        session_config={},
        runtime_config={"duplex_first_append_context_tokens": 5, "duplex_vision_tile_pixels": TILE_448},
        seq=1,
        turn_seq=1,
        payload={"audio": two_seconds, "format": "pcm_f32le", "video_frames": [frame, frame]},
        final=False,
    )

    assert len(calls) == 1
    # 5 context + one unit of 12 - 1, plus the sliced base frame (3 blocks) and the composite (1 block).
    assert len(prompt["prompt_token_ids"]) == 5 + 12 - 1 + 4 * TOKENS_PER_BLOCK


# ---- where the tile comes from ----


@dataclass(frozen=True)
class _ModelConfig:
    model: object = None
    hf_config: object = None


_ABSENT = object()


def _checkpoint(tmp_path: Path, processor: object = _ABSENT, config_side: int | None = None) -> _ModelConfig:
    """A local checkpoint: ``processor`` is ``preprocessor_config.json``'s ``scale_resolution``.

    ``config_side`` is the tile ``config.json`` carries, in the released
    checkpoint's layout (``slice_config`` dict plus ``image_size``). It goes on
    ``hf_config`` as vLLM's ``ModelConfig`` would load it, so a test can make
    the two sources disagree.
    """
    if processor is not _ABSENT:
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({"image_processor_type": "MiniCPMVImageProcessor", "scale_resolution": processor})
        )
    hf_config = None
    if config_side is not None:
        hf_config = SimpleNamespace(
            slice_config={"max_slice_nums": 1, "scale_resolution": config_side}, image_size=config_side
        )
    return _ModelConfig(model=str(tmp_path), hf_config=hf_config)


def test_the_tile_is_read_from_the_released_checkpoint_layout(tmp_path: Path) -> None:
    assert _model_vision_tile_pixels(_checkpoint(tmp_path, processor=448, config_side=448)) == TILE_448


def test_the_tile_follows_the_processor_not_config_json(tmp_path: Path) -> None:
    """Stage0 slices with the processor, so ``config.json`` alone cannot move the tile."""
    assert _model_vision_tile_pixels(_checkpoint(tmp_path, processor=448, config_side=560)) == TILE_448
    assert _model_vision_tile_pixels(_checkpoint(tmp_path, processor=336, config_side=448)) == 336 * 336


def test_a_nested_processor_config_wins_like_it_does_for_the_processor(tmp_path: Path) -> None:
    """transformers 5 saves the image processor nested in ``processor_config.json`` and reads that first."""
    model_config = _checkpoint(tmp_path, processor=448, config_side=448)
    nested = tmp_path / "processor_config.json"

    nested.write_text(json.dumps({"processor_class": "MiniCPMOProcessor"}))
    assert _model_vision_tile_pixels(model_config) == TILE_448

    nested.write_text(json.dumps({"image_processor": {"scale_resolution": 336}}))
    assert _model_vision_tile_pixels(model_config) == 336 * 336

    (tmp_path / "preprocessor_config.json").unlink()
    assert _model_vision_tile_pixels(model_config) == 336 * 336


def test_a_config_json_override_does_not_shrink_the_reservation(tmp_path: Path) -> None:
    """A 500x500 base frame with a 560 tile in ``config.json`` and 448 in the processor.

    Reading ``config.json`` reserved two blocks (132 tokens) for a frame the
    processor slices into three plus the composite (264 tokens).
    """
    runtime_config: dict[str, object] = {}
    _apply_default_scheduler_policy(
        runtime_config,
        config=DuplexSessionConfig(),
        tokenizer=None,
        model_config=_checkpoint(tmp_path, processor=448, config_side=560),
    )
    frame = _frame((500, 500))
    expected_blocks = _blocks_the_model_makes((500, 500), BASE_FRAME_MAX_SLICES, 448) + _blocks_the_model_makes(
        (500, 500), 1, 448
    )

    reserved = _duplex_vision_tokens(
        {"video_frames": [frame, frame]}, tile_pixels=_duplex_vision_tile_pixels(runtime_config)
    )

    assert reserved == expected_blocks * TOKENS_PER_BLOCK == 264


@pytest.mark.parametrize("value", [0, -1, "448", 448.0, True, None], ids=repr)
def test_a_nonsense_processor_tile_is_no_tile(tmp_path: Path, value: object) -> None:
    assert _model_vision_tile_pixels(_checkpoint(tmp_path, processor=value, config_side=448)) is None


def test_without_a_processor_config_there_is_no_tile(tmp_path: Path) -> None:
    """``config.json`` alone is not what Stage0 slices with, so it is not used."""
    assert _model_vision_tile_pixels(_checkpoint(tmp_path, config_side=448)) is None
    (tmp_path / "preprocessor_config.json").write_text("not json")
    assert _model_vision_tile_pixels(_ModelConfig(model=str(tmp_path))) is None


def test_no_model_config_means_no_tile() -> None:
    assert _model_vision_tile_pixels(None) is None
    assert _model_vision_tile_pixels(_ModelConfig()) is None


def test_the_tile_reaches_the_budget_through_the_runtime_config(tmp_path: Path) -> None:
    """``_apply_default_scheduler_policy`` writes it; ``build_duplex_data_plane_prompt`` reads it back."""
    runtime_config: dict[str, object] = {}

    _apply_default_scheduler_policy(
        runtime_config,
        config=DuplexSessionConfig(),
        tokenizer=None,
        model_config=_checkpoint(tmp_path, processor=448),
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
