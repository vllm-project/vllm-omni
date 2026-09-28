# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pybase64 as base64

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.entrypoints.duplex.warmup import (
    _WARMUP_AUDIO_SECONDS,
    _WARMUP_FRAME_HEIGHT,
    _WARMUP_FRAME_WIDTH,
    _WARMUP_SAMPLE_RATE_HZ,
    _short_pcm_f32le_b64,
    _warmup_jpeg_b64,
    startup_warmup_kind,
)


class _Plugin:
    def __init__(self, required: frozenset[str]) -> None:
        self._required = required

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        del max_sessions
        return DuplexCapabilities(required_input_modalities=self._required)


def test_video_required_model_warms_up_one_turn_by_default() -> None:
    plugin = _Plugin(frozenset({"video"}))
    assert startup_warmup_kind(plugin, 0) == "video_turn"


def test_negative_warmup_frames_disables_the_video_turn() -> None:
    plugin = _Plugin(frozenset({"video"}))
    assert startup_warmup_kind(plugin, -1) is None


def test_audio_primary_model_stays_opt_in() -> None:
    plugin = _Plugin(frozenset({"audio"}))
    assert startup_warmup_kind(plugin, 0) is None
    assert startup_warmup_kind(plugin, 4) == "silent_frames"


def test_startup_turn_audio_is_short_and_not_silent() -> None:
    raw = base64.b64decode(_short_pcm_f32le_b64())
    assert len(raw) == int(_WARMUP_SAMPLE_RATE_HZ * _WARMUP_AUDIO_SECONDS) * 4
    assert any(raw)


def test_startup_turn_frame_is_a_camera_sized_jpeg() -> None:
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(base64.b64decode(_warmup_jpeg_b64())))
    assert image.size == (_WARMUP_FRAME_WIDTH, _WARMUP_FRAME_HEIGHT)
    assert image.format == "JPEG"
