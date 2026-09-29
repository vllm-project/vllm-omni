# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-model regression for the scheduler-owned Thinker sliding window."""

import asyncio

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    duplex_camera_frames,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_window_e2e import run_window_turn
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.omni, pytest.mark.advanced_model]


def _assert_complete(result, *, require_audio=False):
    assert result["created"] == 1
    assert result["committed"] > 0
    assert result["done"] > 0
    assert result["closed"] == 1
    assert result["errors"] == []
    if require_audio:
        assert result["audio_bytes"] > 0


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("mode,reference", [("basic", False), ("context", False), ("context", True)])
def test_window_rebuild_and_next_session(omni_server, mode, reference):
    # Two independent sessions exercise admission after final-unit execution.
    for _ in range(2):
        result = asyncio.run(
            run_window_turn(
                url=realtime_url(omni_server),
                model=omni_server.model,
                input_wav=validated_input_wav(),
                mode=mode,
                ref_audio=resolve_ref_audio() if reference else None,
            )
        )
        _assert_complete(result, require_audio=reference)


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("mode,camera", [("basic", False), ("context", False), ("context", True)])
def test_window_continuous_input(omni_server, tmp_path, mode, camera):
    # 24 repeats cover over two minutes and many canonical KV replacements.
    frames = duplex_camera_frames(seconds=4, cache_dir=tmp_path / "camera") if camera else None
    result = asyncio.run(
        run_window_turn(
            url=realtime_url(omni_server),
            model=omni_server.model,
            input_wav=validated_input_wav(),
            mode=mode,
            repeats=24,
            ref_audio=resolve_ref_audio(),
            video_frames=frames,
        )
    )
    _assert_complete(result, require_audio=True)
    assert isinstance(result["input_seconds"], (int, float)) and result["input_seconds"] > 120
    if camera:
        assert isinstance(result["frames_sent"], int) and result["frames_sent"] >= 120


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("mode", ["off", "basic", "context"])
def test_window_buffered_flush(omni_server, mode):
    # Burst appends cross processor chunk boundaries and leave a final tail.
    result = asyncio.run(
        run_window_turn(
            url=realtime_url(omni_server),
            model=omni_server.model,
            input_wav=validated_input_wav(),
            mode=mode,
            ref_audio=resolve_ref_audio(),
            buffered_flush=True,
        )
    )
    _assert_complete(result, require_audio=True)
