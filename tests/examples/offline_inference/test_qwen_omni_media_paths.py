# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check which local media the Qwen offline CLIs pass to their query builders."""

import sys
from importlib import import_module
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MIXED_INPUTS = {
    "video_path": "clip.mp4",
    "image_path": "frame.jpg",
    "audio_path": "speech.wav",
    "num_frames": 3,
    "sampling_rate": 8000,
}


@pytest.mark.parametrize(
    ("model", "query_type", "inputs"),
    [
        ("qwen2_5_omni", "use_mixed_modalities", _MIXED_INPUTS),
        ("qwen3_omni", "use_mixed_modalities", _MIXED_INPUTS),
        ("qwen2_5_omni", "use_multi_audios", {"audio_path": "first.wav", "sampling_rate": 8000}),
    ],
)
def test_query_receives_local_media(monkeypatch, model, query_type, inputs):
    example = import_module(f"examples.offline_inference.{model}.end2end")
    argv = ["end2end.py", "--query-type", query_type]
    for name, value in inputs.items():
        argv.extend([f"--{name.replace('_', '-')}", str(value)])
    monkeypatch.setattr(sys, "argv", argv)

    # Stop at the query builder, before decoding media or starting the engine.
    query = Mock(side_effect=StopIteration)
    monkeypatch.setitem(example.query_map, query_type, query)
    with pytest.raises(StopIteration):
        example.main(example.parse_args())
    query.assert_called_once_with(**inputs)
