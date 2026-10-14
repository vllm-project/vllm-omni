# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

REPO_ROOT = Path(__file__).parents[3]


def _load_example(name: str, relative_path: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def text_to_video() -> ModuleType:
    return _load_example(
        "magi2_text_to_video_example",
        "examples/offline_inference/text_to_video/text_to_video.py",
    )


@pytest.mark.parametrize(
    ("extra_body", "expected_size", "expected_frames", "expected_fps"),
    [
        (None, (896, 512), 125, 12.5),
        ({"resolution": "540p"}, (896, 512), 125, 12.5),
        ({"resolution": "272p"}, (448, 256), 125, 12.5),
        ({"resolution": "1080p"}, (1920, 1088), 249, 25.0),
    ],
)
def test_shared_text_to_video_uses_native_defaults(
    text_to_video: ModuleType,
    extra_body: dict[str, object] | None,
    expected_size: tuple[int, int],
    expected_frames: int,
    expected_fps: float,
) -> None:
    preset = text_to_video._detect_preset(
        "sand-ai/MAGI-2-preview",
        "Magi2Pipeline",
        extra_body,
    )
    assert (preset["width"], preset["height"]) == expected_size
    assert preset["num_frames"] == expected_frames
    assert preset["num_inference_steps"] == 100
    assert preset["fps"] == expected_fps


@pytest.mark.parametrize(
    ("requested_size", "expected_frames", "expected_fps"),
    [
        ((1920, 1088), 249, 25.0),
        ((448, 256), 125, 12.5),
        ((896, 512), 125, 12.5),
    ],
)
def test_shared_text_to_video_takes_the_preset_of_the_requested_size(
    text_to_video: ModuleType,
    requested_size: tuple[int, int],
    expected_frames: int,
    expected_fps: float,
) -> None:
    preset = text_to_video._detect_preset("sand-ai/MAGI-2-preview", "Magi2Pipeline", None, requested_size)
    assert (preset["width"], preset["height"]) == requested_size
    assert preset["num_frames"] == expected_frames
    assert preset["fps"] == expected_fps


class _GeneratedError(Exception):
    pass


class _CapturingOmni:
    sampling_params: list[Any] = []

    def __init__(self, **kwargs: object) -> None:
        del kwargs

    def generate(self, prompt: object, sampling_params: object) -> object:
        del prompt
        self.sampling_params.append(sampling_params)
        raise _GeneratedError


@pytest.mark.parametrize(
    ("script", "extra_args"),
    [
        ("examples/offline_inference/text_to_video/text_to_video.py", []),
        ("examples/offline_inference/image_to_video/image_to_video.py", ["--image", "{image}"]),
    ],
    ids=["text_to_video", "image_to_video"],
)
def test_shared_examples_run_a_sized_1080p_request_at_the_refined_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: str,
    extra_args: list[str],
) -> None:
    image = tmp_path / "first_frame.png"
    Image.new("RGB", (32, 24), "white").save(image)
    module = _load_example(f"magi2_{Path(script).stem}_main", script)
    monkeypatch.setattr(module, "Omni", _CapturingOmni)
    monkeypatch.setattr(module, "get_model_class_name", lambda omni: "Magi2Pipeline")
    monkeypatch.setattr(_CapturingOmni, "sampling_params", [])
    argv = [script, "--model", "sand-ai/MAGI-2-preview", "--model-class-name", "Magi2Pipeline"]
    argv += ["--prompt", "A fox walks through snow", "--width", "1920", "--height", "1088"]
    argv += [arg.format(image=image) for arg in extra_args]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(_GeneratedError):
        module.main()

    (sampling,) = _CapturingOmni.sampling_params
    assert (sampling.width, sampling.height) == (1920, 1088)
    assert sampling.num_frames == 249
    assert sampling.fps == 25.0
    assert sampling.num_inference_steps == 100
