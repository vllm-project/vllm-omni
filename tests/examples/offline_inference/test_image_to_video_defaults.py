# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
import importlib.util
import sys
from typing import Any

import pytest

from tests.examples.helpers import EXAMPLES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.cache
def _load_example_module() -> Any:
    """Load the shared I2V example without executing ``main()``."""
    path = EXAMPLES / "offline_inference" / "image_to_video" / "image_to_video.py"
    spec = importlib.util.spec_from_file_location("image_to_video_defaults_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(model_name: str, model_class_name: str | None = None) -> Any:
    return _load_example_module().resolve_model_generation_defaults(model_name, model_class_name)


@pytest.mark.parametrize(
    (
        "model_name",
        "model_class_name",
        "fps",
        "guidance_scale",
        "num_frames",
        "num_inference_steps",
        "flow_shift",
        "max_area",
        "mod_value",
        "is_ltx2",
    ),
    [
        # #3519: LTX2 advertised defaults (121 frames / 40 steps / 24 fps) must
        # actually apply instead of silently falling back to the Wan2.2 numbers.
        ("/path/to/LTX-2", None, 24, None, 121, 40, None, 512 * 768, 32, True),
        # The issue's original repro selected the class name explicitly.
        ("", "LTX2ImageToVideoPipeline", 24, None, 121, 40, None, 512 * 768, 32, True),
        ("diffusers/LTX-2.3-Diffusers", None, 24, None, 121, 30, None, 512 * 768, 32, True),
        ("ltx2-distilled", "LTX2DistilledPipeline", 24, None, 121, 8, None, 1024 * 1536, 64, True),
        ("Efficient-Large-Model/SANA-Video_2B_480p_diffusers", None, 16, 6.0, 81, 50, 5.0, 480 * 832, 16, False),
        ("sana-video-720p", None, 16, 6.0, 81, 50, 5.0, 704 * 1280, 32, False),
        ("nvidia/Cosmos3-Nano", None, 24, 6.0, 189, 35, 10.0, 1280 * 720, 16, False),
        ("nvidia/Cosmos3-Edge", None, 24, 5.0, 189, 35, 3.0, 480 * 832, 16, False),
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", None, 16, 5.0, 81, 50, 5.0, 480 * 832, 16, False),
        (
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
            None,
            16,
            5.0,
            81,
            50,
            5.0,
            480 * 832,
            16,
            False,
        ),
    ],
)
def test_resolve_model_generation_defaults(
    model_name: str,
    model_class_name: str | None,
    fps: float,
    guidance_scale: float | None,
    num_frames: int,
    num_inference_steps: int,
    flow_shift: float | None,
    max_area: int,
    mod_value: int,
    is_ltx2: bool,
) -> None:
    defaults = _resolve(model_name, model_class_name)
    assert (
        defaults.fps,
        defaults.guidance_scale,
        defaults.num_frames,
        defaults.num_inference_steps,
        defaults.flow_shift,
        defaults.max_area,
        defaults.mod_value,
        defaults.is_ltx2,
    ) == (
        fps,
        guidance_scale,
        num_frames,
        num_inference_steps,
        flow_shift,
        max_area,
        mod_value,
        is_ltx2,
    )


def test_cli_defaults_are_none_so_model_defaults_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitted CLI flags must fall back to the model-aware generation defaults."""
    mod = _load_example_module()
    monkeypatch.setattr(sys, "argv", ["image_to_video.py", "--model", "/path/to/LTX-2"])
    args = mod.parse_args()
    assert args.num_frames is None
    assert args.num_inference_steps is None
    assert args.guidance_scale is None
    assert args.fps is None
    assert args.frame_rate is None
    defaults = mod.resolve_model_generation_defaults(str(args.model).lower(), args.model_class_name)
    assert defaults == mod.ModelGenerationDefaults(24, None, 121, 40, None, 512 * 768, 32, True)
