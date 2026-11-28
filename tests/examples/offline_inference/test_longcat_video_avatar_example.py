# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import importlib.util

import pytest

from tests.examples.helpers import EXAMPLES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@functools.cache
def _load_longcat_example():
    path = EXAMPLES / "offline_inference" / "longcat_video" / "end2end.py"
    spec = importlib.util.spec_from_file_location("longcat_video_avatar_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stage", ["at2v", "ai2v"])
@pytest.mark.parametrize("dimensions", [[], ["--height", "512", "--width", "768"]])
def test_longcat_example_accepts_default_or_paired_dimensions(stage, dimensions):
    args = _load_longcat_example().parse_args(["--stage", stage, *dimensions])
    assert (args.height, args.width) == ((512, 768) if dimensions else (None, None))


@pytest.mark.parametrize("dimensions", [["--height", "512"], ["--width", "768"]])
def test_longcat_example_rejects_one_sided_dimensions(dimensions, capsys):
    with pytest.raises(SystemExit) as exc:
        _load_longcat_example().parse_args(dimensions)
    assert exc.value.code == 2
    assert "--height and --width must be provided together" in capsys.readouterr().err


@pytest.mark.parametrize("height,width", [(0, 768), (512, -1)])
def test_longcat_example_rejects_nonpositive_dimensions(height, width, capsys):
    with pytest.raises(SystemExit) as exc:
        _load_longcat_example().parse_args(["--height", str(height), "--width", str(width)])
    assert exc.value.code == 2
    assert "--height and --width must be positive" in capsys.readouterr().err
