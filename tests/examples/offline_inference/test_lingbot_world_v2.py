# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from examples.offline_inference.diffusion import lingbot_world_v2 as example

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_offline_ulysses_argument(tmp_path):
    argv = ["--image", "frame.jpg", "--action-dir", "forward", "--prompt", "A lake"]
    paths = example.LingBotPaths(
        tmp_path / "frame.jpg", tmp_path / "forward", tmp_path, Path("forward"), 9, tmp_path / "out.mp4"
    )
    assert example.parse_args(argv).ulysses_degree == 1
    args = example.parse_args([*argv, "--ulysses-degree", "4"])
    kwargs = example.build_omni_kwargs(args, paths)
    assert kwargs["ulysses_degree"] == 4 and kwargs["tensor_parallel_size"] == 1
    args.ulysses_degree = 0
    with pytest.raises(ValueError, match="--ulysses-degree"):
        example.build_omni_kwargs(args, paths)
