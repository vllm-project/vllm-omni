# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path

import pytest

from examples.offline_inference.diffusion import ar_diffusion_streaming_decode as streaming_example
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


@pytest.mark.parametrize(("num_blocks", "num_frames"), [(10, 117), (11, 129)])
def test_streaming_action_script_accepts_long_rollouts(tmp_path, num_blocks, num_frames):
    action_script = tmp_path / "actions.json"
    action_script.write_text(json.dumps([[["w"], [], ["a"]] for _ in range(num_blocks)]))

    script = streaming_example._load_action_script(action_script)

    assert len(script) == num_blocks
    assert (
        streaming_example._validate_against_checkpoint(script, frames_per_block=3, temporal_compression=4) == num_frames
    )


def test_streaming_action_script_rejects_wrong_frame_count_in_eleventh_block(tmp_path):
    action_script = tmp_path / "actions.json"
    action_script.write_text(json.dumps([[[], [], []] for _ in range(10)] + [[[], []]]))
    script = streaming_example._load_action_script(action_script)

    with pytest.raises(ValueError, match="block 10 holds 2 frame action lists"):
        streaming_example._validate_against_checkpoint(script, frames_per_block=3, temporal_compression=4)
