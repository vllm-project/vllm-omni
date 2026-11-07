# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 tests for the BWM example rollout with a mocked model boundary (CPU).

``rollout`` must return one frame per action, including tails of 1 to 3
frames, and send the pipeline 4n+1-frame chunks with one action per frame.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ACTION_DIM = 14
HEIGHT, WIDTH = 32, 48


def _load_example_module():
    path = Path(__file__).resolve().parents[4] / "examples" / "offline_inference" / "bwm" / "bwm_world_model.py"
    spec = importlib.util.spec_from_file_location("bwm_world_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bwm_world_model"] = module
    spec.loader.exec_module(module)
    return module


class FakeOmni:
    """Returns ``np`` video shaped like the BWM post-process output; chunk
    frame ``j`` is filled with value ``j``."""

    def __init__(self, num_history_frames: int):
        self.num_history_frames = num_history_frames
        self.chunks: list[int] = []

    def generate(self, prompt, sampling_params):
        n = sampling_params.num_frames
        mm = prompt["multi_modal_data"]
        assert (n - 1) % 4 == 0
        assert mm["action"].shape == (n, ACTION_DIM)
        assert mm["video"].shape == (self.num_history_frames, HEIGHT, WIDTH, 3)
        assert sampling_params.output_type == "np"
        self.chunks.append(n)
        values = (np.arange(n, dtype=np.float32) + 0.5) / 255.0
        video = np.broadcast_to(values[None, :, None, None, None], (1, n, HEIGHT, WIDTH, 3)).copy()
        return [SimpleNamespace(images=[video])]


@pytest.fixture(scope="module")
def example():
    return _load_example_module()


@pytest.mark.parametrize("total_frames", [1, 2, 3, 4, 49, 50, 51, 52, 53, 143])
def test_rollout_returns_one_frame_per_action(example, total_frames):
    omni = FakeOmni(example.NUM_HISTORY_FRAMES)
    first_frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    action = np.random.default_rng(0).uniform(-1, 1, (total_frames, ACTION_DIM)).astype(np.float32)

    frames = example.rollout(omni, first_frame, action)

    assert len(frames) == total_frames
    assert frames[0] is first_frame
    # Generated frames come from past the history prefix of each chunk.
    assert all(int(f[0, 0, 0]) >= example.NUM_HISTORY_FRAMES for f in frames[1:])


@pytest.mark.parametrize(("total_frames", "tail_chunk"), [(50, 13), (51, 13), (52, 13), (53, 13), (2, 13), (4, 13)])
def test_short_tail_uses_padded_4n_plus_1_chunk(example, total_frames, tail_chunk):
    omni = FakeOmni(example.NUM_HISTORY_FRAMES)
    first_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    action = np.zeros((total_frames, ACTION_DIM), dtype=np.float32)

    example.rollout(omni, first_frame, action)

    assert omni.chunks[-1] == tail_chunk
