# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Qwen3-Omni thinker passes CUDA masks; the check gives the same answers there with a fixed number of syncs."""

import random
import warnings

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.common.test_audio_in_video import AUDIO, VIDEO, _audio_in_video, _random_request
from vllm_omni.model_executor.models.common.audio_in_video import check_interleaved_audio_video

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
]


def _run(tokens, device):
    ids = torch.tensor(tokens, dtype=torch.long, device=device)
    is_video, is_audio = ids == VIDEO, ids == AUDIO
    return check_interleaved_audio_video(is_video, is_audio, int(is_video.sum()), int(is_audio.sum()))


def test_cuda_matches_cpu():
    from vllm.model_executor.models.qwen2_5_omni_thinker import check_interleaved_audio_video as reference

    rng = random.Random(0)
    batches: list[list[int]] = [sum((_random_request(rng) for _ in range(rng.randint(1, 4))), []) for _ in range(200)]
    batches.append(_audio_in_video(6600, 750))
    for tokens in batches:
        ids = torch.tensor(tokens, dtype=torch.long)
        is_video, is_audio = ids == VIDEO, ids == AUDIO
        expected = reference(is_video, is_audio, int(is_video.sum()), int(is_audio.sum()))
        assert _run(tokens, "cpu") is expected
        assert _run(tokens, "cuda") is expected


def _device_syncs(tokens):
    ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
    is_video, is_audio = ids == VIDEO, ids == AUDIO
    num_video, num_audio = int(is_video.sum()), int(is_audio.sum())
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            result = check_interleaved_audio_video(is_video, is_audio, num_video, num_audio)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    return result, sum("called a synchronizing CUDA operation" in str(w.message) for w in caught)


def test_device_syncs_do_not_grow_with_the_prompt():
    # vLLM's loop syncs twice per placeholder token; this check syncs for the nonzero size and the final flag.
    assert _device_syncs(_audio_in_video(3000, 250)) == (True, 2)
    assert _device_syncs(_audio_in_video(6000, 500) * 2) == (True, 2)
