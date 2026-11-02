# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio decoded after a request stopped is queued, in order, for the scheduler."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.moss_tts.tail_audio_sender import TailAudioSender

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_pieces_wait_for_their_decode_and_keep_submission_order():
    scheduler = SimpleNamespace(omni_deferred_finishes={"r": {}}, omni_tail_outputs=deque())
    sender = TailAudioSender(scheduler, torch.device("cuda"))
    side = torch.cuda.Stream()
    pieces = []
    with torch.cuda.stream(side):
        for value in (1.0, 2.0, 3.0):
            torch.cuda._sleep(5_000_000)
            pcm = torch.full((2, 4), value, device="cuda")
            event = torch.cuda.Event()
            event.record(side)
            pieces.append((pcm, event))
    sr = torch.tensor(48000, dtype=torch.int32)
    assert sender.is_deferred("r") and not sender.is_deferred("x")
    sender.submit("r", [pieces[0]], sr, False)
    sender.submit("r", pieces[1:], sr, True)
    sender.close()
    first, last = scheduler.omni_tail_outputs
    assert first[0] == "r" and first[2] is False
    assert first[1]["model_outputs"].unique().tolist() == [1.0] and not first[1]["model_outputs"].is_cuda
    assert last[2] is True and last[1]["model_outputs"][0].tolist() == [2.0] * 4 + [3.0] * 4


def test_an_empty_final_is_queued_without_audio():
    scheduler = SimpleNamespace(omni_deferred_finishes={"e": {}}, omni_tail_outputs=deque())
    sender = TailAudioSender(scheduler, torch.device("cuda"))
    sender.submit("e", [], torch.tensor(48000, dtype=torch.int32), True)
    sender.close()
    (only,) = scheduler.omni_tail_outputs
    assert only == ("e", None, True, None)


def test_outputs_after_an_early_first_frame_carry_the_marker():
    from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY

    scheduler = SimpleNamespace(omni_deferred_finishes={"m": {}}, omni_tail_outputs=deque())
    sender = TailAudioSender(scheduler, torch.device("cuda"))
    sr = torch.tensor(48000, dtype=torch.int32)
    sender.submit("m", [(torch.ones((2, 4), device="cuda"), None)], sr, False, after_first=True)
    # Even a terminal output without audio must follow the sent first frame.
    sender.submit("m", [], sr, True, after_first=True)
    sender.close()
    audio, final = scheduler.omni_tail_outputs
    assert bool(audio[1][FIRST_AUDIO_REQUIRED_KEY]) and audio[1]["model_outputs"].shape == (2, 4)
    assert final[2] is True and set(final[1]) == {FIRST_AUDIO_REQUIRED_KEY}
