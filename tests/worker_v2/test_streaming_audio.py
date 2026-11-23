# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import numpy as np
import pytest

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_single_frame_start_ramp
from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SPF = 4


def _chunk_frames(buffer: StreamingAudioBuffer, frames: int, direct_first: bool = False) -> list[int]:
    request = buffer.add("req")
    if direct_first:
        request.accept_first_audio()
    sizes = []
    for i in range(frames):
        out = request.push(np.zeros(SPF, dtype=np.float32), ended=i == frames - 1)
        if out is not None:
            sizes.append(out.numel() // SPF)
    return sizes


def test_without_ramp_second_chunk_waits_for_chunk_frames():
    assert _chunk_frames(StreamingAudioBuffer(SPF, 25), 60) == [1, 25, 25, 9]


def test_ramp_sizes_chunks_by_emission_index():
    buffer = StreamingAudioBuffer(SPF, 25, [1, 2, 4, 8, 16, 25])
    assert _chunk_frames(buffer, 90) == [1, 2, 4, 8, 16, 25, 25, 9]


def test_ramp_counts_direct_first_audio_as_first_chunk():
    buffer = StreamingAudioBuffer(SPF, 25, [1, 2, 4])
    # The first frame goes out on the first-audio queue, not through push().
    assert _chunk_frames(buffer, 12, direct_first=True) == [2, 4, 5]


@pytest.mark.parametrize("raw", [[1, 2, 4, 8, 16, 25], "1,2,4,8,16,25"])
def test_single_stage_ramp_accepts_first_entry_of_one(raw):
    assert parse_single_frame_start_ramp({"codec_chunk_ramp": raw}, steady=25) == [1, 2, 4, 8, 16, 25]


def test_single_stage_ramp_rejects_first_entry_above_one():
    ramp = parse_single_frame_start_ramp({"codec_chunk_ramp": [2, 4, 25]}, steady=25)
    assert ramp is None
    # Disabled ramp: the stream falls back to one frame, then steady chunks.
    assert _chunk_frames(StreamingAudioBuffer(SPF, 25, ramp or ()), 12) == [1, 11]
