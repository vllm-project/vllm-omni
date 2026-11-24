# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch import nn

from tests.model_executor.models.moss_tts.test_streaming_terminal_batch import session
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("buffered", [False, True])
def test_long_prefix_rounds_preserve_audio_and_release_slots(mocker, buffered):
    state = session()
    decoder = object.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder._stream_max_step_frames = 15
    decoder._stream_state_capacity = 8
    decoder._stream_req_slots = {}
    decoder._ensure_stream_session = lambda: state
    state._gpu_output = buffered
    decoder._codec = state._codec
    step = mocker.spy(state, "step")
    entry = mocker.spy(decoder, "_decode_streaming_batch")
    lengths = [55, 49, 3]
    buffers = {i: torch.empty(1, length) for i, length in enumerate(lengths)}
    output = decoder._decode_streaming_batch(
        [(i, str(i), torch.ones(2, length), True) for i, length in enumerate(lengths)],
        output_buffers=buffers,
    )
    for i, length in enumerate(lengths):
        torch.testing.assert_close(output[i], torch.arange(2.0, 2 * length + 1, 2.0)[None])
    assert entry.call_count == 1
    if buffered:
        assert all(output[i].data_ptr() == buffers[i].data_ptr() for i in output)
    assert not decoder._stream_req_slots
    assert any(len(call.args[0]) > 1 for call in step.call_args_list)
    assert step.call_count < 9
    reused = decoder._decode_streaming_batch([(0, "new", torch.ones(2, 1), True)])
    torch.testing.assert_close(reused[0], torch.tensor([[2.0]]))
