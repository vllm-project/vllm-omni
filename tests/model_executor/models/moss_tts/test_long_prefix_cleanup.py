# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch import nn

from tests.model_executor.models.moss_tts.test_streaming_terminal_batch import session
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("interrupt", [False, True])
def test_long_rounds_external_cleanup_and_slot_reuse(mocker, interrupt):
    state = session()
    decoder = object.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder._stream_max_step_frames = 15
    decoder._stream_state_capacity = 8
    decoder._stream_req_slots = {}
    decoder._stream_session = state
    decoder._codec_stream = None
    decoder._codec = state._codec
    decoder._ensure_stream_session = lambda: state
    original = state.step
    calls = 0

    def step(*args, **kwargs):
        nonlocal calls
        calls += 1
        if interrupt and calls == 2:
            raise RuntimeError("injected codec failure")
        return original(*args, **kwargs)

    mocker.patch.object(state, "step", side_effect=step)
    items = [(0, "a", torch.ones(2, 35), False), (1, "b", torch.ones(2, 31), False)]
    if interrupt:
        with pytest.raises(RuntimeError, match="injected codec failure"):
            decoder._decode_streaming_batch(items)
    else:
        output = decoder._decode_streaming_batch(items)
        for index, _, codes, _ in items:
            torch.testing.assert_close(output[index], torch.arange(2.0, 2 * codes.shape[1] + 1, 2.0)[None])
    assert set(decoder._stream_req_slots) == {"a", "b"}
    assert state._codec.state.count_nonzero() > 0
    decoder.on_requests_finished({"a", "b"})
    decoder.on_requests_finished({"a", "b"})
    assert not decoder._stream_req_slots
    assert state._codec.state.count_nonzero() == 0
    reused = decoder._decode_streaming_batch([(0, "new", torch.ones(2, 1), True)])
    torch.testing.assert_close(reused[0], torch.tensor([[2.0]]))
    assert not decoder._stream_req_slots
