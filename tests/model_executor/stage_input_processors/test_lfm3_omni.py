# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.model_executor.stage_input_processors.lfm3_omni import (
    temporal_to_codec_async_chunk,
    temporal_to_codec_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request(name):
    return Mock(external_req_id=name, is_finished=lambda: False)


def test_async_matches_full_payload_and_flushes_empty_terminal_chunk():
    manager = Mock(request_payload={})
    frames = torch.arange(8 * 12).reshape(8, 12)
    output = []
    for frame in frames:
        chunk = temporal_to_codec_async_chunk(manager, {"codes.audio": frame[None]}, request("a"))
        if chunk is not None:
            output.append(chunk.codes.audio)
    terminal = temporal_to_codec_async_chunk(manager, {}, request("a"), is_finished=True)
    assert terminal.meta.finished
    assert terminal.meta.stream_finished
    assert terminal.codes.audio.shape == (12, 0)
    assert terminal.codes.audio.view(torch.uint8).numel() == 0
    assert temporal_to_codec_async_chunk(manager, {}, request("a"), is_finished=True) is None
    full = temporal_to_codec_full_payload(manager, {"codes.audio": frames}, request("a"))
    torch.testing.assert_close(torch.cat(output, dim=1), full["codes"]["audio"])


def test_empty_full_payload_can_be_serialized_as_bytes():
    full = temporal_to_codec_full_payload(None, {}, request("a"))
    assert full["codes"]["audio"].shape == (12, 0)
    assert full["codes"]["audio"].view(torch.uint8).numel() == 0


def test_single_frame_payloads_survive_connector_serialization():
    """The final chunk of a 4k+1-frame stream, and a one-frame full payload, are (12, 1) transposes."""
    manager = Mock(request_payload={})
    frames = torch.arange(5 * 12).reshape(5, 12)
    for frame in frames:
        temporal_to_codec_async_chunk(manager, {"codes.audio": frame[None]}, request("a"))
    final = temporal_to_codec_async_chunk(manager, {}, request("a"), is_finished=True).codes.audio
    full = temporal_to_codec_full_payload(None, {"codes.audio": frames[:1]}, request("a"))["codes"]["audio"]
    for codes, expected in ((final, frames[4:].T), (full, frames[:1].T)):
        restored = OmniSerializer.deserialize(OmniSerializer.serialize({"codes": codes}))["codes"]
        torch.testing.assert_close(restored, expected)


def test_interleaved_requests_do_not_share_pending_frames():
    manager = Mock(request_payload={})
    for name, value in (("a", 3), ("b", 7)):
        assert (
            temporal_to_codec_async_chunk(manager, {"codes.audio": torch.full((2, 12), value)}, request(name)) is None
        )
    for name, value in (("b", 7), ("a", 3)):
        chunk = temporal_to_codec_async_chunk(manager, {}, request(name), is_finished=True)
        torch.testing.assert_close(chunk.codes.audio, torch.full((12, 2), value))


def test_reject_terminal_code_before_codec_decode():
    with pytest.raises(ValueError, match="terminal"):
        temporal_to_codec_full_payload(None, {"codes.audio": torch.full((1, 12), 2016)}, request("a"))
