# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released PCM sample arithmetic and real model data-plane encoding."""

from __future__ import annotations

import base64
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.model_executor.models.lychee_fd.test_native_duplex_stream import REQ, context, result
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.entrypoints.duplex.audio_encoding import encode_audio
from vllm_omni.model_executor.models.lychee_fd.duplex.audio_encoding import make_lychee_audio_encoder
from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SAMPLES = np.array([-1.2, -1, -0.5, -1.9 / 32767, -0.5 / 32767, 0, 1.9 / 32767, 0.5, 1, 1.2], dtype=np.float32)
EXPECTED = np.array([-32767, -32767, -16383, -1, 0, 0, 1, 16383, 32767, 32767], dtype="<i2")


def decoded_samples(encoded):
    pcm = base64.b64decode(encoded)
    samples = np.frombuffer(pcm, dtype="<i2")
    assert len(pcm) == samples.size * 2
    return samples


@pytest.mark.parametrize("speed", [None, 1.0])
@pytest.mark.parametrize("response_format", ["pcm16", "pcm", "PCM16"])
@pytest.mark.parametrize("kind", ["tensor", "numpy"])
def test_default_speed_pcm_uses_released_clip_scale_truncation(speed, response_format, kind):
    fallback = Mock(side_effect=AssertionError("Shared RAW quantization must not encode default Lychee PCM"))
    encoder = make_lychee_audio_encoder(fallback)
    samples = torch.from_numpy(SAMPLES).reshape(2, 5) if kind == "tensor" else SAMPLES.reshape(2, 5)
    encoded = encoder(samples, 24000, response_format, speed)
    actual = decoded_samples(encoded)
    assert actual.size == 10
    np.testing.assert_array_equal(actual, EXPECTED)
    fallback.assert_not_called()


@pytest.mark.parametrize("response_format,speed", [("wav", None), ("flac", 1.0), ("pcm16", 2.0), ("pcm", 0.5)])
def test_other_formats_and_nondefault_speed_delegate_the_original_injected_encoder(response_format, speed):
    fallback = Mock(return_value="delegated-audio")
    audio = torch.from_numpy(SAMPLES)
    actual = make_lychee_audio_encoder(fallback)(audio, 24000, response_format, speed)
    assert actual == "delegated-audio"
    fallback.assert_called_once_with(audio, 24000, response_format, speed)


def test_real_lychee_plane_pcm_frames_metadata_final_and_other_owner_cleanup():
    plugin = LycheeDuplexPlugin(encode_audio)
    plane = plugin.data_plane
    owner = {
        "response_id": "pcm-response",
        "response_number": 1,
        "execution_epoch": 0,
        "session_epoch": 0,
        "chunk_seq": 0,
        "final": False,
        "tick": 10,
        "num_samples": 10,
    }
    payload = {"audio": torch.from_numpy(SAMPLES), "sr": torch.tensor(24000), "lychee_t2w": owner}
    (event,) = tuple(plane.project(result(payload), context=context()))
    np.testing.assert_array_equal(decoded_samples(event["audio"]), EXPECTED)
    assert decoded_samples(event["audio"]).size == 10
    assert event["audio_format"] == "pcm16" and event["sample_rate_hz"] == 24000
    assert event["audio_duration_ms"] == round(10 * 1000 / 24000)
    assert event["audio_complete"] is False and event["end_of_turn"] is False
    assert event["model_metadata"]["model_response_id"] == "pcm-response"
    assert event["model_metadata"]["codec_chunk_seq"] == 0
    assert tuple(plane.project(result(payload), context=context())) == ()
    stale = {**payload, "lychee_t2w": {**owner, "chunk_seq": 1}}
    assert tuple(plane.project(result(stale), context=context(epoch=1))) == ()

    foreign = duplex_resource_request_id(DuplexFence("other"), "stage0")
    foreign_payload = {
        "audio": torch.from_numpy(SAMPLES[:3]),
        "sr": 24000,
        "lychee_t2w": {**owner, "response_id": "other-pcm", "num_samples": 3},
    }
    foreign_result = {
        "data_plane_outputs": [
            SimpleNamespace(
                request_id=foreign, outputs=[SimpleNamespace(multimodal_output=foreign_payload, finish_reason=None)]
            )
        ]
    }
    (other_event,) = tuple(plane.project(foreign_result, context=context()))
    np.testing.assert_array_equal(decoded_samples(other_event["audio"]), EXPECTED[:3])
    final = {"audio": torch.empty(0), "lychee_t2w": {**owner, "num_samples": 0, "chunk_seq": 1, "final": True}}
    (completion,) = tuple(plane.project(result(final), context=context()))
    assert completion["audio"] == "" and completion["audio_complete"] is True
    assert completion["end_of_turn"] is True and completion["preserve_request"] is True
    plane.mark_terminal(REQ)
    assert tuple(plane.project(result(stale), context=context())) == ()
    plane.close_session("probe", active_request_id=REQ)
    assert (foreign, "other-pcm", 0) in plane._audio_seq
    assert not any(key[0] == REQ for key in plane._audio_seq)


def test_lychee_plane_keeps_nondefault_speed_injected_encoding_semantics():
    fallback = Mock(return_value="delegated-audio")
    plane = LycheeDuplexPlugin(fallback).data_plane
    owner = {"response_number": 1, "execution_epoch": 0, "session_epoch": 0, "chunk_seq": 0, "final": True}
    payload = {"audio": torch.from_numpy(SAMPLES), "lychee_t2w": owner}
    (event,) = tuple(plane.project(result(payload), context=replace(context(), speed=2.0)))
    assert event["audio"] == "delegated-audio" and event["audio_format"] == "pcm16"
    assert event["audio_complete"] is True
    args = fallback.call_args.args
    np.testing.assert_array_equal(args[0].numpy(), SAMPLES)
    assert args[1:] == (24000, "pcm16", 2.0)
