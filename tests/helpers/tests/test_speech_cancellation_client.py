# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU transport/harness contracts only; these do not prove TTS inference."""

import pytest

from tests.helpers.assertions import assert_audio_speech_equal
from tests.helpers.runtime import OfflineOmniClient, OmniResponse, OnlineOmniClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cancellation_closes_response_and_preserves_odd_byte_prefix(mocker):
    response = mocker.MagicMock()
    response.__enter__.return_value = response
    response.headers = {"content-type": "audio/pcm"}
    response.iter_content.return_value = iter([b"\x01", b"\x02\x03\x04\x05", b"\x06\x07"])
    post = mocker.patch("tests.helpers.client.requests.post", return_value=response)
    client = object.__new__(OnlineOmniClient)
    client.base_url = "http://127.0.0.1:1"
    result = client.send_audio_speech_cancel_request(
        {
            "json": {"stream": True, "stream_format": "audio", "response_format": "pcm"},
            "cancel_after_audio_bytes": 5,
        }
    )
    assert result.audio_bytes == b"\x01\x02\x03\x04"
    assert result.cancelled and not result.success
    response.__exit__.assert_called_once()
    assert post.call_args.kwargs["stream"] is True


def test_cancellation_does_not_label_complete_eof_as_cancelled(mocker):
    response = mocker.MagicMock()
    response.__enter__.return_value = response
    response.headers = {"content-type": "audio/pcm"}
    response.iter_content.return_value = iter([b"\x01\x02"])
    mocker.patch("tests.helpers.client.requests.post", return_value=response)
    client = object.__new__(OnlineOmniClient)
    client.base_url = "http://127.0.0.1:1"
    with pytest.raises(AssertionError):
        client.send_audio_speech_cancel_request(
            {
                "json": {"stream": True, "stream_format": "audio", "response_format": "pcm"},
            }
        )
    response.__exit__.assert_called_once()


def test_worker_probe_restores_hooks_after_real_runner_failure(mocker):
    runner = mocker.Mock()
    runner.generate.side_effect = RuntimeError("inference failure")
    client = OfflineOmniClient(runner)
    with pytest.raises(RuntimeError, match="inference failure"):
        client.send_worker_probe_request(
            {
                "probe_method": "observe",
                "probe_config": {"token_ids": [10, 3]},
                "prompt": {"prompt_token_ids": [1]},
                "sampling_params_list": [],
            }
        )
    assert runner.omni.engine.collective_rpc.call_args_list[-1].kwargs["args"] == (False, {})


def test_partial_audio_attribution_rejects_equal_length_wrong_request():
    expected = OmniResponse(audio_bytes=b"\x01\x00" * 12, audio_format="audio/pcm")
    prefix = OmniResponse(audio_bytes=b"\x01\x00" * 2, audio_format="audio/pcm")
    wrong = OmniResponse(audio_bytes=b"\x02\x00" * 2, audio_format="audio/pcm")
    assert_audio_speech_equal(prefix, expected, sample_rate=32000, prefix=True)
    with pytest.raises(AssertionError, match="different trajectory/request"):
        assert_audio_speech_equal(wrong, expected, sample_rate=32000, prefix=True)
