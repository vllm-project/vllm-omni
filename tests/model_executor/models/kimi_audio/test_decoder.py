# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Acoustic block dispatch and request lifecycle, without loading acoustic weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.kimi_audio.kimi_audio_decoder import KimiAudioDecoder


@pytest.fixture
def streams():
    return [Mock(detokenize_streaming=Mock(side_effect=lambda codes, **kwargs: codes.float())) for _ in range(3)]


@pytest.fixture
def decoder(streams):
    decoder = KimiAudioDecoder(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            quant_config=None,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=384, kimia_token_offset=256)),
        )
    )
    decoder.detokenizer = Mock()
    decoder.detokenizer.new_stream.side_effect = streams
    return decoder


def test_complete_request_dispatches_each_code_once(decoder, streams):
    codes = torch.arange(65)
    output = decoder(codes, seq_token_counts=[65], request_ids=["a"])

    calls = streams[0].detokenize_streaming.call_args_list
    assert [call.args[0].shape for call in calls] == [(1, 30), (1, 30), (1, 5)]
    assert [call.kwargs for call in calls] == [
        {"upsample_factor": 4, "is_final": False},
        {"upsample_factor": 4, "is_final": False},
        {"upsample_factor": 4, "is_final": True},
    ]
    torch.testing.assert_close(torch.cat([call.args[0] for call in calls], dim=1).reshape(-1), codes)
    torch.testing.assert_close(output.multimodal_outputs["model_outputs"][0], codes.float())
    assert output.multimodal_outputs["sr"][0].item() == 24000
    streams[0].clear_states.assert_called_once_with()
    assert decoder._streams == {}


def test_interleaved_requests_keep_separate_streams_and_cleanup(decoder, streams):
    decoder(
        torch.arange(60),
        seq_token_counts=[30, 30],
        request_ids=["a", "b"],
        runtime_additional_information=[{"meta": {"stream_finished": False, "audio_seed": seed}} for seed in (42, 43)],
    )
    assert [call.args for call in decoder.detokenizer.new_stream.call_args_list] == [(42,), (43,)]
    assert decoder._streams == {"a": (streams[0], 1), "b": (streams[1], 1)}

    decoder(
        torch.tensor([60, 61]),
        seq_token_counts=[2],
        request_ids=["a"],
        runtime_additional_information=[{"meta": {"stream_finished": True, "chunk_seq": 1}}],
    )
    assert decoder.detokenizer.new_stream.call_count == 2
    assert streams[0].detokenize_streaming.call_count == 2
    assert streams[0].detokenize_streaming.call_args.kwargs["is_final"] is True
    streams[0].clear_states.assert_called_once_with()
    streams[1].clear_states.assert_not_called()
    assert decoder._streams == {"b": (streams[1], 1)}

    decoder.on_requests_finished({"b"})
    decoder.on_requests_finished({"b"})
    streams[1].clear_states.assert_called_once_with()
    assert decoder._streams == {}


def test_failed_batch_cleans_advanced_streams_but_preserves_other_requests(decoder, streams):
    decoder(
        torch.arange(90),
        seq_token_counts=[30, 30, 30],
        request_ids=["a", "b", "c"],
        runtime_additional_information=[{"meta": {"stream_finished": False}} for _ in range(3)],
    )
    streams[1].detokenize_streaming.side_effect = RuntimeError("acoustic failure")

    with pytest.raises(RuntimeError, match="acoustic failure"):
        decoder(
            torch.arange(60),
            seq_token_counts=[30, 30],
            request_ids=["a", "b"],
            runtime_additional_information=[{"meta": {"stream_finished": False, "chunk_seq": 1}} for _ in range(2)],
        )

    assert streams[0].detokenize_streaming.call_count == 2
    streams[0].clear_states.assert_called_once_with()
    streams[1].clear_states.assert_called_once_with()
    streams[2].clear_states.assert_not_called()
    assert decoder._streams == {"c": (streams[2], 1)}
