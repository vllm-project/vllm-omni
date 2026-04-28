# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import talker2code2wav_async_chunk, text2flow


def _source_output(request_id: str, prompt_ids: list[int], out_ids: list[int], mm: dict):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        outputs=[SimpleNamespace(token_ids=out_ids, cumulative_token_ids=out_ids, multimodal_output=mm)],
    )


def _transfer_manager(
    *,
    chunk_frames: int = 2,
    pre_lookahead_frames: int = 0,
    stream_scale_factor: int = 1,
    max_chunk_frames: int | None = None,
):
    if max_chunk_frames is None:
        max_chunk_frames = chunk_frames
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_pre_lookahead_frames": pre_lookahead_frames,
                    "codec_max_chunk_frames": max_chunk_frames,
                    "codec_stream_scale_factor": stream_scale_factor,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )


def test_text2flow_supports_batched_source_outputs():
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                _source_output("req-0", [10, 11], [1, 2, 3], {"speech_token": torch.tensor([[1, 2]])}),
                _source_output("req-1", [20, 21], [4, 5], {"speech_token": torch.tensor([[3, 4]])}),
            ]
        )
    ]

    outputs = text2flow(stage_list=stage_list, engine_input_source=[0], prompt=None)

    assert len(outputs) == 2
    assert outputs[0]["prompt_token_ids"] == [1, 2, 3]
    assert outputs[1]["prompt_token_ids"] == [4, 5]
    assert outputs[0]["additional_information"]["ids"]["prompt"] == [10, 11]
    assert outputs[1]["additional_information"]["ids"]["prompt"] == [20, 21]


def test_talker2code2wav_async_chunk_final_payload_uses_absolute_token_offset():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        external_req_id="rid-0",
        output_token_ids=[1, 2, 6562, 3],
        additional_information={
            "speech_token": [torch.tensor([[11, 12, 13]])],
            "speech_feat": [torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])],
            "embedding": [torch.tensor([[0.5, 0.6]])],
        },
        is_finished=lambda: True,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload is not None
    assert payload["meta"]["finished"].item() is True
    assert payload["codes"]["audio"] == [1, 2, 3]
    assert payload["token_offset"] == 0
    assert payload["left_context_size"] == 0
    assert payload["req_id"] == ["rid-0"]
    assert payload["stream_finished"].item() is True
    assert "speech_token" in payload
    assert "speech_feat" in payload
    assert "embedding" in payload


def test_talker2code2wav_async_chunk_emits_eof_when_finished_without_valid_codes():
    transfer_manager = _transfer_manager(chunk_frames=25)
    request = SimpleNamespace(
        external_req_id="rid-eof",
        output_token_ids=[6561, 6562],  # all filtered out
        additional_information={},
        is_finished=lambda: True,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload is not None
    assert payload["codes"]["audio"] == []
    assert payload["meta"]["finished"].item() is True


def test_talker2code2wav_async_chunk_does_not_reemit_without_new_tokens():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        external_req_id="rid-stable",
        output_token_ids=[1, 2],
        additional_information={},
        is_finished=lambda: False,
    )

    payload1 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    payload2 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )

    assert payload1 is not None
    assert payload1["codes"]["audio"] == [1, 2]
    assert payload1["token_offset"] == 0
    assert payload2 is None


def test_talker2code2wav_async_chunk_waits_for_prelookahead_and_emits_cumulative_prefix():
    transfer_manager = _transfer_manager(pre_lookahead_frames=1)
    request = SimpleNamespace(
        external_req_id="rid-pre",
        output_token_ids=[1, 2],
        additional_information={},
        is_finished=lambda: False,
    )

    payload_pending = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    request.output_token_ids = [1, 2, 3]
    payload_ready = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )

    assert payload_pending is None
    assert payload_ready is not None
    assert payload_ready["codes"]["audio"] == [1, 2, 3]
    assert payload_ready["token_offset"] == 0
    assert payload_ready["meta"]["finished"].item() is False


def test_talker2code2wav_async_chunk_final_flush_uses_previous_token_offset():
    transfer_manager = _transfer_manager(pre_lookahead_frames=1)
    request = SimpleNamespace(
        external_req_id="rid-tail",
        output_token_ids=[3, 4, 5],
        additional_information={},
        is_finished=lambda: False,
    )

    payload_stream = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    request.output_token_ids = [3, 4, 5, 6]
    payload_final = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload_stream is not None
    assert payload_stream["meta"]["finished"].item() is False
    assert payload_stream["codes"]["audio"] == [3, 4, 5]
    assert payload_stream["token_offset"] == 0
    assert payload_final is not None
    assert payload_final["meta"]["finished"].item() is True
    assert payload_final["codes"]["audio"] == [3, 4, 5, 6]
    assert payload_final["token_offset"] == 2


def test_talker2code2wav_async_chunk_respects_prompt_token_pad_on_first_chunk():
    transfer_manager = _transfer_manager(pre_lookahead_frames=1)
    request = SimpleNamespace(
        external_req_id="rid-pad",
        output_token_ids=[8, 9, 10],
        additional_information={
            "speech_token": [torch.tensor([[1, 2, 3]])],
        },
        is_finished=lambda: False,
    )

    payload_pending = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    request.output_token_ids = [8, 9, 10, 11]
    payload_ready = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )

    assert payload_pending is None
    assert payload_ready is not None
    assert payload_ready["codes"]["audio"] == [8, 9, 10, 11]
    assert payload_ready["token_offset"] == 0


def test_talker2code2wav_async_chunk_emits_terminal_eof_without_duplicate_audio():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        external_req_id="rid-eof-tail",
        output_token_ids=[3, 4],
        additional_information={},
        is_finished=lambda: False,
    )

    payload_stream = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    payload_final = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload_stream is not None
    assert payload_stream["meta"]["finished"].item() is False
    assert payload_stream["codes"]["audio"] == [3, 4]
    assert payload_final is not None
    assert payload_final["meta"]["finished"].item() is True
    assert payload_final["codes"]["audio"] == []
