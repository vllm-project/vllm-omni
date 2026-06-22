# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.step_audio_editx import (
    _extract_ref_payload,
    talker2code2wav_async_chunk,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Entry:
    def __init__(self, list_data=None, scalar_data=None):
        self.list_data = list_data
        self.scalar_data = scalar_data


def _additional_information(ref_audio="ref.wav"):
    return SimpleNamespace(entries={"ref_audio": _Entry(list_data=[ref_audio])})


def _transfer_manager(*, chunk_size: int = 3):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": chunk_size}}),
    )


def _request(req_id: str, tokens: list[int], *, finished: bool = False, ref_audio="ref.wav"):
    return SimpleNamespace(
        external_req_id=req_id,
        output_token_ids=tokens,
        additional_information=_additional_information(ref_audio),
        is_finished=lambda: finished,
    )


def _full_payload_request(ref_audio="ref.wav"):
    return SimpleNamespace(additional_information=_additional_information(ref_audio))


def _mm(ref_tokens: list[int] | torch.Tensor | None = None):
    if ref_tokens is None:
        ref_tokens = [65536, 65537, 65538]
    if not isinstance(ref_tokens, torch.Tensor):
        ref_tokens = torch.tensor(ref_tokens, dtype=torch.long)
    return {"codes": {"ref": ref_tokens}}


def test_extract_ref_payload_offsets_reference_codes() -> None:
    out = _extract_ref_payload(_mm([65536, 65537, 65538]))

    assert out.tolist() == [0, 1, 2]


def test_extract_ref_payload_rejects_unoffset_reference_codes() -> None:
    with pytest.raises(RuntimeError, match="offset by 65536"):
        _extract_ref_payload(_mm([1, 2, 3]))


def test_async_chunk_accumulates_until_chunk_size_and_sends_conditioning_once() -> None:
    transfer_manager = _transfer_manager(chunk_size=3)
    req_id = "rid"

    first = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_mm([65540, 65541]),
        request=_request(req_id, [65536, 42], finished=False),
    )
    assert first is None

    second = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_mm([65540, 65541]),
        request=_request(req_id, [65536, 42, 65537, 65538], finished=False),
    )

    assert second is not None
    assert second.codes.audio.tolist() == [0, 1, 2]
    assert second.codes.ref.tolist() == [4, 5]
    assert second.latent == ["ref.wav"]
    assert second.meta.finished.item() is False
    assert second.meta.stream_finished is False
    assert second.meta.req_id == req_id

    third = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_mm([65540, 65541]),
        request=_request(req_id, [65536, 42, 65537, 65538, 65539, 65540, 65541], finished=False),
    )

    assert third is not None
    assert third.codes.audio.tolist() == [3, 4, 5]
    assert third.codes.ref is None
    assert third.latent is None


def test_async_chunk_uses_only_new_output_tokens() -> None:
    transfer_manager = _transfer_manager(chunk_size=2)
    req_id = "rid-new-only"

    first = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_mm(),
        request=_request(req_id, [65536, 65537], finished=False),
    )
    assert first is not None
    assert first.codes.audio.tolist() == [0, 1]

    second = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_mm(),
        request=_request(req_id, [65536, 65537, 65538, 65539], finished=False),
    )
    assert second is not None
    assert second.codes.audio.tolist() == [2, 3]


def test_async_chunk_flushes_remaining_tokens_and_clears_state_on_finish() -> None:
    transfer_manager = _transfer_manager(chunk_size=4)
    req_id = "rid-finish"

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_mm(),
        request=_request(req_id, [65536, 65537], finished=True),
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == [0, 1]
    assert payload.meta.finished.item() is True
    assert payload.meta.stream_finished is True
    assert req_id not in transfer_manager.request_payload


def test_async_chunk_errors_when_finished_without_audio_codes() -> None:
    transfer_manager = _transfer_manager(chunk_size=4)

    with pytest.raises(RuntimeError, match="produced no audio codec tokens"):
        talker2code2wav_async_chunk(
            transfer_manager=transfer_manager,
            multimodal_output=_mm(),
            request=_request("rid-empty", [1, 2, 3], finished=True),
        )


def test_token_only_processor_builds_placeholder_prompt_with_ref_audio_and_ref_code() -> None:
    output = SimpleNamespace(
        token_ids=[1, 65536, 65537, 2, 65538],
        multimodal_output=_mm([65540, 65541]),
    )
    talker_output = SimpleNamespace(finished=True, outputs=[output])

    prompts = talker2code2wav_token_only(
        [talker_output],
        prompt={"additional_information": {"ref_audio": "ref.wav"}},
    )

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt["prompt_token_ids"] == [0, 0, 0]
    assert prompt["additional_information"]["ref_audio"] == "ref.wav"
    assert prompt["additional_information"]["codes"]["ref"].tolist() == [4, 5]


def test_token_only_processor_skips_unfinished_outputs() -> None:
    prompts = talker2code2wav_token_only(
        [SimpleNamespace(finished=False, outputs=[])],
        prompt={"additional_information": {"ref_audio": "ref.wav"}},
    )

    assert prompts == []


def test_full_payload_processor_offsets_audio_and_ref_codes() -> None:
    payload = talker2code2wav_full_payload(
        transfer_manager=None,
        pooling_output={
            "codes.audio": torch.tensor([[65536], [65537], [3], [42], [65539]]),
            "codes.ref": torch.tensor([[65540, 65541]]),
        },
        request=_full_payload_request(),
    )

    assert payload is not None
    assert payload["codes"]["audio"].tolist() == [0, 1, 3]
    assert payload["codes"]["ref"].tolist() == [[4, 5]]
    assert payload["meta"]["finished"].item() is True


def test_full_payload_processor_rejects_unoffset_ref_codes() -> None:
    with pytest.raises(RuntimeError, match="offset by 65536"):
        talker2code2wav_full_payload(
            transfer_manager=None,
            pooling_output={
                "codes.audio": torch.tensor([65536, 65537]),
                "codes.ref": torch.tensor([4, 5]),
            },
            request=_full_payload_request(),
        )


def test_full_payload_processor_returns_none_without_audio_codes() -> None:
    payload = talker2code2wav_full_payload(
        transfer_manager=None,
        pooling_output={
            "codes.audio": torch.tensor([1, 2, 3]),
            "codes.ref": torch.tensor([65540, 65541]),
        },
        request=_full_payload_request(),
    )

    assert payload is None
