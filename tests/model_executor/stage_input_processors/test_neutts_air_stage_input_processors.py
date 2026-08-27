# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.neutts_air import (
    NEUTTS_SPEECH_GENERATION_START_TOKEN_ID,
    NEUTTS_SPEECH_TOKEN_OFFSET,
    filter_speech_codes,
    llm2neucodec_async_chunk,
    llm2neucodec_sync,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _speech_tokens(codes: list[int]) -> list[int]:
    return [NEUTTS_SPEECH_TOKEN_OFFSET + code for code in codes]


def _request(
    request_id: str,
    reference_codes: list[int],
    target_codes: list[int],
    *,
    finished: bool = False,
):
    prompt_token_ids = [
        100,
        NEUTTS_SPEECH_GENERATION_START_TOKEN_ID,
        *_speech_tokens(reference_codes),
    ]
    all_token_ids = [
        *prompt_token_ids,
        *_speech_tokens(target_codes),
    ]
    if finished:
        all_token_ids.append(151670)
    return SimpleNamespace(
        request_id=f"internal-{request_id}",
        external_req_id=request_id,
        prompt_token_ids=prompt_token_ids,
        all_token_ids=all_token_ids,
        is_finished=lambda: finished,
    )


def _transfer_manager(**extra):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _payload_codes(payload) -> list[int]:
    assert payload is not None
    assert payload.codes is not None
    assert payload.codes.audio is not None
    return payload.codes.audio.tolist()


def test_filter_speech_codes_rebases_valid_boundary_tokens():
    token_ids = [151671, 151681, 217206]

    assert filter_speech_codes(token_ids) == [0, 10, 65535]


def test_filter_speech_codes_ignores_tokens_outside_codec_range():
    token_ids = [151643, 151669, 151670, 217207, 217651]

    assert filter_speech_codes(token_ids) == []


def test_filter_speech_codes_preserves_order_and_duplicates():
    token_ids = [151681, 151671, 151681]

    assert filter_speech_codes(token_ids) == [10, 0, 10]


def test_llm2neucodec_sync_supports_batched_source_outputs():
    source_outputs = [
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[151671, 151681])]),
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[151691])]),
    ]

    outputs = llm2neucodec_sync(source_outputs)

    assert len(outputs) == 2
    assert outputs[0]["prompt_token_ids"] == [0, 10]
    assert outputs[1]["prompt_token_ids"] == [20]


def test_async_chunk_waits_for_25_frames_plus_5_lookforward_frames():
    manager = _transfer_manager()
    request = _request("rid-wait", list(range(60)), list(range(29)))

    payload = llm2neucodec_async_chunk(manager, None, request)

    assert payload is None
    assert manager.code_prompt_token_ids["rid-wait"] == []


def test_async_chunk_matches_official_30_55_63_lifecycle():
    manager = _transfer_manager()
    reference = list(range(60))
    target = list(range(100, 163))

    first = llm2neucodec_async_chunk(
        manager,
        None,
        _request("rid-flow", reference, target[:30]),
    )
    assert _payload_codes(first) == reference[9:] + target[:30]
    assert first.meta is not None
    assert first.meta.left_context_size == 51
    assert first.meta.right_holdback_size == 3
    assert first.meta.num_processed_tokens == 25
    assert not bool(first.meta.stream_finished.item())
    assert manager.code_prompt_token_ids["rid-flow"] == target[:25]

    second = llm2neucodec_async_chunk(
        manager,
        None,
        _request("rid-flow", reference, target[:55]),
    )
    assert _payload_codes(second) == reference[34:] + target[:55]
    assert second.meta is not None
    assert second.meta.left_context_size == 51
    assert second.meta.right_holdback_size == 3
    assert second.meta.num_processed_tokens == 25
    assert not bool(second.meta.stream_finished.item())
    assert manager.code_prompt_token_ids["rid-flow"] == target[:50]

    final = llm2neucodec_async_chunk(
        manager,
        None,
        _request("rid-flow", reference, target[:63], finished=True),
        is_finished=True,
    )
    assert _payload_codes(final) == reference[59:] + target[:63]
    assert final.meta is not None
    assert final.meta.left_context_size == 50
    assert final.meta.right_holdback_size == 0
    assert final.meta.num_processed_tokens == 13
    assert bool(final.meta.finished.item())
    assert bool(final.meta.stream_finished.item())
    assert manager.code_prompt_token_ids["rid-flow"] == target[:63]


def test_async_chunk_finished_without_remaining_codes_emits_cleanup_wakeup():
    manager = _transfer_manager()
    target = list(range(50))
    manager.code_prompt_token_ids["rid-eof"] = target.copy()

    payload = llm2neucodec_async_chunk(
        manager,
        None,
        _request("rid-eof", list(range(60)), target, finished=True),
        is_finished=True,
    )

    # One old context code wakes Stage 1; num_processed_tokens=0 means it is
    # not new speech and is only used to clean the streaming decoder cache.
    assert _payload_codes(payload) == [target[-1]]
    assert payload.meta is not None
    assert payload.meta.left_context_size == 1
    assert payload.meta.num_processed_tokens == 0
    assert bool(payload.meta.finished.item())
    assert bool(payload.meta.stream_finished.item())


def test_async_chunk_keeps_request_progress_isolated():
    manager = _transfer_manager()
    reference = list(range(60))

    assert (
        llm2neucodec_async_chunk(
            manager,
            None,
            _request("rid-a", reference, list(range(30))),
        )
        is not None
    )
    assert (
        llm2neucodec_async_chunk(
            manager,
            None,
            _request("rid-b", reference, list(range(29))),
        )
        is None
    )
    assert len(manager.code_prompt_token_ids["rid-a"]) == 25
    assert manager.code_prompt_token_ids["rid-b"] == []


def test_async_chunk_rejects_prompt_without_generation_marker():
    manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="internal-bad",
        external_req_id="rid-bad",
        prompt_token_ids=[100, *_speech_tokens([0, 1])],
        all_token_ids=[100, *_speech_tokens([0, 1, 2])],
        is_finished=lambda: False,
    )

    with pytest.raises(ValueError, match="SPEECH_GENERATION_START"):
        llm2neucodec_async_chunk(manager, None, request)
