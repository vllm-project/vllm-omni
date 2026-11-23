# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from dataclasses import dataclass, field

import pytest

from vllm_omni.model_executor.stage_input_processors.mimo_audio import llm2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Connector:
    config: dict[str, object] = field(default_factory=dict)


@dataclass
class _TransferManager:
    connector: _Connector = field(default_factory=_Connector)
    code_prompt_token_ids: defaultdict[str, list[list[int]]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class _Request:
    external_req_id: str


def test_text_only_step_ignores_explicit_null_audio_codes() -> None:
    transfer_manager = _TransferManager()
    request = _Request(external_req_id="text-only")

    payload = llm2code2wav_async_chunk(
        transfer_manager,
        {"codes": {"audio": None}},
        request,
    )

    assert payload is None
    assert transfer_manager.code_prompt_token_ids[request.external_req_id] == []


def test_text_only_terminal_step_emits_finished_sentinel() -> None:
    transfer_manager = _TransferManager()
    request = _Request(external_req_id="text-only")

    payload = llm2code2wav_async_chunk(
        transfer_manager,
        {"codes": {"audio": None}},
        request,
        is_finished=True,
    )

    assert payload is not None
    assert payload.codes.audio.numel() == 0
    assert payload.meta.finished.item() is True
