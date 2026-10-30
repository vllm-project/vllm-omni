# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""function_call_output enters the resident request as a later frame batch."""

from __future__ import annotations

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin import (
    NemotronVoiceChatDuplexPlugin,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


def test_parse_function_call_reads_the_data_plane_event():
    plugin = NemotronVoiceChatDuplexPlugin(lambda *_args: None)
    call = plugin.parse_function_call(
        {
            "function_call": True,
            "call_id": "call_ab12",
            "name": "get_weather",
            "arguments": '{"city":"Taipei"}',
        }
    )
    assert call == {
        "call_id": "call_ab12",
        "name": "get_weather",
        "arguments": '{"city":"Taipei"}',
    }
    assert plugin.parse_function_call({"text": "hello"}) is None


def test_function_call_output_queues_a_later_frame_batch():
    plugin = NemotronVoiceChatDuplexPlugin(lambda *_args: None)
    plugin._tokenizers["test-tokenizer"] = (None, _Tokenizer())
    updated = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="nemotron"),
        {"nvc_tokenizer_ref": "test-tokenizer", "nvc_function_response_generation": 0},
        {"type": "function_call_output", "call_id": "call_1", "output": "20"},
    )
    batches = updated["nvc_function_response_batches"]
    assert updated["nvc_function_response_generation"] == 1
    assert len(batches) == 1
    assert batches[0]["generation"] == 1
    assert batches[0]["token_ids"]
    assert "role" not in updated
