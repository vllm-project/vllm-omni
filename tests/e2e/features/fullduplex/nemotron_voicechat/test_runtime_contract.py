# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine.contracts import DuplexInputMode
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.nemotron_voicechat.runtime import (
    NemotronVoiceChatDuplexRuntimeExtension,
)
from vllm_omni.experimental.fullduplex.nemotron_voicechat.serving_adapter import (
    NemotronVoiceChatServingRuntimeAdapter,
    _render_tool_response,
)
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _frame() -> dict[str, object]:
    raw = np.zeros(1280, dtype=np.float32).tobytes()
    return {
        "type": "audio",
        "audio": base64.b64encode(raw).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }


def _plan(extension, *, input_seq: int):
    return extension.plan_append(
        request_id="req",
        fence=DuplexFence("sid", incarnation=2, epoch=3),
        session_config={},
        runtime_config={
            "nvc_prompt_token_ids": [0, 42, 1],
            "nvc_text_pad_id": 12,
            "nvc_max_model_len": 8192,
        },
        seq=input_seq,
        turn_seq=input_seq,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=_frame(),
        final=False,
        sampling_params=SamplingParams(),
    )


def test_first_append_prefills_prompt_then_each_append_consumes_one_frame() -> None:
    extension = NemotronVoiceChatDuplexRuntimeExtension()

    params = extension.configure_sampling_params(
        runtime_config={}, defaults=(SamplingParams(temperature=0.7, max_tokens=4),)
    )[0]
    assert params.temperature == 0.0 and params.max_tokens == 1 and params.top_p == 1.0 and params.top_k == 0

    first = _plan(extension, input_seq=1)
    later = _plan(extension, input_seq=2)

    assert first.prompt["prompt_token_ids"] == [0, 42, 1, 12]
    assert later.prompt["prompt_token_ids"] == [12]
    assert first.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 1
    assert later.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 2


def test_append_rejects_stage0_context_overflow() -> None:
    with pytest.raises(ValueError, match="max_model_len"):
        _plan(NemotronVoiceChatDuplexRuntimeExtension(), input_seq=8190)


def test_function_output_becomes_versioned_nvidia_channel_tokens() -> None:
    encoded: list[str] = []
    adapter = NemotronVoiceChatServingRuntimeAdapter(lambda *_: None)
    adapter._tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: encoded.append(text) or [31, 32, 33],
    )

    first = adapter.runtime_config_for_function_output(
        {"type": "function_call_output", "call_id": "call-1", "output": '{"result":20}'},
        {},
    )
    second = adapter.runtime_config_for_function_output(
        {"type": "function_call_output", "call_id": "call-2", "output": "plain text"},
        first,
    )

    assert _render_tool_response('{"result":20}') == '<TOOL_RESPONSE>[{"result":20}]</TOOL_RESPONSE>'
    assert encoded == [
        '<TOOL_RESPONSE>[{"result":20}]</TOOL_RESPONSE>',
        '<TOOL_RESPONSE>["plain text"]</TOOL_RESPONSE>',
    ]
    assert first["nvc_function_response_generation"] == 1
    assert second["nvc_function_response_generation"] == 2
    assert second["nvc_function_response_token_ids"] == [31, 32, 33]
    assert second["nvc_function_response_call_id"] == "call-2"
    assert second["nvc_function_response_batches"] == [
        {"generation": 1, "call_id": "call-1", "token_ids": [31, 32, 33]},
        {"generation": 2, "call_id": "call-2", "token_ids": [31, 32, 33]},
    ]

    session = {"function_response_generation": 0, "forced_function_tokens": []}
    NemotronVoiceChatThinkerForConditionalGeneration._sync_forced_function_response(session, second)

    assert session["function_response_generation"] == 2
    assert session["forced_function_token"] == 31
    assert session["forced_function_tokens"] == [31, 32, 33, 31, 32, 33]
