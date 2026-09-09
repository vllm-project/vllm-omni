# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.contracts import DuplexInputMode
from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.runtime import (
    NemotronVoiceChatDuplexRuntimeExtension,
)
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.serving_adapter import (
    NemotronVoiceChatServingRuntimeAdapter,
    _render_tool_response,
)
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import ModelInputError

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


@pytest.mark.parametrize("available", [False, True])
def test_nemotron_native_append_capability_does_not_enable_model_replay(monkeypatch, available) -> None:
    monkeypatch.setattr("vllm_omni.engine.kv_append.scheduler_native_append_available", lambda: available)
    capabilities = NemotronVoiceChatServingRuntimeAdapter.capabilities(max_sessions=1)
    assert capabilities.supports_scheduler_native_append is available
    assert capabilities.supports_prompt_replay is False
    assert not hasattr(NemotronVoiceChatDuplexRuntimeExtension(), "prepare_recovery_prompt")


@pytest.mark.parametrize("computed", [0, 3])
def test_nemotron_rejects_historical_recompute_before_advancing_perception(computed) -> None:
    model = SimpleNamespace(_sessions={"req": {}})
    with pytest.raises(ModelInputError, match="native_duplex_recompute_unsupported"):
        NemotronVoiceChatThinkerForConditionalGeneration._preprocess_duplex(
            model,
            request_id="req",
            input_ids=torch.zeros(1, dtype=torch.long),
            info={"_omni_num_computed_tokens": computed},
            duplex={
                "source_input_seq": 2,
                "kv_append_start": 4,
                "runtime_config": {"nvc_text_pad_id": 12},
            },
        )
    assert model._sessions == {"req": {}}


def test_nemotron_missing_continuation_state_fails_request_locally() -> None:
    model = SimpleNamespace(_sessions={})
    with pytest.raises(ModelInputError, match="no retained model state"):
        NemotronVoiceChatThinkerForConditionalGeneration._preprocess_duplex(
            model,
            request_id="req",
            input_ids=torch.zeros(1, dtype=torch.long),
            info={"_omni_num_computed_tokens": 4},
            duplex={
                "source_input_seq": 2,
                "kv_append_start": 4,
                "runtime_config": {"nvc_text_pad_id": 12},
            },
        )
    assert not model._sessions


def test_nemotron_retained_continuation_uses_current_frame_and_previous_sample() -> None:
    model = SimpleNamespace(
        _sessions={"req": {"func_token": 5, "prefill_embeds": torch.zeros(4, 4)}},
        _duplex_previous_text_tokens={"req": 6},
        _duplex_stable_frame=lambda *args: torch.ones(1, 4),
        _sync_forced_function_response=lambda *args: None,
        _fuse=lambda text, frame, function: frame + text.reshape(1, 1) + function.reshape(1, 1),
    )
    _, embeddings, _ = NemotronVoiceChatThinkerForConditionalGeneration._preprocess_duplex(
        model,
        request_id="req",
        input_ids=torch.zeros(1, dtype=torch.long),
        info={"_omni_num_computed_tokens": 4},
        duplex={
            "source_input_seq": 2,
            "kv_append_start": 4,
            "runtime_config": {"nvc_text_pad_id": 12},
        },
    )
    assert torch.equal(embeddings, torch.full((1, 4), 12.0))


def test_function_output_becomes_versioned_nvidia_channel_tokens() -> None:
    encoded: list[str] = []

    def encode(text: str, **_kwargs) -> list[int]:
        encoded.append(text)
        return [31, 32, 33]

    adapter = NemotronVoiceChatServingRuntimeAdapter(lambda *_: None)
    adapter._tokenizer = SimpleNamespace(
        encode=encode,
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
