# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invalid inputs on Qwen3-Omni: ``POST /v1/chat/completions``, ``WS /v1/video/chat/stream``, ``WS /v1/realtime``."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.omni]

_SKIP_ISSUE_3649 = pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/3649")


def _minimal_chat_json(omni_server: OmniServer) -> dict[str, object]:
    """Minimal valid chat body; individual tests override one offending field."""
    return {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


_QWEN3_OMNI_SERVER = [
    pytest.param(
        OmniServerParams(
            model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            stage_config_path=get_deploy_config_path("qwen3_omni_moe.yaml"),
            server_args=["--async-chunk"],
        ),
        id="qwen3_omni",
    ),
]


def _chat_completions_request_without_expectations(omni_server: OmniServer, case_id: str) -> dict[str, Any]:
    """Build ``send_chat_completions_http_request`` kwargs excluding ``err_code`` / ``err_message``."""
    if case_id == "malformed_json":
        return {"raw_body": "{not-json", "timeout": 120}
    if case_id == "missing_messages":
        return {"json": {"model": omni_server.model, "stream": False}, "timeout": 120}
    if case_id == "model_mismatch":
        return {
            "json": {
                "model": "this-model-is-not-served-on-this-instance",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            "timeout": 120,
        }
    body = _minimal_chat_json(omni_server)
    if case_id == "stream_wrong_type":
        body["stream"] = "not-a-boolean"  # type: ignore[assignment]
    elif case_id == "temperature_wrong_type":
        body["temperature"] = "hot"  # type: ignore[assignment]
    elif case_id == "max_tokens_wrong_type":
        body["max_tokens"] = "not-an-int"  # type: ignore[assignment] # type: ignore[assignment]
    elif case_id == "modalities_list_bad":
        body["modalities"] = [123]  # type: ignore[assignment]
    elif case_id == "response_format_json_schema_incomplete":
        body["response_format"] = {"type": "json_schema"}
    elif case_id == "logprobs_wrong_type":
        body["logprobs"] = "yes"  # type: ignore[assignment]
    elif case_id == "logprobs_top_without_enabled":
        body["logprobs"] = False
        body["top_logprobs"] = 5
    elif case_id == "speaker_unknown":
        body["speaker"] = "zz_invalid_qwen3_omni_chat_speaker_xyz"
    else:
        raise AssertionError(f"unknown chat completions invalid case_id {case_id!r}")
    return {"json": body, "timeout": 120}


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/chat/completions
# ─────────────────────────────────────────────────────────────────────────────


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "case_id, err_code, err_message",
    [
        pytest.param(
            "malformed_json",
            400,
            ("json_invalid", "decode error"),
            id="malformed_json",
        ),
        pytest.param("missing_messages", 400, ("messages", "missing", "Field required"), id="missing_messages"),
        pytest.param(
            "model_mismatch",
            404,
            ("model", "does not exist", "NotFoundError"),
            id="model_mismatch",
        ),
        pytest.param("stream_wrong_type", 400, ("stream", "bool_parsing", "not-a-boolean"), id="invalid_stream_type"),
        pytest.param(
            "temperature_wrong_type",
            400,
            ("temperature", "float_parsing", "a valid number"),
            id="invalid_temperature_type",
        ),
        pytest.param(
            "max_tokens_wrong_type", 400, ("max_tokens", "int_parsing", "integer"), id="invalid_max_tokens_type"
        ),
        pytest.param(
            "modalities_list_bad",
            400,
            ("modalities", "value_error", ""),
            id="modalities_list_bad_element",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "response_format_json_schema_incomplete",
            400,
            ("response_format", "value_error", "json_schema"),
            id="invalid_response_format_json_schema",
        ),
        pytest.param("logprobs_wrong_type", 400, "logprobs", id="logprobs_wrong_type", marks=_SKIP_ISSUE_3649),
        pytest.param(
            "logprobs_top_without_enabled",
            400,
            ("top_logprobs", "logprobs", "True"),
            id="logprobs_top_without_logprobs_true",
        ),
        pytest.param(
            "speaker_unknown",
            400,
            ("Invalid speaker", "Supported"),
            id="speaker_unknown_preset",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_chat_completions_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    case_id: str,
    err_code: int | tuple[int, ...],
    err_message: str | tuple[str, ...],
) -> None:
    cfg = _chat_completions_request_without_expectations(omni_server, case_id)
    cfg["err_code"] = err_code
    cfg["err_message"] = err_message
    openai_client.send_chat_completions_http_request(cfg)[0]


# ─────────────────────────────────────────────────────────────────────────────
# WS /v1/video/chat/stream
# ─────────────────────────────────────────────────────────────────────────────

# First-frame JSON with ``omni_server.model`` for ``test_video_chat_stream_invalid_requests``.
_VIDEO_CHAT_WS_SESSION_MODALITIES_SCALAR = object()


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "send_frames_spec, err_message",
    [
        pytest.param("{not-json", "Invalid JSON", id="invalid_json"),
        pytest.param(
            _VIDEO_CHAT_WS_SESSION_MODALITIES_SCALAR,
            "Invalid session config",
            id="session_modalities_scalar",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_video_chat_stream_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    send_frames_spec: Any,
    err_message: str,
) -> None:
    if send_frames_spec is _VIDEO_CHAT_WS_SESSION_MODALITIES_SCALAR:
        send_frames = json.dumps({"type": "session.config", "model": omni_server.model, "modalities": "text"})
    else:
        send_frames = send_frames_spec
    assert isinstance(send_frames, str)
    openai_client.send_video_chat_stream_ws_request(
        {
            "send_frames": send_frames,
            "timeout": 120,
            "ws_max_size": None,
            "ws_json_type": "error",
            "err_message": err_message,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# WS /v1/realtime
# ─────────────────────────────────────────────────────────────────────────────


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_realtime_rejected_when_async_chunk_enabled(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    openai_client.send_realtime_ws_request(
        {
            "timeout": 120,
            "ws_max_size": None,
            "ws_json_type": "error",
            "ws_error_code": "unsupported",
            "err_message": "async_chunk",
        }
    )
