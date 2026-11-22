# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Invalid inputs on omni chat.

Covers ``POST /v1/chat/completions``, ``POST /v1/chat/completions/batch``,
``POST /v1/completions``, ``WS /v1/video/chat/stream``, and ``WS /v1/realtime``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.omni]


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
    if case_id == "modalities_list_bad":
        body["modalities"] = [123]
    elif case_id == "logprobs_wrong_type":
        body["logprobs"] = "yes"
    elif case_id == "logprobs_top_without_enabled":
        body["logprobs"] = False
        body["top_logprobs"] = 5
    elif case_id == "speaker_unknown":
        body["speaker"] = "zz_invalid_qwen3_omni_chat_speaker_xyz"
    elif case_id == "audio_format_unsupported":
        body["modalities"] = ["text", "audio"]
        body["audio"] = {"voice": "alloy", "format": "aac"}
    elif case_id == "audio_data_url_corrupt":
        body["messages"] = dummy_messages_from_mix_data(
            audio_data_url="data:audio/invalid;base64,AAAA",
            content_text="What is the capital of China? Answer in 20 words.",
        )
        body["stream"] = True
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
            "model_mismatch",
            404,
            ("model", "does not exist", "NotFoundError"),
            id="model_mismatch",
        ),
        pytest.param(
            "modalities_list_bad",
            400,
            ("modalities", "list of strings"),
            id="modalities_list_bad_element",
        ),
        pytest.param("logprobs_wrong_type", 400, ("logprobs", "boolean"), id="logprobs_wrong_type"),
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
        pytest.param(
            "audio_format_unsupported",
            400,
            ("Invalid audio format", "aac", "Supported formats"),
            id="unsupported_audio_format",
        ),
        pytest.param(
            "audio_data_url_corrupt",
            400,
            "Invalid or corrupted audio data",
            id="invalid_audio_data_url",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_chat_completions_invalid_requests(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    case_id: str,
    err_code: int | tuple[int, ...],
    err_message: str | tuple[str, ...],
) -> None:
    cfg = _chat_completions_request_without_expectations(omni_server, case_id)
    cfg["err_code"] = err_code
    cfg["err_message"] = err_message
    online_client.send_chat_completions_http_request(cfg)[0]


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/completions
# ─────────────────────────────────────────────────────────────────────────────


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_completions_rejected_for_thinker_talker(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
) -> None:
    """Thinker-talker rejects ``POST /v1/completions`` with 400.

    The handoff segments the sequence with ChatML markers. A bare completion
    prompt does not carry them, so this route stays closed.
    """
    online_client.send_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "Hello, how are you?",
                "max_tokens": 10,
            },
            "timeout": 120,
            "err_code": 400,
            "err_message": "thinker-talker handoff",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/chat/completions/batch
# ─────────────────────────────────────────────────────────────────────────────


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "sad_opts, err_message",
    [
        pytest.param({"stream": "not-a-bool"}, None, id="stream_wrong_type"),
        pytest.param({"n": 2}, "n > 1", id="n_greater_than_one"),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_batched_completions_with_bad_values(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    sad_opts: dict[str, object],
    err_message: str | None,
) -> None:
    """Bad batched chat options (wrong ``stream`` type, ``n=2``) return 400."""
    messages = [
        [{"role": "user", "content": "Say hello."}],
        [{"role": "user", "content": "Say goodbye."}],
    ]
    request_config: dict[str, object] = {
        "json": {
            "model": omni_server.model,
            "messages": messages,
            **sad_opts,
        },
        "timeout": 120,
        "err_code": 400,
    }
    if err_message is not None:
        request_config["err_message"] = err_message
    online_client.send_batched_chat_completions_http_request(request_config)


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
    online_client: OnlineOmniClient,
    send_frames_spec: Any,
    err_message: str,
) -> None:
    if send_frames_spec is _VIDEO_CHAT_WS_SESSION_MODALITIES_SCALAR:
        send_frames = json.dumps({"type": "session.config", "model": omni_server.model, "modalities": "text"})
    else:
        send_frames = send_frames_spec
    assert isinstance(send_frames, str)
    online_client.send_video_chat_stream_ws_request(
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
@pytest.mark.parametrize(
    "send_frames_spec, ws_error_code, err_message",
    [
        pytest.param("{not-json", "invalid_event", "Invalid or unrecognized client event", id="invalid_json"),
        pytest.param(
            json.dumps({"type": "session.update"}),
            "invalid_event",
            "Invalid or unrecognized client event",
            id="session_update_missing_session",
        ),
        pytest.param(
            json.dumps({"type": "session.update", "session": []}),
            "invalid_event",
            "Invalid or unrecognized client event",
            id="session_update_invalid_session",
        ),
        pytest.param(
            json.dumps({"type": "input_audio_buffer.commit"}),
            "invalid_request_error",
            "Input audio buffer is empty",
            id="commit_without_audio",
        ),
        pytest.param(
            json.dumps({"type": "unknown.event"}),
            "invalid_event",
            "Invalid or unrecognized client event",
            id="unknown_event_type",
        ),
        pytest.param(
            json.dumps({"type": "input_audio_buffer.append", "audio": "not-valid-base64!!!"}),
            "invalid_request_error",
            "Invalid base64 audio data",
            id="invalid_audio_append",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_realtime_invalid_requests(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    send_frames_spec: Any,
    ws_error_code: str,
    err_message: str | tuple[str, ...],
) -> None:
    online_client.send_realtime_ws_request(
        {
            "send_frames": send_frames_spec,
            "timeout": 120,
            "ws_max_size": None,
            "ws_skip_types": ["session.created"],
            "ws_json_type": "error",
            "ws_error_code": ws_error_code,
            "err_message": err_message,
        }
    )
