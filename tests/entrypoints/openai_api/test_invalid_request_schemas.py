# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-schema validation through the actual API router, without a model engine.

Model/media-specific failures stay in the real-service reliability suite.
"""

import logging
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.openai.api_server import build_openai_app, make_arg_parser, router

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def client():
    # Reuse production's upstream exception conversion as well as Omni's actual
    # routes. Do not copy request-validation logic or invent a test-only mapper.
    args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    root_logger = logging.getLogger()
    original_level = root_logger.level
    handler_levels = [(handler, handler.level) for handler in root_logger.handlers]
    try:
        upstream_app = build_openai_app(args, ("generate",))
    finally:
        # SageMaker initialization can change the process-wide logging levels.
        root_logger.setLevel(original_level)
        for handler, level in handler_levels:
            handler.setLevel(level)
    app = FastAPI(exception_handlers=upstream_app.exception_handlers)
    app.include_router(router)
    app.state.api_server_count = 1
    app.state.args = args
    # No engine or serving object: accepting an invalid request is a failure.
    with TestClient(app) as test_client:
        yield test_client


def _assert_bad_request(response, expected):
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert isinstance(error["message"], str) and error["message"]
    assert error.get("type") and error.get("code") is not None
    fragments = expected if isinstance(expected, tuple) else (expected,)
    assert all(str(fragment).lower() in response.text.lower() for fragment in fragments), response.text


@pytest.mark.parametrize(
    "body_spec, err_message",
    [
        pytest.param({}, ("prompt", "Field required", "Missing"), id="missing_prompt"),
        pytest.param({"prompt": 123}, ("prompt", "string_type", "valid string"), id="prompt_wrong_type"),
        pytest.param(
            {"use_system_prompt": "bogus"},
            ("use_system_prompt", "value_error", "dynamic"),
            id="use_system_prompt_invalid",
        ),
        pytest.param(
            {"num_inference_steps": 0},
            ("num_inference_steps", "'greater_than_equal", "1"),
            id="num_inference_steps_below_min",
        ),
        pytest.param(
            {"num_inference_steps": 201},
            ("num_inference_steps", "less_than_equal", "200"),
            id="num_inference_steps_above_max",
        ),
        pytest.param(
            {"guidance_scale": -1.0}, ("guidance_scale", "greater_than_equal", "0"), id="guidance_scale_negative"
        ),
        pytest.param(
            {"guidance_scale": 21.0}, ("guidance_scale", "less_than_equal", "20"), id="guidance_scale_above_max"
        ),
        pytest.param(
            {"true_cfg_scale": -1.0}, ("true_cfg_scale", "greater_than_equal", "0"), id="true_cfg_scale_negative"
        ),
        pytest.param(
            {"true_cfg_scale": 21.0}, ("true_cfg_scale", "less_than_equal", "20"), id="true_cfg_scale_above_max"
        ),
        pytest.param({"layers": 1}, ("layers", "value_error", "2", "10"), id="layers_below_min"),
        pytest.param({"layers": 11}, ("layers", "value_error", "2", "10"), id="layers_above_max"),
        pytest.param(
            {"vae_use_slicing": "wrong_type"},
            ("vae_use_slicing", "bool_parsing", "validation error"),
            id="vae_use_slicing_wrong_type",
        ),
        pytest.param({"user": 123}, ("user", "string_type", "valid string"), id="user_wrong_type"),
        pytest.param(
            {"negative_prompt": ["noise"]},
            ("negative_prompt", "string_type", "valid string"),
            id="negative_prompt_wrong_type",
        ),
        pytest.param(
            {"generator_device": 1},
            ("generator_device", "string_type", "valid string"),
            id="generator_device_wrong_type",
        ),
    ],
)
def test_schema_images_generations_invalid_requests(client: TestClient, body_spec: Any, err_message: Any) -> None:
    body = {"model": "test", "prompt": "a red apple", **body_spec}
    if not body_spec:
        body.pop("prompt")
    response = client.post("/v1/images/generations", json=body)
    _assert_bad_request(response, err_message)


@pytest.mark.parametrize(
    "overrides, err_message",
    [
        pytest.param({"input": "   "}, ("cannot be empty", "input"), id="input_whitespace_only"),
        pytest.param({"input": 123}, ("input", "a valid string"), id="input_wrong_type"),
        pytest.param(
            {"response_format": "mpeg"},
            ("response_format", "literal_error", "Input should be"),
            id="response_format_invalid",
        ),
        pytest.param({"speed": 0.1}, ("speed", "greater_than_equal", "0.25"), id="speed_too_low"),
        pytest.param({"speed": 99.0}, ("speed", "less_than_equal", "4"), id="speed_too_high"),
        pytest.param({"stream_format": 1}, ("stream_format", "literal_error", "audio"), id="stream_format_wrong_type"),
        pytest.param(
            {"stream_format": "sse"},
            ("sse", "stream_format", "value_error", "not a supported stream_format"),
            id="stream_format_sse_blocked",
        ),
        pytest.param({"audio_length": -1.0}, ("audio_length", "greater_than", "0"), id="audio_length_negative"),
        pytest.param({"audio_length": 0}, ("audio_length", "greater_than", "0"), id="audio_length_zero"),
        pytest.param(
            {"negative_prompt": ["noise"]}, ("negative_prompt", "a valid string"), id="negative_prompt_wrong_type"
        ),
        pytest.param(
            {"guidance_scale": -1.0},
            ("guidance_scale", "greater_than_equal", "0"),
            id="guidance_scale_negative",
        ),
        pytest.param(
            {"num_inference_steps": 0},
            ("num_inference_steps", "greater_than_equal", "1"),
            id="num_inference_steps_zero",
        ),
        pytest.param(
            {"num_inference_steps": -1},
            ("num_inference_steps", "greater_than_equal", "1"),
            id="num_inference_steps_negative",
        ),
        pytest.param(
            {"num_inference_steps": 6000},
            ("num_inference_steps", "less_than_equal", "1000"),
            id="num_inference_steps_above_max",
        ),
        pytest.param(
            {"guidance_scale": 1001.0},
            ("guidance_scale", "less_than_equal", "1000"),
            id="guidance_scale_above_max",
        ),
    ],
)
def test_schema_audio_generate_invalid_field_values(client: TestClient, overrides: Any, err_message: Any) -> None:
    body = {"model": "test", "input": "Hello", "ref_audio": "https://example.com/ref.wav"}
    body.update(overrides)
    response = client.post("/v1/audio/generate", json=body)
    _assert_bad_request(response, err_message)


@pytest.mark.parametrize(
    "endpoint, json_body, err_message",
    [
        pytest.param(
            "sleep",
            {"stage_ids": [], "level": 2},
            ("stage_ids", "too_short", "at least 1"),
            id="sleep_empty_stage_ids",
        ),
        pytest.param(
            "sleep",
            {"stage_ids": "not-a-list", "level": 2},
            ("stage_ids", "list_type", "a valid list"),
            id="sleep_invalid_stage_ids_type",
        ),
        pytest.param(
            "sleep",
            {"stage_ids": [0], "level": "high"},
            ("level", "int_parsing", "a valid integer"),
            id="sleep_invalid_level_type",
        ),
        pytest.param(
            "sleep",
            {"stage_ids": [0], "level": -1},
            ("level", "greater"),
            id="sleep_negative_level",
        ),
        pytest.param("wakeup", {}, ("stage_ids", "missing", "Field required"), id="wakeup_missing_stage_ids"),
        pytest.param(
            "wakeup",
            {"stage_ids": None},
            ("stage_ids", "list_type", "a valid list"),
            id="wakeup_invalid_stage_ids_type",
        ),
        pytest.param(
            "wakeup",
            {"stage_ids": []},
            ("stage_ids", "too_short", "at least 1"),
            id="wakeup_empty_stage_ids",
        ),
    ],
)
def test_schema_omni_sleep_wakeup_invalid_requests(
    client: TestClient, endpoint: Any, json_body: Any, err_message: Any
) -> None:
    path = f"/v1/omni/{endpoint}"
    response = client.post(path, json=json_body)
    _assert_bad_request(response, err_message)


@pytest.mark.parametrize(
    "form_extra, err_message",
    [
        pytest.param({"seconds": "0"}, ("seconds", "string_pattern_mismatch", "1"), id="seconds_zero"),
        pytest.param({"size": "bad"}, ("size", "string_pattern_mismatch"), id="size_bad"),
        pytest.param(
            {"num_inference_steps": "0"},
            ("num_inference_steps", "greater_than_equal", "1"),
            id="num_inference_steps_zero",
        ),
        pytest.param(
            {"guidance_scale": "25"}, ("guidance_scale", "less_than_equal", "20"), id="guidance_scale_above_max"
        ),
    ],
)
def test_schema_post_videos_invalid_requests(client: TestClient, form_extra: Any, err_message: Any) -> None:
    response = client.post("/v1/videos", data={"prompt": "a cat", **form_extra})
    _assert_bad_request(response, err_message)


@pytest.mark.parametrize(
    "form_extra, err_message",
    [
        pytest.param({"seconds": "0"}, ("seconds", "string_pattern_mismatch", "1"), id="seconds_zero"),
        pytest.param({"size": "bad"}, ("size", "string_pattern_mismatch"), id="size_bad"),
        pytest.param(
            {"num_inference_steps": "201"},
            ("num_inference_steps", "less_than_equal", "200"),
            id="num_inference_steps_above_max",
        ),
    ],
)
def test_schema_post_videos_sync_invalid_requests(client: TestClient, form_extra: Any, err_message: Any) -> None:
    response = client.post("/v1/videos/sync", data={"prompt": "a cat", **form_extra})
    _assert_bad_request(response, err_message)


@pytest.mark.parametrize(
    "overrides, err_message",
    [
        pytest.param(
            {"instructions": 123}, ("instructions", "string_type", "valid string"), id="instructions_wrong_type"
        ),
        pytest.param(
            {"response_format": "mpeg"}, ("response_format", "literal_error", "wav"), id="response_format_invalid"
        ),
        pytest.param({"stream_format": 1}, ("stream_format", "literal_error", "audio"), id="stream_format_wrong_type"),
        pytest.param({"stream": "wrong_type"}, ("stream", "bool_parsing", "validation error"), id="stream_wrong_type"),
        pytest.param(
            {"task_type": "InvalidEnum"}, ("task_type", "literal_error", "CustomVoice"), id="task_type_invalid"
        ),
        pytest.param({"ref_text": ["x"]}, ("ref_text", "string_type", "valid string"), id="ref_text_wrong_type"),
        pytest.param(
            {"x_vector_only_mode": "wrong"},
            ("x_vector_only_mode", "valid boolean"),
            id="x_vector_only_mode_string_not_boolean",
        ),
        pytest.param(
            {"speaker_embedding": [0.01] * 1024},
            ("mutually exclusive", "speaker_embedding", "ref_audio"),
            id="speaker_embedding_with_ref_audio_mutually_exclusive",
        ),
        pytest.param({"max_new_tokens": 0}, ("max_new_tokens", "greater_than_equal"), id="max_new_tokens_below_min"),
        pytest.param(
            {"seed": -(2**63) - 1},
            ("seed", "greater_than_equal", "-9223372036854775808"),
            id="seed_below_min",
        ),
        pytest.param({"seed": 2**63}, ("seed", "less_than_equal", "9223372036854775807"), id="seed_above_max"),
        pytest.param(
            {"initial_codec_chunk_frames": -1},
            ("initial_codec_chunk_frames", "greater_than_equal", "0"),
            id="initial_codec_chunk_frames_negative",
        ),
    ],
)
def test_schema_speech_invalid_field_values(client: TestClient, overrides: Any, err_message: Any) -> None:
    body = {"model": "test", "input": "Hello", "ref_audio": "https://example.com/ref.wav"}
    body.update(overrides)
    response = client.post("/v1/audio/speech", json=body)
    _assert_bad_request(response, err_message)


@pytest.mark.parametrize(
    "case_id, err_message",
    [
        pytest.param("malformed_json", ("json_invalid", "decode error"), id="malformed_json"),
        pytest.param("missing_messages", ("messages", "missing", "Field required"), id="missing_messages"),
        pytest.param("stream_wrong_type", ("stream", "bool_parsing", "not-a-boolean"), id="invalid_stream_type"),
        pytest.param(
            "temperature_wrong_type", ("temperature", "float_parsing", "a valid number"), id="invalid_temperature_type"
        ),
        pytest.param("max_tokens_wrong_type", ("max_tokens", "int_parsing", "integer"), id="invalid_max_tokens_type"),
        pytest.param(
            "response_format_json_schema_incomplete",
            ("response_format", "BadRequestError", "json_schema"),
            id="invalid_response_format_json_schema",
        ),
    ],
)
def test_schema_chat_completions(client: TestClient, case_id: str, err_message: Any) -> None:
    body: dict[str, Any] = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    if case_id == "malformed_json":
        response = client.post(
            "/v1/chat/completions", content="{not-json", headers={"Content-Type": "application/json"}
        )
    else:
        overrides: dict[str, dict[str, Any]] = {
            "stream_wrong_type": {"stream": "not-a-boolean"},
            "temperature_wrong_type": {"temperature": "hot"},
            "max_tokens_wrong_type": {"max_tokens": "not-an-int"},
            "response_format_json_schema_incomplete": {"response_format": {"type": "json_schema"}},
        }
        if case_id == "missing_messages":
            body.pop("messages")
        else:
            body.update(overrides[case_id])
        response = client.post("/v1/chat/completions", json=body)
    _assert_bad_request(response, err_message)
