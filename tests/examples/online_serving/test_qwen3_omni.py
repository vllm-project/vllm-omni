"""
Online serving tests: Qwen3-Omni.
See examples/online_serving/qwen3_omni/README.md
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from pathlib import Path

import pytest

from tests.examples.helpers import (
    extract_content_after_keyword,
    extract_last_audio_saved_path,
    run_cmd,
    strip_audio_saved_to_lines,
    strip_trailing_audio_saved_line,
)
from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_file_to_text, cosine_similarity_text
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.example, pytest.mark.omni]

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]


stage_configs = [get_deploy_config_path("ci/qwen3_omni_moe.yaml")]


example_dir = str(Path(__file__).parent.parent.parent.parent / "examples" / "online_serving")
# Create parameter combinations for model and stage config
test_params = [
    OmniServerParams(model=model, port=8091, stage_config_path=stage_config)
    for model in models
    for stage_config in stage_configs
]
common_args = ["python", os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py")]


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_001(omni_server) -> None:
    command = common_args + [
        "--query-type",
        "use_image",
        "--model",
        omni_server.model,
    ]

    result = run_cmd(command)
    text_content_tmp = extract_content_after_keyword("Chat completion output from text:", result)
    text_content = strip_trailing_audio_saved_line(text_content_tmp)
    wav_path = extract_last_audio_saved_path(result)
    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path}")
    print(f"text content is: {text_content}")
    print(f"audio content is: {audio_content}")

    assert "cherry blossom" in text_content, "The output does not contain any of the keywords."

    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.8, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_002(omni_server) -> None:
    command = common_args + [
        "--query-type",
        "use_video",
        "--model",
        omni_server.model,
        "--prompt",
        "What are the main activities shown in this video?",
    ]
    result = run_cmd(command)

    text_content_tmp = extract_content_after_keyword("Chat completion output from text:", result)
    text_content = strip_trailing_audio_saved_line(text_content_tmp)

    # Verify text output same as audio output
    wav_path = extract_last_audio_saved_path(result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path}")
    print(f"text content is: {text_content}")
    print(f"audio content is: {audio_content}")
    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords."
    )
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.8, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_003(omni_server) -> None:
    command = ["bash", os.path.join(example_dir, "qwen3_omni/run_curl_multimodal_generation.sh"), "use_image"]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Output of request:", result)

    # Verify text output same as audio output
    print(f"text content is: {text_content}")
    assert "cherry blossom" in text_content, "The output does not contain any of the keywords."
    # TODO: Verify the E2E latency after confirmation baseline.


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_001(omni_server) -> None:
    command = common_args + [
        "--query-type",
        "use_image",
        "--model",
        omni_server.model,
        "--modalities",
        "text",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output
    print(f"text content is: {text_content}")
    assert "cherry blossom" in text_content, "The output does not contain any of the keywords."
    # TODO: Verify the E2E latency after confirmation baseline.


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_002(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_image",
        "--modalities",
        "audio",
    ]

    result = run_cmd(command)
    # Verify text output same as audio output
    wav_path = extract_last_audio_saved_path(result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path}")
    print(f"audio content is: {audio_content}")
    assert "cherry blossom" in audio_content, "The output does not contain any of the keywords."

    # TODO: Verify the E2E latency after confirmation baseline.


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_003(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_image",
        "--modalities",
        "audio,text",
    ]

    result = run_cmd(command)

    text_content_tmp = extract_content_after_keyword("Chat completion output from text:", result)
    text_content = strip_trailing_audio_saved_line(text_content_tmp)

    # Verify text output same as audio output
    wav_path = extract_last_audio_saved_path(result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path}")
    print(f"text content is: {text_content}")
    assert "cherry blossom" in audio_content, "The output does not contain any of the keywords."
    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.8, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_stream_001(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_image",
        "--stream",
    ]

    result = run_cmd(command)

    text_content_tmp = extract_content_after_keyword("content:", result)
    text_content = strip_audio_saved_to_lines(text_content_tmp)

    # Verify text output same as audio output
    wav_path = extract_last_audio_saved_path(result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path}")
    print(f"text content is: {text_content}")
    assert "cherry blossom" in audio_content, "The output does not contain any of the keywords."
    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.8, "The audio content is not same as the text"
    # TODO: Verify the E2E latency after confirmation baseline.


# TODO: test local web ui
