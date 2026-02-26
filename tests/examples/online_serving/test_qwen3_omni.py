"""
Example Online tests for Qwen3-Omni model.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import convert_audio_file_to_text, cosine_similarity_text
from tests.utils import hardware_test

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]


stage_configs = [str(Path(__file__).parent.parent.parent / "e2e" / "stage_configs" / "qwen3_omni_ci.yaml")]

example_dir = str(Path(__file__).parent.parent.parent.parent / "examples" / "online_serving" / "qwen3_omni")
# Create parameter combinations for model and stage config
test_params = [(8091, model, stage_config) for model in models for stage_config in stage_configs]


def run_cmd(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command)

    all_output = result.stdout
    print(f"All output:\n{all_output}")
    return all_output


def extract_content_after_keyword(keywords, text):
    matches = re.findall(rf"{keywords}\s*(.+)", text, re.DOTALL)

    if not matches:
        raise AssertionError(f"Keywords {keywords} not found in provided text output")
    return matches[0]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_001(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "use_image",
    ]

    result = run_cmd(command)
    text_content = extract_content_after_keyword("Chat completion output from text:", result)
    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    print(f"audio content is: {audio_content}")

    assert "cherry blossom" in text_content, "The output does not contain any of the keywords."

    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_002(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "use_video",
        "--prompt",
        "What are the main activities shown in this video?",
    ]
    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    print(f"audio content is: {audio_content}")
    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords."
    )
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_003(omni_server) -> None:
    command = ["bash", os.path.join(example_dir, "run_curl_multimodal_generation.sh"), "use_image"]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Output of request:", result)

    # Verify text output same as audio output
    print(f"text content is: {text_content}")
    assert "cherry blossom" in text_content, "The output does not contain any of the keywords."
    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_001(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "use_image",
        "--modalities",
        "text",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output
    print(f"text content is: {text_content}")
    assert "cherry blossom" in text_content, "The output does not contain any of the keywords."
    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_002(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "use_image",
        "--modalities",
        "audio",
    ]

    run_cmd(command)
    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"audio content is: {audio_content}")
    assert "cherry blossom" in audio_content, "The output does not contain any of the keywords."

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_003(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "use_image",
        "--modalities",
        "audio,text",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    assert "cherry blossom" in audio_content, "The output does not contain any of the keywords."
    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_stream_001(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "use_image",
        "--stream",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("content:", result)

    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    assert "cherry blossom" in audio_content, "The output does not contain any of the keywords."
    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"
    # TODO: Verify the E2E latency after confirmation baseline.


# TODO: test local web ui
