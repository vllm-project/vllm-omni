"""
Example online tests for Qwen2.5-Omni-7B model.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import OmniServer, convert_audio_file_to_text, cosine_similarity_text

models = ["Qwen/Qwen2.5-Omni-7B"]


stage_configs = [str(Path(__file__).parent.parent.parent / "e2e" / "stage_configs" / "qwen2_5_omni_ci.yaml")]

example_dir = str(Path(__file__).parent.parent.parent.parent / "examples" / "online_serving" / "qwen2_5_omni")
# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses module scope so the server starts only once per test module.
    Multi-stage initialization can take 10-20+ minutes.
    """
    model, stage_config_path = request.param

    print(f"Starting OmniServer with model: {model}")
    print("This may take 10-20+ minutes for initialization...")

    with OmniServer(
        model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "120"], port=8091
    ) as server:
        print("OmniServer started successfully")
        yield server
        print("OmniServer stopped")


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


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_001(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "mixed_modalities",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    print(f"audio content is: {audio_content}")

    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in text_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in text_content, "The output does not contain any of the keywords in audio description."

    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_002(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "mixed_modalities",
        "--prompt",
        "Analyze all the media content and provide a comprehensive summary.",
    ]
    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in text_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in text_content, "The output does not contain any of the keywords in audio description."

    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_003(omni_server) -> None:
    command = ["bash", os.path.join(example_dir, "run_curl_multimodal_generation.sh"), "mixed_modalities"]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Output of request:", result)

    # Verify text output same as audio output
    print(f"text content is: {text_content}")
    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in text_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in text_content, "The output does not contain any of the keywords in audio description."

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_001(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "mixed_modalities",
        "--modalities",
        "text",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output
    print(f"text content is: {text_content}")
    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in text_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in text_content, "The output does not contain any of the keywords in audio description."

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_002(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "mixed_modalities",
        "--modalities",
        "audio",
    ]

    run_cmd(command)
    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"audio content is: {audio_content}")
    assert all(keyword in audio_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in audio_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in audio_content, "The output does not contain any of the keywords in audio description."

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_003(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "mixed_modalities",
        "--modalities",
        "audio,text",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("Chat completion output from text:", result)

    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in text_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in text_content, "The output does not contain any of the keywords in audio description."

    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.skip(reason="There is a known issue with stream error.")
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_stream_001(omni_server) -> None:
    command = [
        "python",
        os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py"),
        "--query-type",
        "mixed_modalities",
        "--stream",
    ]

    result = run_cmd(command)

    text_content = extract_content_after_keyword("content:", result)

    # Verify text output same as audio output
    audio_content = convert_audio_file_to_text(output_path="./audio_0.wav")
    print(f"text content is: {text_content}")
    assert all(keyword in text_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in text_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in text_content, "The output does not contain any of the keywords in audio description."

    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"
    # TODO: Verify the E2E latency after confirmation baseline.


# TODO: test local web ui
