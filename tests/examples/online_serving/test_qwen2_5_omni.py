"""
Online serving tests: Qwen2.5-Omni-7B.
See examples/online_serving/qwen2_5_omni/README.md
"""

import os

from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from pathlib import Path

import pytest

from tests.conftest import OmniServerParams, convert_audio_file_to_text, cosine_similarity_text
from tests.examples.conftest import (
    extract_content_after_keyword,
    run_cmd,
    strip_trailing_audio_saved_line,
)
from tests.utils import hardware_test

pytestmark = [pytest.mark.advanced_model, pytest.mark.example]

models = ["Qwen/Qwen2.5-Omni-7B"]


stage_configs = [str(Path(__file__).parent.parent.parent / "e2e" / "stage_configs" / "qwen2_5_omni_ci.yaml")]

if current_omni_platform.is_xpu():
    stage_configs = [
        str(Path(__file__).parent.parent.parent / "e2e" / "stage_configs" / "xpu" / "qwen2_5_omni_ci.yaml")
    ]

example_dir = str(Path(__file__).parent.parent.parent.parent / "examples" / "online_serving")
# Create parameter combinations for model and stage config
test_params = [
    OmniServerParams(model=model, port=8091, stage_config_path=stage_config)
    for model in models
    for stage_config in stage_configs
]

common_args = ["python", os.path.join(example_dir, "openai_chat_completion_client_for_multimodal_generation.py")]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_001(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_mixed_modalities",
    ]

    result = run_cmd(command)

    text_content_tmp = extract_content_after_keyword("Chat completion output from text:", result)
    text_content = strip_trailing_audio_saved_line(text_content_tmp)

    # Verify text output same as audio output
    wav_path = extract_content_after_keyword("Audio saved to", result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path.strip()}")
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


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_002(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_mixed_modalities",
        "--prompt",
        "Analyze all the media content and provide a comprehensive summary.",
    ]
    result = run_cmd(command)

    text_content_tmp = extract_content_after_keyword("Chat completion output from text:", result)
    text_content = strip_trailing_audio_saved_line(text_content_tmp)

    # Verify text output same as audio output
    wav_path = extract_content_after_keyword("Audio saved to", result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path.strip()}")
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


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_send_multimodal_request_003(omni_server) -> None:
    command = [
        "bash",
        os.path.join(example_dir, "qwen2_5_omni/run_curl_multimodal_generation.sh"),
        "mixed_modalities",
    ]

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


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_001(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_mixed_modalities",
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


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_002(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_mixed_modalities",
        "--modalities",
        "audio",
    ]

    result = run_cmd(command)
    # Verify text output same as audio output
    wav_path = extract_content_after_keyword("Audio saved to", result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path.strip()}")
    print(f"audio content is: {audio_content}")
    assert all(keyword in audio_content for keyword in ["baby", "book"]), (
        "The output does not contain any of the keywords in video description."
    )
    # There is currently an issue with incorrect image descriptions.
    # assert "cherry blossom" in audio_content, "The output does not contain any of the keywords in image description."
    assert "lamb" in audio_content, "The output does not contain any of the keywords in audio description."

    # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_modality_control_003(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_mixed_modalities",
        "--modalities",
        "audio,text",
    ]

    result = run_cmd(command)

    text_content_tmp = extract_content_after_keyword("Chat completion output from text:", result)
    text_content = strip_trailing_audio_saved_line(text_content_tmp)

    # Verify text output same as audio output
    wav_path = extract_content_after_keyword("Audio saved to", result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path.strip()}")
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


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_stream_001(omni_server) -> None:
    command = common_args + [
        "--model",
        omni_server.model,
        "--query-type",
        "use_mixed_modalities",
        "--stream",
    ]

    result = run_cmd(command)

    text_content_tmp = extract_content_after_keyword("content:", result)
    text_content = strip_trailing_audio_saved_line(text_content_tmp)

    # Verify text output same as audio output
    wav_path = extract_content_after_keyword("Audio saved to", result)
    audio_content = convert_audio_file_to_text(output_path=f"./{wav_path.strip()}")
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
