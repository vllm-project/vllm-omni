"""
E2E offline tests for MiniCPM-o 4.5 model with multimodal input and audio / text output.
"""

import copy
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
import torch
from vllm.sampling_params import RequestOutputKind

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path

models = ["openbmb/MiniCPM-o-4_5"]

_CI_DEPLOY = get_deploy_config_path("minicpmo_4_5_batching.yaml")


test_params = [(model, None, {"deploy_config": _CI_DEPLOY, "trust_remote_code": True}) for model in models]


def get_question(prompt_type: str = "text") -> str:
    prompts = {
        "text": "What is the capital of China? Answer in 20 words.",
        "audio": "Describe the audio briefly.",
        "image": "What color are the squares in this image?",
        "video": "Describe the video briefly.",
        "mix": "Describe what is in the image and audio.",
    }
    return prompts.get(prompt_type, prompts["text"])


@pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/5437")
@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing text, generating text output."""
    request_config = {"prompts": get_question("text"), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing audio, generating text output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    request_config = {"prompts": get_question("audio"), "audios": (audio, 16000), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing image, generating text output."""
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {"prompts": get_question("image"), "images": image, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating text output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing text, generating audio output through the talker token2wav path."""
    request_config = {"prompts": get_question("text"), "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/5437")
@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing mixed modalities (image + audio), generating audio output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {
        "prompts": get_question("mix"),
        "audios": (audio, 16000),
        "images": image,
        "modalities": ["audio"],
    }
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)


def _delta_outputs(
    omni_runner,
    *,
    prompt,
    modalities,
    audios=None,
    images=None,
    videos=None,
):
    omni_inputs = omni_runner.get_omni_inputs(
        prompts=prompt,
        audios=audios,
        images=images,
        videos=videos,
        modalities=modalities,
    )
    sampling_params_list = copy.deepcopy(omni_runner.get_default_sampling_params_list())
    for stage_id, sampling_params in enumerate(sampling_params_list):
        if hasattr(sampling_params, "output_kind"):
            # Audio streaming starts at the Talker. The Thinker must hand off
            # its complete TTS span and aligned hidden states in one output.
            sampling_params.output_kind = (
                RequestOutputKind.FINAL_ONLY if stage_id == 0 and "audio" in modalities else RequestOutputKind.DELTA
            )
    return omni_runner.omni.generate(omni_inputs, sampling_params_list, use_tqdm=False)


def _text_chunks(outputs) -> list[str]:
    chunks = []
    for stage_output in outputs:
        if stage_output.final_output_type != "text":
            continue
        text = stage_output.request_output.outputs[0].text
        if text:
            chunks.append(text)
    return chunks


def _audio_chunks(outputs) -> list[torch.Tensor]:
    chunks = []
    for stage_output in outputs:
        if stage_output.final_output_type != "audio":
            continue
        audio = (stage_output.multimodal_output or {}).get("audio")
        if audio is None:
            continue
        values = audio if isinstance(audio, list) else [audio]
        pieces = [torch.as_tensor(value).detach().float().cpu().reshape(-1) for value in values]
        pieces = [piece for piece in pieces if piece.numel()]
        if pieces:
            chunks.append(torch.cat(pieces))
    return chunks


def _assert_terminal_output(outputs, output_type: str) -> None:
    assert any(output.final_output_type == output_type and output.finished for output in outputs), (
        f"No terminal {output_type} output received"
    )


def _assert_chunked_audio(outputs) -> None:
    chunks = _audio_chunks(outputs)
    assert len(chunks) >= 2, f"Expected at least two audio chunks, got {len(chunks)}"
    waveform = torch.cat(chunks)
    assert waveform.numel() > 0, "Generated audio is empty"
    assert torch.isfinite(waveform).all(), "Generated audio contains non-finite samples"
    assert waveform.abs().max() > 0.01, "Generated audio appears silent"
    _assert_terminal_output(outputs, "audio")


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text_async_chunk_streaming(omni_runner, run_level) -> None:
    outputs = _delta_outputs(
        omni_runner,
        prompt=get_question("text"),
        modalities=["text"],
    )

    chunks = _text_chunks(outputs)
    assert len(chunks) >= 2, f"Expected at least two text chunks, got {len(chunks)}"
    _assert_terminal_output(outputs, "text")
    if run_level in {"advanced_model", "full_model"}:
        assert "beijing" in "".join(chunks).lower()


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio_async_chunk_streaming(omni_runner) -> None:
    outputs = _delta_outputs(
        omni_runner,
        prompt=get_question("text"),
        modalities=["audio"],
    )

    _assert_chunked_audio(outputs)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_text_audio_async_chunk_streaming(omni_runner) -> None:
    audio = generate_synthetic_audio(5, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    outputs = _delta_outputs(
        omni_runner,
        prompt="What is recited in the audio? What is in this image? Describe the video briefly.",
        audios=(audio, 16000),
        images=generate_synthetic_image(24, 24)["np_array"],
        videos=generate_synthetic_video(24, 24, 20)["np_array"],
        modalities=["text", "audio"],
    )

    text_chunks = _text_chunks(outputs)
    assert text_chunks, "Expected a final text output"
    _assert_terminal_output(outputs, "text")
    _assert_chunked_audio(outputs)
