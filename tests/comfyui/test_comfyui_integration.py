import multiprocessing
import time
import traceback
from collections.abc import Iterable
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
import torch
from comfy_api.input import AudioInput, VideoInput
from comfyui_vllm_omni.nodes import (
    VLLMOmniComprehension,
    VLLMOmniGenerateImage,
    VLLMOmniTTS,
    VLLMOmniVoiceClone,
)
from PIL import Image
from vllm import SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.outputs import OmniRequestOutput


class ServerCase(NamedTuple):
    served_model: str
    stage_list: list
    stage_configs: list[dict]
    outputs: list[OmniRequestOutput]


def _build_image(size: tuple[int, int] = (64, 64), color: str = "red") -> Image.Image:
    return Image.new("RGB", size, color=color)


def _build_text_output(text: str = "This is a test response.") -> OmniRequestOutput:
    completion_output = CompletionOutput(
        index=0,
        text=text,
        token_ids=[1, 2, 3],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    request_output = RequestOutput(
        request_id="test_req_text",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
        metrics=None,
        lora_request=None,
    )
    return OmniRequestOutput(
        request_id="test_req_text",
        finished=True,
        final_output_type="text",
        request_output=request_output,
    )


def _build_audio_chat_output(num_samples: int = 24000) -> OmniRequestOutput:
    completion_output = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    completion_output.multimodal_output = {"audio": [torch.zeros(1, num_samples)]}
    request_output = RequestOutput(
        request_id="test_req_audio_chat",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
        metrics=None,
        lora_request=None,
    )
    return OmniRequestOutput(
        request_id="test_req_audio_chat",
        finished=True,
        final_output_type="audio",
        request_output=request_output,
    )


def _build_audio_speech_output(num_samples: int = 24000) -> OmniRequestOutput:
    return OmniRequestOutput.from_diffusion(
        request_id="test_req_audio_speech",
        images=[],
        multimodal_output={"audio": torch.zeros(num_samples), "sr": 24000},
        final_output_type="audio",
    )


def _build_diffusion_image_output_for_images_endpoint() -> OmniRequestOutput:
    return OmniRequestOutput.from_diffusion(
        request_id="test_req_img_dalle",
        images=[_build_image()],
        final_output_type="image",
    )


def _build_diffusion_image_output_for_chat_endpoint() -> OmniRequestOutput:
    request_output = MagicMock()
    request_output.images = [_build_image(color="blue")]
    request_output.finished = True
    return OmniRequestOutput(
        request_id="test_req_img_chat",
        finished=True,
        final_output_type="image",
        request_output=request_output,
    )


def _build_mock_outputs(outputs: Iterable[OmniRequestOutput]):
    async def _mock_generate(*args, **kwargs):
        for output in outputs:
            yield output

    return _mock_generate


@pytest.fixture
def server_case(request) -> ServerCase:
    return request.param


@pytest.fixture
def mock_async_omni(server_case: ServerCase):
    async def _mock_preprocess_chat(self, *args, **kwargs):
        return ([{"role": "user", "content": "test"}], [{"prompt": "test prompt"}])

    # Need to mock AsyncOmni itself (not only its generate method) because
    # 1. The API layer uses its stage_list and stage_configs attributes
    # 2. Its __init__ method has slow side effects (model & config loading).
    with (
        patch("vllm_omni.entrypoints.openai.api_server.AsyncOmni") as MockAsyncOmni,
        patch(
            "vllm_omni.entrypoints.openai.serving_chat.OmniOpenAIServingChat._preprocess_chat",
            new=_mock_preprocess_chat,
        ),
    ):
        mock_instance = AsyncMock()
        mock_instance.generate = _build_mock_outputs(server_case.outputs)

        mock_instance.stage_list = server_case.stage_list
        mock_instance.stage_configs = server_case.stage_configs
        mock_instance.default_sampling_params_list = [
            SamplingParams() if stage.get("stage_type") != "diffusion" else MagicMock()
            for stage in server_case.stage_configs
        ]
        mock_instance.errored = False
        mock_instance.dead_error = RuntimeError("Mock engine error")
        mock_instance.model_config = MagicMock(max_model_len=4096, io_processor_plugin=None)
        mock_instance.io_processor = MagicMock()
        mock_instance.input_processor = MagicMock()
        mock_instance.shutdown = MagicMock()
        mock_instance.get_vllm_config = AsyncMock(return_value=None)
        mock_instance.get_supported_tasks = AsyncMock(return_value=["generate"])
        mock_instance.get_tokenizer = AsyncMock(return_value=None)

        MockAsyncOmni.return_value = mock_instance
        yield MockAsyncOmni


@pytest.fixture
def api_server(unused_tcp_port_factory, server_case: ServerCase, mock_async_omni):
    """Set up a API server in background process from command line with parametrized model name and mocked AsyncOmni."""
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cmd = OmniServeCommand()
    cmd.subparser_init(subparsers)

    port = unused_tcp_port_factory()
    args = parser.parse_args(["serve", server_case.served_model, "--omni", "--port", str(port)])

    def run_server():
        try:
            cmd.cmd(args)
        except Exception:
            traceback.print_exc()

    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    # Wait for the server to be ready by polling the health endpoint.
    wait_time = 30
    wait_poll_interval = 1
    for _ in range(wait_time // wait_poll_interval):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(wait_poll_interval)
    else:
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=5)
        pytest.fail(f"API server failed to start within {wait_time} seconds")

    yield f"http://127.0.0.1:{port}/v1"

    if server_process.is_alive():
        server_process.terminate()
    server_process.join(timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_case,model,image_input",
    [
        pytest.param(
            ServerCase(
                served_model="Tongyi-MAI/Z-Image-Turbo",
                stage_list=["diffusion"],
                stage_configs=[{"stage_type": "diffusion"}],
                outputs=[_build_diffusion_image_output_for_images_endpoint()],
            ),
            "Tongyi-MAI/Z-Image-Turbo",
            False,
            id="text-to-image-dalle-endpoint",
        ),
        pytest.param(
            ServerCase(
                served_model="ByteDance-Seed/BAGEL-7B-MoT",
                stage_list=["diffusion"],
                stage_configs=[{"stage_type": "diffusion"}],
                outputs=[_build_diffusion_image_output_for_chat_endpoint()],
            ),
            "ByteDance-Seed/BAGEL-7B-MoT",
            False,
            id="text-to-image-bagel-chat-endpoint",
        ),
        pytest.param(
            ServerCase(
                served_model="Qwen/Qwen-Image-Edit",
                stage_list=["diffusion"],
                stage_configs=[{"stage_type": "diffusion"}],
                outputs=[_build_diffusion_image_output_for_images_endpoint()],
            ),
            "Qwen/Qwen-Image-Edit",
            True,
            id="image-to-image-dalle-endpoint",
        ),
        pytest.param(
            ServerCase(
                served_model="ByteDance-Seed/BAGEL-7B-MoT",
                stage_list=["diffusion"],
                stage_configs=[{"stage_type": "diffusion"}],
                outputs=[_build_diffusion_image_output_for_chat_endpoint()],
            ),
            "ByteDance-Seed/BAGEL-7B-MoT",
            True,
            id="image-to-image-bagel-chat-endpoint",
        ),
    ],
    indirect=["server_case"],
)
async def test_image_generation_node(api_server: str, model: str, image_input: bool):
    node = VLLMOmniGenerateImage()

    kwargs = {
        "url": api_server,
        "model": model,
        "prompt": "A beautiful sunset",
        "width": 512,
        "height": 512,
    }
    if image_input:
        kwargs["image"] = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    result = await node.generate(**kwargs)

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == (1, 64, 64, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_case",
    [
        pytest.param(
            ServerCase(
                served_model="Qwen/Qwen2.5-Omni-7B",
                stage_list=[
                    MagicMock(is_comprehension=True, model_stage="llm"),
                    MagicMock(is_comprehension=False, model_stage="llm"),
                    MagicMock(is_comprehension=False, model_stage="llm"),
                ],
                stage_configs=[
                    {"stage_type": "llm"},
                    {"stage_type": "llm"},
                    {"stage_type": "llm"},
                ],
                outputs=[_build_audio_chat_output(), _build_text_output("Comprehension response")],
            ),
            id="multimodal-comprehension",
        )
    ],
    indirect=["server_case"],
)
async def test_comprehension_node(api_server: str):
    node = VLLMOmniComprehension()

    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    video = VideoInput(b"mock_video_for_test")  # type: ignore[reportAbstractUsage]
    audio: AudioInput = {"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000}

    text_response, audio_response = await node.generate(
        url=api_server,
        model="Qwen/Qwen2.5-Omni-7B",
        prompt="Describe all modalities.",
        image=image,
        audio=audio,
        video=video,
        output_text=True,
        output_audio=True,
        use_audio_in_video=True,
    )

    assert text_response == "Comprehension response"
    assert isinstance(audio_response, dict)
    assert audio_response["sample_rate"] == 24000
    assert isinstance(audio_response["waveform"], torch.Tensor)
    assert audio_response["waveform"].shape == (1, 1, 24000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_case,node_cls,call_kwargs",
    [
        pytest.param(
            ServerCase(
                served_model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                stage_list=["llm"],
                stage_configs=[{"stage_type": "llm"}],
                outputs=[_build_audio_speech_output()],
            ),
            VLLMOmniTTS,
            {
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "input": "Hello from TTS test",
                "voice": "Vivian",
                "response_format": "wav",
                "speed": 1.0,
                "model_specific_params": None,
            },
            id="tts",
        ),
        pytest.param(
            ServerCase(
                served_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                stage_list=["llm"],
                stage_configs=[{"stage_type": "llm"}],
                outputs=[_build_audio_speech_output()],
            ),
            VLLMOmniVoiceClone,
            {
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "input": "Hello from voice clone test",
                "voice": "Vivian",
                "response_format": "wav",
                "speed": 1.0,
                "ref_audio": {"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000},
                "ref_text": "Reference transcript",
                "x_vector_only_mode": False,
                "model_specific_params": None,
            },
            id="tts-voice-clone",
        ),
    ],
    indirect=["server_case"],
)
async def test_tts_nodes(api_server: str, node_cls, call_kwargs: dict):
    node = node_cls()
    result = await node.generate(url=api_server, **call_kwargs)

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["sample_rate"] == 24000
    assert isinstance(result[0]["waveform"], torch.Tensor)
    assert result[0]["waveform"].shape == (1, 1, 24000)
