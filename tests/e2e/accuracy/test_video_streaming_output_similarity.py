"""
Online serving smoke test for streaming video output similarity.
"""

import json
from pathlib import Path

import pytest

from tests.e2e.accuracy.helpers import assert_video_similarity_metrics, probe_video
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import DiffusionResponse, OmniServer, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "BestWishYsh/Helios-Distilled"
PROMPT = "A serene lakeside sunrise with mist over the water."
WIDTH = 640
HEIGHT = 384
FPS = 16
NUM_FRAMES = 65
NUM_INFERENCE_STEPS = 5
SEED = 42
MIN_PSNR_DB = 28.0
MIN_SSIM = 0.95

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})

HELIOS_DISTILLED_EXTRA_PARAMS = {
    "is_enable_stage2": True,
    "pyramid_num_stages": 3,
    "pyramid_num_inference_steps_list": [1, 1, 1],
    "is_amplify_first_chunk": True,
}


def _server_args(*, streaming_output: bool) -> list[str]:
    args = [
        "--stage-init-timeout",
        "600",
        "--init-timeout",
        "900",
        "--log-stats",
    ]
    if streaming_output:
        args.append("--diffusion-streaming-output")
    return args


def _request_config(model: str, *, serialize_extra_params: bool) -> dict:
    extra_params = (
        json.dumps(HELIOS_DISTILLED_EXTRA_PARAMS) if serialize_extra_params else HELIOS_DISTILLED_EXTRA_PARAMS
    )
    return {
        "model": model,
        "form_data": {
            "prompt": PROMPT,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": 1.0,
            "seed": SEED,
            "extra_params": extra_params,
        },
    }


def _client_for(server: OmniServer, *, run_level: str) -> OpenAIClientHandler:
    return OpenAIClientHandler(
        host=server.host,
        port=server.port,
        api_key="EMPTY",
        run_level=run_level,
        log_stats=server.log_stats,
    )


def _write_video_artifact(response: DiffusionResponse, output_path: Path) -> Path:
    assert response.videos, "No video artifact returned"
    output_path.write_bytes(response.videos[0])
    return output_path


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            MODEL,
            id="helios_distilled",
            marks=SINGLE_CARD_MARKS,
        )
    ],
)
def test_helios_streaming_output_matches_non_streaming(
    model: str,
    model_prefix: str,
    run_level: str,
    tmp_path: Path,
    record_property,
):
    model = model_prefix + model

    with OmniServer(model, _server_args(streaming_output=True), use_omni=True) as server:
        client = _client_for(server, run_level=run_level)
        streaming_response = client.send_streaming_video_diffusion_request(
            _request_config(server.model, serialize_extra_params=False)
        )[0]

    streaming_path = _write_video_artifact(streaming_response, tmp_path / "helios_streaming.mp4")
    record_property("helios_streaming_video", str(streaming_path))
    record_property("helios_streaming_latency_s", streaming_response.e2e_latency)

    with OmniServer(model, _server_args(streaming_output=False), use_omni=True) as server:
        client = _client_for(server, run_level=run_level)
        non_streaming_response = client.send_video_diffusion_request(
            _request_config(server.model, serialize_extra_params=True)
        )[0]

    non_streaming_path = _write_video_artifact(non_streaming_response, tmp_path / "helios_non_streaming.mp4")
    record_property("helios_non_streaming_video", str(non_streaming_path))
    record_property("helios_non_streaming_latency_s", non_streaming_response.e2e_latency)

    streaming_metadata = probe_video(streaming_path)
    non_streaming_metadata = probe_video(non_streaming_path)
    assert streaming_metadata == non_streaming_metadata, (
        f"Video metadata mismatch for helios_streaming_output:\n"
        f"streaming={streaming_metadata}\n"
        f"non_streaming={non_streaming_metadata}\n"
        f"streaming_path={streaming_path}\n"
        f"non_streaming_path={non_streaming_path}"
    )

    ssim, psnr = assert_video_similarity_metrics(
        label="helios_streaming_output",
        online_path=streaming_path,
        offline_path=non_streaming_path,
        ssim_threshold=MIN_SSIM,
        psnr_threshold=MIN_PSNR_DB,
    )
    record_property("helios_streaming_vs_non_streaming_psnr_db", psnr)
    record_property("helios_streaming_vs_non_streaming_ssim_percent", ssim * 100.0)
