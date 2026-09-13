# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WF-07 contracts: remote generation, unchanged timing, and real MP4 audio."""

import json
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import pytest
from comfyui_vllm_omni.utils import format as media
from comfyui_vllm_omni.utils.models import _minimaxh3_params_builder

WORKFLOWS = Path(__file__).resolve().parents[4] / "apps/ComfyUI-vLLM-Omni/example_workflows"
NAME = "vLLM-Omni MiniMax H3 Video Upscale"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_workflow_keeps_remote_audio_and_fps():
    graph = json.loads((WORKFLOWS / f"{NAME}.json").read_text())
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    generated = next(n for n in nodes.values() if n["type"] == "VLLMOmniGenerateVideo")
    components = next(n for n in nodes.values() if n["type"] == "GetVideoComponents")
    composed = next(n for n in nodes.values() if n["type"] == "CreateVideo")
    upscale = next(n for n in nodes.values() if n["type"] == "ImageUpscaleWithModel")

    def source(n, input_name):
        slot = next(i for i in n["inputs"] if i["name"] == input_name)
        edge = links[slot["link"]]
        return edge[1], edge[2]

    assert source(components, "video") == (generated["id"], 0)
    assert source(upscale, "image") == (components["id"], 0)
    assert source(composed, "images") == (upscale["id"], 0)
    assert source(composed, "audio") == (components["id"], 1)
    assert source(composed, "fps") == (components["id"], 2)
    # A frame-by-frame upscale must not change the H3 time grid.
    api = json.loads((WORKFLOWS / f"{NAME}.api.json").read_text())
    inputs = api[str(generated["id"])]["inputs"]
    assert inputs["fps"] == 24
    assert (inputs["num_frames"] - 5) % 17 == 0
    assert (inputs["width"], inputs["height"]) == (1344, 768)
    assert api[str(composed["id"])]["inputs"]["audio"] == [str(components["id"]), 1]
    assert api[str(composed["id"])]["inputs"]["fps"] == [str(components["id"]), 2]


def test_h3_named_ratio_reaches_server_without_mutating_params():
    params = {"flow_shift": 6.0, "audio_flow_shift": 3.0, "aspect_ratio": "16:9", "type": "minimax_h3"}
    before = dict(params)
    result = _minimaxh3_params_builder(params, extra_params={"task": "t2va"})
    assert result["flow_shift"] == 6.0
    assert json.loads(result["extra_params"]) == {"task": "t2va", "audio_flow_shift": 3.0, "aspect_ratio": "16:9"}
    assert params == before
    legacy = _minimaxh3_params_builder({"flow_shift": 12.0, "audio_flow_shift": 3.0}, extra_params={"task": "fl2va"})
    assert json.loads(legacy["extra_params"]) == {"task": "fl2va", "audio_flow_shift": 3.0}


def make_mp4(with_audio):
    buffer = BytesIO()
    with av.open(buffer, "w", format="mp4") as output:
        video = output.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        sound = output.add_stream("aac", rate=32000) if with_audio else None
        if sound is not None:
            sound.layout = "stereo"
        for i in range(24):
            pixels = np.full((64, 64, 3), i * 8, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 24)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode(None):
            output.mux(packet)
        if sound is not None:
            t = np.arange(32000, dtype=np.float32) / 32000
            samples = np.stack([0.2 * np.sin(2 * np.pi * frequency * t) for frequency in (440, 880)])
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate, frame.pts, frame.time_base = 32000, 0, Fraction(1, 32000)
            for packet in sound.encode(frame):
                output.mux(packet)
            for packet in sound.encode(None):
                output.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize("with_audio", [True, False])
def test_mp4_response_preserves_audio_presence(monkeypatch, with_audio):
    # Only the ComfyUI wrapper is replaced; encoding, demux and decoding use real PyAV.
    monkeypatch.setattr(media.InputImpl, "VideoFromComponents", lambda components: components)
    result = media.bytes_to_video(make_mp4(with_audio))
    assert result["images"].shape == (24, 64, 64, 3)
    assert result["frame_rate"] == Fraction(24)
    if with_audio:
        audio = result["audio"]
        assert audio is not None
        assert audio["sample_rate"] == 32000
        assert audio["waveform"].shape[:2] == (1, 2)
        assert abs(audio["waveform"].shape[-1] - 32000) <= 1024
        assert audio["waveform"].square().mean().item() > 0.005
    else:
        assert result["audio"] is None
