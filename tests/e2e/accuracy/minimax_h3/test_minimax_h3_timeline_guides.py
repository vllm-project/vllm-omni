# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end check that MiniMax H3 timeline guides condition the generated clip.

One offline T2VA request on the real FL2VA checkpoint (two GPUs, distributed
layerwise offload with DP2, a small canvas and eight steps) carries a still
guide in the middle, a clip guide at the tail and an audio guide at the start. The
generated video must reproduce each guide at its frames, and the soundtrack
must carry the guide tone. The clip is passed as a lossless file so the ffmpeg
decode path runs too. Guides are offline-only, so this uses ``Omni`` rather
than the HTTP server.
"""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner

pytestmark = [pytest.mark.benchmark, pytest.mark.diffusion, pytest.mark.full_model]

MODEL_REPO_ID = "MiniMaxAI/MiniMax-H3"
MODEL_ENV_VAR = "VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"

WIDTH = 640
HEIGHT = 352
FPS = 24
# The shortest output H3 accepts (4 s), aligned to the 17k + 5 frame grid.
DURATION_SECONDS = 4.0
NUM_FRAMES = 107
# Guides are clean conditions, so a short schedule already reproduces them.
NUM_INFERENCE_STEPS = 8
SEED = 1101
AUDIO_SAMPLE_RATE = 32000

IMAGE_GUIDE_FRAME = 36
CLIP_GUIDE_FRAMES = 22
TONE_HZ = 440.0
TONE_SECONDS = 2.0

# Mean absolute pixel error (0-255) between a guide and its generated frames.
# On 2x H200 this request measured 2.63 (still) and 3.12 (clip); the same
# request without guides measured 86.95 and 89.26.
MAX_GUIDE_MAE = 10.0
# Share of output power within 10 Hz of the tone over the guided audio span:
# 0.9989 with the audio guide and 0.0033 without it on the same setup.
MIN_TONE_POWER_FRACTION = 0.5

PROMPT = (
    "A static camera holds on a softly lit abstract backdrop of smooth color gradients "
    "that slowly drift and blend. A steady, pure sine tone hums throughout."
)


def _model_name() -> str:
    configured = os.environ.get(MODEL_ENV_VAR)
    if configured:
        return configured
    return str(Path(snapshot_download(repo_id=MODEL_REPO_ID, allow_patterns=["FL2VA/**"])) / "FL2VA")


def _gradient_frame(phase: float) -> np.ndarray:
    """A smooth, low-frequency RGB pattern the VAE reconstructs closely."""
    y = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)[None, :]
    red = 0.5 + 0.4 * np.sin(2 * np.pi * (x + phase))
    green = 0.5 + 0.4 * np.cos(2 * np.pi * (y - 0.5 * phase))
    blue = 0.5 + 0.4 * np.sin(np.pi * (x + y) + 2 * np.pi * phase)
    frame = np.stack([np.broadcast_to(channel, (HEIGHT, WIDTH)) for channel in (red, green, blue)], axis=-1)
    return np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _write_lossless_clip(frames: np.ndarray, path: Path) -> None:
    """Encode RGB frames at 24 FPS with FFV1 so decoding returns the same pixels."""
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
    )


def _video_frames(output: Any) -> np.ndarray:
    images = output.images
    first = np.asarray(images[0])
    frames = first if len(images) == 1 and first.ndim == 4 else np.stack([np.asarray(i) for i in images])
    assert frames.dtype == np.uint8 and frames.shape == (NUM_FRAMES, HEIGHT, WIDTH, 3), frames.shape
    return frames


def _audio_waveform(output: Any) -> np.ndarray:
    audio = output.multimodal_output["audio"]
    audio = audio.float().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio, dtype=np.float32)
    assert audio.ndim == 3 and audio.shape[:2] == (1, 2), audio.shape
    return audio[0]


def _mae(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual.astype(np.float32) - expected.astype(np.float32)).mean())


def _tone_power_fraction(waveform: np.ndarray, seconds: float) -> float:
    mono = waveform[:, : int(seconds * AUDIO_SAMPLE_RATE)].mean(axis=0)
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / AUDIO_SAMPLE_RATE)
    band = (freqs >= TONE_HZ - 10.0) & (freqs <= TONE_HZ + 10.0)
    return float(power[band].sum() / power[1:].sum())


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
def test_minimax_h3_timeline_guides_condition_frames_and_audio(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 timeline guide test requires CUDA.")

    image_guide = _gradient_frame(0.0)
    clip_guide = np.stack([_gradient_frame(0.5 + 0.02 * index) for index in range(CLIP_GUIDE_FRAMES)])
    clip_path = tmp_path / "clip_guide.mkv"
    _write_lossless_clip(clip_guide, clip_path)
    t = np.arange(int(TONE_SECONDS * AUDIO_SAMPLE_RATE), dtype=np.float32) / AUDIO_SAMPLE_RATE
    tone = np.repeat(0.3 * np.sin(2 * np.pi * TONE_HZ * t)[None], 2, axis=0)
    prompt = {
        "prompt": PROMPT,
        "multi_modal_data": {
            "timeline_guides": [
                {"frame_index": IMAGE_GUIDE_FRAME, "image": Image.fromarray(image_guide)},
                {"frame_index": -CLIP_GUIDE_FRAMES, "video": str(clip_path)},
                {"frame_index": 0, "audio": (torch.from_numpy(tone), AUDIO_SAMPLE_RATE)},
            ],
        },
    }

    with OmniRunner(
        _model_name(),
        trust_remote_code=True,
        # The same DLO/DP2 layout as the online H3 DLO tests: each replica
        # streams the transformer layers, so two 80 GB cards suffice.
        num_gpus=2,
        tensor_parallel_size=1,
        data_parallel_size=2,
        ulysses_degree=1,
        ring_degree=1,
        text_encoder_tp_size=1,
        vae_patch_parallel_size=1,
        enable_distributed_layerwise_offload=True,
        vae_parallel_mode="tile",
        vae_use_tiling=True,
        diffusion_compile_dynamic=False,
        stage_init_timeout=1800,
    ) as runner:
        sampling_params = copy.deepcopy(runner.get_default_sampling_params_list())
        diffusion = sampling_params[0]
        diffusion.width = WIDTH
        diffusion.height = HEIGHT
        diffusion.fps = FPS
        diffusion.num_inference_steps = NUM_INFERENCE_STEPS
        diffusion.seed = SEED
        diffusion.extra_args = {
            "task": "t2va",
            "duration": DURATION_SECONDS,
            "aspect_ratio": "16:9",
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        }
        outputs = runner.omni.generate(prompt, sampling_params)

    assert len(outputs) == 1
    frames = _video_frames(outputs[0])
    waveform = _audio_waveform(outputs[0])

    image_mae = _mae(frames[IMAGE_GUIDE_FRAME], image_guide)
    clip_mae = _mae(frames[-CLIP_GUIDE_FRAMES:], clip_guide)
    tone_fraction = _tone_power_fraction(waveform, TONE_SECONDS)
    print(f"image_mae={image_mae:.2f} clip_mae={clip_mae:.2f} tone_fraction={tone_fraction:.4f}")

    assert image_mae < MAX_GUIDE_MAE, f"frame {IMAGE_GUIDE_FRAME} ignores the image guide: MAE {image_mae:.2f}"
    assert clip_mae < MAX_GUIDE_MAE, f"last {CLIP_GUIDE_FRAMES} frames ignore the clip guide: MAE {clip_mae:.2f}"
    assert tone_fraction > MIN_TONE_POWER_FRACTION, f"audio guide tone missing: band power {tone_fraction:.4f}"
