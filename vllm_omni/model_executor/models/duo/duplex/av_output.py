# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One bounded JPEG/PCM packet, with a shared integer AV clock."""

import io

import numpy as np
import pybase64
from PIL import Image

from vllm_omni.protocol.duplex.av import AVChunkClock


def encode_av_chunk(payload, metadata, *, max_bytes=32 * 1024 * 1024, jpeg_quality=85):
    rgb = payload["rgb"]
    wave = payload["waveform"]
    rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    wave = wave.detach().cpu().numpy() if hasattr(wave, "detach") else np.asarray(wave)
    clock = AVChunkClock(
        int(metadata["block"]),
        int(metadata["frame_start"]),
        int(metadata["frame_stop"]),
        int(metadata["audio_start_sample"]),
        int(metadata["audio_stop_sample"]),
        int(metadata["fps"]),
        int(metadata["sample_rate"]),
    )
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 4
        or rgb.shape[0] not in (4, 17)
        or rgb.shape[-1] != 3
        or clock.frame_stop - clock.frame_start != len(rgb)
        or wave.shape != (2, clock.audio_stop_sample - clock.audio_start_sample)
        or not np.isfinite(wave).all()
        or rgb.nbytes + wave.nbytes > max_bytes
    ):
        raise ValueError("Invalid/oversize synchronized AV chunk")
    frames = []
    for i, frame in enumerate(rgb):
        out = io.BytesIO()
        Image.fromarray(frame).save(out, format="JPEG", quality=jpeg_quality)
        frames.append(dict(pts=clock.frame_start + i, data=pybase64.b64encode(out.getvalue()).decode("ascii")))
    pcm = (np.clip(wave.T, -1, 1) * 32767).round().astype("<i2").tobytes()
    audio = pybase64.b64encode(pcm).decode("ascii")
    video = dict(codec="jpeg", width=int(rgb.shape[2]), height=int(rgb.shape[1]), frames=frames, **clock.to_wire())
    if len(audio) + sum(len(f["data"]) for f in frames) > max_bytes:
        raise ValueError("Encoded AV chunk exceeds output budget")
    return dict(
        audio_data=audio,
        audio_format="pcm16",
        sample_rate_hz=clock.sample_rate_hz,
        channels=2,
        video=video,
        audio_duration_ms=1000 * (clock.audio_stop_sample - clock.audio_start_sample) / clock.sample_rate_hz,
        audio_text_mark=False,
        text_requires_complete_audio=True,
    )
