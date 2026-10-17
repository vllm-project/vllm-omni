# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail fast on unusable MiniCPM Daily-Omni media dependencies, before model startup.

Run ``python -m vllm_omni.benchmarks.media_preflight`` in the benchmark environment.
An optional ``--video`` / ``--audio`` pair exercises downloaded media instead of
an offline, two-second codec fixture. This does not load model weights.
"""

from __future__ import annotations

import argparse
import tempfile
import wave
from pathlib import Path


def check_minicpm_media_dependencies(*, include_audio: bool) -> None:
    """Import the required backends outside the per-sample exception handler."""
    try:
        import av  # noqa: F401
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401

        if include_audio:
            import soundfile  # noqa: F401
            from vllm.multimodal.media.audio import load_audio  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MiniCPM Daily-Omni media dependencies are unavailable. Install the matching vLLM release "
            "and vllm-omni (`pip install -e .`) in the benchmark environment; PyAV (av), Pillow, "
            "NumPy and soundfile must be usable. Run `python -m vllm_omni.benchmarks.media_preflight` "
            "before starting the model."
        ) from exc


def _write_fixture(directory: Path) -> tuple[Path, Path]:
    """Write actual MPEG-4 video and PCM audio for a tiny offline decoder probe."""
    import av
    import numpy as np

    video = directory / "probe.mp4"
    audio = directory / "probe.wav"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for level in (48, 192):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), level, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        samples = np.sin(2 * np.pi * 440 * np.arange(32000) / 16000) * 16000
        output.writeframes(samples.astype("<i2").tobytes())
    return video, audio


def probe_media(video: Path, audio: Path | None) -> tuple[int, int]:
    """Exercise the same extraction and serialization functions as the benchmark."""
    from vllm_omni.benchmarks.data_modules.daily_omni_dataset import (
        DailyOmniDataset,
        _numpy_to_wav_bytes,
        _pil_to_jpeg_bytes,
    )

    frames, segments = DailyOmniDataset._extract_minicpm_frame_audio_segments(
        video, audio_path=audio, include_audio=True, max_num_frames=2
    )
    if not frames or len(frames) != len(segments) or any(len(segment) == 0 for segment in segments):
        raise RuntimeError("MiniCPM media probe produced empty or mismatched image/audio segments")
    for frame in frames:
        if not _pil_to_jpeg_bytes(frame).startswith(b"\xff\xd8"):
            raise RuntimeError("MiniCPM media probe failed JPEG serialization")
    for segment in segments:
        if not _numpy_to_wav_bytes(segment).startswith(b"RIFF"):
            raise RuntimeError("MiniCPM media probe failed WAV serialization")
    return len(frames), len(segments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--audio", type=Path, help="Separate audio; otherwise read the video's audio stream")
    args = parser.parse_args(argv)
    if args.audio and not args.video:
        parser.error("--audio requires --video")
    check_minicpm_media_dependencies(include_audio=True)
    with tempfile.TemporaryDirectory(prefix="omni-media-preflight-") as directory:
        video, audio = (args.video, args.audio) if args.video else _write_fixture(Path(directory))
        frames, segments = probe_media(video, audio)
    print(f"MiniCPM media preflight passed: {frames} frames, {segments} audio segments (JPEG/WAV).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
