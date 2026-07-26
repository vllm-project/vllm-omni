#!/usr/bin/env python3
"""Generate deterministic offline image and audio smoke-test inputs."""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_path = args.output_dir / "competition_smoke.png"
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 224, 224), fill="navy")
    draw.ellipse((80, 80, 176, 176), fill="yellow")
    image.save(image_path, format="PNG")

    audio_path = args.output_dir / "competition_smoke.wav"
    sample_rate = 16000
    frames = bytearray()
    for index in range(sample_rate):
        sample = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)

    print(image_path)
    print(audio_path)


if __name__ == "__main__":
    main()
