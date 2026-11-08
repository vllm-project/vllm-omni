# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode saved benchmark videos after timing and record their media properties."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def validate_videos(directory: Path, config_path: Path) -> dict:
    import av

    (config,) = json.loads(config_path.read_text())
    (workload,) = config["benchmark_params"]
    if "media_expectations" in config:
        expected = config["media_expectations"]
    else:
        body = workload["extra_body"]
        expected = {"width": body["width"], "height": body["height"], "frames": body["num_frames"], "fps": body["fps"]}
    count = workload["num_prompts"] + workload["num_warmups"]
    paths = sorted(directory.rglob("*.mp4"))
    if len(paths) != count:
        raise ValueError(f"Expected {count} videos including warmups, found {len(paths)}")
    videos = []
    for path in paths:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            actual = {
                "width": stream.width,
                "height": stream.height,
                "fps": float(stream.average_rate),
                "frames": sum(1 for _ in container.decode(stream)),
            }
        wanted = {key: expected[key] for key in actual}
        if actual != wanted:
            raise ValueError(f"Unexpected media in {path.name}: {actual}")
        if expected.get("require_audio"):
            import numpy as np

            with av.open(str(path)) as container:
                if not container.streams.audio:
                    raise ValueError(f"Missing audio in {path.name}")
                stream = container.streams.audio[0]
                samples = 0
                peak = 0.0
                sample_rate = stream.codec_context.sample_rate
                channels = stream.codec_context.channels
                for frame in container.decode(stream):
                    data = frame.to_ndarray().astype(float)
                    if not np.isfinite(data).all() or frame.sample_rate != sample_rate:
                        raise ValueError(f"Invalid audio in {path.name}")
                    peak = max(peak, float(np.abs(data).max()))
                    samples += frame.samples
            if samples <= 0 or sample_rate <= 0 or peak <= 0:
                raise ValueError(f"Empty or silent audio in {path.name}")
            duration = samples / sample_rate
            if (
                sample_rate != expected["audio_sample_rate"]
                or channels != expected["audio_channels"]
                or abs(duration - actual["frames"] / actual["fps"]) > expected["audio_video_tolerance_s"]
            ):
                raise ValueError(f"Unexpected audio rate, channels or duration in {path.name}")
            actual.update(
                audio_samples=samples, audio_sample_rate=sample_rate, audio_channels=channels, audio_duration_s=duration
            )
        videos.append(
            {
                "path": str(path.relative_to(directory)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                **actual,
            }
        )
    return {
        "schema_version": 1,
        "purpose": "comparison_media_validation",
        "runner_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "build_id": os.environ.get("BUILDKITE_BUILD_ID"),
        "job_id": os.environ.get("BUILDKITE_JOB_ID"),
        "test_name": config["test_name"],
        "num_prompts": workload["num_prompts"],
        "num_warmups": workload["num_warmups"],
        "status": "passed",
        "files": videos,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_videos(args.directory, args.config)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Decoded {len(result['files'])} complete videos with the requested dimensions, frames and fps")
