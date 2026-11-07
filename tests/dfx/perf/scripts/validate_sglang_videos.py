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
    expected = workload["extra_body"]
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
        wanted = {key: expected["num_frames" if key == "frames" else key] for key in actual}
        if actual != wanted:
            raise ValueError(f"Unexpected media in {path.name}: {actual}")
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
