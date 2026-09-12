# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Upload a Cosmos multiview JSON manifest's local conditioning files."""

import argparse
import copy
import json
import mimetypes
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx


def prepare_request(manifest: dict[str, Any], base_dir: Path) -> tuple[dict[str, str], list[Path]]:
    """Accept an offline manifest or a video API request containing extra_params."""
    extra = copy.deepcopy(manifest.get("extra_params", {}))
    if "multiview" not in extra:
        extra["multiview"] = copy.deepcopy(manifest["multiview"])
    extra.setdefault("wsm", manifest.get("wsm", True))
    paths = []
    for view in extra["multiview"]["views"]:
        for role in ("control", "vision"):
            if f"{role}_reference_index" in view:
                raise ValueError("The client expects local media paths, not pre-existing upload indexes.")
            fields = [field for field in (f"{role}_path", role) if view.get(field) is not None]
            if not fields:
                continue
            if len(fields) != 1:
                raise ValueError(f"Specify only one of {role} and {role}_path per camera.")
            value = view.pop(fields[0])
            if not isinstance(value, str):
                raise ValueError(f"{role} must be a local file path for HTTP uploads.")
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            if not path.is_file():
                raise FileNotFoundError(path)
            view[f"{role}_reference_index"] = len(paths)
            paths.append(path)
    data = {"prompt": str(manifest.get("prompt", "")), "extra_params": json.dumps(extra)}
    for key in (
        "model",
        "fps",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "flow_shift",
        "seed",
        "negative_prompt",
        "width",
        "height",
    ):
        if manifest.get(key) is not None:
            data[key] = str(manifest[key])
    for alias, key in (("num_steps", "num_inference_steps"), ("guidance", "guidance_scale"), ("shift", "flow_shift")):
        if key not in data and manifest.get(alias) is not None:
            data[key] = str(manifest[alias])
    return data, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--server", default="http://localhost:8091")
    parser.add_argument("--sync", action="store_true", help="Use /v1/videos/sync instead of a background job")
    parser.add_argument(
        "--num-inference-steps", type=int, help="Number of diffusion steps, overriding the manifest value"
    )
    parser.add_argument("--output", type=Path, default=Path("multiview.mp4"))
    parser.add_argument("--timeout", type=float, default=3600, help="HTTP and job polling timeout in seconds")
    args = parser.parse_args()
    if args.num_inference_steps is not None and args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    data, paths = prepare_request(json.loads(args.manifest.read_text()), args.manifest.resolve().parent)
    if args.num_inference_steps is not None:
        data["num_inference_steps"] = str(args.num_inference_steps)
    api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    server = args.server.rstrip("/")
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        with ExitStack() as stack:
            files = [
                (
                    "input_references",
                    (
                        path.name,
                        stack.enter_context(path.open("rb")),
                        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    ),
                )
                for path in paths
            ]
            response = client.post(server + "/v1/videos" + ("/sync" if args.sync else ""), data=data, files=files)
            response.raise_for_status()
        if not args.sync:
            job = response.json()
            job_url = server + "/v1/videos/" + job["id"]
            print(f"Submitted {job['id']}", flush=True)
            deadline = time.monotonic() + args.timeout
            while job["status"] not in ("completed", "failed"):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Polling timed out; the job is still available at {job_url}")
                time.sleep(1)
                response = client.get(job_url)
                # Failed jobs use an error HTTP status while still returning job metadata.
                if response.is_error:
                    try:
                        failed_job = response.json()
                    except ValueError:
                        response.raise_for_status()
                    if isinstance(failed_job, dict) and failed_job.get("status") == "failed":
                        raise RuntimeError(f"Video generation failed: {failed_job.get('error')}")
                    response.raise_for_status()
                job = response.json()
            if job["status"] == "failed":
                raise RuntimeError(f"Video generation failed: {job.get('error')}")
            response = client.get(job_url + "/content")
            response.raise_for_status()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(response.content)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
