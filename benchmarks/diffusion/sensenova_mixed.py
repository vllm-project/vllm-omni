# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure first-hit and repeat latency for mixed SenseNova-U1.5 requests.

Start the server with ``TORCH_LOGS=recompiles VLLM_LOGGING_LEVEL=DEBUG`` and redirect its output to a
file. Pass that file with ``--server-log`` to count serving-time recompiles and
decode-graph captures alongside the per-request latency measurements.
The OpenAI chat endpoint normalizes diffusion fields in
``vllm_omni/entrypoints/openai/serving_chat.py`` and
``diffusion_request_utils.py``; ``cfg_scale`` aliases ``true_cfg_scale``.
"""

import argparse
import base64
import io
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import requests
from PIL import Image

DEFAULT_CASES = ("t2i:1024x1024", "t2t", "i2t", "t2i:1536x1536")
IMAGE_SEED = 42
_RESOLUTION = re.compile(r"^(\d+)x(\d+)$")


def _parse_case(case: str) -> tuple[str, int | None, int | None]:
    kind, _, resolution = case.partition(":")
    if kind in {"t2t", "i2t"} and not resolution:
        return kind, None, None
    if kind in {"t2i", "t2i-think"}:
        match = _RESOLUTION.fullmatch(resolution)
        if match:
            width, height = map(int, match.groups())
            if width > 0 and height > 0:
                return kind, width, height
    raise ValueError(f"Invalid case {case!r}; use t2i:WxH, t2i-think:WxH, t2t or i2t")


def _image_data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), (127, 127, 127)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _request_payload(case: str, *, steps: int, cfg_scale: float, image_uri: str) -> dict:
    kind, width, height = _parse_case(case)
    text = "A red cube on a white table" if kind.startswith("t2i") else "Describe the scene briefly."
    content = [{"type": "text", "text": text}]
    if kind == "i2t":
        content.append({"type": "image_url", "image_url": {"url": image_uri}})
    payload = {
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text" if kind in {"t2t", "i2t"} else "image"],
    }
    if kind.startswith("t2i"):
        payload.update(width=width, height=height, num_inference_steps=steps, seed=IMAGE_SEED, cfg_scale=cfg_scale)
        if kind == "t2i-think":
            payload["think"] = True
    else:
        payload["max_tokens"] = 2
    return payload


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="sensenova/SenseNova-U1.5-8B-MoT")
    parser.add_argument("--model-revision", help="Revision passed to the server")
    parser.add_argument("--server-sha", help="vLLM-Omni commit used by the server")
    parser.add_argument("--hardware", help="GPU and topology used by the server")
    parser.add_argument("--warmup-config", default="none", help="Server warmup profile label or JSON")
    parser.add_argument("--case", action="append", help="Request case; repeat to set the alternating sequence")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--server-log", type=Path, help="Server log receiving TORCH_LOGS=recompiles")
    parser.add_argument("--output", type=Path, help="Write the JSON result to this file")
    args = parser.parse_args()
    if args.rounds < 1 or args.steps < 1 or args.cfg_scale <= 0:
        parser.error("--rounds, --steps and --cfg-scale must be positive")

    cases = tuple(args.case or DEFAULT_CASES)
    for case in cases:
        _parse_case(case)
    image_uri = _image_data_uri()
    log_offset = args.server_log.stat().st_size if args.server_log else None
    rows = []
    with requests.Session() as session:
        for round_index in range(args.rounds):
            for case in cases:
                payload = _request_payload(case, steps=args.steps, cfg_scale=args.cfg_scale, image_uri=image_uri)
                started = time.perf_counter()
                response = session.post(
                    f"{args.base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=args.timeout
                )
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                body = response.json()
                if not body.get("choices"):
                    raise RuntimeError(f"No choices returned for {case}: {body}")
                rows.append({"round": round_index, "case": case, "latency_s": round(elapsed, 3)})
                print(f"round={round_index} case={case} latency={elapsed:.3f}s", flush=True)

    latencies = [row["latency_s"] for row in rows]
    report = {
        "model": args.model,
        "model_revision": args.model_revision,
        "benchmark_sha": _git_sha(),
        "server_sha": args.server_sha,
        "hardware": args.hardware,
        "warmup_config": args.warmup_config,
        "cases": cases,
        "rounds": args.rounds,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "image_seed": IMAGE_SEED,
        "p50_s": statistics.median(latencies),
        "p100_s": max(latencies),
        "by_case": {
            case: {
                "p50_s": statistics.median(row["latency_s"] for row in rows if row["case"] == case),
                "p100_s": max(row["latency_s"] for row in rows if row["case"] == case),
            }
            for case in cases
        },
        "requests": rows,
    }
    if args.server_log:
        with args.server_log.open("rb") as stream:
            stream.seek(log_offset)
            server_output = stream.read().decode("utf-8", errors="replace")
        report["serving_recompiles"] = len(re.findall(r"Recompiling function", server_output))
        report["serving_decode_graph_captures"] = len(re.findall(r"Captured decode graph", server_output))

    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
