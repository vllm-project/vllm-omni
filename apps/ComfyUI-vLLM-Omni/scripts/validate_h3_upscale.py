#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run WF-07 through real ComfyUI HTTP APIs and verify both saved MP4s."""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def media_info(path):
    result = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    return result["streams"]


def waveform(path, rate):
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    return np.frombuffer(raw, dtype="<f4")


def audio_alignment(a, b, rate):
    count = min(len(a), len(b))
    a, b = a[:count].astype(np.float64), b[:count].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    values = correlate(b, a, method="fft")
    lags = correlation_lags(len(b), len(a))
    allowed = np.abs(lags) <= int(rate * 0.2)
    lag = int(lags[allowed][np.argmax(values[allowed])])
    if lag > 0:
        a, b = a[:-lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[:lag]
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"lag_ms": lag * 1000 / rate, "correlation": float(np.dot(a, b) / norm) if norm else 0.0}


def verify(source, output):
    before, after = media_info(source), media_info(output)
    sv = next(s for s in before if s["codec_type"] == "video")
    dv = next(s for s in after if s["codec_type"] == "video")
    sa = [s for s in before if s["codec_type"] == "audio"]
    da = [s for s in after if s["codec_type"] == "audio"]
    checks = {
        "width_doubled": dv["width"] == 2 * sv["width"],
        "height_doubled": dv["height"] == 2 * sv["height"],
        "same_frame_count": dv["nb_frames"] == sv["nb_frames"],
        "h3_frame_constraint": (int(dv["nb_frames"]) - 5) % 17 == 0,
        "24_fps_preserved": Fraction(dv["r_frame_rate"]) == Fraction(sv["r_frame_rate"]) == 24,
        "video_duration_preserved": abs(float(dv["duration"]) - float(sv["duration"])) < 1e-5,
        "generated_audio_present": len(sa) == len(da) == 1,
    }
    alignment = {}
    if sa and da:
        rate = int(sa[0]["sample_rate"])
        checks["sample_rate_preserved"] = sa[0]["sample_rate"] == da[0]["sample_rate"]
        checks["channels_preserved"] = sa[0]["channels"] == da[0]["channels"]
        checks["audio_duration_within_one_frame"] = abs(float(sa[0]["duration"]) - float(da[0]["duration"])) <= 1 / 24
        a, b = waveform(source, rate), waveform(output, rate)
        count = min(len(a), len(b))
        third = count // 3
        alignment = {
            "all": audio_alignment(a, b, rate),
            "early": audio_alignment(a[:third], b[:third], rate),
            # Match time windows rather than codec-dependent padded tails.
            "late": audio_alignment(a[count - third : count], b[count - third : count], rate),
        }
        checks["audio_content_preserved"] = alignment["all"]["correlation"] > 0.98
        checks["audio_offset_within_one_frame"] = abs(alignment["all"]["lag_ms"]) <= 1000 / 24
        checks["audio_drift_within_one_frame"] = (
            abs(alignment["early"]["lag_ms"] - alignment["late"]["lag_ms"]) <= 1000 / 24
        )
    return {
        "source_streams": before,
        "output_streams": after,
        "checks": checks,
        "audio_alignment": alignment,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--server-url", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--lora-path", help="Path as seen by the remote vLLM-Omni server")
    parser.add_argument("--output-dir", type=Path, default=Path("wf07-evidence"))
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).resolve().parents[1] / "example_workflows/vLLM-Omni MiniMax H3 Video Upscale.api.json"
    graph = json.loads(template.read_text())
    graph["1"]["inputs"].update(url=args.server_url, model=args.model)
    if args.lora_path:
        graph["9"]["inputs"]["local_path"] = args.lora_path
    # These IDs are the two SaveVideo outputs in the shipped WF-07 template.
    output_nodes = {"generated": "10", "upscaled": "6"}
    payload = {"prompt": graph, "client_id": str(uuid.uuid4())}
    (args.output_dir / "submitted_prompt.json").write_text(json.dumps(payload, indent=2))
    started = time.monotonic()
    submitted = request(args.comfy_url.rstrip("/") + "/prompt", payload)
    (args.output_dir / "submission.json").write_text(json.dumps(submitted, indent=2))
    if submitted.get("node_errors"):
        raise RuntimeError(submitted["node_errors"])
    prompt_id = submitted["prompt_id"]
    print(f"Queued real H3 workflow: {prompt_id}", flush=True)
    while time.monotonic() - started < args.timeout:
        history = request(f"{args.comfy_url.rstrip('/')}/history/{prompt_id}").get(prompt_id)
        if history:
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
            if history["status"]["status_str"] != "success":
                raise RuntimeError(history["status"])
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"ComfyUI prompt {prompt_id} has not finished")
    paths = {}
    for label, node_id in output_nodes.items():
        item = history["outputs"][node_id]["images"][0]
        query = urllib.parse.urlencode({k: item.get(k, "") for k in ("filename", "subfolder", "type")})
        path = args.output_dir / f"{label}.mp4"
        with urllib.request.urlopen(f"{args.comfy_url.rstrip('/')}/view?{query}", timeout=120) as response:
            path.write_bytes(response.read())
        paths[label] = path
    report = verify(paths["generated"], paths["upscaled"])
    requested = graph["1"]["inputs"]
    generated_video = next(stream for stream in report["source_streams"] if stream["codec_type"] == "video")
    report["checks"].update(
        requested_width=generated_video["width"] == requested["width"],
        requested_height=generated_video["height"] == requested["height"],
        requested_frame_count=int(generated_video["nb_frames"]) == requested["num_frames"],
    )
    report["passed"] = all(report["checks"].values())
    report.update(
        prompt_id=prompt_id,
        elapsed_seconds=round(time.monotonic() - started, 3),
        source="Real remote vLLM-Omni H3 output, saved before frame upscale",
        file_sha256={name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()},
    )
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
