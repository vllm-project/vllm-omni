#!/usr/bin/env python3
"""Capture a controlled MiniCPM-o request with the online NPU profiler."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .client import build_payload, run_stream_request


def service_root(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized[:-3] if normalized.endswith("/v1") else normalized


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {
            "sha": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--short")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "branch": None, "dirty": None}


async def set_profile(
    client: httpx.AsyncClient,
    *,
    root_url: str,
    start: bool,
    stages: list[int],
) -> dict[str, Any]:
    action = "start_profile" if start else "stop_profile"
    response = await client.post(f"{root_url}/{action}", json={"stages": stages})
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"{action} returned {result!r}")
    return result


async def run_one(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    payload: dict[str, Any],
    name: str,
    input_modality: str,
    with_audio: bool,
    output_wav: Path | None,
) -> dict[str, Any]:
    return await run_stream_request(
        client,
        endpoint=endpoint,
        payload=payload,
        request_name=name,
        input_modality=input_modality,
        with_audio=with_audio,
        output_wav=output_wav,
    )


async def capture(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_url = service_root(args.base_url)
    endpoint = f"{root_url}/v1/chat/completions"
    with_audio = args.output_mode == "text_audio"
    payload = build_payload(
        model=args.model,
        prompt=args.prompt,
        input_modality=args.input_modality,
        media=args.media,
        with_audio=with_audio,
        seed=args.seed,
        thinker_max_tokens=args.thinker_max_tokens,
        talker_max_tokens=args.talker_max_tokens,
    )
    timeout = httpx.Timeout(args.timeout)
    headers = {"Authorization": "Bearer EMPTY"}
    warmups: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    profile_started = False
    stop_result: dict[str, Any] | None = None
    profile_error: str | None = None

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        health = await client.get(f"{root_url}/health")
        health.raise_for_status()
        for index in range(args.warmups):
            record = await run_one(
                client,
                endpoint=endpoint,
                payload=payload,
                name=f"warmup-{index + 1}",
                input_modality=args.input_modality,
                with_audio=with_audio,
                output_wav=None,
            )
            warmups.append(record)
            if not record["success"]:
                profile_error = f"warmup {index + 1} failed"
                break

        if profile_error is None:
            try:
                await set_profile(client, root_url=root_url, start=True, stages=args.stages)
                profile_started = True
                for index in range(args.requests):
                    record = await run_one(
                        client,
                        endpoint=endpoint,
                        payload=payload,
                        name=f"profile-{index + 1}",
                        input_modality=args.input_modality,
                        with_audio=with_audio,
                        output_wav=(output_dir / f"profile-{index + 1}.wav") if with_audio else None,
                    )
                    records.append(record)
                    if not record["success"]:
                        profile_error = f"profile request {index + 1} failed"
                        break
            except Exception as exc:
                profile_error = f"{type(exc).__name__}: {exc}"
            finally:
                if profile_started:
                    try:
                        stop_result = await set_profile(
                            client,
                            root_url=root_url,
                            start=False,
                            stages=args.stages,
                        )
                    except Exception as exc:
                        stop_error = f"{type(exc).__name__}: {exc}"
                        profile_error = f"{profile_error}; stop failed: {stop_error}" if profile_error else stop_error

    root = Path(__file__).resolve().parents[3]
    result = {
        "schema_version": 1,
        "metric_scope": "profiler_diagnostic_not_score",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(root),
        "command": shlex.join([sys.executable, *sys.argv]),
        "base_url": args.base_url,
        "profile_stages": args.stages,
        "input_modality": args.input_modality,
        "output_mode": args.output_mode,
        "workload": {
            "model": args.model,
            "prompt": args.prompt,
            "media": args.media,
            "seed": args.seed,
            "thinker_max_tokens": args.thinker_max_tokens,
            "talker_max_tokens": args.talker_max_tokens,
            "warmups": args.warmups,
            "requests": args.requests,
        },
        "warmups": warmups,
        "records": records,
        "profile_stop": stop_result,
        "error": profile_error,
        "passed": profile_error is None and len(records) == args.requests,
    }
    path = output_dir / "profile_capture.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0 if result["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--input-modality", choices=("text", "image", "audio", "video"), default="text")
    parser.add_argument("--media")
    parser.add_argument("--output-mode", choices=("text", "text_audio"), default="text_audio")
    parser.add_argument("--prompt", default="Say one short sentence about efficient multimodal inference.")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinker-max-tokens", type=int, default=128)
    parser.add_argument("--talker-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.input_modality != "text" and not args.media:
        parser.error("--media is required for non-text input")
    if args.warmups < 0 or args.requests < 1:
        parser.error("--warmups must be >= 0 and --requests must be >= 1")
    raise SystemExit(asyncio.run(capture(args)))


if __name__ == "__main__":
    main()
