#!/usr/bin/env python3
# Local perf harness for PR #4580 parallelism matrix (NOT for commit).
# Times each autoregressive chunk of DreamZero generation for a given deploy config.
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path

os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

# Reuse the canonical example's observation builder + Omni wiring.
from export_prediction_video import WORKER_EXTENSION, _build_observations  # noqa: E402

from vllm_omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402


def _stats(secs: list[float]) -> dict:
    xs = sorted(secs)
    n = len(xs)
    return {
        "count": n,
        "mean_ms": round(1000 * statistics.fmean(xs), 1),
        "median_ms": round(1000 * statistics.median(xs), 1),
        "p90_ms": round(1000 * xs[min(n - 1, int(0.9 * n))], 1),
        "min_ms": round(1000 * xs[0], 1),
        "max_ms": round(1000 * xs[-1], 1),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GEAR-Dreams/DreamZero-DROID")
    ap.add_argument("--deploy-config", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, default=repo_root / "outputs" / "dreamzero" / "assets")
    ap.add_argument("--num-chunks", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2, help="leading chunks excluded from steady-state (idx0=prefill)")
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    label = args.label or args.deploy_config.stem
    session_id = f"bench-{uuid.uuid4().hex[:8]}"
    _, observations = _build_observations(
        args.video_dir,
        args.prompt,
        session_id,
        num_chunks=args.num_chunks,
        repeat_chunk_observations=True,
    )

    build_t0 = time.perf_counter()
    omni = Omni(
        model=args.model,
        deploy_config=str(args.deploy_config),
        enforce_eager=True,
        worker_extension_cls=WORKER_EXTENSION,
    )
    init_s = time.perf_counter() - build_t0

    per_chunk: list[float] = []
    for index, obs in enumerate(observations):
        sp = OmniDiffusionSamplingParams(
            extra_args={"reset": index == 0, "session_id": obs["session_id"], "robot_obs": obs}
        )
        t0 = time.perf_counter()
        result = omni.generate(obs["prompt"], sampling_params_list=[sp])
        dt = time.perf_counter() - t0
        if not result:
            raise RuntimeError(f"no output for chunk {index}")
        per_chunk.append(dt)
        print(f"[{label}] chunk {index:2d}  {dt * 1000:8.1f} ms", flush=True)

    steady = per_chunk[args.warmup:] or per_chunk[-1:]
    summary = {
        "label": label,
        "deploy_config": str(args.deploy_config),
        "num_chunks": args.num_chunks,
        "warmup": args.warmup,
        "init_s": round(init_s, 1),
        "chunk0_prefill_ms": round(1000 * per_chunk[0], 1),
        "total_gen_s": round(sum(per_chunk), 2),
        "steady_state": _stats(steady),
        "steady_chunks_per_s": round(len(steady) / sum(steady), 4) if sum(steady) else None,
        "all_chunk_ms": [round(1000 * x, 1) for x in per_chunk],
    }
    print("BENCH_SUMMARY " + json.dumps(summary), flush=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
