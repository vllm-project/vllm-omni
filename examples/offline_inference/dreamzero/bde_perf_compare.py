# SPDX-License-Identifier: Apache-2.0
"""Timed DreamZero rollout for BDE-vs-main comparison.

Runs one session of (prefill + N chunk) forwards, timing each `omni.generate`
(the per-forward E2E), saving the decoded latents for a precision check, and
dumping a timing JSON. Works on both the BDE branch (with BDE_KV_ENABLE=1) and a
clean main checkout (baseline) — it only uses helpers from export_prediction_video,
which exist on both.

    BDE_KV_ENABLE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/models \\
      python examples/offline_inference/dreamzero/bde_perf_compare.py \\
      --num-chunks 12 --tag bde \\
      --latents outputs/bde_parity/perf_bde.pt --timing outputs/bde_parity/perf_bde.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_prediction_video as E  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GEAR-Dreams/DreamZero-DROID")
    ap.add_argument("--deploy-config", type=Path, default=Path("vllm_omni/deploy/dreamzero.yaml"))
    ap.add_argument("--video-dir", type=Path, default=Path("/models/dreamzero-assets"))
    ap.add_argument("--prompt", default="A robot arm manipulates objects on a table.")
    ap.add_argument("--session-id", default="perf-cmp")
    ap.add_argument("--num-chunks", type=int, default=12)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--latents", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--timing", type=Path, default=None)
    args = ap.parse_args()

    _, base = E._build_observations(args.video_dir, prompt=args.prompt, session_id=args.session_id)
    observations = [base[0]] + [base[1]] * args.num_chunks
    n = len(observations)
    print(f"[perf:{args.tag}] {n} forwards (1 prefill + {args.num_chunks} chunk) "
          f"BDE_KV_ENABLE={os.environ.get('BDE_KV_ENABLE', '0')}")

    t0 = time.perf_counter()
    omni = E.Omni(
        model=args.model,
        deploy_config=str(args.deploy_config),
        enforce_eager=True,
        worker_extension_cls=E.WORKER_EXTENSION,
    )
    load_s = time.perf_counter() - t0
    print(f"[perf:{args.tag}] engine load: {load_s:.2f}s")

    outputs = []
    per_forward = []
    for index, obs in enumerate(observations):
        sp = E.OmniDiffusionSamplingParams(
            extra_args={"reset": index == 0, "session_id": obs["session_id"], "robot_obs": obs}
        )
        s = time.perf_counter()
        result = omni.generate(obs["prompt"], sampling_params_list=[sp])  # blocking -> E2E
        dt = time.perf_counter() - s
        if not result:
            raise RuntimeError(f"No output for forward {index}")
        outputs.append(result[0])
        per_forward.append(dt)
        print(f"[perf:{args.tag}] forward {index:2d} ({'prefill' if index == 0 else 'chunk'}): {dt*1000:8.1f} ms")

    prefill = per_forward[0]
    steady = per_forward[2:] if len(per_forward) > 2 else per_forward[1:]  # drop prefill + 1 warmup
    mean_steady = sum(steady) / len(steady)
    summary = {
        "tag": args.tag,
        "bde_kv_enable": os.environ.get("BDE_KV_ENABLE", "0"),
        "num_forwards": n,
        "load_s": load_s,
        "prefill_s": prefill,
        "steady_mean_s": mean_steady,
        "steady_min_s": min(steady),
        "steady_max_s": max(steady),
        "total_gen_s": sum(per_forward),
        "per_forward_s": per_forward,
    }
    print(f"[perf:{args.tag}] prefill={prefill*1000:.1f}ms  "
          f"steady_mean={mean_steady*1000:.1f}ms (n={len(steady)})  total={sum(per_forward):.2f}s")

    # Decode once (after timing) so latents + the final video are both saved.
    latents = torch.cat([E._extract_latents(o) for o in outputs], dim=2)
    if args.latents is not None:
        args.latents.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), args.latents)
        print(f"[perf:{args.tag}] SAVED_LATENTS={args.latents} shape={tuple(latents.shape)}")
    if args.video is not None:
        frames = E._decode_with_worker(omni, latents)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        E._write_mp4(args.video, frames, fps=args.fps)
        print(f"[perf:{args.tag}] SAVED_MP4={args.video} frames={frames.shape[0]}")
    if args.timing is not None:
        args.timing.parent.mkdir(parents=True, exist_ok=True)
        args.timing.write_text(json.dumps(summary, indent=2))
        print(f"[perf:{args.tag}] SAVED_TIMING={args.timing}")


if __name__ == "__main__":
    main()
