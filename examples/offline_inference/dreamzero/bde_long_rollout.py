# SPDX-License-Identifier: Apache-2.0
"""Long DreamZero rollout in one session — to exercise BDE KV eviction.

The stock export submits 2 forwards (prefill + 1 chunk = ~12 frames), which never
reaches the model's attention window (max_attention_size). This harness submits
``--num-chunks`` chunk observations in a single session so the cumulative KV grows
past the window and the model (and BDE) must evict — letting us check that BDE's
paged eviction stays parity-exact with the model-local path under real eviction.

KV-off vs KV-on is toggled by the BDE_KV_ENABLE env var (same as the export):

    BDE_KV_ENABLE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/models \\
      python examples/offline_inference/dreamzero/bde_long_rollout.py \\
      --num-chunks 8 --output outputs/bde_parity/long_on.mp4
"""

import argparse
import os
import sys
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
    ap.add_argument("--session-id", default="long-rollout")
    ap.add_argument("--num-chunks", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("outputs/bde_parity/long.mp4"))
    ap.add_argument("--latents", type=Path, default=None,
                    help="if set, torch.save the raw decoded latents here for exact parity checks")
    args = ap.parse_args()

    # base = [prefill_obs, chunk_obs]; extend the chunk into a long session.
    _, base = E._build_observations(args.video_dir, prompt=args.prompt, session_id=args.session_id)
    observations = [base[0]] + [base[1]] * args.num_chunks
    print(f"[long-rollout] submitting {len(observations)} forwards in session {args.session_id!r}")

    omni, outputs = E._run_generation(args.model, args.deploy_config, observations)
    latents = torch.cat([E._extract_latents(o) for o in outputs], dim=2)
    if args.latents is not None:
        args.latents.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), args.latents)
        print(f"SAVED_LATENTS={args.latents}  shape={tuple(latents.shape)}")
    frames = E._decode_with_worker(omni, latents)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    E._write_mp4(args.output, frames, fps=5)
    print(f"SAVED_MP4={args.output}  frames={frames.shape[0]}")


if __name__ == "__main__":
    main()
