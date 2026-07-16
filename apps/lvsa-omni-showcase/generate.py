#!/usr/bin/env python
"""Generate a long video through an LVSA-enabled vLLM-Omni server.

A thin client over the Omni video API — start the server first with
``serve_lvsa.sh``. The sparse attention runs server-side; this just submits a
job and saves the result.

Uses the **asynchronous** endpoint (``POST /v1/videos`` → poll
``GET /v1/videos/{id}`` → download ``GET /v1/videos/{id}/content``) rather than
``POST /v1/videos/sync``. The sync endpoint is capped by a fixed server-side
timeout (``VIDEO_SYNC_TIMEOUT_S``, 600 s) and is documented as
"for benchmark and testing scenarios" — long videos (LVSA's use case) routinely
exceed it. The async job has no such cap.

  python generate.py --frames 481 --height 480 --width 832 \
      --prompt "A golden retriever running through a sunlit forest." \
      --out dog_forest_lvsa.mp4
"""

import argparse
import os
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("OMNI_SERVER", "http://localhost:8000"))
    ap.add_argument("--prompt", default="A dog running through a sunlit forest.")
    ap.add_argument("--frames", type=int, default=161)  # Wan 2x horizon
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--out", default="lvsa_long_video.mp4")
    ap.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status polls.")
    ap.add_argument(
        "--max-wait", type=float, default=7200.0, help="Give up if the job hasn't finished after this many seconds."
    )
    args = ap.parse_args()

    base = args.server.rstrip("/")
    form = {
        "prompt": args.prompt,
        "size": f"{args.width}x{args.height}",
        "num_frames": str(args.frames),
        "fps": str(args.fps),
        "num_inference_steps": str(args.steps),
        "guidance_scale": str(args.guidance),
    }

    # ── 1. submit the async job ──────────────────────────────────────────────
    print(f"[lvsa-app] POST {base}/v1/videos  {args.width}x{args.height} {args.frames}f steps={args.steps}")
    resp = requests.post(f"{base}/v1/videos", data=form, timeout=120)
    resp.raise_for_status()
    job = resp.json()
    vid = job["id"]
    print(f"[lvsa-app] job {vid} ({job.get('status', '?')})")

    # ── 2. poll until completed / failed ─────────────────────────────────────
    deadline = time.time() + args.max_wait
    last = None
    while True:
        if time.time() > deadline:
            raise SystemExit(f"[lvsa-app] timed out after {args.max_wait:.0f}s (job {vid} still running)")
        time.sleep(args.poll_interval)
        r = requests.get(f"{base}/v1/videos/{vid}", timeout=60)
        if r.status_code != 200:
            # retrieve_video returns a non-200 JSON when the job has FAILED
            raise SystemExit(f"[lvsa-app] job failed: {r.text}")
        st = r.json()
        status = st.get("status")
        progress = st.get("progress", 0)
        if (status, progress) != last:
            print(f"[lvsa-app] {status}  {progress}%")
            last = (status, progress)
        if status == "completed":
            break
        if status == "failed":
            raise SystemExit(f"[lvsa-app] job failed: {st.get('error')}")

    # ── 3. download the rendered mp4 ─────────────────────────────────────────
    c = requests.get(f"{base}/v1/videos/{vid}/content", timeout=600)
    c.raise_for_status()
    with open(args.out, "wb") as f:
        f.write(c.content)
    print(f"[lvsa-app] saved {args.out}")


if __name__ == "__main__":
    main()
