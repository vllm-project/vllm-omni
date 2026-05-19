#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GR00T-N1.7 parity check against the DROID demo dataset (F10).

Replay a real LeRobot demo episode through the running vllm-omni OpenPI
server and compare the predicted action trajectory against the dataset's
ground-truth actions.  **Zero client-side dependencies** beyond pandas /
imageio / openpi-client / websockets — no checkpoint stats, no Qwen3VL
processor, no SE(3) math.  All normalize/denormalize/SE(3) composition
happens server-side per issue #3553 ("raw observations in, absolute
actions out").

Prerequisites:
    1. A running vllm-omni server with the GR00T-N1.7 deploy:
       ``examples/online_serving/gr00t/run_server.sh``
    2. ``pip install pandas pyarrow imageio[ffmpeg] openpi-client websockets``
    3. A LeRobot-format dataset directory (e.g. the DROID demo bundled
       under Isaac-GR00T's ``demo_data/droid_sample``).

Usage:
    python3 examples/online_serving/gr00t/parity_eval.py \\
        --dataset /path/to/lerobot/demo \\
        --traj-id 1 --action-horizon 8 --steps 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import websockets.sync.client

# openpi_client package name shadows this dir; pop sys.path[0] before import.
_example_dir = str(Path(__file__).resolve().parent)
_removed = False
if sys.path and sys.path[0] == _example_dir:
    sys.path.pop(0)
    _removed = True
try:
    from openpi_client import msgpack_numpy
finally:
    if _removed:
        sys.path.insert(0, _example_dir)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gr00t-parity")

DEFAULT_URL = "ws://127.0.0.1:8000/v1/realtime/robot/openpi"
DEFAULT_EMBODIMENT = "oxe_droid_relative_eef_relative_joint"


def _load_modality(dataset_path: str) -> dict:
    with open(os.path.join(dataset_path, "meta", "modality.json")) as f:
        return json.load(f)


def _load_tasks(dataset_path: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(os.path.join(dataset_path, "meta", "tasks.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            out[int(d["task_index"])] = d["task"]
    return out


def _read_video_frame(dataset_path: str, view: str, episode_id: int, frame_idx: int) -> np.ndarray:
    """Return frame ``frame_idx`` from ``observation.images.<view>``
    as HWC uint8 RGB.  DROID videos are AV1-encoded; route through
    ffmpeg via imageio so plain opencv builds work too."""
    video_path = os.path.join(
        dataset_path, "videos", "chunk-000",
        f"observation.images.{view}", f"episode_{episode_id:06d}.mp4",
    )
    return np.asarray(iio.imread(video_path, index=frame_idx, plugin="pyav"))


def _send_inference(
    ws_url: str,
    session_id: str,
    embodiment: str,
    modality: dict,
    raw_state: dict[str, list[float]],
    images: dict[str, np.ndarray],
    prompt: str,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """One OpenPI websocket roundtrip → dict[action_key, absolute ndarray]."""
    obs = {
        "session_id": session_id,
        "embodiment": embodiment,
        "modality_config": modality,
        "prompt": prompt,
        "state": raw_state,
        "images": images,
    }
    if seed is not None:
        obs["seed"] = int(seed)
    with websockets.sync.client.connect(ws_url, max_size=64 * 1024 * 1024) as conn:
        _ = msgpack_numpy.unpackb(conn.recv())  # discard server metadata
        conn.send(msgpack_numpy.packb(obs))
        actions = msgpack_numpy.unpackb(conn.recv())
    if not isinstance(actions, dict):
        raise RuntimeError(
            f"Expected dict-of-ndarrays per issue #3553; got {type(actions).__name__}"
        )
    return actions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="LeRobot-format dataset directory (e.g. the DROID "
                        "demo bundled under Isaac-GR00T's demo_data).")
    p.add_argument("--traj-id", type=int, default=1,
                   help="Episode index (matches episode_000001.parquet etc.).")
    p.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--action-horizon", type=int, default=8,
                   help="Pred-action steps to consume open-loop before re-querying.")
    p.add_argument("--steps", type=int, default=40,
                   help="Max dataset frames to replay.")
    p.add_argument("--seed", type=int, default=0,
                   help="Per-request seed for the flow-matching initial-noise "
                        "sample.  GR00T-N1.7 actions are noise-sensitive even "
                        "after 4 Euler steps, so a fixed seed is required for "
                        "reproducible MSE/MAE numbers.  Use --seed -1 to "
                        "disable pinning (production behavior).")
    args = p.parse_args()

    modality = _load_modality(args.dataset)
    tasks = _load_tasks(args.dataset)

    state_keys = list(modality["state"].keys())
    action_keys = list(modality["action"].keys())
    video_keys = list(modality["video"].keys())
    log.info(f"[F10] modality state={state_keys}, action={action_keys}, video={video_keys}")

    parquet_path = os.path.join(
        args.dataset, "data", "chunk-000", f"episode_{args.traj_id:06d}.parquet"
    )
    df = pd.read_parquet(parquet_path)
    n = min(args.steps, len(df))
    step_indices = list(range(0, n, args.action_horizon))
    log.info(f"[F10] traj {args.traj_id}: {n} frames, "
             f"{len(step_indices)} inference calls, horizon={args.action_horizon}")

    session_id = str(uuid.uuid4())
    pred_chunks: dict[str, list[np.ndarray]] = {k: [] for k in action_keys}
    gt_chunks: dict[str, list[np.ndarray]] = {k: [] for k in action_keys}

    for call_idx, frame_i in enumerate(step_indices):
        t0 = time.time()
        row = df.iloc[frame_i]

        # Raw state per modality key (no client-side normalize).
        raw_state = {
            k: [float(x) for x in np.asarray(row[f"observation.state.{k}"], dtype=np.float32).flatten()]
            for k in state_keys
        }

        # Last video frame per camera view.
        images: dict[str, np.ndarray] = {}
        for view in video_keys:
            original_key = modality["video"][view].get("original_key", f"observation.images.{view}")
            view_id = original_key.replace("observation.images.", "")
            images[view] = _read_video_frame(args.dataset, view_id, args.traj_id, frame_i)

        prompt = tasks.get(int(row["task_index"]), "")
        actions_dict = _send_inference(
            args.url, session_id, args.embodiment, modality,
            raw_state, images, prompt,
            seed=(args.seed if args.seed >= 0 else None),
        )
        dt = time.time() - t0
        max_abs = max(float(np.abs(np.asarray(v)).max()) for v in actions_dict.values())
        log.info(f"  [step {call_idx + 1}/{len(step_indices)} frame={frame_i}] "
                 f"pred max-abs={max_abs:.3f}  ({dt:.2f}s)")

        # Server returns absolute actions per key; just collect GT chunks
        # aligned with the open-loop horizon.
        for key in action_keys:
            denorm = np.asarray(actions_dict[key], dtype=np.float32)
            for j in range(args.action_horizon):
                gt_frame = frame_i + j
                if gt_frame >= n:
                    break
                gt_row = df.iloc[gt_frame]
                gt_val = np.asarray(gt_row[f"action.{key}"], dtype=np.float32).flatten()
                pred_chunks[key].append(denorm[j])
                gt_chunks[key].append(gt_val)

    log.info("\n" + "=" * 80)
    log.info("F10: PHYSICAL-SPACE PARITY (vllm-omni GR00T-N1.7 vs DROID ground truth)")

    total_mse_num = 0.0
    total_mae_sum = 0.0
    total_den = 0
    for key in action_keys:
        pred = np.stack(pred_chunks[key])
        gt = np.stack(gt_chunks[key])
        m = min(pred.shape[0], gt.shape[0])
        pred, gt = pred[:m], gt[:m]
        if gt.ndim == 1:
            gt = gt[:, None]
        if pred.ndim == 1:
            pred = pred[:, None]
        mse = float(np.mean((pred - gt) ** 2))
        mae = float(np.mean(np.abs(pred - gt)))
        max_abs_delta = float(np.max(np.abs(pred - gt)))
        log.info(f"  {key:20s}  pairs={m:4d}  MSE={mse:.6f}  MAE={mae:.6f}  "
                 f"max-abs-delta={max_abs_delta:.4f}")
        total_mse_num += mse * pred.size
        total_mae_sum += mae * pred.size
        total_den += pred.size

    log.info(f"  {'OVERALL':20s}  MSE={total_mse_num / max(total_den, 1):.6f}  "
             f"MAE={total_mae_sum / max(total_den, 1):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
