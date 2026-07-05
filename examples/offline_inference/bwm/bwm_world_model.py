# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Boundless-World-Model (BWM) offline example: action-conditioned rollout.

Given a robot manipulation episode (lerobot layout: mp4 + parquet actions,
as shipped in the BWM repo's ``demo/`` folder), autoregressively generates
the manipulation video from the first frame and the action trajectory,
mirroring the reference ``scripts/infer.py`` rollout: each window feeds 9
history frames (generated so far) plus the next 48 future actions.

Usage:
    # 1. Assemble the model directory (once):
    python examples/offline_inference/bwm/download_bwm.py --output-dir models/BWM

    # 2. Get demo data (once):
    git clone https://github.com/boundless-large-model/boundless-world-model /tmp/bwm-repo

    # 3. Roll out:
    python examples/offline_inference/bwm/bwm_world_model.py \
        --model models/BWM --episode-dir /tmp/bwm-repo/demo \
        --episode 0 --output bwm_rollout.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# Column layout of lerobot "observation.state" (26 dims) and the 14-dim
# end-effector subset ("eef_abs") the released BWM checkpoint is trained on.
# Mirrors LoadCobotAction in the reference repo.
EEF_INDICES = [7, 8, 9, 10, 11, 12, 6, 20, 21, 22, 23, 24, 25, 19]

NUM_FRAMES = 57  # chunk length (pixel frames), release default
NUM_HISTORY_FRAMES = 9  # history frames carried between windows


def load_episode(episode_dir: Path, episode_index: int):
    meta = None
    with open(episode_dir / "demo.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            if entry["episode_index"] == episode_index:
                meta = entry
                break
    if meta is None:
        raise ValueError(f"Episode {episode_index} not found in {episode_dir}/demo.jsonl")

    reader = imageio.get_reader(str(episode_dir / meta["video"]))
    first_frame = reader.get_data(meta["start_frame"])
    reader.close()

    import pyarrow.parquet as pq

    table = pq.read_table(str(episode_dir / meta["action"]), columns=["observation.state"])
    state = np.asarray(table.to_pydict()["observation.state"], dtype=np.float32)
    start, end = int(meta["start_frame"]), int(meta["end_frame"])
    state = state[start : end + 1]
    if state.shape[1] == 26:
        state = state[:, EEF_INDICES]

    stat = json.loads((episode_dir / "stat.json").read_text())["state_pose"]
    lo = np.asarray(stat.get("p01", stat["min"]), dtype=np.float32)
    hi = np.asarray(stat.get("p99", stat["max"]), dtype=np.float32)
    action = np.clip(2 * (state - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)
    return first_frame, action, meta


def history_indices(num_generated: int, history: int) -> list[int]:
    """Reference `_build_autoregressive_history_indices`: first frame plus
    the most recent frames, padded with frame 0 early in the rollout."""
    if num_generated < history:
        return [0] * (history - num_generated) + list(range(num_generated))
    if history == 1:
        return [num_generated - 1]
    return [0] + list(range(num_generated - (history - 1), num_generated))


def extract_video(outputs) -> np.ndarray:
    """Pull the video array out of Omni generate() results.

    The post-process function returns ``{"video": <np array>}``; depending on
    the entrypoint version it surfaces via ``images`` or nested request
    outputs.
    """
    out = outputs[0]
    if getattr(out, "is_pipeline_output", False) and getattr(out, "request_output", None) is not None:
        out = out.request_output
    candidates = getattr(out, "images", None) or []
    for item in candidates:
        if isinstance(item, dict) and "video" in item:
            return np.asarray(item["video"])
        if isinstance(item, np.ndarray):
            return item
    if isinstance(out, dict) and "video" in out:
        return np.asarray(out["video"])
    raise ValueError(f"Could not find video frames in output (type {type(out)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Assembled BWM directory (see download_bwm.py)")
    parser.add_argument("--episode-dir", type=Path, required=True, help="BWM demo folder (demo.jsonl layout)")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", default="bwm_rollout.mp4")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-windows", type=int, default=0, help="0 = full episode")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    first_frame, action, meta = load_episode(args.episode_dir, args.episode)
    total_frames = action.shape[0]
    height, width = first_frame.shape[:2]
    print(f"Episode {args.episode}: {total_frames} action frames, {height}x{width}, task={meta['task']}")

    omni = Omni(
        model=args.model,
        model_class_name="BoundlessWorldModelPipeline",
        enforce_eager=args.enforce_eager,
    )

    generated: list[np.ndarray] = [first_frame]
    future_per_window = NUM_FRAMES - NUM_HISTORY_FRAMES
    window = 0
    while len(generated) < total_frames:
        if args.max_windows and window >= args.max_windows:
            break
        hist_idx = history_indices(len(generated), NUM_HISTORY_FRAMES)
        future_start = len(generated)
        future_count = min(future_per_window, total_frames - future_start)
        if future_count < 4:
            break  # remainder too short for a latent frame
        # Window action condition: actions at the history frames + future
        # actions, padded with the last action up to the chunk length.
        cond = np.concatenate([action[hist_idx], action[future_start : future_start + future_count]], axis=0)
        # Chunk length: history + future, rounded UP to the 4n+1 frame grid
        # (rounding down would produce zero new frames on short tail windows).
        num_frames = 1 + ((cond.shape[0] - 1 + 3) // 4) * 4
        if cond.shape[0] < num_frames:
            cond = np.concatenate([cond, np.repeat(cond[-1:], num_frames - cond.shape[0], axis=0)], axis=0)

        history_frames = np.stack([generated[i] for i in hist_idx], axis=0)
        print(
            f"[window {window}] history={hist_idx[0]}..{hist_idx[-1]} "
            f"future=[{future_start}, {future_start + future_count}) chunk={num_frames}"
        )

        outputs = omni.generate(
            {
                "prompt": meta.get("prompt", ""),
                "multi_modal_data": {"video": history_frames, "action": cond},
            },
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1.0,
                seed=args.seed + window,
                output_type="np",
            ),
        )
        video = extract_video(outputs)
        frames = (np.asarray(video[0] if video.ndim == 5 else video) * 255).clip(0, 255).astype(np.uint8)
        # Drop the history prefix; keep newly generated future frames.
        new_frames = frames[NUM_HISTORY_FRAMES : NUM_HISTORY_FRAMES + future_count]
        generated.extend(list(new_frames))
        window += 1

    writer = imageio.get_writer(args.output, fps=args.fps, quality=5)
    for frame in generated:
        writer.append_data(frame)
    writer.close()
    print(f"Wrote {len(generated)} frames -> {args.output}")


if __name__ == "__main__":
    main()
