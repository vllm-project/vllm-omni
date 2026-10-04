# Boundless-World-Model (BWM)

Action-conditioned video world model for robotic manipulation, built on
Wan2.2-TI2V-5B. Given history frames and a normalized 14-dim end-effector
action trajectory, BWM generates the resulting manipulation video.

See the recipe for full context: [`recipes/BLM/Boundless-World-Model.md`](../../../recipes/BLM/Boundless-World-Model.md).

## 1. Assemble the model directory (once)

```bash
python download_bwm.py --output-dir models/BWM
```

Downloads the diffusers-format Wan2.2-TI2V-5B base and the BWM checkpoint,
converts the fine-tuned DiT weights to diffusers naming, splits out the
action encoder, and writes `model_index.json`.

## 2. Get demo data (once)

```bash
git clone https://github.com/boundless-large-model/boundless-world-model /tmp/bwm-repo
```

The `demo/` folder ships three RoboTwin episodes in lerobot layout
(mp4 + parquet actions + `stat.json` normalization statistics).

## 3. Autoregressive rollout

```bash
python bwm_world_model.py \
    --model models/BWM \
    --episode-dir /tmp/bwm-repo/demo \
    --episode 0 \
    --output bwm_rollout.mp4
```

The script mirrors the reference rollout: each window conditions on 9
history frames (first frame + 8 most recent) and the next 48 future
actions, generating 57-frame chunks until the episode's action trajectory
is exhausted.

## Request contract

The pipeline is request-level (like Cosmos3 `forward_dynamics`):

```python
omni.generate(
    {
        "prompt": "",  # unused: BWM runs with the text pathway disabled
        "multi_modal_data": {
            "video": history_frames,   # (T, H, W, C) uint8, T = 9
            "action": action_window,   # (57, 14) float in [-1, 1]
        },
    },
    OmniDiffusionSamplingParams(
        height=672, width=896, num_frames=57,
        num_inference_steps=50, guidance_scale=1.0, seed=42,
    ),
)
```

Actions are one-per-pixel-frame, `1 + 4 * (latent_frames - 1)` per chunk,
normalized to [-1, 1] with the dataset's p01/p99 bounds client-side.
If `num_frames` is omitted, the chunk length is derived from the action
trajectory length. History frames are resized to `height`/`width` when
they differ. The pipeline is single-device (no CFG by design; parallel
configs are rejected at startup).
