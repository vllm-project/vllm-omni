# Boundless-World-Model (BWM)

> Action-conditioned video world model for robotic manipulation (Wan2.2-TI2V-5B)

## Summary

- Vendor: BLM-Lab
- Model: `BLM-Lab/Boundless-World-Model`
- Task: Action-conditioned video generation (robot world model / forward dynamics): history frames + normalized end-effector action trajectory in, manipulation video out
- Mode: Offline generation via the `Omni` API (request-level, one chunk per request; autoregressive rollouts loop client-side)
- Maintainer: Community

## When to use this recipe

Use this recipe to run BWM as a learned simulator for robotic manipulation:
given the first frame (or the last 9 frames of a running rollout) and a
14-dim end-effector action trajectory, BWM generates the physically
consistent resulting video. BWM ranks first among open-source models on the
WorldArena Track 1 / Track 2 Data Engine leaderboards (May 2026).

This is a concrete model integration under the world-model track
([RFC #1987](https://github.com/vllm-project/vllm-omni/issues/1987)),
request-level like Cosmos3 `forward_dynamics`.

## References

- Model weights: <https://huggingface.co/BLM-Lab/Boundless-World-Model>
- Reference implementation: <https://github.com/boundless-large-model/boundless-world-model>
- Base model: <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>
- Related example: [`examples/offline_inference/bwm/`](../../examples/offline_inference/bwm/)
- Pipeline: [`vllm_omni/diffusion/models/bwm/pipeline_bwm.py`](../../vllm_omni/diffusion/models/bwm/pipeline_bwm.py)

## Model assembly

The upstream release ships a single Wan-native-format checkpoint
(fine-tuned DiT + action encoder). Assemble a servable diffusers-layout
directory once:

```bash
python examples/offline_inference/bwm/download_bwm.py --output-dir models/BWM
```

This downloads the diffusers-format Wan2.2-TI2V-5B base (transformer
config/weights + VAE, ~11 GB), the BWM checkpoint (~9.4 GB), converts the
fine-tuned DiT weights to diffusers naming, and writes `model_index.json`
with `_class_name: BoundlessWorldModelPipeline`. No text encoder is needed:
BWM runs with the text pathway disabled (the cross-attention context is the
action embedding).

## Hardware Support

## GPU

### 1x H200 141GB (offline inference)

#### Environment

- OS: Ubuntu 22.04+
- Python: 3.12
- Driver / runtime: NVIDIA CUDA environment
- vLLM version: match the repository requirements from your current checkout
- vLLM-Omni version or commit: use the commit you are deploying from

#### Command

```bash
# Demo data (lerobot layout: mp4 + parquet actions + normalization stats)
git clone https://github.com/boundless-large-model/boundless-world-model /tmp/bwm-repo

python examples/offline_inference/bwm/bwm_world_model.py \
    --model models/BWM \
    --episode-dir /tmp/bwm-repo/demo \
    --episode 0 \
    --output bwm_rollout.mp4
```

#### Verification

The script prints one line per autoregressive window and writes
`bwm_rollout.mp4`: 143 frames at 480x640 for demo episode 0, in 3 windows.
The output resolution follows the input frame (the demo video is 480x640;
supply a 672x896 video to render at 672x896), and the frame count follows
the episode's 143 action frames (the last window's chunk is padded to 57
frames with the final action, then trimmed). Compare against the reference
implementation's output for the same episode
(`bash scripts/infer_example.sh` in the BWM repo, which renders at 672x896
and stops at 141 frames).

#### Notes

- Memory usage (1x H200, bf16, command above): per-forward allocator peak, max over the 3 windows (the runner resets peak stats before each forward, so model load is excluded), 16.6 GiB `torch.cuda.max_memory_allocated` and 18.8 GiB `torch.cuda.max_memory_reserved`; whole-device peak sampled by nvidia-smi every 200 ms over the full run, 19.5 GiB. With a 672x896 input video: 22.2 GiB allocated, 26.1 GiB reserved, 26.8 GiB per nvidia-smi. `--enforce-eager` gives the same peaks
- Release defaults: 57-frame chunks, 9 history frames, 50 denoise steps, flow shift 5.0, no CFG (`guidance_scale=1.0`)
- Actions must be normalized to [-1, 1] with the dataset statistics client-side (`stat.json`, p01/p99 bounds); the example handles this
- Known limitations: request-level serving only (interactive/session serving would build on the AR-diffusion engine, RFC #4366/#4480); single view; `eef_abs` (14-dim end-effector) action space as released
