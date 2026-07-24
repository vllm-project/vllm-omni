# DreamX-World-5B-Cam

> Image + caption + camera/action-controlled video generation (Wan2.2 TI2V-5B + PRoPE)

## Summary

- Vendor: GD-ML (AMAP)
- Model: `GD-ML/DreamX-World-5B-Cam`
- Task: Image-to-video world generation with explicit 6-DoF camera/action control
- Mode: Offline generation via the shared `image_to_video` example / `Omni` API
- Maintainer: Community

## When to use this recipe

Use this to generate camera-controllable videos from a single start image + a
caption + a sequence of camera action commands. The pipeline class is
`WanCameraPipeline`: a Wan2.2 TI2V-5B image-to-video backbone with a per-block
PRoPE camera self-attention branch.

The released checkpoint is **transformer-only**; the VAE / text-encoder /
tokenizer load from the base `Wan-AI/Wan2.2-TI2V-5B-Diffusers` by default
(override via `model_config["base_model_path"]`).

Camera action tokens (composable, e.g. `"wj"` = push in + pan left):

| Token | Control | Token | Control |
|-------|---------|-------|---------|
| `w` | push in | `s` | pull out |
| `a` | move left | `d` | move right |
| `i` | tilt up | `k` | tilt down |
| `j` | pan left | `l` | pan right |

`action_seq` / `action_speed_list` are declared in
[`vllm_omni/model_extras/dreamx_world.py`](../../vllm_omni/model_extras/dreamx_world.py)
and passed via `--extra-body`.

## References

- Model: https://huggingface.co/GD-ML/DreamX-World-5B-Cam
- Upstream: https://github.com/AMAP-ML/DreamX-World
- Base model: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers
- Integration issue: https://github.com/vllm-project/vllm-omni/issues/4570

## Hardware Support

## GPU

### 1x H100 80GB

#### Environment

- OS: Linux; Python 3.12
- GPU: 1x NVIDIA H100 80GB HBM3, driver 580.126.09
- vLLM version: 0.24.0 (torch 2.11.0+cu130)
- vLLM-Omni: editable install from the repo root (`pip install -e .`)
- `GD-ML/DreamX-World-5B-Cam` plus base `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
  are fetched on first run (~39 GB)

#### Command

Reproduces the second item in upstream
[`configs/dreamx/eval.json`](https://github.com/AMAP-ML/DreamX-World/blob/master/configs/dreamx/eval.json)
(`demo/007.jpg`, `w` → `wj`). Clone the upstream repo for the start frame. The
model id auto-detects `WanCameraPipeline` (no `--model-class-name` needed).

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model GD-ML/DreamX-World-5B-Cam \
  --image /path/to/DreamX-World/demo/007.jpg \
  --prompt "Style: Minecraft. A serene Minecraft landscape at sunset, featuring a blocky cliffside overlooking a calm ocean. In the foreground, grassy terrain with yellow flowers and red soil leads up to a rugged cliff composed of layered red and gray blocks. Sparse trees grow on rocky outcrops, adding life to the structured environment. The midground reveals the cliff's dramatic descent into the water, while the background showcases a vast ocean reflecting the warm hues of the setting sun. The sky is painted in gradients of orange, pink, and pale blue, with pixelated clouds drifting above. The lighting casts soft shadows and enhances the textured, cubic surfaces, creating a peaceful and immersive atmosphere that blends natural beauty with digital artistry." \
  --height 704 --width 1280 --num-frames 121 --fps 24 \
  --num-inference-steps 50 --guidance-scale 3.0 --flow-shift 3.0 --seed 42 \
  --extra-body '{"action_seq": ["w", "wj"], "action_speed_list": [4, 6]}' \
  --output dreamx_i2v.mp4
```

#### Verification

A 121-frame 704×1280 MP4 is written to `dreamx_i2v.mp4`; the camera pushes in
(`w`), then pushes in while panning left (`wj`) — the same action sequence as
the upstream reference for this `eval.json` item. Exact viewpoints differ
slightly by design: vLLM-Omni's action-frame allocation intentionally diverges
from upstream's (see Notes).

#### Notes

- **Measured (1x H100 80GB, command above, warm, mean of 3 runs):**
  - I2V 704×1280, 121 frames, 50 steps → **~196 s** end-to-end
    (denoise 3.71 s/step, VAE decode ~6.8 s); engine init ~57 s.
  - Upstream DreamX reference on the same GPU, inputs, seed and sampler
    (UniPC): ~292 s end-to-end, init ~152 s.
- **Memory:** ~49.7 GB peak device memory (vs ~45.1 GB upstream) — fits a
  single 80 GB GPU with headroom. The camera (PRoPE) branch requires
  `sequence_parallel_size == 1` and `pipeline_parallel_size == 1`.
- `num_frames` must satisfy the 1+4k pattern (e.g. 81, 121); it is snapped
  automatically. `121` frames = 5s @ 24fps; `81` = 5s @ 16fps.
- Camera control is **required**: `action_seq` + `action_speed_list` must be
  provided; the pipeline raises if they are missing (use the base
  `WanPipeline` for plain image-to-video).
- Frame 0 is the identity pose; the remaining `num_frames - 1` frames are
  split near-evenly across actions (durations differ by at most 1), so
  `num_frames >= len(action_seq) + 1`. Upstream instead gives every action
  `ceil(num_frames / len(action_seq))` frames and truncates the tail; the
  vLLM-Omni scheduling avoids under-representing the final action, at the cost
  of slightly different trajectories than upstream for the same inputs.
- The long-horizon autoregressive `DreamX-World-5B` model is out of scope.
