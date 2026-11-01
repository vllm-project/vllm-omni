# Video Depth Anything video depth estimation — H200

## Summary

- Vendor: ByteDance / Depth Anything team
- Model: `depth-anything/Video-Depth-Anything-{Small,Base,Large}` (relative depth),
  `depth-anything/Metric-Video-Depth-Anything-{Small,Base,Large}` (metric depth)
- Task: Temporally consistent per-frame depth for videos of any length
- Mode: Online serving (`POST /v1/videos/depth`) and offline `Omni.generate`
- Hardware: 1x NVIDIA H200 141GB
- Maintainer: Community

## When to use this recipe

Serve Video Depth Anything to turn a video into a depth video without the
frame-to-frame flicker of single-image depth models. Typical uses are depth
conditioning for video generation and editing, relighting, compositing,
3D-aware effects and robotics preprocessing.

## Supported model contract

| Item | Contract |
| ---- | -------- |
| Endpoint | `POST /v1/videos/depth` ([Video Depth API](../../docs/serving/video_depth_api.md)) |
| Input | One uploaded video (up to 512 MiB and 2048 frames, any container PyAV decodes); every frame is processed in decode order |
| Options | `input_size` (network short side, multiple of 14, default 518), `max_res` (downscale above this long side, default 1280, `-1` keeps the input size) |
| Output | `[T, H, W]` depth at the (possibly downscaled) input size: an H.264 depth video (grayscale or inferno) and optional raw float16 depth (NPZ) |
| Depth type | Relative checkpoints: affine-invariant inverse depth (larger is nearer). Metric checkpoints: depth in meters |
| Checkpoints | The six released `.pth` files, each served from its Hub repository or a local directory holding exactly one of them. The variant comes from the file name |
| Licenses | Code Apache-2.0. Small weights Apache-2.0; Base and Large weights CC-BY-NC-4.0 |
| Not covered | Streaming (frame-by-frame) inference, the reference's `--fp32`-only mode as a per-request switch (use `--dtype float32` for the server), sequence parallelism |

## References

- Upstream: <https://github.com/DepthAnything/Video-Depth-Anything>, <https://huggingface.co/depth-anything>
- Paper: Video Depth Anything: Consistent Depth Estimation for Super-Long Videos (CVPR 2025)
- Reference implementation used for parity: `DepthAnything/Video-Depth-Anything@4f5ae23`

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141 GB
- Number of devices: 1
- Device interconnect: not used
- Qualification scope: single-request online serving

## Software environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.15
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.2, PyTorch 2.13.0
- vLLM version: 0.32.0
- vLLM-Omni version or commit: `9a095b910` plus this change

## Command

```bash
vllm serve depth-anything/Video-Depth-Anything-Small --omni --port 8091
```

Swap in any of the other five repositories to serve that variant. Weights stay
in FP32 and the network runs under FP16 autocast, as the reference does by
default; add `--dtype float32` for full FP32.

## Verification

```bash
curl -s http://localhost:8091/v1/videos/depth \
  -F "video=@davis_rollercoaster.mp4" -F "colormap=inferno" \
  | tee result.json | jq '{width, height, num_frames, fps, metric, depth_min, depth_max}'
jq -r .video result.json | base64 -d > depth.mp4
```

Expected for the reference repository's `assets/example_videos/davis_rollercoaster.mp4`:
`960x540`, 70 frames at 24 fps, `metric: false`, `depth_min` 0 and
`depth_max` about 7.55.

## Notes

- Speed (Small, FP16 autocast, warm server, end to end over HTTP including
  upload, decoding and the H.264 encode):

  | Video | Grayscale video | Inferno video + raw depth |
  | ----- | --------------- | ------------------------- |
  | `davis_rollercoaster`, 70 frames 960x540 | 2.8 s | 3.4 s |
  | `Tokyo-Walk`, 195 frames 1280x720 | 8.4 s | 13.1 s |

  Metric-Small takes the same time. The first request after startup is
  about 2 s slower while cuDNN selects its kernels.
- Memory usage: 0.11 GiB of Small weights; about 4.1 GiB of GPU memory in use
  after a 720p request. Peak activation memory depends on `input_size` and the
  aspect ratio, not on the video length: the model runs one 32-frame window at
  a time.
- Accuracy: with TF32 disabled, the port matches the reference
  `infer_video_depth` in FP32 to a relative L2 error of about 1e-7 for single
  windows and 6e-7 to 1.3e-6 for whole videos. Small, Metric-Small and Large
  were checked on `davis_rollercoaster` (70 frames, 960x540) and `Tokyo-Walk`
  (195 frames, 1280x720), with 10, 32, 54 and 70-frame cuts that exercise
  every window and alignment path; Base and Metric-Large on
  `davis_rollercoaster`. Metric-Base was checked for strict loading and a
  40-frame run only.
- Precision: against the reference in FP32 on `Tokyo-Walk`, relative L2 error

  | Model | Reference FP16 | This port, FP16 | This port, BF16 |
  | ----- | -------------- | --------------- | --------------- |
  | Small | 3.09e-4 | 2.61e-4 | 3.24e-3 |
  | Metric-Small | 2.79e-3 | 2.75e-3 | 2.66e-2 |
  | Large | 5.63e-4 | 5.52e-4 | 3.93e-3 |

  BF16 drifts about ten times more, so the pipeline computes in FP16 for any
  `--dtype` except `float32`. The served float16 depth of `Tokyo-Walk` (Small)
  is within 3.4e-4 of the FP32 reference, of which float16 storage alone
  accounts for 2.1e-4.
- Known limitations: one video per request, at most 2048 frames; the whole
  output is returned at once, so `return_depth=true` on long high-resolution
  videos produces large responses (about 150 MB of NPZ for 195 frames at
  720p).
