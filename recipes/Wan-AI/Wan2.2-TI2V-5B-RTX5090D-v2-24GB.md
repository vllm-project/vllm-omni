# Wan2.2 TI2V 5B — RTX 5090 D v2 24GB

> Single-GPU, native 720p text-to-video and image-to-video serving on a 24 GB SM 12.0 card

## Summary

- Vendor: Wan-AI
- Model: [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- Task: Unified text-to-video (T2V) and image-to-video (I2V) generation
- Mode: Online serving with the OpenAI-compatible Videos API, one request at a time
- Hardware: 1x NVIDIA GeForce RTX 5090 D v2 24GB (SM 12.0), Ubuntu on WSL2
- Recommended deployment: online FP8 transformer, layerwise DiT offload, VAE
  tiling, eager mode
- Maintainer: Community

## When to use this recipe

Use this recipe to serve both tasks at the model's native 1280x704 and 121
frames on one 24 GB SM 12.0 GPU. On this machine only online FP8 completes at
the native shape. BF16 does not fit in VRAM without offload, and with offload
it exhausts the 26 GB of host memory available to WSL2. Layerwise offload of the
DiT keeps the FP8 profile well below the VRAM limit at almost no latency cost.
For the same model on an 80 GB GPU in BF16, see the
[A100 recipe](./Wan2.2-TI2V-5B-A100.md).

## Supported model contract

| Task | Required input | Qualified output | Profile |
| --- | --- | --- | --- |
| T2V | Text prompt; omit `input_reference` | 1280x704, 121 frames, 24 fps, 50 steps | Online FP8 + layerwise DiT offload + VAE tiling |
| I2V | Text prompt plus one image uploaded as `input_reference` | 1280x704, 121 frames, 24 fps, 50 steps | Same profile |

Wan video lengths follow the temporal VAE contract (`4k+1` frames). Each
request produced one output with seed 42. Other shapes, such as portrait
`704x1280`, were not run on this card.

## References

- Model card: [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- Online T2V example:
  [`examples/online_serving/text_to_video`](../../examples/online_serving/text_to_video)
- Online I2V example:
  [`examples/online_serving/image_to_video`](../../examples/online_serving/image_to_video)
- Videos API: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Consumer-device tracking issue, item A1:
  [vllm-project/vllm-omni#10008](https://github.com/vllm-project/vllm-omni/issues/10008)
- Full measurement report, including the failed BF16 and module-offload rows:
  [#10008 comment](https://github.com/vllm-project/vllm-omni/issues/10008#issuecomment-6871930957)

## Checkpoint and setup

The measurements used snapshot revision
`b8fff7315c768468a5333511427288870b2e9635`:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --revision b8fff7315c768468a5333511427288870b2e9635
```

The I2V requests use `examples/i2v_input.JPG`, which ships in that snapshot.

## Hardware

- Accelerator model and per-device memory: NVIDIA GeForce RTX 5090 D v2,
  24,455 MiB reported by `nvidia-smi`, SM 12.0. The Windows desktop held
  87–121 MiB at the start of each run.
- Number of devices: 1
- Device interconnect: none (single device)
- Host memory: 26 GB WSL2 memory limit and 7 GB swap. Host memory is material
  here: the recommended profile used 16.1–16.7 GB of host RSS, and the BF16
  offload attempts ran out of it.
- Qualification scope: one machine. A fresh server for each configuration, one
  warmup request, then three sequential steady requests. The recommended
  profile was repeated on a newer main commit, see
  [Qualification evidence](#qualification-evidence).

This file does not cover other 24 GB cards such as the RTX 5090 Laptop GPU or
the 32 GB RTX 5090 and RTX 5090 D.

## Software environment

- OS: Ubuntu 26.04.1 LTS on WSL 3.0.3.0 (kernel 6.18.54.1)
- Python: 3.12.14
- Driver / runtime: NVIDIA 610.88, PyTorch 2.14.0+cu130
- vLLM: 0.33.0
- vLLM-Omni: commit `18747cd29ce4493cc9c48169d401bcedc96ae700` for the
  configuration matrix; commit `c71efee1041c32821aac1a1bc3af1d7e0fcfd7f2` for
  the current-main check
- Diffusers / Transformers: 0.40.0 / 5.19.0

## Command

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
  --port 8092 \
  --dtype bfloat16 \
  --vae-use-tiling \
  --enforce-eager \
  --diffusion-quantization-config '{"transformer":{"method":"fp8"}}' \
  --init-timeout 1800 \
  --diffusion-offload-config '{"mode":"layer","components":["dit"]}'
```

`--init-timeout 1800` is a safety margin; the server became ready in 27–46 s.
The layerwise offload streams the DiT only, and the UMT5 text encoder stays on
the GPU.

Generate a text-only video:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F "prompt=A small sailboat glides across a calm lake at sunrise, with soft ripples on the water." \
  -F "negative_prompt=blurry, low quality, static, distorted, artifacts, watermark, text" \
  -F "width=1280" -F "height=704" \
  -F "num_frames=121" -F "fps=24" \
  -F "num_inference_steps=50" -F "guidance_scale=5.0" \
  -F "flow_shift=5.0" -F "seed=42" \
  --output wan22-ti2v-5b-t2v.mp4
```

Generate an image-conditioned video from the same server, with the image from
the model snapshot:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F "input_reference=@/path/to/snapshot/examples/i2v_input.JPG" \
  -F "prompt=The subject in the image moves naturally while the camera slowly pushes in, cinematic, smooth natural motion." \
  -F "negative_prompt=blurry, low quality, static, distorted, artifacts, watermark, text" \
  -F "width=1280" -F "height=704" \
  -F "num_frames=121" -F "fps=24" \
  -F "num_inference_steps=50" -F "guidance_scale=5.0" \
  -F "flow_shift=5.0" -F "seed=42" \
  --output wan22-ti2v-5b-i2v.mp4
```

A request takes about four minutes on this card, within the default 600 s
limit of `/v1/videos/sync` (`VLLM_OMNI_VIDEO_SYNC_TIMEOUT`).

## Verification

```bash
for video in wan22-ti2v-5b-t2v.mp4 wan22-ti2v-5b-i2v.mp4; do
  ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,r_frame_rate,nb_read_frames \
    -of default=noprint_wrappers=1 "$video"
done
```

Each file is expected to report:

```text
width=1280
height=704
r_frame_rate=24/1
nb_read_frames=121
```

## Qualification evidence

### Native 1280x704 and 121 frames (commit `18747cd29`)

Eager mode, VAE tiling, 50 steps, guidance 5.0, `flow_shift` 5.0, seed 42.
"Host RAM" marks a run that a watchdog stopped when `MemAvailable` fell below
1.5 GB, because running this WSL2 instance out of memory has crashed it
before. Those runs might have finished slowly with swap; that was not tested.

| Task | Precision / offload | Result | Steady median (s) | Worker peak, allocated / reserved (GiB) | Whole-device peak (MiB) | Host RSS peak (GB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| T2V | FP8 / layerwise (recommended) | 3/3 | 239.0 | 15.23 / 17.88 | 19,080 | 16.6 |
| T2V | FP8 / none | 3/3 | 229.8 | 19.65 / 22.26 | 23,582 | 5.2 |
| T2V | BF16 / none | CUDA OOM on the first request | — | — | 22,958 | 12.5 |
| T2V | BF16 / layerwise | Host RAM after one completed request, 0/3 steady | — | 15.38 / 16.36 (first request) | 19,346 | 24.1 |
| I2V | FP8 / layerwise (recommended) | 3/3 | 256.6 | 16.73 / 19.87 | 21,116 | 16.7 |
| I2V | FP8 / none | 3/3 | 251.8 | 21.00 / 22.92 | 23,686 | 6.1 |
| I2V | BF16 / none | CUDA OOM on the first request | — | — | 23,206 | 11.3 |
| I2V | BF16 / layerwise | Host RAM after one completed request, 0/3 steady | — | 17.04 / 18.63 (first request) | 21,400 | 24.0 |

Without VAE tiling, FP8 T2V without offload ran out of memory in VAE decode
after denoising had finished. Within each completed row, all MP4s have the same
SHA-256, and the FP8 layerwise and FP8 no-offload rows decode to the same
frames.

How these were measured:

- Worker peaks come from the CUDA allocator, read at the pipeline's stage
  boundaries by a local measurement-only patch. The same patch caps the worker
  with `torch.cuda.set_per_process_memory_fraction(0.96)` (22.93 GiB), so that
  exceeding VRAM raises a CUDA OOM instead of spilling into shared system
  memory under WSL2. The patch changes no computation.
- The whole-device peak is an `nvidia-smi` sample taken once per second and
  includes the Windows desktop.
- Host RSS is the summed RSS of the server's process tree, sampled once per
  second.

### FP8 against BF16

Computed per frame on decoded MP4s with the same request and seed. At the
native shape the BF16 reference is the one valid MP4 that each BF16 layerwise
run produced before it ran out of host memory. The thresholds are the smoke and
stricter levels of the
[quantization guide](../../docs/user_guide/quantization/overview.md#output-metrics): PSNR at
least 20 / 25 dB, MAE at most 12 / 6.

| Shape | PSNR mean / min (dB) | MAE | PSNR level | MAE level |
| --- | ---: | ---: | --- | --- |
| T2V 1280x704, 121 frames | 27.13 / 26.02 | 5.44 | stricter | stricter |
| I2V 1280x704, 121 frames | 25.26 / 20.91 | 9.08 | stricter (mean) | smoke |
| T2V 832x480, 81 frames | 15.44 / 14.97 | 17.46 | fails smoke | fails smoke |

In a side-by-side visual review, FP8 was almost indistinguishable from BF16 in
all three pairs, and every clip looked normal. At 832x480 the metrics and the
visual review disagree, so FP8 is qualified here only at the native shape.

### Current-main check (commit `c71efee10`)

The recommended profile was repeated for both tasks at commit `c71efee10`
with the same runner and patch: one warmup request, then three steady
requests. Both produced MP4s with the same SHA-256 as at `18747cd29`.

| Task | Steady median, range (s) | Worker peak, allocated / reserved (GiB) | Whole-device peak (MiB) | Host RSS peak (GB) | MP4 vs. `18747cd29` |
| --- | --- | ---: | ---: | ---: | --- |
| T2V | 229.68 (229.62–229.72) | 15.22 / 17.96 | 19,172 | 16.1 | Identical |
| I2V | 251.76 (251.70–251.78) | 16.73 / 20.07 | 21,338 | 16.1 | Identical |

## Notes

- Memory: the recommended profile reserves up to 17.96 GiB (T2V) and
  20.07 GiB (I2V) at its peak. FP8 without offload also completes, but I2V
  then reserves 22.92 GiB of the 22.93 GiB cap, which leaves no headroom.
- Host memory: BF16 at the native shape needs more than the 26 GB of host
  memory available to WSL2 here, with either module or layerwise offload. A
  host with more memory may run it; that was not tested.
- `--vae-use-tiling` is required at the native shape.
- The module-offload rows in the linked report used a local change that makes
  the offload copies blocking, so module offload is not provided as a profile
  here.
- `--enforce-eager` was used in every run; compiled mode was not measured.

## Supported features

| Feature | Status on this hardware | Shared guide |
| --- | --- | --- |
| Online FP8 transformer | Qualified at the native shape; fails the metric thresholds at 832x480 | [FP8](../../docs/user_guide/quantization/fp8.md) |
| BF16 | Does not run at the native shape on this machine (VRAM, or host memory with offload) | — |
| Int8 W8A8 | Not supported on SM 12.0 with vLLM 0.33.0 | [Int8](../../docs/user_guide/quantization/int8.md) |
| Layerwise offload (DiT) | Qualified with FP8 (recommended) | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| Module offload | Not qualified on the stock path | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| VAE tiling | Required at the native shape | [VAE parallelism and tiling](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Cache-DiT / TeaCache | Not measured | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| Compiled mode / CUDA Graph | Not measured | — |
| Concurrent requests | Not measured; all runs were sequential | — |
| TP / SP / CFG parallelism | Not applicable to this single-GPU profile | [Parallelism overview](../../docs/user_guide/diffusion/parallelism/overview.md) |
