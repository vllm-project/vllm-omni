# Wan2.1 T2V 1.3B — RTX 5090 D v2 24GB

> Single-GPU text-to-video serving on a 24 GB SM 12.0 card, BF16 without offload

## Summary

- Vendor: Wan-AI
- Model: [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- Task: Text-to-video
- Mode: Online serving with the OpenAI-compatible Videos API, one request at a time
- Hardware: 1x NVIDIA GeForce RTX 5090 D v2 24GB (SM 12.0), Ubuntu on WSL2
- Recommended deployment: BF16, no offload, VAE tiling, eager mode
- Maintainer: Community

## When to use this recipe

Use this recipe to serve Wan2.1 T2V 1.3B on one 24 GB SM 12.0 GPU. At
832x480 and 81 frames the BF16 pipeline reserves 17.10 GiB at its peak, so it
needs neither offload nor quantization on this card. Online FP8 is an optional
speed-up at 832x480.

## Supported model contract

| Task | Input | Output | Profile |
| --- | --- | --- | --- |
| Text-to-video | One text prompt | 832x480 at 17, 33 or 81 frames, or 480x272 at 17 frames; 16 fps; 30 steps | BF16, no offload (recommended) |
| Text-to-video | One text prompt | Same shapes | Online FP8 transformer, no offload (optional) |

Pass `size` and `num_frames` explicitly; only the shapes above were run. Wan
video lengths follow the temporal VAE contract (`4k+1` frames). Both 17-frame
shapes showed distortion in their final frames in BF16 and FP8 alike, so use 33
frames or more when output quality matters.

## References

- Model card: [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- Online text-to-video example:
  [`examples/online_serving/text_to_video`](../../examples/online_serving/text_to_video)
- Videos API: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Consumer-device tracking issue, item A1:
  [vllm-project/vllm-omni#10008](https://github.com/vllm-project/vllm-omni/issues/10008)
- Full measurement report, including the module-offload rows:
  [#10008 comment](https://github.com/vllm-project/vllm-omni/issues/10008#issuecomment-6871920279)

## Checkpoint and setup

The measurements used snapshot revision
`0fad780a534b6463e45facd96134c9f345acfa5b`:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --revision 0fad780a534b6463e45facd96134c9f345acfa5b
```

## Hardware

- Accelerator model and per-device memory: NVIDIA GeForce RTX 5090 D v2,
  24,455 MiB reported by `nvidia-smi`, SM 12.0. The Windows desktop held
  279–407 MiB at the start of each run.
- Number of devices: 1
- Device interconnect: none (single device)
- Host memory: 26 GB WSL2 memory limit and 7 GB swap. Without offload the
  server used at most 4.4 GB of host RSS.
- Qualification scope: one machine. A fresh server for each configuration, one
  warmup request, then ten sequential steady requests. A shorter check (one
  warmup, three steady requests) was repeated on a newer main commit, see
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
vllm serve Wan-AI/Wan2.1-T2V-1.3B-Diffusers --omni \
  --port 8091 \
  --dtype bfloat16 \
  --vae-use-tiling \
  --enforce-eager
```

For the optional FP8 profile, add
`--diffusion-quantization-config '{"transformer":{"method":"fp8"}}'`.

## Verification

```bash
curl --fail-with-body -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F "prompt=A small sailboat glides across a calm lake at sunrise, with soft ripples on the water." \
  -F "size=832x480" -F "num_frames=81" -F "fps=16" \
  -F "num_inference_steps=30" -F "guidance_scale=5.0" -F "seed=42" \
  --output wan21-t2v-832x480-81.mp4

ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames \
  -of default=noprint_wrappers=1 wan21-t2v-832x480-81.mp4
```

Expected output:

```text
width=832
height=480
r_frame_rate=16/1
nb_read_frames=81
```

On this card the BF16 request returns in about 90 s after the server's first
request.

## Qualification evidence

### Configuration matrix (commit `18747cd29`)

Eager mode, VAE tiling, 30 steps, guidance 5.0, seed 42, no offload. Every row
completed 10 of 10 steady requests, and all MP4s within a row have the same
SHA-256.

| Shape | Precision | Cold start (s) | Steady median (s) | Worker peak, allocated / reserved (GiB) | Whole-device peak (MiB) | Host RSS peak (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 480x272, 17 frames | BF16 | 22.0 | 3.41 | 14.22 / 14.84 | 16,144 | 3.5 |
| 480x272, 17 frames | FP8 | 83.9 | 4.58 | 12.98 / 13.52 | 14,905 | 3.1 |
| 832x480, 17 frames | BF16 | 22.1 | 11.52 | 14.26 / 14.96 | 16,274 | 3.7 |
| 832x480, 17 frames | FP8 | 47.7 | 9.50 | 13.03 / 13.53 | 14,919 | 3.5 |
| 832x480, 33 frames | BF16 | 29.9 | 25.34 | 14.32 / 15.43 | 16,748 | 3.3 |
| 832x480, 33 frames | FP8 | 50.3 | 20.91 | 13.08 / 14.02 | 15,417 | 3.7 |
| 832x480, 81 frames | BF16 | 34.3 | 90.57 | 15.10 / 17.10 | 18,584 | 3.9 |
| 832x480, 81 frames | FP8 | 53.4 | 81.45 | 13.86 / 15.59 | 17,023 | 4.2 |

How these were measured:

- Worker peaks come from the CUDA allocator, read at the pipeline's stage
  boundaries by a local measurement-only patch. The same patch caps the worker
  with `torch.cuda.set_per_process_memory_fraction(0.96)` (22.93 GiB), so that
  exceeding VRAM raises a CUDA OOM instead of spilling into shared system
  memory under WSL2. The cap was never reached. The patch changes no
  computation.
- The whole-device peak is an `nvidia-smi` sample taken once per second and
  includes the Windows desktop.
- Host RSS is the summed RSS of the server's process tree, sampled once per
  second.
- Cold start ends when `/health` returns 200.

### FP8 against BF16

Computed per frame on the decoded steady MP4s from the rows above. The
thresholds are the smoke and stricter levels of the
[quantization guide](../../docs/user_guide/quantization/overview.md#output-metrics): PSNR at
least 20 / 25 dB, MAE at most 12 / 6. The MP4s are lossy, so encoder
differences are included.

| Shape | PSNR mean / min (dB) | MAE | PSNR level | MAE level |
| --- | ---: | ---: | --- | --- |
| 480x272, 17 frames | 20.10 / 18.66 | 14.45 | smoke | fails smoke |
| 832x480, 17 frames | 26.02 / 25.27 | 6.54 | stricter | smoke |
| 832x480, 33 frames | 27.69 / 27.55 | 7.06 | stricter | smoke |
| 832x480, 81 frames | 25.50 / 23.80 | 7.46 | stricter (mean) | smoke |

In a side-by-side visual review, FP8 was almost indistinguishable from BF16 at
all four shapes. The weak 480x272 metrics compare two clips that are both
distorted in their final frames.

### Current-main check (commit `c71efee10`)

The two 832x480, 81-frame rows were repeated at commit `c71efee10` with the
same runner and patch: one warmup request, then three steady requests. Both
produced MP4s with the same SHA-256 as at `18747cd29`.

| Precision | Steady median, range (s) | Worker peak, allocated / reserved (GiB) | Whole-device peak (MiB) | Host RSS peak (GB) | MP4 vs. `18747cd29` |
| --- | --- | ---: | ---: | ---: | --- |
| BF16 | 90.27 (90.16–90.27) | 15.10 / 17.04 | 18,226 | 4.4 | Identical |
| FP8 | 81.18 (81.11–81.58) | 13.86 / 15.55 | 16,692 | 3.9 | Identical |

The Windows desktop held 100 MiB at the start of these runs, against 279–407
MiB for the matrix, so the whole-device peaks of the two runs are not directly
comparable.

## Notes

- Under WSL2, a run that exceeds VRAM may not raise an OOM. The Windows driver
  can back the overflow with system memory, and the request becomes much
  slower. Windows Task Manager shows this as rising "Shared GPU memory".
- Offload is not needed on this card. The module-offload rows in the linked
  report used a local change that makes the offload copies blocking. The stock
  offload path was not run on this machine, so offload is not provided as a
  profile here.
- In the matrix, FP8 cold starts took 47–84 s against 22–34 s for BF16. In the
  later current-main check the FP8 server was ready in 26 s. The difference was
  not investigated.
- `--enforce-eager` was used in every run; compiled mode was not measured.

## Supported features

| Feature | Status on this hardware | Shared guide |
| --- | --- | --- |
| BF16, no offload | Qualified at the shapes above (recommended) | — |
| Online FP8 transformer | Qualified at the shapes above, see the FP8 table | [FP8](../../docs/user_guide/quantization/fp8.md) |
| Int8 W8A8 | Not supported on SM 12.0 with vLLM 0.33.0 | [Int8](../../docs/user_guide/quantization/int8.md) |
| VAE tiling | Enabled in every row | [VAE parallelism and tiling](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| CPU offload | Not needed; not qualified on the stock path | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| Cache-DiT / TeaCache | Not measured | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| Compiled mode / CUDA Graph | Not measured | — |
| Concurrent requests | Not measured; all runs were sequential | — |
| TP / SP / CFG parallelism | Not applicable to this single-GPU profile | [Parallelism overview](../../docs/user_guide/diffusion/parallelism/overview.md) |
