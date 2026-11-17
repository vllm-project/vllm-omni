# Wan2.2 TI2V 5B — RTX PRO 4500 Blackwell 32GB

> Single-GPU, native 720p text-to-video and image-to-video serving on a 32 GB workstation card

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- Task: Unified text-to-video (T2V) and image-to-video (I2V) generation
- Mode: Online serving with the OpenAI-compatible Videos API
- Hardware: 1x NVIDIA RTX PRO 4500 Blackwell, 32,623 MiB
- Maintainer: Community

## When to use this recipe

Use this recipe to serve both T2V and I2V from one dense 5B checkpoint on a
single 32 GB Blackwell workstation GPU. It qualifies the model's native 720p
profile without CPU offload or multi-GPU parallelism, using the same commands
and sampling parameters as the
[A100 80GB profile](./Wan2.2-TI2V-5B-A100.md) so the two results are directly
comparable. The synchronous endpoint is used below because it returns the MP4
and request timing in one response; the asynchronous Videos API uses the same
server and sampling parameters.

## Supported model contract

| Task | Required input | Qualified output | Endpoint |
| --- | --- | --- | --- |
| T2V | Text prompt; omit `input_reference` | 1280x704, 121 frames, 24 fps | `POST /v1/videos/sync` |
| I2V | Text prompt plus one image uploaded as `input_reference` | 1280x704, 121 frames, 24 fps | `POST /v1/videos/sync` |

The upstream checkpoint supports both tasks at 720p and 24 fps. For native
landscape 720p, use `1280x704`; portrait generation uses `704x1280`. Wan video
lengths follow the temporal VAE contract (`4k+1` frames). This recipe validates
one output per request with 50 denoising steps and seed 42.

## References

- Upstream model card:
  <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>
- A100 80GB profile for the same model:
  [`Wan2.2-TI2V-5B-A100.md`](./Wan2.2-TI2V-5B-A100.md)
- Online T2V example:
  [`examples/online_serving/text_to_video`](../../examples/online_serving/text_to_video)
- Online I2V example:
  [`examples/online_serving/image_to_video`](../../examples/online_serving/image_to_video)
- Offline I2V example:
  [`examples/offline_inference/image_to_video`](../../examples/offline_inference/image_to_video)
- Community recipe tracker:
  [vllm-project/vllm-omni#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware

- Accelerator model and per-device memory: NVIDIA RTX PRO 4500 Blackwell, 32,623 MiB
- Number of devices: 1
- Device interconnect: not applicable to this single-GPU profile
- Host memory: 62 GB allocated to the pod (RunPod Secure Cloud, 28 vCPU AMD EPYC 7663);
  CPU offload was not used
- Qualification scope: BF16, batch size 1, cuDNN attention (`CUDNN_ATTN`), eager mode,
  VAE tiling, T2V and I2V at 1280x704 / 121 frames / 24 fps

## Software environment

- OS: Ubuntu 24.04.5 LTS, Linux 6.8.0
- Python: 3.12.3
- NVIDIA driver / CUDA: 595.91.07 / CUDA 13.2
- PyTorch: 2.14.0+cu130
- Transformers / Diffusers: 5.19.0 / 0.40.0
- vLLM: 0.33.0
- vLLM-Omni: commit `30466ff9a83a7c195b241206b7761325b9bd327e`

Install the matching vLLM 0.33.0 CUDA 13 runtime and vLLM-Omni from source as
described in the [installation guide](../../docs/getting_started/installation/README.md),
including the pinned cuDNN runtime step.

## Command

Start the server from the repository root:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --omni \
  --port 8092 \
  --vae-use-tiling \
  --enforce-eager \
  --init-timeout 1800 \
  --stage-init-timeout 1800
```

In another terminal, download the public reference image:

```bash
wget -O cherry_blossom.jpg \
  https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

Generate a text-only video:

```bash
curl --fail-with-body -D t2v-headers.txt \
  -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F "prompt=Two anthropomorphic cats in comfortable boxing gear and bright gloves fight intensely on a spotlighted stage, cinematic lighting, smooth motion." \
  -F "negative_prompt=blurry, low quality, static, distorted, artifacts, watermark, text" \
  -F "width=1280" -F "height=704" \
  -F "num_frames=121" -F "fps=24" \
  -F "num_inference_steps=50" -F "guidance_scale=5.0" \
  -F "flow_shift=5.0" -F "seed=42" \
  --output wan22-ti2v-5b-t2v.mp4
```

Generate an image-conditioned video from the same server:

```bash
curl --fail-with-body -D i2v-headers.txt \
  -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F "input_reference=@cherry_blossom.jpg" \
  -F "prompt=Cherry blossoms sway gently in the breeze while petals drift across the path, cinematic, smooth natural motion." \
  -F "negative_prompt=blurry, low quality, static, distorted, artifacts, watermark, text" \
  -F "width=1280" -F "height=704" \
  -F "num_frames=121" -F "fps=24" \
  -F "num_inference_steps=50" -F "guidance_scale=5.0" \
  -F "flow_shift=5.0" -F "seed=42" \
  --output wan22-ti2v-5b-i2v.mp4
```

## Verification

Check both MP4 files with `ffprobe`:

```bash
for video in wan22-ti2v-5b-t2v.mp4 wan22-ti2v-5b-i2v.mp4; do
  ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration \
    -of default=noprint_wrappers=1 "$video"
done
```

Each file reported:

```text
codec_name=h264
width=1280
height=704
r_frame_rate=24/1
duration=5.041667
nb_read_frames=121
```

The synchronous response headers and an independent 200 ms `nvidia-smi`
sampler produced these results on the first T2V and I2V requests after
server startup:

| Task | HTTP | `X-Inference-Time-S` | Server peak | Observed device peak | MP4 size |
| --- | ---: | ---: | ---: | ---: | ---: |
| T2V | 200 | 529.817 s | 25,478 MB | 26,135 MiB | 6,680,214 bytes |
| I2V | 200 | 563.678 s | 29,370 MB | 29,766 MiB | 9,887,628 bytes |

Both outputs were decoded in full and visually inspected. The T2V output
contained the prompted boxing cats. The I2V first frame preserved the uploaded
cherry-blossom scene, and later frames showed motion.

## Notes

- Memory: model loading used 21.2239 GiB. The measured I2V device peak was
  29,766 MiB, leaving about 2.8 GiB free on this 32 GB card. Native 720p runs
  without CPU or layerwise offload, but with little headroom; concurrent
  requests and higher resolutions are not qualified on this profile.
- Speed: per-clip inference time was about 1.85x the A100 80GB profile
  (529.8 s vs 288.3 s for T2V, 563.7 s vs 299.4 s for I2V) at identical settings.
- Attention backend: vLLM-Omni selects `CUDNN_ATTN` by default on Blackwell
  sm_120 (cuDNN 9.25.1, head_dim 128). The A100 profile used FlashAttention.
- Startup: model loading took 349.2 s (241.9 s reading weights from a
  network-backed volume), which exceeds the default 300 s stage-init timeout.
  This qualification therefore used `--init-timeout 1800 --stage-init-timeout 1800`.
  Local NVMe storage should load considerably faster.
- The log shows a harmless warning that `transformer_2` is declared offloadable
  but is absent; TI2V-5B has a single transformer.
- `--vae-use-tiling` and `--enforce-eager` are kept from the A100 profile so
  the two recipes are directly comparable. Remove `--enforce-eager` only after
  separately warming and measuring compiled mode.
- `flow_shift=5.0` is the 720p setting. The shared 480p examples use 12.0.
- The generated MP4 contains video only; this checkpoint does not generate audio.

## Supported features

| Feature | Status for this profile | Shared guide |
| --- | --- | --- |
| OpenAI-compatible Videos API | Qualified for synchronous T2V and I2V | [Videos API](../../docs/serving/videos_api.md) |
| VAE tiling | Enabled and qualified on one RTX PRO 4500 | [VAE parallelism and tiling](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Cache-DiT / TaylorSeer | Available but not enabled in the measurements above | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| TeaCache | Available but not qualified in this recipe | [TeaCache](../../docs/user_guide/diffusion/cache_acceleration/teacache.md) |
| CPU or layerwise offload | Not needed; the I2V peak leaves about 2.8 GiB free on 32 GB | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| TP / SP / CFG parallelism | Not applicable to this single-GPU profile | [Parallelism overview](../../docs/user_guide/diffusion/parallelism/overview.md) |
