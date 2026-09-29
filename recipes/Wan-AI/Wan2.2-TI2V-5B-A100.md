# Wan2.2 TI2V 5B — A100 80GB

> Single-GPU, native 720p text-to-video and image-to-video serving

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- Task: Unified text-to-video (T2V) and image-to-video (I2V) generation
- Mode: Online serving with the OpenAI-compatible Videos API
- Hardware: 1x NVIDIA A100-SXM4-80GB
- Maintainer: Community

## When to use this recipe

Use this recipe to serve both T2V and I2V from one dense 5B checkpoint on one
A100 80GB. It qualifies the model's native 720p profile without CPU offload or
multi-GPU parallelism. The synchronous endpoint is used below because it
returns the MP4 and request timing in one response; the asynchronous Videos API
uses the same server and sampling parameters.

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
- Online T2V example:
  [`examples/online_serving/text_to_video`](../../examples/online_serving/text_to_video)
- Online I2V example:
  [`examples/online_serving/image_to_video`](../../examples/online_serving/image_to_video)
- Offline I2V example:
  [`examples/offline_inference/image_to_video`](../../examples/offline_inference/image_to_video)
- Community recipe tracker:
  [vllm-project/vllm-omni#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware

- Accelerator model and per-device memory: NVIDIA A100-SXM4-80GB, 81,920 MiB
- Number of devices: 1
- Device interconnect: not applicable to this single-GPU profile
- Host memory: 1 TiB installed on the qualification host; CPU offload was not used
- Qualification scope: BF16, batch size 1, native FlashAttention, eager mode,
  VAE tiling, T2V and I2V at 1280x704 / 121 frames / 24 fps

## Software environment

- OS: Ubuntu 24.04.3 LTS, Linux 6.8.0
- Python: 3.12.3
- NVIDIA driver / host toolkit: 550.127.05 / CUDA 12.8
- PyTorch: 2.13.0+cu129
- Transformers / Diffusers: 5.14.1 / 0.40.0
- vLLM: 0.28.0+cu129
- vLLM-Omni: commit `ff6e906a25de1ca3122c0d0d5250fde5a0ddbc7b`

The default vLLM 0.28.0 wheel targets CUDA 13.0. Driver 550 cannot run that
variant, so this qualification used the official CUDA 12.9 wheel:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  --torch-backend=auto
uv pip install -e .
```

## Command

Start the server from the repository root:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --omni \
  --port 8092 \
  --vae-use-tiling \
  --enforce-eager
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
sampler produced these steady-state results after server warmup:

| Task | HTTP | `X-Inference-Time-S` | Server peak | Observed device peak | MP4 size |
| --- | ---: | ---: | ---: | ---: | ---: |
| T2V | 200 | 288.267 s | 26,546 MB | 27,533 MiB | 7,321,179 bytes |
| I2V | 200 | 299.439 s | 29,766 MB | 30,333 MiB | 15,667,146 bytes |

Both outputs were decoded in full and visually inspected. The T2V output
contained the prompted boxing cats. The I2V first frame preserved the uploaded
cherry-blossom scene (5.81 mean absolute pixel error after resize and H.264
encoding), and later frames showed motion.

## Notes

- Memory usage: model loading used 21.2239 GiB. The measured I2V device peak
  was 30,333 MiB, leaving more than 50 GiB free on this A100.
- `--vae-use-tiling` is retained in the known-good command even though this
  profile has ample headroom; it bounds decode memory at 720p.
- `--enforce-eager` removes graph-compilation variability from the measured
  profile. Remove it only after separately warming and measuring compiled mode.
- `flow_shift=5.0` is the 720p setting. The shared 480p examples use 12.0.
- The generated MP4 contains video only; this checkpoint does not generate audio.
- The first request includes neither model-load time nor the server's startup
  warmup. Server initialization took approximately 26 seconds with a warm cache.

## Supported features

| Feature | Status for this profile | Shared guide |
| --- | --- | --- |
| OpenAI-compatible Videos API | Qualified for synchronous T2V and I2V | [Videos API](../../docs/serving/videos_api.md) |
| VAE tiling | Enabled and qualified on one A100 | [VAE parallelism and tiling](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Cache-DiT / TaylorSeer | Available but not enabled in the measurements above | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| TeaCache | Available but not qualified in this recipe | [TeaCache](../../docs/user_guide/diffusion/cache_acceleration/teacache.md) |
| CPU or layerwise offload | Not needed for the qualified 80GB profile | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| TP / SP / CFG parallelism | Not applicable to this single-GPU profile | [Parallelism overview](../../docs/user_guide/diffusion/parallelism/overview.md) |
