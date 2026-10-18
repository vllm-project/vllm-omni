# Wan2.2 TI2V 5B — H20 96GB

> Single-GPU 720p text-to-video and image-to-video serving

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- Task: Unified text-to-video (T2V) and image-to-video (I2V) generation
- Mode: Online serving with the OpenAI-compatible Videos API
- Hardware: 1x NVIDIA H20 96GB
- Maintainer: Community

## When to use this recipe

Use this recipe to serve the unified Wan2.2 TI2V-5B checkpoint on one H20. The
same server handles T2V requests without a reference image and I2V requests
with one uploaded reference image. The profile keeps the model in BF16, uses
one diffusion worker, and enables VAE tiling for bounded decode memory.

This file records the reproducible H20 command, validation procedure, and one
measured qualification run. Rerun the two requests in the
[Verification](#verification) section when validating another H20 host before
comparing latency or peak memory.

## Supported model contract

| Task | Required input | Model output contract | Endpoint |
| --- | --- | --- | --- |
| T2V | Text prompt; omit `input_reference` | 1280x704, 121 frames, 24 fps | `POST /v1/videos/sync` |
| I2V | Text prompt and one image in `input_reference` | 1280x704, 121 frames, 24 fps | `POST /v1/videos/sync` |

The upstream checkpoint supports 720p generation at 24 fps. Use `1280x704` for
landscape output or `704x1280` for portrait output. Wan's temporal VAE expects
frame counts of the form `4k+1`; the commands below use 121 frames (about five
seconds at 24 fps), 50 denoising steps, and seed 42.

## References

- Upstream model card:
  <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>
- Online video API example:
  [`examples/online_serving/image_to_video`](../../examples/online_serving/image_to_video)
- Replica serving example:
  [`examples/online_serving/replica_data_parallel`](../../examples/online_serving/replica_data_parallel)
- Single-GPU deploy defaults:
  [`vllm_omni/deploy/wan2_2_ti2v.yaml`](../../vllm_omni/deploy/wan2_2_ti2v.yaml)
- Videos API reference: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)
- Supported model table: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Diffusion feature matrix: [`docs/user_guide/diffusion_features.md`](../../docs/user_guide/diffusion_features.md)
- Community recipe tracker:
  [vllm-project/vllm-omni#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware

- Accelerator model and per-device memory: NVIDIA H20, 97,871 MiB reported by
  `nvidia-smi`
- Number of devices: 1
- Device interconnect: not applicable to this single-GPU profile
- Host memory: 1.9 TiB installed (`free -h`); CPU offload was not used
- Qualification scope: BF16, batch size 1, TP=1, eager mode, VAE tiling,
  T2V and I2V at 1280x704 / 121 frames / 24 fps

The qualification host used driver `570.86.10`. Do not transfer latency or
memory numbers from the A100 recipe to this profile.

## Software environment

The qualification host used the following environment:

- OS: Ubuntu 22.04.5 LTS, Linux 5.15.0-91-generic
- Python: 3.12.13
- NVIDIA driver: 570.86.10
- PyTorch: 2.13.0+cu130
- Transformers / Diffusers: 5.14.1 / 0.40.0
- vLLM: 0.31.0
- vLLM-Omni: `0.31.0rc2.dev48+gc548a110a`, commit
  `c548a110a5a9bc278c39cfa81e673a8d505a24dd`

Record these values again on another H20 host before serving:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version
python - <<'PY'
import importlib.metadata as metadata

for package in ("torch", "vllm", "vllm-omni", "diffusers", "transformers"):
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PY
git rev-parse HEAD
```

Qualification used a local Diffusers snapshot of this model. The command also
accepts the public model ID or another local copy with the same Diffusers
layout.

## Command

Set `MODEL` to either the Hugging Face model ID or a local Diffusers directory
such as `/path/to/Wan2.2-TI2V-5B-Diffusers`. The local directory must contain
the `transformer`, `text_encoder`, `tokenizer`, `vae`, and `scheduler`
subdirectories.

Start the server from the repository root:

```bash
MODEL="${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --omni \
  --port 8092 \
  --deploy-config vllm_omni/deploy/wan2_2_ti2v.yaml
```

The deploy file selects one multiprocessing worker (`tensor_parallel_size: 1`),
eager execution, VAE tiling, and the default seed. Do not add a vLLM
`--tensor-parallel-size` or `--enable-expert-parallel` flag to this `--omni`
command; Omni parallelism is defined by the deploy configuration.

## Verification

Run the following from a separate shell after placing the reference image at
`$OUT/cherry_blossom.jpg`. The image used here is available at
<https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg>.
The output directory is ignored by Git, so generated videos, headers, and logs
do not become recipe changes.

```bash
set -euo pipefail

OUT="${OUT:-outputs/wan22-ti2v-h20}"
mkdir -p "$OUT"

curl --fail-with-body -D "$OUT/t2v.headers" \
  -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F "prompt=Two anthropomorphic cats in comfortable boxing gear and bright gloves fight intensely on a spotlighted stage, cinematic lighting, smooth motion." \
  -F "negative_prompt=blurry, low quality, static, distorted, artifacts, watermark, text" \
  -F "width=1280" -F "height=704" \
  -F "num_frames=121" -F "fps=24" \
  -F "num_inference_steps=50" -F "guidance_scale=5.0" \
  -F "flow_shift=5.0" -F "seed=42" \
  --output "$OUT/t2v.mp4"

curl --fail-with-body -D "$OUT/i2v.headers" \
  -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F "input_reference=@$OUT/cherry_blossom.jpg" \
  -F "prompt=Cherry blossoms sway gently in the breeze while petals drift across the path, cinematic, smooth natural motion." \
  -F "negative_prompt=blurry, low quality, static, distorted, artifacts, watermark, text" \
  -F "width=1280" -F "height=704" \
  -F "num_frames=121" -F "fps=24" \
  -F "num_inference_steps=50" -F "guidance_scale=5.0" \
  -F "flow_shift=5.0" -F "seed=42" \
  --output "$OUT/i2v.mp4"
```

Check the response status, inference time, reported peak memory, and encoded
video properties:

```bash
grep -iE '^(HTTP/|x-inference-time-s:|x-peak-memory-mb:)' \
  "$OUT/t2v.headers" "$OUT/i2v.headers"

for video in "$OUT/t2v.mp4" "$OUT/i2v.mp4"; do
  ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration \
    -of default=noprint_wrappers=1 "$video"
done
```

Both requests should return HTTP 200 and produce decodable MP4 files with
1280x704 video, 121 frames, and 24 fps. Record the two `X-Inference-Time-S`
and `X-Peak-Memory-MB` values from the headers. During one request per mode,
sample the physical device independently:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 | tee "$OUT/nvidia-smi.csv"
```

Stop the sampler after both warm requests complete. Report the maximum
`memory.used` value and the sampling interval with the results.

## Qualification results

The following single bounded run was measured after server startup and model
warmup, with one request per mode, 50 denoising steps, seed 42, batch size 1,
and a one-second `nvidia-smi` sampling interval. It is qualification evidence,
not a throughput benchmark; model loading and the approximately 148-second
server initialization are excluded from the request times.

| Task | HTTP | `X-Inference-Time-S` | `X-Peak-Memory-MB` | `nvidia-smi` peak | MP4 validation |
| --- | ---: | ---: | ---: | ---: | --- |
| T2V | 200 | 420.913 s | 25,502 MB | 26,132 MiB | H.264, 1280x704, 121 frames, 24 fps |
| I2V | 200 | 436.277 s | 29,394 MB | 30,024 MiB | H.264, 1280x704, 121 frames, 24 fps |

## Notes

- `--vae-use-tiling` is enabled by `vllm_omni/deploy/wan2_2_ti2v.yaml` to bound
  decode memory at 720p.
- `--enforce-eager` makes the single-request measurement repeatable by avoiding
  CUDA graph capture during the qualification run. Remove it only after a
  separate compiled-mode measurement.
- `flow_shift=5.0` is the upstream 720p setting. The 480p examples use a
  different flow-shift value and are not this profile.
- The checkpoint produces video only; it does not generate an audio track.
- The T2V MP4 was 5,810,451 bytes and the I2V MP4 was 9,481,007 bytes. Both
  were decoded through all 121 frames with PyAV; the first and last frames were
  visually inspected. T2V showed the prompted boxing cats, and I2V preserved
  the cherry-blossom scene while adding falling-petal motion.
- `X-Peak-Memory-MB` is the request-scoped engine metric. The `nvidia-smi`
  value is an independent one-second physical-device sample and includes the
  resident model.
- H20 throughput, latency, and memory depend on the driver, PyTorch/vLLM build,
  thermal state, and exact vLLM-Omni commit. Keep those values with the table.

## Supported features

| Feature | Status for this profile | Shared guide |
| --- | --- | --- |
| OpenAI-compatible Videos API | Qualified for synchronous T2V and I2V on one H20 | [Videos API](../../docs/serving/videos_api.md) |
| VAE tiling | Enabled and exercised by the single-GPU deploy config | [VAE parallelism and tiling](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Cache-DiT / TaylorSeer | Available but not enabled in this recipe | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| TeaCache | Available but not qualified in this recipe | [TeaCache](../../docs/user_guide/diffusion/cache_acceleration/teacache.md) |
| CPU or layerwise offload | Not needed for the 96GB target; not qualified here | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| TP / SP / CFG parallelism | Not used in this single-GPU profile | [Parallelism overview](../../docs/user_guide/diffusion/parallelism/overview.md) |
