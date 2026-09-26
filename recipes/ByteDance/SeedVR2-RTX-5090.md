# SeedVR2 video restoration — RTX 5090

## Summary

- Vendor: ByteDance
- Model: SeedVR2 3B
- Task: restore or upscale an uploaded video
- Mode: native offline/HTTP pipeline, one Euler step
- Hardware: NVIDIA GeForce RTX 5090, 32 GiB per device
- Maintainer: 0z5a

## When to use this recipe

Run short, constant-frame-rate videos with the native SeedVR2 pipeline. This
recipe covers the whole-clip path. Long-video batching, overlap, and color
correction from external applications are separate execution semantics.

## Supported model contract

| Property | Contract |
| --- | --- |
| Model directory | See [checkpoint files and hashes](../../docs/models/seedvr2.md#model-directory) |
| Input | One uploaded video; blank prompt; positive dimensions divisible by 16 |
| Output | MP4 at requested geometry, original frame count and source frame rate |
| Audio | First mono/stereo track, aligned by PTS; re-encoded as AAC |
| Sampling | One Euler step, CFG=1, per-request seed |
| Precision | 3B DiT and VAE FP16 |
| Parallelism | Head-sharded Ulysses window attention at SP>1 via `ulysses_degree`; replicated weights |
| Frame padding | Internal 4n+1 padding, cropped back; five returns five, six returns six |

## References

- [Canonical SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- [Model guide](../../docs/models/seedvr2.md)
- [Integration RFC](https://github.com/vllm-project/vllm-omni/issues/7723)

## Hardware

RTX 5090 with 32 GiB per GPU, PCIe, one GPU for the default command or two/four
for SP. Each GPU must fit a complete model copy and its activations. Host RAM
must accommodate checkpoint staging per rank. Admission uses a padded
five-frame 848×480 output pixel budget by default, allowing longer clips at
smaller resolutions, plus a 257-frame decoder-work cap. The validated SP4
configuration with VAE tiling and `vae_patch_parallel_size=4` uses a padded
five-frame 2560×1472 budget. The model guide lists exact bounds and which
profiles have completed GPU validation.

## Software environment

Linux x86-64, Python 3.12, PyTorch 2.13.0+cu130, CUDA 13.0, vLLM 0.29.0, and
PyAV in the existing environment. Use the vLLM-Omni branch containing this recipe.

## Command

Set `MODEL_DIR` to the authorized checkpoint directory documented in the model
guide; it must also contain `model_index.json` selecting `SeedVR2Pipeline`.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" --omni \
  --model-class-name SeedVR2Pipeline --dtype float16 --enforce-eager \
  --num-gpus 1 --host 127.0.0.1 --port 8098
```

For SP=2, expose two GPUs and replace the GPU count with:

```bash
--num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":2}}'
```

Use degree 4 and four visible GPUs for SP=4. Combine any feature-specific
settings below into the same stage-overrides object; do not repeat that flag.
The validated 16-frame 720p original-size profile is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_DIR" --omni \
  --model-class-name SeedVR2Pipeline --dtype float16 --enforce-eager \
  --vae-use-tiling --num-gpus 4 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":4,"vae_patch_parallel_size":4,"vae_parallel_mode":"spatial_shard_height"}}' \
  --host 127.0.0.1 --port 8098
```

Use `size=1280x720` for a landscape 1280×720 input or `size=720x1280` for a
portrait 720×1280 input. Both require 16 decoded source frames and retain the
source frame rate; no upscaling is requested.

SP>1 automatically selects the specialized window attention runtime; no separate
attention flag is needed.

## Verification

```bash
curl --fail-with-body http://127.0.0.1:8098/v1/videos/sync \
  -F 'prompt= ' -F 'input_references=@input.mp4;type=video/mp4' \
  -F 'size=224x128' -F 'num_inference_steps=1' \
  -F 'guidance_scale=1' -F 'seed=7723' --output restored.mp4
python - <<'PY'
import av
with av.open('restored.mp4') as video:
    stream = video.streams.video[0]
    print(stream.width, stream.height, stream.average_rate,
          sum(1 for _ in video.decode(video=0)), len(video.streams.audio))
PY
```

Expect HTTP 200 and a completely decodable 224×128 video. Check the frame count,
FPS, timestamps, and audio against the input. Repeat the same seed and compare
decoded frames; changing the seed should change the restored video. Inspect
matching first/middle/last frames at the same display scale. AAC is re-encoded,
so input/output audio packet hashes need not be identical.

The released 3B checkpoint also passed these four-GPU SP4 HTTP cases with the
configuration above. Each input had 16 frames at 24 FPS; the output retained
the original size, 16 decoded frames, timestamps, and decodable audio. All four
workers and the server exited normally after each run.

| 1× input → output | Earlier six-frame admission | Revised HTTP result | Request time |
| --- | --- | --- | ---: |
| 1280×720, 16 frames → 1280×720 | Rejected | HTTP 200; media checks passed | 3.553 s |
| 720×1280, 16 frames → 720×1280 | Rejected | HTTP 200; media checks passed | 3.388 s |

These are single requests on a shared host, not performance comparisons.

For transformer-only SP=1/2/4 parity, set `VLLM_TEST_SEEDVR2_MODEL` to the 3B
safetensors file and run:

```bash
python -m pytest -o addopts='' -v tests/diffusion/models/seedvr2/test_seedvr2_e2e.py
```

This checkpoint-gated test does not validate the HTTP server or an optional
optimization. Feature validation must use its enabled configuration, full model
outputs, and the actual backend selected by the worker.

## Supported features

| Feature | Status |
| --- | --- |
| [Window SP](../../docs/models/seedvr2.md) | Model-local regular/shifted window attention |
| RoPE table cache | Reuses window-local angle tables without a device-to-host cache-key read |
| Grouped SDPA index cache | Reuses per-layout row indices across layers; packed-varlen is separate |
| W8A8 video projections | Optional `additional_config.seedvr2_activation_quantization=fp8` or `int8`; no default speedup claim |
| CPU offload, LoRA, compiled execution, CFG/TP/PP | Unsupported |
| VFR or multichannel audio | Unsupported |
| VAE temporal/spatial tiling | Available with `--vae-use-tiling` |
| Quantization | Experimental opt-in W8A8 video projections |

## Measurement scope

Compare identical inputs, seeds, dimensions, GPU counts, and warmups. Separate
correctness from performance: use repeated interleaved HTTP measurements before
claiming a stable gain. Keep raw logs, tensors, and generated media outside the
source diff; link selected visuals from the PR's test results.

## Temporal tiling and VAE patch parallelism

Use `--vae-use-tiling` for causal temporal chunks. On a single GPU this also
bounds large convolution workspace with exact spatial halos. To shard VAE
activations across the same two SP ranks, use:

```bash
--vae-use-tiling --num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":2,"vae_patch_parallel_size":2,"vae_parallel_mode":"spatial_shard_height"}}'
```

The VAE patch degree must equal the SP degree. Height sharding is supported;
width sharding and batch slicing are not. Compare clips that cross a temporal
chunk boundary as well as five/six-frame clips, and inspect frames adjacent to
chunk boundaries. Reduced peak memory does not by itself establish lower latency.
High-resolution clips can still exceed device capacity; validate the intended
frame count and output size on the target GPUs.
