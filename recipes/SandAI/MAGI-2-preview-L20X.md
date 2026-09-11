# MAGI-2 Preview — NVIDIA L20X

> Native text/image-to-video-and-audio generation with the released Preview stage

## Summary

- Vendor: SandAI
- Model: `sand-ai/MAGI-2-preview`
- Runtime: Native vLLM-Omni pipeline; no SandAI runtime import
- Modes: Offline shared examples and OpenAI-compatible video serving
- Hardware: NVIDIA L20X 140 GiB
- Recommended deployment: Four-GPU resident SP4 (`TP=1`, `SP=4`)
- Maintainer: Community

## Supported model contract

### Tasks and inputs

| Task | Text prompt | Media condition | Shared entrypoint |
| --- | --- | --- | --- |
| T2VA | Required | None | [`text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py) |
| I2VA | Required | Exactly one still image | [`image_to_video.py`](../../examples/offline_inference/image_to_video/image_to_video.py) |

The I2VA path preserves the source aspect ratio and applies MAGI-2's native
resize/pad transform. Video or audio conditions, multiple images, and fused
per-rank request batches are not supported.

### Output

| Category | Supported Preview specification |
| --- | --- |
| Duration | Exactly 10 seconds |
| Video | 125 frames at 12.5 fps |
| Native tiers | `272p` (448x256) and `540p` (896x512) |
| Optional final resize | `output_width` and `output_height` must be supplied together |
| Audio | 44.1 kHz stereo, generated with the video |
| Refiner | The separate 1080p refiner is not included in this integration |

### Profiles provided by this recipe

| Profile | Devices | Purpose | Qualification |
| --- | ---: | --- | --- |
| Resident SP4 | 4 | Recommended fidelity/default deployment | Released smoke and reference parity |
| Rank-local DLO SP4 | 4 | Full-quality lower-HBM transformer streaming | Released checkpoint, 100-step T2VA and I2VA |
| DLO DP4 / DP2SP2 | 4 | Concurrent request throughput | Bounded one- and four-step coverage |
| Ordinary layerwise | 1 | Memory-constrained single request | Released checkpoint, one and four steps |
| Online resident SP4 | 4 | OpenAI-compatible video API | Serving path covered |

## References

- Official model card: <https://huggingface.co/sand-ai/MAGI-2-preview>
- Architecture blog: <https://sand.ai/blog/magi-2-preview>
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Diffusion feature matrix: [`docs/user_guide/diffusion_features.md`](../../docs/user_guide/diffusion_features.md)

## Prepare the checkpoint

The native Preview path uses approximately 274 GiB on disk. The separate
refiner is not required.

```bash
export MAGI2_CKPT_ROOT=/path/to/MAGI-2-preview

hf download sand-ai/MAGI-2-preview \
  preview/ text_encoder/ vae/ turbo_vae/ stable-audio-open-1.0/ \
  --revision 2dea51b64db47ee5b4402d36fd90829a0c58913b \
  --local-dir "$MAGI2_CKPT_ROOT"
```

Install vLLM-Omni and ensure `ffmpeg` is on `PATH`. No second model runtime is
required.

## Hardware support

### Locally qualified NVIDIA configuration

| Item | Value |
| --- | --- |
| Accelerator | NVIDIA L20X |
| Per-device memory | 143,771 MiB (about 140.4 GiB) |
| Device count | 4; the one-device profile uses one of the same GPUs |
| Device interconnect | All-to-all NV18 links (`nvidia-smi topo -m`) |
| Host memory | 2.0 TiB |

The official SandAI runtime requires eight NVIDIA Hopper GPUs. That official
eight-GPU system was unavailable for this PR; compatible eight-worker
topologies pass configuration validation but are not locally runtime-qualified.
No H100, H200, B200, or B300 runtime claim is made here.

### Software environment

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, Linux 5.10.134 |
| Python | 3.12.13 |
| NVIDIA driver / CUDA runtime | 570.133.20 / 13.0 |
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.26.0 |
| vLLM-Omni | This PR branch; evidence commits `a5af2a8c` and `42bbfb57` |
| Precision | BF16 |

Set `MAGI2_DETERMINISTIC=1` before worker startup when deterministic kernels
are required. The setting is fixed for the worker lifetime.

Each worker process supports one MAGI-2 pipeline instance. Pipeline startup
sets process-wide deterministic state, so multi-pipeline tests and deployments
must use separate worker processes rather than constructing multiple instances
inside one process.

MAGI-2 uses vLLM's bundled FlashAttention 3 kernels by default on Hopper
(compute capability 9.x), FlashAttention 4 on supported Blackwell devices
with FA3 as the fallback, and FlashAttention 2 on earlier architectures. Set
`MAGI2_FLASH_ATTN_VERSION=2`, `3`, or `4` before worker startup to force an
available kernel for diagnostics or output comparisons.

## Offline generation

### Recommended four-GPU T2VA

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python examples/offline_inference/text_to_video/text_to_video.py \
  --model "$MAGI2_CKPT_ROOT" \
  --model-class-name Magi2Pipeline \
  --prompt "A red fox walks through fresh snow while wind moves the pine branches." \
  --height 512 --width 896 --num-frames 125 \
  --num-inference-steps 100 --fps 12.5 \
  --tensor-parallel-size 1 --ulysses-degree 4 \
  --extra-body '{"seconds":10,"resolution":"540p"}' \
  --output magi2_540p_t2va.mp4
```

For a load-and-kernel smoke, use one inference step. A one-step output is not a
quality evaluation.

### Recommended four-GPU I2VA

Use the same geometry and parallelism with the shared I2V entrypoint:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "$MAGI2_CKPT_ROOT" \
  --model-class-name Magi2Pipeline \
  --image /path/to/first_frame.png \
  --prompt "The fox looks up, then walks forward as snow falls around it." \
  --height 512 --width 896 --num-frames 125 \
  --num-inference-steps 100 --fps 12.5 \
  --tensor-parallel-size 1 --ulysses-degree 4 \
  --extra-body '{"seconds":10,"resolution":"540p"}' \
  --output magi2_540p_i2va.mp4
```

### One-device layerwise offload

MAGI-2 already stages Qwen, the image VAE, TurboVAE, and Oobleck from pinned
CPU memory. Ordinary layerwise offload streams the Preview DiT blocks so the
complete transformer does not need to fit in HBM at once.

```bash
export CUDA_VISIBLE_DEVICES=0

python examples/offline_inference/text_to_video/text_to_video.py \
  --model "$MAGI2_CKPT_ROOT" \
  --model-class-name Magi2Pipeline \
  --prompt "A red fox walks through fresh snow while wind moves the pine branches." \
  --height 256 --width 448 --num-frames 125 \
  --num-inference-steps 4 --fps 12.5 \
  --enable-cpu-offload --enable-layerwise-offload \
  --extra-body '{"seconds":10,"resolution":"272p"}' \
  --output magi2_272p_cpu_layerwise.mp4
```

Standalone `--enable-cpu-offload` is rejected because staging whole modules
does not make the complete Preview DiT fit on one qualified GPU. When both
flags are supplied, the shared offloader selects layerwise mode and MAGI's
native auxiliary staging remains active.

## Online serving

The OpenAI-compatible serving defaults are the full `540p` preset and `100`
denoising steps. The model always generates a 10-second, 125-frame clip at
12.5 fps; pass `resolution=272p` and `num_inference_steps=1` for a low-cost
load smoke. Requests with another duration are rejected before generation.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve "$MAGI2_CKPT_ROOT" --omni \
  --model-class-name Magi2Pipeline \
  --num-gpus 4 \
  --tensor-parallel-size 1 \
  --ulysses-degree 4 \
  --port 8091
```

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F 'prompt=A red fox walks through fresh snow.' \
  -F 'seconds=10' \
  -F 'size=896x512' \
  -F 'num_frames=125' \
  -F 'fps=12.5' \
  -F 'num_inference_steps=100' \
  -F 'seed=42' \
  -F 'extra_params={"resolution":"540p"}' \
  -o magi2_online.mp4
```

For I2VA, add `-F 'input_reference=@first_frame.png;type=image/png'`.

## Supported features

Use the shared guides for launch syntax and feature semantics. The flags below
are the MAGI-2-specific qualified combinations.

| Feature | MAGI-2 status / topology | Guide |
| --- | --- | --- |
| Sequence parallel | Resident SP4 default; compatible SP8 configuration | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) |
| Tensor parallel | TP4 or TP2SP2 on four workers | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| DLO | DP4/DP2SP2 AllGather; SP4 rank-local requires `--dlo-no-use-allgather` | [Distributed layerwise offload](../../docs/design/feature/distributed_layerwise_offload.md) |
| HSDP | HSDP4+SP4; alternative to TP and DLO | [HSDP](../../docs/user_guide/diffusion/parallelism/hsdp.md) |
| CFG parallel | CFG2xSP2 on four workers; CFG2xSP4 requires eight | [CFG parallel](../../docs/user_guide/diffusion/parallelism/cfg_parallel.md) |
| VAE patch parallel | TurboVAE tile decode across the complete worker group | [VAE parallelism](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Cache-DiT | Repeated Preview transformer layers; approximate | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| One-device offload | Ordinary layerwise DiT plus native auxiliary CPU staging | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| Quantization | Not supported | [Feature matrix](../../docs/user_guide/diffusion_features.md) |
| Ring / pipeline parallel | Not supported | [Feature matrix](../../docs/user_guide/diffusion_features.md) |

DP request waves must contain exactly `data_parallel_size` requests with the
same explicit step count. Data-parallel replicas require TP=1. SP-only DLO
cannot AllGather because SP ranks own different MoE-head shards.

## Request fields

Common geometry and sampling values use the shared CLI flags. The native
Preview pipeline accepts these model-specific `--extra-body` fields:

| Field | Meaning |
| --- | --- |
| `seconds` | Must be `10`. |
| `resolution` | `272p` or `540p`; default `540p`. |
| `output_width`, `output_height` | Optional final resize; supply both. |
| `deterministic` | Must match `MAGI2_DETERMINISTIC` at worker startup. |

Use `--image` in the shared I2V entrypoint rather than the lower-level
`image_path` field.

## Verification

```bash
ffprobe -v error \
  -show_entries stream=codec_type,width,height,avg_frame_rate,nb_frames,sample_rate,channels \
  -of json magi2_540p_t2va.mp4
```

A 540p result contains 125 896x512 frames at 12.5 fps and stereo 44.1 kHz
audio. A 272p result contains 125 448x256 frames with the same duration, rate,
and audio contract.

## Local qualification evidence

### Released-checkpoint E2E

| Profile | Workload | Steps | E2E | Peak HBM | Peak host PSS | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| DLO SP4 | 540p T2VA | 100 | 563.855 s | See PR evidence | See PR evidence | `3437d078e...f16fdea` |
| DLO SP4 | 540p I2VA | 100 | 556.533 s | See PR evidence | See PR evidence | `b8c2a4ac...ff2de8` |
| Layerwise 1 GPU | 272p T2VA | 1 | 8.86 s | 49.29 GiB | Not sampled | Valid video + audio |
| Layerwise 1 GPU | 272p T2VA | 4 | 42.55 s | 49.29 GiB | 371.97 GiB | Valid video + audio |

The one-device four-step host-PSS sampler scans the full process tree and adds
observable CPU overhead, so its stage timing is qualification evidence rather
than a latency benchmark.

### Distributed correctness

- Resident TP4, TP2SP2, and SP4 match a single-rank native oracle within the
  documented BF16 tolerance; resident SP4 is the reference-aligned default.
- Deterministic rank-local DLO SP4 output matched resident SP4 exactly.
- HSDP4+SP4, HSDP4+CFG2xSP2, Cache-DiT forced hits, and TurboVAE PP4 have
  focused four-rank coverage. TurboVAE PP4 matched resident decode exactly.
- Full 100-step 540p quality generation is qualified only for four-device DLO
  SP4. Eight-worker layouts remain configuration-only on this host.

These are bounded qualification runs, not committed benchmarks. Generated
artifacts stay outside the repository, and this integration adds no
model-specific example or benchmark script.
