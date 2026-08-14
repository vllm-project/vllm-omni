# VAE Parallelism Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Production Optimization Profiles](#production-optimization-profiles)
- [MiniMax-H3 Qualification](#minimax-h3-qualification)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

VAE parallelism distributes VAE (Variational AutoEncoder) decode/encode work across multiple GPUs. This guide covers VAE patch/tile parallelism, which splits latent space into spatial tiles or patches, and Wan spatial-shard decode, which shards decoder feature maps along height or width.

This is particularly useful for:
- **High-resolution image generation** where VAE decode can become a memory bottleneck
- **Memory-constrained environments** where the VAE decode activation peak exceeds available VRAM
- **Multi-GPU setups** where you want to leverage distributed resources for the VAE stage

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).


VAE patch parallelism uses two strategies based on image size:

| Strategy | Use Case | How It Works | Overlap Handling | Output Quality |
|----------|----------|--------------|------------------|----------------|
| **Tiled Decode** | Large images (triggers VAE tiling) | Distributes existing VAE tiling computation across ranks. Each rank decodes a subset of overlapping tiles. | Uses VAE's native `blend_v` and `blend_h` functions to seamlessly merge overlapping regions | Bit-identical (same logic as single-GPU tiling) |
| **Patch Decode** | Small images (no VAE tiling) | Splits latent into spatial patches with halos. Each rank decodes one patch with boundary context. | Halo regions provide edge context; core regions are directly stitched without blending | Near-identical (diff < 0.5%, visually imperceptible) |


Most VAE Patch Parallel implementations reuse the DiT process group (`dit_group`).
For architectures declared capable by runtime metadata, the generic runtime
also creates deterministic, contiguous VAE process groups sized by
`vae_patch_parallel_size`. MiniMax-H3 resolves its group while applying the
parallel configuration and therefore uses this subgroup without model-specific
code changes. This means:

- **Shared ranks**: VAE patch parallelism uses the same GPU ranks as DiT parallelism (Tensor Parallel, Sequence Parallel, etc.)
- **Combined usage**: VAE patch parallelism is typically used together with other parallelism methods
- **Configuration alignment**: `vae_patch_parallel_size` must evenly divide the diffusion world size
- **MiniMax-H3 independence**: the VAE group can be smaller than, and is independent of, the DiT TP/USP group

---

## Quick Start

### Basic Usage

Simplest working example:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

# TP=2 for DiT, VAE patch parallel also uses these 2 GPUs
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=2,          # Enable tensor parallelism for DiT
        vae_patch_parallel_size=2,       # Enable VAE patch parallelism
    ),
    vae_use_tiling=True,  # Required for VAE patch parallelism
)

outputs = omni.generate(
    "a futuristic city at sunset, high resolution, 8k",
    OmniDiffusionSamplingParams(
        num_inference_steps=9,
        height=1152,  # High resolution benefits from VAE patch parallel
        width=1152,
    ),
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/text_to_image/`:

```bash
# Text-to-Image with Z-Image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Tongyi-MAI/Z-Image-Turbo \
    --prompt "a futuristic city at sunset" \
    --height 1152 \
    --width 1152 \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-use-tiling
```

### Online Serving

You can enable VAE patch parallelism in online serving via `--vae-patch-parallel-size`:

```bash
# Text-to-Image with Z-Image, TP=2 + VAE patch parallel=2
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091 \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-use-tiling
```

### MiniMax-H3 with a Smaller VAE Group

MiniMax-H3 can use a VAE group that is smaller than its DiT parallel world. For
example, a four-rank `TP=2, USP=2` deployment can decode with two-rank VAE
groups. The deterministic rank partition is `[0, 1]` and `[2, 3]`:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
    --omni \
    --trust-remote-code \
    --num-gpus 4 \
    --tensor-parallel-size 2 \
    --usp 2 \
    --text-encoder-tp-size 4 \
    --vae-patch-parallel-size 2 \
    --vae-parallel-mode tile \
    --vae-use-tiling
```

Every rank belongs to exactly one VAE group. All ranks create the groups in the
same order, preventing collective-order mismatches during initialization.

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_patch_parallel_size` | int | 1 | Number of GPUs in each VAE patch/tile-parallel group. Set to 2 or higher to enable. It must evenly divide the diffusion world size. The runtime supplies the independent deterministic group during component configuration; legacy executors that bind their group during construction continue to reuse the DiT group. |
| `vae_parallel_mode` | str | `"tile"` | VAE parallel decode strategy: `"tile"` (default tile/patch parallel decode), `"spatial_shard_height"`, or `"spatial_shard_width"` (spatially-sharded decode, Wan only). See [Spatially-Sharded Decode](#spatially-sharded-decode-wan). |

Additional requirements:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_use_tiling` | bool | False | Must be set to `True` when using VAE patch parallelism. |

The capability-driven production runtime also accepts:

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `vae_optimization_profile` | `safe`, `optimized`, `diagnostic`, `student` | `safe` | Selects production-safe defaults and validation gates. |
| `vae_stack_tiling` | `auto`, `true`, `false` | Profile default | Batches a rank's native VAE tiles into fewer decoder calls. Explicit unsupported values fail during startup. |
| `vae_compile` | `auto`, `true`, `false` | Profile default | Compiles stable video-decoder regions. Each block has a bounded shape cache and falls back eagerly after a compilation/runtime failure. |
| `vae_compile_max_shape_buckets` | positive integer | 4 | Maximum compiled input-shape buckets per stable decoder block. Later shapes remain eager. |

!!! note "Automatic VAE Tiling"
    When `vae_patch_parallel_size > 1` and the model has a distributed VAE (`DistributedVaeMixin`), the system automatically sets `vae_use_tiling=True` if not already enabled.

## Production Optimization Profiles

| Profile | Stack tiles | VAE compile | VAE group | Intended use |
|---------|-------------|-------------|-----------|--------------|
| `safe` | Off | Off | Existing/configured | Default fallback and correctness reference |
| `optimized` | Validated `auto` | Validated `auto` | Qualified configured size | Production candidate |
| `diagnostic` | Explicitly configurable | Explicitly configurable | Configurable | Component timing and qualification |
| `student` | Model-specific | Model-specific | Model-specific | Post-training research; rejected unless a compatible artifact contract exists |

`safe` rejects attempts to turn on stacked tiles or VAE compile. Use
`optimized` for automatic selection, or `diagnostic` to force one fast path at
a time. `auto` falls back to the eager/sequential implementation when the
architecture or runtime cannot qualify the optimization. Explicit `true`
fails during startup when support is absent.

MiniMax-H3 declares native tiled decode, stacked tiles, bounded regional VAE
compile, and independent VAE process groups. It does not declare spatial
sharding. Wan declares tiled decode and spatial sharding, but not the H3 fast
paths.

Example optimized H3 server:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
    --omni --trust-remote-code \
    --num-gpus 4 --tensor-parallel-size 2 --usp 2 \
    --vae-patch-parallel-size 2 --vae-parallel-mode tile \
    --vae-optimization-profile optimized \
    --vae-compile-max-shape-buckets 4
```

Both stacked-tile `auto` and explicit `true` validate the latent shape, ensure
that each VAE rank has at least two tiles, and check accelerator memory
headroom. An explicit request still takes the sequential path when a particular
request shape or the available memory cannot safely use stacking. A stacked decode
failure clears the failed allocation and retries sequentially for that request.
The original tile mode is restored in a `finally` block, so subsequent requests
do not inherit failed state.

VAE compilation is limited to repeated `TransformerBlock` regions of the H3
video decoder. Audio VAE and tiling control flow remain eager. Each shape bucket
is compiled at most once; compilation or execution failure permanently routes
that bucket to eager execution, and shapes beyond the configured bucket bound
also remain eager. This path is independent of DiT compilation, so
`--enforce-eager --vae-compile true` keeps the DiT eager while compiling only
the qualified VAE regions.

## MiniMax-H3 Qualification

Enable `diagnostic` to return component stage durations without adding profiler
synchronization to the normal `safe` or `optimized` serving paths:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
    --omni --trust-remote-code \
    --vae-optimization-profile diagnostic \
    --vae-stack-tiling true \
    --vae-compile false
```

Diagnostic `stage_durations` include:

- `video_vae.decode_latent`
- `audio_vae.decode_latent`
- `video_vae.tiled_decode`
- `video_vae.tile_decode`
- `video_vae.tile_communication`
- `video_vae.tile_merge`
- the existing aggregate pipeline `decode`

Per-request diagnostic logs additionally record latent shape and SHA-256
fingerprint, VAE parallel size,
optimization/profile mode, tile count, cold/warm state, and whether sequential
fallback occurred. Existing response metadata reports peak device memory.

Use identical prompts, latent-producing seeds, shapes, and sampling settings for
the safe reference and candidate. The qualification tool records cold/first and
warm latency, component/VAE time, end-to-end time, peak memory, media hashes,
video PSNR, audio MAE, and a post-run recovery request:

```bash
# Run against the safe server first.
python benchmarks/diffusion/vae_optimization_benchmark.py \
    --profile-name safe \
    --output-dir /tmp/h3-vae-safe \
    --runs 3

# Restart with the candidate profile, then compare against the safe report.
python benchmarks/diffusion/vae_optimization_benchmark.py \
    --profile-name optimized \
    --output-dir /tmp/h3-vae-optimized \
    --reference-report /tmp/h3-vae-safe/safe-report.json \
    --runs 3 \
    --max-end-to-end-regression-pct 5 \
    --min-video-psnr-db 40 \
    --max-video-mae 1 \
    --max-video-seam-band-mae 1 \
    --max-video-seam-excess-ratio 1.25 \
    --server-log /path/to/server.log \
    --max-rank-imbalance-pct 15 \
    --max-audio-mae 0.01 \
    --max-av-sync-delta-s 0.1
```

The audio gate operates on normalized samples decoded from the served MP4.
The default `0.01` MAE accommodates the measured codec-level repeat variation;
release qualification should retain the stricter of this value and the
same-profile repeat baseline. The synchronization gate compares decoded video
and audio durations and permits at most 100 ms by default.

The H3 video defaults require at least 40 dB PSNR and at most one 8-bit code
value of global MAE. Seam MAE is measured only in four-pixel bands around the
native decoder tile boundaries derived from tile size 256, minimum overlap 64,
and VAE ratio 16; it must remain below one code value and no more than 1.25×
the non-seam MAE. Override the tile geometry flags when qualifying a different
artifact instead of treating the maximum error on an arbitrary image row or
column as a tile seam.

When both reports contain component timings, the report also computes VAE
share, VAE speedup, Amdahl-predicted end-to-end speedup, and observed
end-to-end speedup. This prevents a large decoder-only percentage from being
presented as the total serving gain. When diagnostic logs are supplied for both
runs, differing latent fingerprints fail qualification before output quality is
interpreted.

For Ref2VA, add `--task ref2va --input-reference /path/to/image-or-video`;
repeat `--input-reference` for mixed/multiple references and optionally pass
`--audio-reference URL`. Reference and candidate runs must use the same ordered
inputs.

Qualification must cover FL2VA and Ref2VA at every release-gated
resolution/duration. Include offload on/off, VAE group sizes, cold/first/warm,
tile seams, audio/video synchronization, and forced compile/OOM/collective
failure recovery. Do not promote `optimized` merely because startup succeeds.

### Decoder student research track

Decoder students stay outside the serving runtime. A candidate must provide a
versioned artifact manifest for the exact H3 video-VAE contract (24 latent
channels, 16× spatial ratio, 4× temporal ratio), checkpoint provenance, and a
`module:callable` offline runner. Evaluate the reference and student with the
same saved latent tensor:

```bash
python benchmarks/diffusion/h3_vae_student_evaluation.py \
    --reference-manifest /path/to/reference.json \
    --candidate-manifest /path/to/student.json \
    --latent /path/to/identical_h3_latent.pt \
    --output /tmp/h3-student-report.json \
    --warmups 1 --runs 3 \
    --min-psnr-db 50 \
    --min-decoder-speedup 1.0
```

The serving `student` profile intentionally fails without such a compatible,
post-trained artifact. Flash-VAED, Turbo-VAED, FlashDecoder, and models with a
different latent space are not accepted as H3 drop-in decoders.

---

## Spatially-Sharded Decode (Wan)

The default `vae_parallel_mode="tile"` distributes whole tiles across ranks. For the **Wan** VAE there is an alternative decode strategy, **spatially-sharded decode**, selected via `vae_parallel_mode="spatial_shard_height"` or `vae_parallel_mode="spatial_shard_width"`.

Instead of assigning independent tiles to ranks, spatial-shard decode shards the decoder feature maps along the height (`spatial_shard_height`) or width (`spatial_shard_width`) dimension and exchanges halo rows/columns between neighboring ranks around the spatial convolutions. This keeps the receptive field correct across shard boundaries, so the result matches the single-GPU decode within numerical tolerance.

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=2,
        vae_patch_parallel_size=2,               # must match the DiT group size
        vae_parallel_mode="spatial_shard_width", # or "spatial_shard_height"
    ),
)
```

Or from the CLI / serving entrypoint:

```bash
vllm serve Wan-AI/Wan2.1-T2V-1.3B-Diffusers --omni \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-parallel-mode spatial_shard_width
```

**Constraints and behavior:**

- Spatial-shard decode is **decode-only** and currently implemented for the **Wan** VAE. Other models ignore `spatial_shard_*` modes.
- It requires `vae_patch_parallel_size` to **match the DiT process group size**. If it does not, the VAE logs a warning and **falls back to tile-parallel decode** at runtime.
- `spatial_shard_height` and `spatial_shard_width` are mutually exclusive for a given VAE instance (the decoder is patched in place for a single split dimension).

For end-to-end latency/throughput, launch serving with the desired `vae_parallel_mode` and use the existing diffusion serving benchmark:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --endpoint /v1/videos --dataset random --task t2v --num-prompts 1 \
    --height 480 --width 832 --num-frames 17 --max-concurrency 1
```

---

## Best Practices

### When to Use

**Good for:**

- High-resolution image generation and long video generation
- Memory-constrained setups where VAE decode causes OOM
- Multi-GPU environments

**Not for:**

- Low-resolution images/videos where VAE decode is not a bottleneck
- Single GPU setups should use vae tiling decode, but not parallel vae tiling decode
- Models that do not support vae patch parallel

---

## Troubleshooting

### Common Issue 1: Model Not Support VAE Patch Parallel

**Symptoms**:
```
WARNING: vae_patch_parallel_size=2 is set but VAE patch parallelism is NOT enabled for xxxPipeline; ignoring.
```

**Root Cause**: VAE Patch Parallelism requires the model's VAE to implement `DistributedVaeMixin`. At startup, `vllm_omni/diffusion/registry.py` checks whether the instantiated pipeline has a `.vae` attribute that is an instance of `DistributedVaeMixin`. If it does not, the setting is silently ignored:

```python
vae_pp_size = od_config.parallel_config.vae_patch_parallel_size
is_distributed_vae = hasattr(model, "vae") and isinstance(model.vae, DistributedVaeMixin)
if vae_pp_size > 1 and not is_distributed_vae:
    logger.warning(
        "vae_patch_parallel_size=%d is set but VAE patch parallelism is NOT enabled for %s; ignoring.",
        vae_pp_size,
        od_config.model_class_name,
    )
```

**Solutions**:

1. **Use a supported model** (recommended): check [Supported Models](../../diffusion_features.md#supported-models) for the VAE-Patch-Parallel column.

2. To add support for a new model, implement `DistributedVaeMixin` on its VAE class (contributions are welcome).


### Common Issue 2: Invalid `vae_patch_parallel_size`

**Symptoms**: Worker startup fails before model-parallel groups are created or model weights are loaded.

**Root Cause**: The requested VAE group is larger than the diffusion world size or does not divide it evenly.

**Recommendation**: Choose a positive divisor of the diffusion world size. For example, a world size of 8 supports VAE group sizes 1, 2, 4, or 8.

Note that the size of DiT process group size equals to:
```text
dit_parallel_size = data_parallel_size
                  × cfg_parallel_size
                  × sequence_parallel_size
                  × pipeline_parallel_size
                  × tensor_parallel_size

```
_sequence_parallel_size = ulysses_degree × ring_degree_

---

## Summary

1. ✅ **Enable VAE Patch Parallelism** - Set `vae_patch_parallel_size`， `vae_use_tiling=True` in `DiffusionParallelConfig` to reduce VAE decode peak memory
2. ✅ **Use Long Sequence** - VAE patch parallelism benefits are most apparent at long sequence decoding
3. ✅ **Combine with other parallelism methods** - Suggest to use together with Tensor Parallel or CFG-Parallel for maximum memory savings
