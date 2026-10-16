# Helios for video generation

## Summary

- Vendor: Helios
- Model: `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled`
- Task: Text-to-video generation
- Mode: Offline inference with the shared text_to_video example
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for running Helios
video generation with vLLM-Omni. The concrete command below focuses on
`BestWishYsh/Helios-Base` text-to-video generation on one NVIDIA H20 GPU via the
shared `text_to_video.py` example. The same example also supports Helios-Mid and
Helios-Distilled through the generic `--extra-body` flag (see below).
Image-to-video and video-to-video require image/video conditioning inputs and
are not covered by the text-to-video example.

## References

- Upstream repository: <https://github.com/PKU-YuanGroup/Helios>
- Model weights:
  <https://huggingface.co/BestWishYsh/Helios-Base>
- Related example under `examples/`:
  [`examples/offline_inference/text_to_video/text_to_video.md`](../../examples/offline_inference/text_to_video/text_to_video.md)

## Hardware Support

## GPU

### 1x NVIDIA H20

#### Environment

- OS: Ubuntu Linux x86_64
- Python: 3.12.12
- Driver / runtime: NVIDIA driver `580.126.20`, CUDA `13.0`
- Hardware: 1x NVIDIA H20 GPU from an 8x H20 host
- vLLM version: `0.19.0`
- vLLM-Omni version or commit: `a3903810`

#### Command

Run the baseline Helios-Base text-to-video example from the repository root:

```bash
cd examples/offline_inference/text_to_video

python text_to_video.py \
  --model BestWishYsh/Helios-Base \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
  --guidance-scale 5.0 \
  --output helios_t2v_base.mp4
```

To use cache-dit acceleration, run on a vLLM-Omni checkout that includes Helios cache-dit support and enable the cache backend:

```bash
cd examples/offline_inference/text_to_video

python text_to_video.py \
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --model BestWishYsh/Helios-Base \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
  --guidance-scale 5.0 \
  --output helios_t2v_base.mp4
```

#### Verification

The script should print the resolved generation configuration, total generation
time, and output path. A successful run ends with output similar to:

```text
Total generation time: <seconds> seconds (<milliseconds> ms)
Saved generated video to helios_t2v_base.mp4
```

#### Important flags and deploy config

- `--model BestWishYsh/Helios-Base` selects the base Helios checkpoint used by
  this recipe.
- `--guidance-scale 5.0` matches the Helios-Base recommendation. Use
  `--guidance-scale 1.0` for Helios-Distilled.
- `--cache-backend cache_dit` enables the cache-dit acceleration path.
- `--enable-cache-dit-summary` prints cache-dit summary information after
  diffusion forward passes.
- No separate deploy config is required for this offline recipe; the shared
  `text_to_video.py` example configures the pipeline through its arguments.
- Helios-specific knobs (declared in `vllm_omni/model_extras/helios.py`) are
  passed via the generic `--extra-body` JSON flag:
    - Helios-Mid: `--extra-body '{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [20, 20, 20], "use_cfg_zero_star": true, "use_zero_init": true, "zero_steps": 1}'`
    - Helios-Distilled: `--extra-body '{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [2, 2, 2], "is_amplify_first_chunk": true}'`

#### Known limitations

- Helios generates video in 33-frame chunks. For best performance, set
  `--num-frames` to a multiple of `33`; non-multiple values are rounded up to
  the nearest multiple of `33`.

### Text encoder online FP8

Enable the UMT5 text encoder explicitly with
`--quantization-config '{"text_encoder":{"method":"fp8"}}'`.
This quantizes the gated FFN input projections (`wi_0`, `wi_1`) using
vLLM's native online FP8 weights and dynamic activations. Attention, FFN
output projections, embeddings and norms retain their original precision.
The video transformer and VAE also retain their original precision.
The checkpoint must contain BF16 or FP16 weights; static activation scales
and pre-quantized FP8 text encoder checkpoints are not supported.

For a distilled video with two-way sequence parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/offline_inference/text_to_video/text_to_video.py \
  --model BestWishYsh/Helios-Distilled \
  --model-class-name HeliosPipeline \
  --quantization-config '{"text_encoder":{"method":"fp8"}}' \
  --ulysses-degree 2 --enable-layerwise-offload --vae-use-tiling \
  --height 384 --width 640 --num-frames 33 --guidance-scale 1.0 \
  --extra-body '{"is_enable_stage2":true,"pyramid_num_inference_steps_list":[2,2,2],"is_amplify_first_chunk":true}' \
  --seed 42 --enforce-eager --output helios_encoder_fp8.mp4
```

FP8 requires an SM89+ NVIDIA GPU. Compare complete video requests against
BF16 with identical prompts, seeds, generation settings and offload settings;
encoder weight compression alone does not establish an E2E speedup.

### 2× L20 validation

Tested on 2026-09-20 with `BestWishYsh/Helios-Distilled` revision
`b991c0379a018f4de3227d95468237f56066f5bb` and implementation `d5d7ae021`.
Two NVIDIA L20 GPUs (46,068 MiB each), Ulysses=2, eager execution,
DiT layerwise offload, resident text encoder, VAE tiling, default FLASH_ATTN.
PyTorch 2.13.0+cu130, vLLM 0.29.0, transformers 5.14.1, diffusers 0.40.0;
driver 595.91.07. Both modes use the same implementation and settings.

Each mode ran one warmup plus six measured full requests: three prompts,
repeated twice, seed 42, 33 frames at 384×640 and 24 FPS, guidance 1.
Distilled stage 2 uses `[2,2,2]` with first-chunk amplification (12 steps).
Loading, startup kernel initialization and video export are excluded from
request latency. Memory is the maximum measured allocated/reserved CUDA
memory per worker, excluding other processes.

| Metric | BF16 | Encoder FP8 | Observed comparison |
| --- | ---: | ---: | --- |
| E2E mean ± sample SD (s) | 50.551 ± 9.335 | 43.795 ± 7.821 | 1.154× |
| Rank-0 encoder mean (ms) | 54.696 | 52.268 | 1.046× |
| Rank-0 peak allocated (GiB) | 14.659 | 12.784 | 1.875 GiB lower |
| Rank-1 peak allocated (GiB) | 14.659 | 12.785 | 1.874 GiB lower |
| Rank-0 peak reserved (GiB) | 15.123 | 13.248 | 1.875 GiB lower |
| Rank-1 peak reserved (GiB) | 15.123 | 13.770 | 1.354 GiB lower |
| Native FP8 linear layers per rank | 0 | 48 | FFN inputs only |

These are observed ratios on a shared host. The large timing variance and
the much smaller encoder-time change do not establish an isolated E2E
quantization speedup. The model uses FP8 for 2,013,265,920 FFN input weights;
attention, FFN outputs and other encoder weights remain BF16. DiT is BF16
and VAE is FP32. Offloaded DiT parameter snapshots only show materialized
blocks, not the complete model size.

All 14 exported videos decode to 33 RGB frames. Repeated outputs match
exactly within each precision mode. Three paired prompt checks:

| Prompt | Mean frame SSIM | Visual observation |
| --- | ---: | --- |
| Golden retriever walking in a sunny meadow | 0.7814 | Dog, meadow and motion retained; texture/pose differences |
| Red sailboat, white sail, blue lake and green hills | 0.8884 | Objects and motion retained; layout/detail differences |
| Two yellow rubber ducks in a turquoise pool | 0.8589 | Both modes generate one duck; baseline count failure retained |

This is three-prompt smoke coverage, not a general quality evaluation.
Cross-attention precomputation is disabled with DiT offload because it
bypasses block transfer hooks; the projected-prompt cache is disabled too.
The earlier cached run reused stale conditioning and is excluded above.
The BF16 shutdown emitted a worker-termination warning after all outputs
were saved; both processes exited before the next run.

Measured E2E samples, in seconds:

- BF16: `58.745, 49.358, 37.028, 42.198, 56.598, 59.378`.
- FP8: `59.126, 42.183, 37.639, 43.867, 39.209, 40.745`.

Use the command above for each of these prompts, then repeat the same
sequence without `--quantization-config`. For repeated timing, retain one
`Omni` instance per precision mode, warm it up once, and measure six
`generate` calls with a newly seeded sampling configuration each time.
