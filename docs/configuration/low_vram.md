# Low-VRAM diffusion preset

The opt-in `fp8` preset composes existing memory optimizations for the native
Z-Image pipeline on one NVIDIA GPU. It is experimental:
configuration tests do not qualify an 8, 12, or 16 GB tier. Resolution, frame
count, host RAM, checkpoint loading peaks, and the desktop's GPU usage still
determine whether a request fits. FP8 also changes numerical output.

## One flag or one profile

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --low-vram-preset fp8
```

The bundled profile additionally pins stage 0 to device 0, limits concurrency
to one, and disables CUDA graphs:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni \
  --deploy-config vllm_omni/deploy/diffusion_low_vram.yaml
```

Use the profile path from a repository checkout. For a Python caller using
the shared `Omni` entrypoint, pass `low_vram_preset="fp8"`.

The preset supplies these **unset** diffusion settings after deploy and CLI
sources merge:

| Setting | Preset default |
| --- | --- |
| `quantization_config` | `fp8`, online quantization through the existing factory |
| `diffusion_offload_config` | `mode: layer`, components `dit` and `text_encoder` |
| `cache_backend` | `none` |
| `vae_use_tiling` | `true` |

Cache acceleration retains intermediate tensors and can increase memory use.
It is deliberately off by default, including when `DIFFUSION_CACHE_BACKEND`
is set. To opt into approximate caching for Z-Image:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --low-vram-preset fp8 \
  --cache-backend tea_cache --cache-config '{"rel_l1_thresh": 0.1}'
```

Explicit stage/CLI settings retain normal precedence. Compatible FP8 options
(for example, ignored layers) may replace preset defaults. Module offload is
rejected while the preset is enabled: the RTX 3090 qualification run generated
uniform gray images with FP8 plus module offload, whereas layer offload produced
normal images. The root cause remains unconfirmed; removing the preset does not
fix that underlying combination.

Layer offload bounds resident block weights and transfers them as needed.
It does not reduce every activation or guarantee a particular VRAM peak.
The preset does not configure TTS stages or automatic idle sleep (RFC A4).

## Known limitation: model-loading peak

Online FP8 quantization runs on the GPU. While weights load, components that
are not quantized (for Z-Image, the BF16 Qwen3 text encoder) are materialized
on the GPU before they are offloaded back to host memory. The startup peak is
therefore much higher than the serving footprint. In the measurement below it
was about 9.5 GiB during loading and about 3.9 GiB while serving requests.
Until loading is staged through host memory, the preset can serve within an
8 GB budget but cannot start within one. A pre-quantized checkpoint, or a
GPU with enough headroom for this loading peak, avoids the issue.

## Measured so far

The following is an exploratory data point, not a tier qualification. It was
measured on an RTX 3090 24 GB (SM86, so FP8 uses the weight-only Marlin
kernel) with Z-Image-Turbo revision `f332072a` at 512x512, 9 steps, guidance 0,
three prompts with seeds 42–44, eager mode and concurrency 1. Memory figures are
whole-device `nvidia-smi` samples taken every 0.5 s.

| Configuration | Load-phase peak (MiB) | Serving peak (MiB) | Process memory after load (GiB) | Request latency (s), first / steady |
| --- | ---: | ---: | ---: | --- |
| BF16 with module offload | 8,897 | 13,647 | 8.57 | 12.09 / 4.6 |
| Preset: FP8 with layer offload | 9,697 | 3,949 | 2.47 | 2.46 / 2.4 |
| FP8 with module offload (rejected) | 9,697 | 7,981 | 4.96 | 6.26 / 3.2 (uniform gray images) |

Compared against BF16, the preset's images score PSNR 19.7–26.4 dB,
SSIM 0.84–0.90 and LPIPS (AlexNet) 0.06–0.13. One of the three images meets all
of the exploratory thresholds that were set before the run (PSNR ≥ 20 dB,
SSIM ≥ 0.85, LPIPS ≤ 0.2). Visual review found the same subject and
composition, with minor differences in detail. With TeaCache at
`rel_l1_thresh=0.1`, the 9-step outputs were byte-identical to TeaCache off,
which suggests that no steps were skipped. This run therefore shows no
TeaCache benefit.

## Interaction matrix

These are configuration-policy results, **not measured quality or memory results**.
The existing backend still checks GPU capability and component support.

| Offload | Quantization | Cache | VAE tiling | Result |
| --- | --- | --- | --- | --- |
| Layer, rank-local | FP8 | None | On | Default for native Z-Image |
| Module | FP8 | Any | On | Rejected: invalid images observed during qualification |
| Layer | FP8 | TeaCache | On | Accepted for Z-Image; needs combined quality validation |
| Layer | GGUF, INT8, other methods | Any | Any | Rejected; remove preset to configure these independently |
| Layer | FP8 | Other backends | On | Rejected by this preset's initial scope |
| Any | FP8 | Any | Off | Rejected |
| Layer with AllGather / multiple GPUs | FP8 | Any | On | Rejected: single-GPU preset |
| Conflicting legacy offload flag | FP8 | Any | On | Rejected; select mode in `diffusion_offload_config` |
| Layer plus full compilation | FP8 | Any | On | Rejected; use regional compilation |
| Layer plus sleep mode | FP8 | Any | On | Rejected pending combined sleep/wake qualification |

Unsupported pipeline classes, non-native loaders, malformed cache JSON, and
incomplete component selections also raise descriptive configuration errors.
Without `low_vram_preset`, existing independent feature behavior is unchanged.

## Reproducible qualification

Before claiming a VRAM tier or quality tolerance, fix a workload and record:

1. Commit, model revision, GPU/VRAM, OS and WSL2 version, driver, torch,
   vLLM and vLLM-Omni versions; exact command and request parameters.
2. A BF16 reference with cache disabled, the same seeds/prompts, resolution,
   steps, frame count and concurrency. Declare PSNR/SSIM/LPIPS thresholds and
   visual acceptance criteria **before** evaluating approximate configurations.
3. Run layer offload with FP8 and tiling; for Z-Image repeat with
   TeaCache disabled/enabled. Compare FP8 without caching separately to isolate
   quantization error from combined quantization/cache error.
4. Record peak allocated and reserved GPU memory per stage and end-to-end,
   whole-device memory from `nvidia-smi`, and host memory. Keep OOMs and failed
   configurations in the report, including whether loading or inference failed.
5. Separate process cold start, first request, and steady-state latency.
   Sleep/wake is intentionally rejected until its combined lifecycle is qualified.

Use the existing [diffusion benchmark](gh-file:benchmarks/diffusion/README.md)
for workload generation and the [FP8 guide](../user_guide/quantization/fp8.md)
for backend prerequisites.
