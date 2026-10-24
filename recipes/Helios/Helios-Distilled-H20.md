# Helios-Distilled attention backends on H20 (historical runtime)

## Summary

- Vendor: Helios
- Model: `BestWishYsh/Helios-Distilled`
- Task: Text-to-video generation
- Mode: Offline, one request at a time
- Hardware: One NVIDIA H20, 96 GB
- Maintainer: Community
- Runtime scope: Historical H20 experiment at `b63e35a` (2026-09-27)

## When to use this recipe

This recipe records a dense attention comparison on Helios-Distilled at runtime
commit `b63e35ab4ffae2f78b556150943ec0e08a444030`, with `fa3-fwd==0.0.3`
installed. All recorded timings, memory figures, output-alignment results and
backend recommendations apply to that historical H20 environment. Current main
selects its provider according to the hardware, attention path and optional
packages; its performance needs separate measurements. This recipe does not
measure concurrent serving throughput or compare the Distilled checkpoint with
Helios-Base.

## Supported model contract

| Item | Profile |
| --- | --- |
| Input | One text prompt and seed per request; empty negative prompt |
| Output | RGB video, 640 × 384, 33 or 66 frames, exported at 16 FPS |
| Sampling | DMD pyramid stages `[2, 2, 2]`, first-chunk amplification enabled, guidance 1.0 |
| Precision | BF16 transformer and text encoder; FP32 Wan VAE |
| Execution | Eager request mode, one GPU, TP/SP/CFG degrees of 1 |
| Excluded | I2V/V2V conditioning, step streaming, batching, quantization, Cache-DiT/TeaCache, CPU offload |

The first-chunk amplification doubles each pyramid stage's step count. The
33-frame profile therefore executes 12 transformer forwards; the 66-frame
profile adds six forwards for the second chunk. The nominal
`--num-inference-steps` argument does not override the explicit pyramid list.

## References

- [Upstream Helios](https://github.com/PKU-YuanGroup/Helios)
- [Checkpoint](https://huggingface.co/BestWishYsh/Helios-Distilled)
- [Helios A2 RFC](https://github.com/vllm-project/vllm-omni/issues/8173)
- [Shared text-to-video example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [Dense attention backends](../../docs/user_guide/diffusion/attention_backends/dense_backends.md)
- [Diffusion feature matrix](../../docs/user_guide/diffusion_features.md)

## Checkpoint setup

Pin the model to revision `b991c0379a018f4de3227d95468237f56066f5bb`:

```bash
hf download BestWishYsh/Helios-Distilled \
  --revision b991c0379a018f4de3227d95468237f56066f5bb \
  --exclude 'transformer_ode/*' \
  --local-dir /data/Helios-Distilled
```

The required original assets occupy approximately 80.5 GB. The native pipeline
loads `transformer`, `text_encoder`, `tokenizer`, `scheduler`, and `vae`;
`transformer_ode` is not used. Budget additional space for the environment and
approximately 8 GB of float-frame evidence when running all three backends
with the default benchmark matrix.
For a disk-only setup, budgeting at least 120 GB for checkpoint, environment,
and results avoids relying on RAM-backed weight staging. This is a storage
budget estimate, not a measured minimum RAM or GPU requirement.

## Hardware

- Accelerator: One NVIDIA H20, 97,871 MiB reported by `nvidia-smi`.
- Host: 16 CPU cores and 150 GB RAM allocated by the provider.
- Interconnect: No inter-GPU communication is used.
- Scope: Single-device offline generation; other GPUs require separate validation.

## Historical software environment

The following versions describe the September 27 experiment at `b63e35a`.

| Component | Validated version |
| --- | --- |
| OS / Python | Ubuntu 22.04 / Python 3.12.3 |
| NVIDIA driver / CUDA runtime | 580.105.08 / 13.0 |
| PyTorch / cuDNN | 2.13.0+cu130 / 9.20.0 |
| vLLM | 0.30.0 |
| vLLM-Omni runtime commit | `b63e35ab4ffae2f78b556150943ec0e08a444030` |
| Transformers / Diffusers | 5.14.1 / 0.40.0 |
| Triton / kernels | 3.7.1 / 0.16.1 |
| FlashAttention 3 forward package | `fa3-fwd==0.0.3` |
| NumPy / scikit-image | 2.3.5 / 0.26.0 |

Use the [source installation guide at the measured runtime](https://github.com/vllm-project/vllm-omni/blob/b63e35ab4ffae2f78b556150943ec0e08a444030/docs/getting_started/installation/README.md)
with the versions above. `fa3-fwd==0.0.3` is part of this historical environment,
not an installation recommendation for current main. At `b63e35a`, the H20 CUDA
resolver tried `fa3_fwd_interface` before other FlashAttention providers. The
benchmark recorded the selected attention wrapper; newer benchmark helpers also
record the resolved `flash_attn_func`, `flash_attn_varlen_func` and
`optional_fa3_fwd_dense_func` module/qualified names under
`attention_implementations.flash_attention_bindings`. A binding absent from an
older runtime is recorded as `null`, so these helpers also work on `b63e35a`.
The September 27 artifact predates this field: it records wrapper counts and
installed package versions, but does not independently establish the bound
provider. Its historical metadata and script hashes are preserved.

### Main after #9201: optional Hopper dense FA3

[PR #7735](https://github.com/vllm-project/vllm-omni/pull/7735) removed
`fa3-fwd` as a required CUDA dependency.
[PR #9201](https://github.com/vllm-project/vllm-omni/pull/9201) restored it as an
optional dense kernel on Hopper (SM90), including H20. For this recipe's BF16,
unmasked Helios self- and cross-attention calls, an importable `fa3-fwd` supplies
`fa3_fwd_interface.flash_attn_func` through `fa.optional_fa3_fwd_dense_func`.
Without that optional binding, these H20 calls use the available
`vllm.vllm_flash_attn` dispatcher. The optional dense route excludes FP16;
masked, packed, piecewise and FP8 paths keep their separate dispatch routes.

`flash_attn_func` can remain `None` even when the optional dense function is
bound, so record all three callable bindings. On the vLLM dispatcher route,
the selected kernel depends on vLLM and device configuration; the module name
alone does not establish a particular FlashAttention version.

With `fa3-fwd` installed, these current-main Helios calls can use the same dense
entry point preferred by the historical `b63e35a` resolver. That does not make
the September timings a measurement of current main. For a new comparison,
rerun all three backends on the same runtime and record the resolved bindings
and software environment.

## Reproduce the historical comparison

To run against the measured runtime, create a separate checkout and copy this
PR's benchmark/analysis helpers and saved summary into it. From this PR's
repository root:

```bash
git worktree add --detach ../vllm-omni-helios-b63e35a \
  b63e35ab4ffae2f78b556150943ec0e08a444030
cp benchmarks/diffusion/benchmark_helios_attention.py \
  benchmarks/diffusion/compare_helios_attention.py \
  benchmarks/diffusion/export_helios_attention.py \
  ../vllm-omni-helios-b63e35a/benchmarks/diffusion/
cp recipes/Helios/Helios-Distilled-H20-results.json \
  ../vllm-omni-helios-b63e35a/recipes/Helios/
cd ../vllm-omni-helios-b63e35a
```

In a separate environment, install this checkout using the historical source
installation guide and version table above, including `fa3-fwd==0.0.3`. The
copied helpers add instrumentation; they do not replace the pinned runtime's
attention resolver. They contain review improvements, so new result provenance
must record their actual script hashes rather than reuse the historical hashes.

This is a Helios-specific offline harness for the fixed A2 workload. Follow-up
Helios comparisons can reuse it with the same measurement contract. Its location
does not settle RFC #8173's shared-harness question: serving, TTFF and concurrency
measurements still require a separately agreed common interface.

Run from the repository root, using a fresh process for each backend:

```bash
for backend in TORCH_SDPA FLASH_ATTN CUDNN_ATTN; do
  python -m benchmarks.diffusion.benchmark_helios_attention \
    --model /data/Helios-Distilled \
    --backend "$backend" \
    --frames 33 66 \
    --seeds 42 7 123 \
    --warmup 1 \
    --repeats 3 \
    --output-dir "helios-attention/$backend"
done
```

Each process performs one excluded warmup per frame count, then three
repetitions of three fixed prompt/seed pairs. The generator is recreated for
every request. Do not run GPU workloads concurrently with the benchmark.
The reported experiment uses the backend order shown above on one device;
it does not include randomized backend order or independent multi-session runs.

The output directory contains:

- `results.json`: package versions, workload, every request's wall time,
  pipeline stage timers, per-forward device timings and latent shapes, worker
  peak reserved memory, and summaries.
- `.npy` files: normalized float32 RGB frames for every measured request.
- `.mp4` files: generated examples from the first repetition of each case.

Wall time surrounds `Omni.generate`. It includes encoding, denoising, VAE
decoding, and output transfer; it excludes engine initialization, disk writes,
and MP4 encoding. Stage timers and transformer forward hooks are enabled consistently for every
backend. Each transformer forward is bracketed by device events on the worker
stream, with synchronization only when collecting the completed request. These
intervals include device-stream gaps inside the forward, and are not a sum of
CUDA kernel durations. The script checks the expected 12/18 forward counts.
RPC collection happens outside the wall-time interval. Mean forward time divides
the measured transformer total by the actual forward count; it averages the
three pyramid resolutions and is not a full-resolution-only step measurement.
The memory metric is the diffusion worker's peak **reserved** allocator
memory for the request, not total device usage or parent-process allocation.

`TORCH_SDPA` lets PyTorch choose a kernel. It is not a forced math-only
reference. `CUDNN_ATTN` explicitly pins cuDNN; the installed implementation
behind `FLASH_ATTN` must be recorded with the environment. Compare raw `.npy`
frames, not compressed MP4s, and compare repeated runs within each backend
before attributing output differences to the backend choice.

## Measured latency and memory

Measured on 2026-09-27 at runtime `b63e35a` with `fa3-fwd==0.0.3` installed
and the historical environment above. Each row summarizes nine
requests (three prompt/seed pairs, three repetitions), after one excluded
warmup for that frame count. All 54 measured requests completed. Values in
parentheses are the minimum and maximum request wall times.

| Backend | Frames | Wall time, ms: median (range) | Transformer GPU ms, median | Mean ms/forward, median | Peak worker reserved, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `TORCH_SDPA` | 33 | 23510.5 (23501.8–23526.0) | 21828.4 | 1819.0 | 45.99 |
| `FLASH_ATTN` | 33 | 21362.0 (21354.7–21367.5) | 19676.3 | 1639.7 | 45.99 |
| `CUDNN_ATTN` | 33 | 21364.2 (21355.0–21374.2) | 19680.7 | 1640.1 | 45.99 |
| `TORCH_SDPA` | 66 | 36045.2 (36028.8–36080.7) | 32734.4 | 1818.6 | 46.47 |
| `FLASH_ATTN` | 66 | 32820.8 (32802.0–32843.0) | 29508.8 | 1639.4 | 46.47 |
| `CUDNN_ATTN` | 66 | 32827.0 (32813.5–32843.0) | 29514.0 | 1639.7 | 46.47 |

On `b63e35a`, relative to `TORCH_SDPA`, `FLASH_ATTN` (with `fa3-fwd==0.0.3` installed) reduced median request latency by **9.14%** for
33 frames and **8.95%** for 66 frames. Transformer device time fell by about
9.85%. cuDNN delivered effectively the same latency as `FLASH_ATTN`: their differences
of 2.2/6.2 ms are smaller than the observed within-backend ranges. Peak worker
reserved memory was unchanged. This is an existing-backend comparison on the
same runtime, not a before/after production-code optimization.

[Machine-readable measurements and alignment](Helios-Distilled-H20-results.json)
include every measured request's timing and float-frame SHA256.

## Prompt-cache isolation

The pinned runtime can reuse stale cross-attention K/V after the allocator
reuses a previous prompt tensor's address. This is tracked by
[the upstream cache-lifetime fix](https://github.com/vllm-project/vllm-omni/pull/8064).
In the initial mixed-prompt diagnostic, repeating the train prompt after the
beach prompt produced beach content (raw-frame MAE 0.26468 versus the first
train output). Those diagnostic timings and outputs are excluded from the
backend comparison.

Before **every** warmup and measured request, the benchmark's worker extension
calls the existing `clear_cross_attention_cache()` method through RPC. The
reset happens outside the timing interval; text projection and K/V construction
inside the request remain included. K/V reuse across denoising steps of that
request stays enabled. This condition is identical for all backends and is
recorded in the metadata. The recipe measures this isolated-request profile;
it does not establish correctness of mixed-prompt serving on the pinned
unpatched runtime.

## Compare output alignment

SSIM is an optional analysis dependency, installed separately from the runtime.
To reproduce the recorded analysis version, run:

```bash
python -m pip install scikit-image==0.26.0
```

Then run:

```bash
python -m benchmarks.diffusion.compare_helios_attention helios-attention
```

`alignment.json` reports raw-frame MAE, RMSE, maximum absolute difference,
PSNR, mean per-frame SSIM, and temporal-difference MAE. It compares the first
repetition of each case against `TORCH_SDPA`, and the remaining repetitions
against the same backend's first output. Exact equality is recorded explicitly;
its infinite PSNR is represented as JSON `null`. The tool rejects incomplete
matrices, differing workload/environment metadata, or arrays whose SHA256 does
not match the recorded contiguous float32 buffer digest. These metrics quantify
numerical alignment; they do not establish perceptual quality on a broad video
benchmark.

### Export the compact result artifact

The checked-in JSON is a summary: it contains each measured request's aggregate
wall/transformer timings, forward count, memory and output hash, plus alignment
metrics. Full `transformer_timings` and `stage_durations_ms` remain in the raw
per-backend `results.json` files; they are not included in the committed summary.
The exporter checks that alignment groups contain exactly the matching
measurement cases, without duplicates, and recomputes latency/memory summaries
from the validated request records. It does not recompute image metrics; use the
comparison tool on the corresponding raw arrays before exporting.
The following exports that summary from the raw matrix:

```bash
python -m benchmarks.diffusion.export_helios_attention helios-attention \
  --provenance run-provenance.json \
  --output helios-attention-summary.json
```

The provenance JSON must supply `date`, `runtime_commit`,
`benchmark_script_sha256`, `comparison_script_sha256`, `model`, `model_revision`,
`checkpoint_verification` and `scope` for the measured run. For re-exporting the
original September 27 raw matrix only, use the checked-in
`recipes/Helios/Helios-Distilled-H20-results.json` as `--provenance`. A new run
must supply its own provenance; the exporter never hashes today's scripts and
attributes them to a historical run.

### Observed alignment on the historical runtime

These results use the same `b63e35a` runtime and environment as the latency table.

All 12 repeated outputs within **each** backend were byte-identical to that
backend's first output for the same prompt/seed/frame count (36/36 comparisons).
Neither alternative was byte-identical to SDPA in any of the six cases.
The table gives ranges across those six first-repetition cases; RGB values
are normalized to `[0, 1]`.

| Backend versus SDPA | MAE | RMSE | PSNR, dB | Mean frame SSIM | Temporal-delta MAE | Largest pixel error |
| --- | --- | --- | --- | --- | --- | ---: |
| `FLASH_ATTN` (`b63e35a`, `fa3-fwd==0.0.3` installed) | 0.00620–0.01472 | 0.01593–0.04238 | 27.46–35.95 | 0.93431–0.98788 | 0.00447–0.01337 | 1.00000 |
| cuDNN | 0.00633–0.01368 | 0.01555–0.04088 | 27.77–36.16 | 0.94409–0.98712 | 0.00500–0.01096 | 0.99778 |

Average alignment does not imply pixel-level equivalence: the largest local
differences approach the full pixel range. These six cases provide a measured
tradeoff, not a broad quality acceptance threshold or a reason to change the
global default. Exact repetitions here describe this single-session setup.

## Historical backend choice and validated command

The reported speedups use explicitly selected `TORCH_SDPA` as the baseline.
On that H20 runtime, automatic selection already preferred `FLASH_ATTN` when a
supported FlashAttention implementation imported successfully. These numbers are therefore
not an improvement over the existing automatic default.

For latency-oriented use of the historical `b63e35a` H20 environment,
`FLASH_ATTN` with `fa3-fwd==0.0.3` installed and `CUDNN_ATTN` had effectively
tied performance. This is a recommendation for that pinned environment only;
current main needs separate performance validation whether it selects optional
Hopper FA3 or the vLLM dispatcher. Evaluate the output differences on your own
prompts before switching an established workload. Keep `TORCH_SDPA` when
reproducing these SDPA reference outputs is the priority. This recipe does not recommend Sage,
quantization, or a backend choice for other hardware.

The following `FLASH_ATTN` command was validated on `b63e35a` with the historical
software versions above. Run it from the historical checkout prepared earlier.
It uses the shared example in a fresh process for one request, so it does not
reuse a previous request's prompt cache:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
python examples/offline_inference/text_to_video/text_to_video.py \
  --model /data/Helios-Distilled \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
  --negative-prompt "" \
  --seed 42 \
  --height 384 --width 640 --num-frames 33 \
  --num-inference-steps 6 --guidance-scale 1.0 \
  --fps 16 --enforce-eager \
  --extra-body '{"is_enable_stage2":true,"pyramid_num_inference_steps_list":[2,2,2],"is_amplify_first_chunk":true}' \
  --output helios-distilled-flash-attn.mp4
```

The output should be a generated 33-frame, 640 × 384 video at 16 FPS. This
one-shot example is a functionality check; use the warmed-up matrix above
for latency comparisons.
