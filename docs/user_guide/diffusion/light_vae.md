# MiniMax-H3 LiteVAE Decoder

LiteVAE accelerates video decoding by replacing the native MiniMax-H3 video
decoder with a checkpoint containing fewer Transformer blocks. It is an
opt-in model change: generated pixels can differ from the original decoder.
It preserves the original video encoder and audio VAE.

Start with the [MiniMax-H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md),
or the [Ascend NPU recipe](../../../recipes/MiniMaxAI/MiniMax-H3-NPU.md),
then add the decoder setting below. The NPU recipe records measurements on
eight Ascend 910B4 NPUs; those results do not qualify other hardware or tasks.

## Enable the decoder

Add this deployment-level setting to the existing H3 serving configuration:

```bash
--additional-config '{"lightvae_decoder_path":"/weights/LightVAE/lightvae-lite-h3-decoder.safetensors"}'
```

Use the official standard, non-INT8 checkpoint. The file must be available to
every process that loads the decoder. The regular H3 pipeline and the separate
decoder pipeline support the setting; encode-only stages ignore it. Loading
requires the original native H3 VAE assets and its existing remote-code opt-in.
The checkpoint is not downloaded automatically.

An explicit LiteVAE setting takes precedence over automatic discovery of a
bundled NVFP4 decoder. Do not combine `lightvae_decoder_path` with the explicit
`video_vae_decoder` setting: they select alternative video decoders and loading
fails if both are provided. Without LiteVAE, the existing decoder selection is
unchanged.

The loader reads the checkpoint's architecture and compact layers, converts
QKV and MLP layouts, validates the native structure and strictly loads the
decoder and `post_quant_conv` on CPU before device placement or offload setup.
It also checks runtime normalization epsilon and native RoPE frequency buffers
against the Lite architecture. Unsupported position embeddings fail loading.
The original video encoder and audio VAE remain in use. Invalid checkpoints
fail loading rather than silently selecting the original decoder.

## Configuration and scope

| Setting | Behavior |
| --- | --- |
| `additional_config.lightvae_decoder_path` | Nonempty local path to the standard H3 Lite decoder safetensors file. |
| Setting omitted | Use the existing H3 decoder selection, including bundled NVFP4 auto-discovery where applicable. |
| LiteVAE selected | Disable bundled NVFP4 auto-discovery and load the supplied Lite checkpoint. |
| Explicit `video_vae_decoder` also selected | Reject the conflicting decoder settings before loading. Omit this key or set it to `null`. |

To return to the original selection, remove `lightvae_decoder_path` and restart
the service. Do not use an empty string or `null` for this key. It is a
deployment setting, not a per-request `extra_params` option. If an existing
`--additional-config` JSON object is in use, merge the key into that object.

The loader requires architecture metadata and compatible native H3 VAE assets;
renaming an arbitrary checkpoint does not make it compatible. The separately
deployed decoder accepts the same setting, but the measured E2E coverage here
is the regular H3 pipeline with T2VA. FL2VA, Ref2VA, and disaggregated execution
need their own runtime and quality validation.

LiteVAE reuses H3's tiling, parallelism, and offload ownership. It does not
introduce a new VAE parallel mode or enable the Wan VAE fast path.

## Measure serving performance and quality

Compare three configurations with identical requests: main, PR with LiteVAE
disabled, and PR with LiteVAE enabled. Keep prompts, seeds, dimensions, frame
count, steps, LoRA, attention, quantization, offload, and VAE tiling fixed.
Warm each configuration before timing, retain individual hot-run samples, and
bracket comparisons with repeated control runs to detect drift.

Keep the following measurements separate:

| Measurement | Boundary |
| --- | --- |
| Isolated decoder benchmark | One resident decoder on one device, decoding the same saved latent through the benchmark's floating-point path. |
| Serving video VAE stage | `video_vae.decode_latent()` including latent denormalization, native decode, and output restoration/conversion. |
| E2E request latency | Client-observed request time including the remaining model stages and response delivery. |

For NPU stage timing, record start/end Events on each rank's current stream
around `decode_latent()`. Resolve elapsed time after completion and retain
each rank's Event and host timings. The maximum of eight rank durations is a
stage critical-path estimate, not a globally synchronized wall-clock interval.
Current-stream Events do not independently measure work on unrelated streams.
This interval excludes DiT denoising, audio VAE decoding, MP4 encoding, and HTTP
delivery; it is not a pure decoder-kernel measurement.

Report speedup as baseline latency divided by LiteVAE latency, and latency
reduction as `1 - LiteVAE / baseline`. Compare instrumentation enabled and
disabled on the same source/configuration separately: PR-off versus main alone
cannot establish instrumentation overhead.

For quality review, retain complete videos from all configurations and inspect
fine detail, text, faces, hands, motion, and temporal consistency. Main/PR-off
MP4 hash equality checks the disabled path; it does not establish LiteVAE-on
quality equivalence. Still frames supplement rather than replace video review.
The official-reference comparison below checks the same Lite model's numerical
implementation, a separate question from original-versus-Lite quality.

## Run an isolated decoder measurement

Use a configured Linux project environment with the target device available.
In Windows CPU development, replace `.venv/bin/python` with
`.venv/Scripts/python.exe`.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync --python .venv/bin/python benchmarks/diffusion/bench_minimax_h3_vae.py \
  --variant original --model-path "$H3_VIDEO_VAE" \
  --latents "$H3_LATENT_FILE" --device npu:0 --dtype fp16 \
  --warmup 5 --repeats 20 --rounds 3 --trust-remote-code \
  --output-dir results/h3-original

uv run --no-sync --python .venv/bin/python benchmarks/diffusion/bench_minimax_h3_vae.py \
  --variant lite --model-path "$H3_VIDEO_VAE" --checkpoint "$H3_LITE_DECODER" \
  --latents "$H3_LATENT_FILE" --device npu:0 --dtype fp16 \
  --warmup 5 --repeats 20 --rounds 3 --trust-remote-code \
  --output-dir results/h3-lite
```

`H3_VIDEO_VAE` points to the native `video_vae` component directory.
`H3_LATENT_FILE` is a safetensors file containing the normalized BCTHW `latent`.
Use the same saved latent and dtype for both runs; `fp16` above is an example.
Run the variants sequentially in separate processes with sufficient device
memory. Record utilization, memory use, and other device activity before,
during, and after each run so interference can be assessed.
This measurement uses one resident decoder and one card.

The measurement explicitly selects the native legacy floating-point temporal
path because the normal streaming optimization can emit quantized uint8 video.
Both native variants therefore use the same float path. Reports record this
choice; serving with streaming output is a separate end-to-end experiment.
The official reference's RGB output is converted to the same clamped `[0,1]`
pixel range once. The metric function rejects invalid ranges and nonfinite data.

## Compare with an official reference

In a separate reference environment, install a fixed Git revision of the
official LightVAE repository using `uv`. Run the same benchmark file with
`--variant official`, the same `--checkpoint`, `--latents` and matching tile
settings, plus an explicit `--device` and `--dtype`. This mode generates a
reference output once and records the reference source revision. CPU is
suitable for small reference cases; the actual reference device and precision
must be included when interpreting numerical differences.

```bash
uv run --no-sync --python .venv/bin/python benchmarks/diffusion/bench_minimax_h3_vae.py \
  --variant compare \
  --reports results/h3-original/report.json results/h3-lite/report.json results/h3-official/report.json \
  --output-dir results/h3-comparison
```

Each run writes a new directory with `report.json` and unquantized
`decoded.safetensors`. Existing directories are not overwritten. Reports retain
raw timing samples, device memory peaks where supported, model loading time,
input and weight hashes, source hashes and package/device information. Capture
CANN and driver versions, card availability and host memory separately with the
experiment environment record. Model loading time excludes weight hashing and
input preparation; it is not total process startup latency.

The comparison checks experimental signatures, runtime devices, package versions,
Omni revision, source hashes, raw timing samples and official-reference weight
identity. Original and Lite runs must use the same software and source code;
missing or inconsistent evidence fails comparison. Source hashes cover the H3
adapter and helpers, plus the loaded remote component's Python source directory.
The official reference may use its separate environment. The numerical gate is at least
40 dB PSNR between the *same Lite model's* reference and native output before
uint8 quantization and MP4 compression. Identical output has `exact=true` and a
null PSNR. This gate does not apply to original-versus-Lite compression quality.
SSIM uses torchmetrics with data range 1, an 11-pixel Gaussian window and sigma
1.5, averaged over frames.

An optional `--ground-truth` safetensors file containing a `[0,1]` BCTHW `video`
adds reconstruction PSNR and SSIM for both decoders. Align preprocessing,
cropping and valid frame counts before producing it.

Exit code 0 means the requested run finished or the numerical comparison
passed; inspect `speedup_verified` separately. Exit code 1 indicates failure;
2 means the required comparison evidence is missing. Verified decode speedup
requires the slowest Lite round median to beat the fastest original round
median. Missing reference, OOM, mismatched devices or incomplete timing cannot
produce a successful speedup claim.
