# H3 LightVAE A2 POC

This POC adds an optional standard LightVAE-Lite video decoder to the native
MiniMax-H3 VAE adapter. Its initial target is Ascend A2. Real A2 accuracy,
memory, multi-card compatibility and performance require hardware validation;
CPU checkpoint tests do not establish those results.

An available A5 server can be used for initial validation. Keep A5 results
separate from the A2 acceptance record; A5 compatibility and performance do not
establish the corresponding A2 results.

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

## Run an isolated decoder measurement

Use a configured Linux A2 project environment. In Windows CPU development,
replace `.venv/bin/python` with `.venv/Scripts/python.exe`.

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
Use the same saved latent for both runs. Select the dtype used by the verified
A2 baseline; `fp16` above is an example, not an A2 validation claim. Run the
variants sequentially in separate processes on the same otherwise-idle device.
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

Complete the three-clip reconstruction comparison, target multi-card tile and
offload validation, and fixed-input H3 end-to-end tests on A2 before deciding
whether to adopt Lite or start the Lite reference-encoder phase.
