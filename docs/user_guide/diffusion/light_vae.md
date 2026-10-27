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
Comparing the native Lite implementation with the same Lite model's official
reference is a separate question from original-versus-Lite quality.
