# MiniMax-H3 LiteVAE Decoder

LiteVAE replaces the MiniMax-H3 video decoder with a smaller Transformer to
reduce decoding latency. The video encoder and audio VAE remain unchanged.
Generated pixels can differ from the original decoder.

## Usage

Add the following to an existing [H3 serving command](gh-file:recipes/MiniMaxAI/MiniMax-H3.md):

```bash
--additional-config '{"lightvae_decoder_path":"/weights/lightvae-lite-h3-decoder.safetensors"}'
```

Use the standard, non-INT8 H3 Lite decoder safetensors checkpoint. It must be
available locally to every decoding process; it is not downloaded automatically.
The original native H3 VAE assets and `--trust-remote-code` are still required.

Merge the key into any existing `--additional-config` JSON object. This is a
service-startup setting, not a per-request parameter. To restore the existing
decoder selection, remove the key and restart the service.

## Decoder selection

| Configuration | Behavior |
| --- | --- |
| LiteVAE key omitted | Preserve existing decoder selection, including bundled NVFP4 auto-discovery. |
| `lightvae_decoder_path` set | Load LiteVAE and skip bundled NVFP4 auto-discovery. |
| LiteVAE and non-null `video_vae_decoder` both set | Reject the conflicting settings before loading. |
| Empty, null, or incompatible LiteVAE checkpoint | Fail loading; do not silently fall back to the original decoder. |

The checkpoint must contain compatible architecture metadata and weights.
The loader validates the native decoder structure, normalization, and RoPE,
then strictly loads weights on CPU before device placement and offload setup.

## Supported scope

- The regular H3 pipeline and separate decoder pipeline accept this setting;
  encode-only stages ignore it.
- LiteVAE reuses H3 tiling, parallelism, and offload configuration.
- Runtime measurements cover T2VA on eight Ascend 910B4 NPUs. FL2VA, Ref2VA,
  and disaggregated execution require separate validation.

## Performance and quality

See the [NPU recipe](gh-file:recipes/MiniMaxAI/MiniMax-H3-NPU.md#litevae-measurements-8-ascend-910b4-npus)
for configuration and measured results. Keep E2E request latency separate from
video VAE decoding time; improving one stage does not yield the same E2E speedup.

Compare original and LiteVAE videos with identical prompts, seeds, and sampling
settings. Review fine detail and temporal consistency, including faces, hands,
and text. Matching original-decoder outputs with LiteVAE disabled checks the
disabled path, not LiteVAE-on quality equivalence.
