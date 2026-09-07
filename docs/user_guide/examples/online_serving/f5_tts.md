# F5-TTS Online Serving

Serve F5-TTS via the `/v1/audio/speech` endpoint.

## Start Server

```bash
vllm serve SWivid/F5-TTS/F5TTS_v1_Base \
    --omni --trust-remote-code --port 8091
```

The 3-segment model ID (`SWivid/F5-TTS/F5TTS_v1_Base`) is required. F5-TTS
runs on the default single-stage diffusion config, so no deploy config is
auto-loaded. To enable the DiT CUDA Graph replay (opt-in), pass the extras
as stage overrides (`vllm_omni/deploy/f5_tts.yaml` holds the reference
values):

```bash
vllm serve SWivid/F5-TTS/F5TTS_v1_Base \
    --omni --trust-remote-code --port 8091 \
    --stage-overrides '{"0":{"extras":{"f5_dit_cudagraph":true}}}'
```

### With Cache Acceleration

Cache-DiT (recommended, 1.62x speedup with zero quality loss):

```bash
vllm serve SWivid/F5-TTS/F5TTS_v1_Base \
    --omni --trust-remote-code --port 8091 \
    --cache-backend cache_dit \
    --cache-config '{"Fn_compute_blocks": 1, "Bn_compute_blocks": 0, "max_warmup_steps": 4}'
```

> TeaCache is **not supported** for F5-TTS: step-level caching is
> incompatible with flow-matching + sway sampling (48+ configs tuned, no
> quality-preserving speedup). Use Cache-DiT instead.

## Client Request

```bash
REF_AUDIO_PATH=ref_audio.wav
REF_BASE64=$(base64 -w 0 "${REF_AUDIO_PATH}")

curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output output.wav \
  -d @- <<EOF
{
  "input": "Hello, this is a test of F5 TTS synthesis.",
  "ref_text": "Reference text matching the audio.",
  "ref_audio": "data:audio/wav;base64,${REF_BASE64}",
  "num_inference_steps": 32,
  "guidance_scale": 2.0,
  "seed": 42
}
EOF
```

## Supported Model Variants

| Model ID | Vocoder |
|----------|---------|
| `SWivid/F5-TTS/F5TTS_v1_Base` | Vocos (recommended) |
| `SWivid/F5-TTS/F5TTS_v1_Base_no_zero_init` | Vocos |
| `SWivid/F5-TTS/F5TTS_Base` | Vocos |
| `SWivid/F5-TTS/F5TTS_Base_bigvgan` | BigVGAN |

## Notes

- F5-TTS is a diffusion-based TTS model; no deploy config is needed
  (single-stage diffusion models use the default stage config; graph
  extras go through `--stage-overrides`)
- Reference audio should be 3-12 seconds, mono, any sample rate (resampled to 24kHz)
- Reduce `num_inference_steps` to 16 for ~2x faster generation with acceptable quality
