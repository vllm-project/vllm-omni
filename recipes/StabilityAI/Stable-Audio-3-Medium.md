# Stable Audio 3 (medium)

> Stable Audio 3 medium: 1.4B DiT text-to-audio diffusion (stereo 44.1 kHz, up to 380 s)

## Summary

- Vendor: StabilityAI
- Model: `stabilityai/stable-audio-3-medium` (gated)
- Task: Text -> audio (music, sound effects, ambient). Initial scope is
  text-to-audio; audio-to-audio editing and inpainting are deferred.
- Mode: Offline inference + online serving (pure diffusion)
- Maintainer: Community

## When to use this recipe

Use this recipe to generate stereo 44.1 kHz audio (music / SFX / ambient) from a
text prompt with Stable Audio 3 medium -- a 1.4B-parameter DiT diffusion model
that scales to long clips (up to 380 s). It is the direct successor to
stable-audio-open-1.0. The small variants (`-small-music`, `-small-sfx`) are
CPU-targeted and lower quality; Large (2.7B) is API-only and out of scope.

## References

- Upstream reference: <https://github.com/Stability-AI/stable-audio-3> (MIT)
- Model card (gated): <https://huggingface.co/stabilityai/stable-audio-3-medium>
- Pipeline: `vllm_omni.diffusion.models.stable_audio_3.pipeline_stable_audio_3.StableAudio3Pipeline`
- Offline example: [`examples/offline_inference/stable_audio_3/`](../../examples/offline_inference/stable_audio_3/) (download helper) plus the shared [`examples/offline_inference/text_to_audio/`](../../examples/offline_inference/text_to_audio/) driver
- Online example: [`examples/online_serving/stable_audio/`](../../examples/online_serving/stable_audio/)
- Related issue: <https://github.com/vllm-project/vllm-omni/issues/3787>

## Hardware Support

## GPU

### 1x RTX 4090 24GB

#### Environment

- OS: Ubuntu 22.04
- Python: 3.10+
- Driver / runtime: CUDA 12.x / 13.x
- vLLM version: 0.22.0
- vLLM-Omni version: 0.1.x

#### Command

`stabilityai/stable-audio-3-medium` is **gated** -- accept the license on the
model page and run `hf auth login` first. The HF repo does not ship the
`model_index.json` / `transformer/config.json` that the engine uses for model
discovery, so prepare a local directory with the helper script (it fetches the
weights and writes both files):

```bash
python examples/offline_inference/stable_audio_3/download_stable_audio_3.py \
    --output-dir ./stable-audio-3-medium
```

Offline (text-to-audio) via the shared `text_to_audio.py` driver:

```bash
python examples/offline_inference/text_to_audio/text_to_audio.py \
    --model ./stable-audio-3-medium \
    --prompt "An ambient drone evolving slowly with shimmering overtones" \
    --audio-length 30.0 \
    --num-inference-steps 100 \
    --guidance-scale 7.0 \
    --output sa3.wav
```

Online (`/v1/audio/generate`):

```bash
vllm-omni serve ./stable-audio-3-medium \
    --model-class-name StableAudio3Pipeline \
    --trust-remote-code \
    --enforce-eager \
    --port 8091

curl -X POST http://localhost:8091/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "An ambient drone with shimmering overtones",
        "audio_length": 30.0,
        "guidance_scale": 7.0,
        "num_inference_steps": 100,
        "seed": 42
    }' --output sa3.wav
```

#### Verification

```bash
STABLE_AUDIO_3_TEST_MODEL=./stable-audio-3-medium pytest -v \
    tests/e2e/offline_inference/test_stable_audio_3_expansion.py \
    tests/e2e/online_serving/test_stable_audio_3_expansion.py
# expect: offline 2 passed, online 1 passed; valid stereo 44.1 kHz, no NaN
```

#### Notes

- Memory usage: ~5.7 GB peak (warm, `torch.compile`, batch 1).
- Latency (100 steps, CFG 7.0, `dpmpp-3m-sde`; RTF = latency / audio seconds):

  | Audio length | E2E latency | RTF   |
  |--------------|-------------|-------|
  | 30 s         | 2.51 s      | 0.084 |
  | 60 s         | 4.17 s      | 0.070 |
  | 120 s        | 7.90 s      | 0.066 |

- Variable-length latents scale correctly (verified 5 s through 60 s; up to 380 s).
- Long clips: add `--enable-cpu-offload` and/or `--enable-layerwise-offload`.
- Multi-GPU: `--use-hsdp` shards the 1.4B DiT with FSDP2. Tensor / sequence /
  CFG parallelism are not implemented for SA3 (audio-diffusion peers ship
  without them).
- Flash Attention 2 is recommended for performance but not required -- the
  PyTorch SDPA / flex-attention fallback works end to end.
- Sampler: `dpmpp-3m-sde` with a `LogSNRShift` schedule (upstream defaults).

#### Known limitations

- Initial scope is text-to-audio only; audio-to-audio editing and inpainting /
  continuation are deferred.
- Cache-DiT acceleration is not wired up yet.

### 1x H200 141GB

Verified by a maintainer on a single H200 (vLLM 0.22.0, transformers 5.8.1):
the same end-to-end suite passes (offline 2 / online 1), and a 10 s clip at
100 steps, CFG 7.0, seed 42 produces valid stereo 44.1 kHz audio.
