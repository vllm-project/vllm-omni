# dots.tts

> Offline continuous-AR TTS at 48 kHz (rednote-hilab)

## Summary

- Vendor: rednote-hilab
- Model: `rednote-hilab/dots.tts-soar`
- Task: Text-to-speech, zero-shot synthesis only
- Mode: Offline end-to-end example only (not yet wired for online serving —
  see [Known limitations](#known-limitations))
- Maintainer: Community

## When to use this recipe

Use this recipe as a known-good starting point for running
`rednote-hilab/dots.tts-soar` offline on vLLM-Omni on consumer-class GPUs.
dots.tts is a ~1.7B-parameter continuous-AR TTS model (Qwen2.5-1.5B base LM
+ 344M DiT flow-matching head + 180M AudioVAE) that emits 48 kHz mono audio.
It follows the same "vLLM-native base LM + side-path computation" pattern as
VoxCPM2 — single-stage pipeline
`Qwen2.5-1.5B base LM → DiT (10-step Euler flow matching) → patch_encoder AR
loopback → AudioVAE (streaming decode)` — with a plain Qwen2 backbone
instead of MiniCPM4, and no FSQ / residual-LM stage.

This is an early integration (see [Known limitations](#known-limitations)
below before depending on it for anything beyond offline experimentation).

## References

- Offline end-to-end script:
  [`examples/offline_inference/text_to_speech/dots_tts/end2end.py`](../../examples/offline_inference/text_to_speech/dots_tts/end2end.py)
- Example guide:
  [`examples/offline_inference/text_to_speech/README.md`](../../examples/offline_inference/text_to_speech/README.md#dotstts)
- Default deploy config:
  [`vllm_omni/deploy/dots_tts.yaml`](../../vllm_omni/deploy/dots_tts.yaml)
- Talker / pipeline source:
  [`vllm_omni/model_executor/models/dots_tts/`](../../vllm_omni/model_executor/models/dots_tts/)
- Upstream: [rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts)

## Hardware Support

This recipe documents one tested 16 GB consumer-GPU configuration. Other
vendor sections (ROCm, NPU) and larger-VRAM configurations are welcome as
community validation lands.

## GPU

### 1 x RTX 5080 16GB (Single GPU, Minimum Recommended)

dots.tts (~1.7B params across the base LM + DiT + AudioVAE + CAM++
speaker encoder, bfloat16) fits comfortably on a single 16 GB GPU. The
bundled default config at
[`vllm_omni/deploy/dots_tts.yaml`](../../vllm_omni/deploy/dots_tts.yaml)
(`gpu_memory_utilization: 0.8`, `max_num_seqs: 4`, `enforce_eager: true`,
`enable_prefix_caching: false`) loads cleanly with ~5.1 GiB for model
weights and ~0.3 GiB peak activation; the remainder of the configured
budget is available for KV cache. Total resident footprint at idle is
roughly **7-8 GiB / 16 GB** — the only tight spot in the full CUDA-Graph
roadmap would be step 8's graph capture (not implemented yet; this
release runs `enforce_eager: true`, so it doesn't apply today).

#### Environment

- OS: Linux (WSL2)
- Python: 3.12
- Driver / runtime: NVIDIA driver 595.95
- torch: 2.11.0+cu130
- vLLM: 0.26.0
- vLLM-Omni: 0.22.1.dev (current `main`)

#### Command

```bash
python examples/offline_inference/text_to_speech/dots_tts/end2end.py \
    --model rednote-hilab/dots.tts-soar \
    --text "Hello, this is a test of dots TTS running on vLLM Omni."
```

The deploy config at
[`vllm_omni/deploy/dots_tts.yaml`](../../vllm_omni/deploy/dots_tts.yaml)
is loaded automatically by the model registry (HF `model_type=dots_tts`).
Pass `--deploy-config <path>` to override.

#### Verification

**T1 — offline zero-shot synthesis**:

```bash
python examples/offline_inference/text_to_speech/dots_tts/end2end.py \
    --model rednote-hilab/dots.tts-soar \
    --text "Hello, this is a test of dots TTS running on vLLM Omni."
```

Observed: `output_audio/output.wav`, 3.52 s @ 48 kHz mono. Single
one-shot process (init → one `generate()` → exit), so the reported
numbers include engine init and first-request warmup, not just steady-state
per-step throughput — same caveat as VoxCPM2's recipe. `Inference: 5.52s`,
`RTF: 1.569`.

Weight-loading breakdown from the same run (all tensors matched, no
missing/extra keys): 951/951 AudioVAE, 244/244 DiT, 270/270
patch_encoder, 198 Qwen2, 938/938 CAM++ speaker encoder.

Whisper transcription of the output matched the input text with no
dropped leading word (confirms the streaming-vocoder patch-boundary fix
described in [Known limitations](#known-limitations)).

#### Notes

- Output: 48 kHz mono WAV.
- Checkpoints: `rednote-hilab/dots.tts-soar` is the validated default
  used throughout this recipe. `rednote-hilab/dots.tts-base` shares the
  same architecture but is unvalidated in this repo. `rednote-hilab/dots.tts-mf`
  (MeanFlow, 2-4 step) is not supported — see below.
- `enforce_eager: true` and `enable_prefix_caching: false` in the deploy
  config are load-bearing, not just conservative defaults: with prefix
  caching enabled, vLLM-Omni's prefix-cache multimodal-output merge path
  does not preserve the `sparse_audio` marker this model relies on to
  route audio output correctly, and generation silently truncates to a
  single ~160 ms patch. Do not override `enable_prefix_caching` for this
  model until that framework-level gap is fixed.

## Known limitations

- **No online serving yet.** dots.tts is not registered in
  `vllm_omni/entrypoints/openai/tts_adapters/` (the framework other TTS
  models in this repo use for `/v1/audio/speech`), so `vllm serve
  rednote-hilab/dots.tts-soar --omni` will start but the OpenAI-compatible
  speech endpoint does not know how to build request params for it yet.
  Offline `Omni()` (as used in `end2end.py`) is the only supported path
  today.
- **Voice cloning is not wired.** The CAM++ x-vector speaker encoder
  weights load and are exercised by `load_weights()`, but nothing in the
  prompt builder consumes a reference audio yet — generation is zero-shot
  only. `end2end.py` has no `--ref-audio`/`--ref-text` flags.
- **`dots.tts-mf` (MeanFlow, 2-4 step) checkpoint is not supported.** Only
  the fixed 10-step Euler DiT sampler used by `dots.tts-soar` /
  `dots.tts-base` is implemented.
- **No CUDA graph capture.** The talker runs fully eager. voxcpm2's three
  captured graphs (base LM decode, CFM solver, VAE decode) have no
  dots.tts equivalent yet.
- **Concurrent requests do not scale.** Each request's 10-step DiT Euler
  integration runs serially in the side path (no cross-request batching,
  unlike voxcpm2's `enable_batched_cfm`). A community review of this
  integration measured no throughput gain at `c=4` concurrent requests
  versus `c=1`.
- **`SamplingParams.seed` does not control audio-generation randomness.**
  The DiT's flow-matching noise is deterministically derived per-request
  from a fixed internal seed (reproducible run-to-run, matching
  voxcpm2's `deterministic_cfm_seed` convention) rather than from the
  caller-supplied `seed` field — the same limitation voxcpm2 has today.
