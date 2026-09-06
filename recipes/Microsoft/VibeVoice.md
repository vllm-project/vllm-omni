# VibeVoice for zero-shot multi-speaker TTS on 1x GPU

## Summary

- Vendor: Microsoft
- Model: `microsoft/VibeVoice-1.5B`
- Task: Zero-shot text-to-speech with reference audio voice cloning
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe to serve VibeVoice-1.5B on a single H100 GPU. VibeVoice
clones any speaker's voice from a short reference audio clip (≤60 s) and
supports up to four speakers per request with independent references. Output
is 24 kHz mono PCM.

## References

- Upstream or canonical docs:
  [microsoft/VibeVoice-1.5B on HuggingFace](https://huggingface.co/microsoft/VibeVoice-1.5B)
- vLLM-Omni model guide:
  [`docs/models/vibevoice.md`](../../docs/models/vibevoice.md)
- Bundled reference audio provenance:
  [`docs/design/vibevoice/ASSET_PROVENANCE.md`](../../docs/design/vibevoice/ASSET_PROVENANCE.md)

## Hardware Support

### GPU

### 1x H100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with H100 80GB
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the server from the repository root:

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --host 127.0.0.1 \
  --port 8000
```

The default deploy config (`vllm_omni/deploy/vibevoice.yaml`) sets TP=1,
`max_num_seqs=4`, `max_model_len=65536`, positive/negative KV cache 8 GiB
each, and enables diffusion + decode CUDA graphs with greedy AR sampling.

#### Verification

Quick API smoke test with a bundled default voice (no `ref_audio` needed):

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Hello, this is a test.",
    "response_format": "wav"
  }' --output test.wav
```

Verify the output is 24 kHz mono and non-empty:

```bash
python -c "import soundfile as sf; w, sr = sf.read('test.wav'); \
  assert sr == 24000 and w.ndim == 1 and len(w) > 0; print('OK', sr, len(w))"
```

Streaming SSE with `finish_reason`:

```bash
curl -N http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Streaming test.",
    "response_format": "pcm",
    "stream": true,
    "stream_format": "sse",
    "max_new_tokens": 2
  }' | grep -c 'speech.audio.delta'
# Expected: 2
```

#### Notes

- Memory usage: ~27 GB total (AR + diffusion + negative KV branch); fits 80 GB cards comfortably.
- Audio output: 24 kHz mono, 3200 samples per token.
- Key flags: `--omni` is required; `--tokenizer` points to the Qwen2.5-1.5B tokenizer (the official checkpoint does not bundle one).
- No request-level seed: VibeVoice uses greedy AR sampling and a global diffusion RNG. The `seed` field is rejected; omit it.
- Reference audio limit: 60 s max per reference, one per speaker. When `ref_audio` is omitted entirely, four bundled Apache-2.0 reference voices are assigned in speaker first-appearance order.
- Known limitations: First request triggers CUDA graph capture (one-time ~10 s latency). `finish_reason="stop"` means natural completion; `finish_reason="length"` means valid truncation at `max_new_tokens`.
