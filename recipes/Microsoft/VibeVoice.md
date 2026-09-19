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
each, and defaults to the full-graph profile: `enforce_eager: false` permits
positive AR graphs and the eligible independent negative executor, with
diffusion + decode CUDA graphs enabled and greedy AR sampling. To use fully
eager execution, set `enforce_eager: true` and disable both
`diffusion_cuda_graph` and `decode_cuda_graph` under
`engine_extras.additional_config.vibevoice_runtime_config` in a custom deploy
YAML, passed with `--deploy-config`.

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
    "max_new_tokens": 128
  }' --output speech.sse
```

Validate the SSE payloads rather than counting transport events. The number
of generated tokens does not equal the number of audio delta events:

```python
import base64
import json
from pathlib import Path

text = Path("speech.sse").read_text().replace("\r\n", "\n")
chunks = []
done = None
for frame in text.split("\n\n"):
    data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
    if not data:
        continue
    event = json.loads(data)
    assert done is None, "Unexpected event after speech.audio.done"
    if event["type"] == "speech.audio.delta":
        chunks.append(base64.b64decode(event["audio"], validate=True))
    elif event["type"] == "speech.audio.done":
        done = event
    else:
        raise AssertionError(f"Unexpected SSE event: {event}")

pcm = b"".join(chunks)
assert pcm and len(pcm) % (3200 * 2) == 0  # mono PCM16, 3200 samples/audio token
assert done is not None and done["finish_reason"] in {"stop", "length"}
print("OK", len(pcm) // 2, "samples", done["finish_reason"])
```

#### Notes

- Memory usage: independent H100 80GB measurements (32 English requests, including 2 warmups) observed full-graph peaks of about 22.6 GiB at B1 and 23.4 GiB at B4 with positive/negative pools of 8 GiB each. These are sampled observations, not a guaranteed memory ceiling or full quality qualification.
- Audio output: 24 kHz mono, 3200 samples per token.
- Key flags: `--omni` is required; `--tokenizer` points to the Qwen2.5-1.5B tokenizer (the official checkpoint does not bundle one).
- No request-level seed: VibeVoice uses greedy AR sampling and a global diffusion RNG. The `seed` field is rejected; omit it.
- Reference audio limit: 60 s max per reference, one per speaker. When `ref_audio` is omitted entirely, four bundled Apache-2.0 reference voices are assigned in speaker first-appearance order.
- Graph warmup: eligible negative AR graphs and configured diffusion graphs are captured at startup; decode graphs are captured lazily when compatible request caches are available. Startup and unprepared paths can incur extra latency; no fixed first-request latency is guaranteed.
- Completion: `finish_reason="stop"` means natural completion; `finish_reason="length"` means valid truncation at `max_new_tokens`.
