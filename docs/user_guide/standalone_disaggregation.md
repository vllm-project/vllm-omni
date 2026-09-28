# Standalone Stage Disaggregation

Standalone mode boots a single pipeline stage as an independent HTTP server.
Each stage can be deployed, scaled, and managed independently by external
infrastructure. No orchestrator is involved — stages are fully independent.

## Quickstart

Boot each stage on a separate GPU, then chain them with the reference
coordinator:

```bash
# Terminal 1: talker (stage 0)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
    --standalone --stage-id 0 --port 8000 --trust-remote-code

# Terminal 2: code2wav (stage 1)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
    --standalone --stage-id 1 --port 8001 --trust-remote-code

# Terminal 3: chain stages
python standalone_disagg_client.py \
    --text "Hello, how are you?" \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --voice vivian
```

## Finding stages and their order

Each model's deploy YAML (`vllm_omni/deploy/<model>.yaml`) lists its stages
in order. For example, `qwen3_tts.yaml` defines stage 0 (talker) and stage 1
(code2wav). Use `--stage-id` to select which stage to boot.

## `/v1/stage/run` endpoint

Standalone stages communicate via `/v1/stage/run`. The behavior depends on
the request body:

- **Without `stage_output`:** runs the model and returns raw multimodal
  output as JSON. The first stage in the pipeline uses this.

```bash
curl http://localhost:8000/v1/stage/run \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "input": "Hello", "voice": "vivian"}'
# → {"stage_output": {"codes": {"audio": [[...], ...]}, ...}, "request_id": "..."}
```

- **With `stage_output`:** accepts the previous stage's output and returns
  the final result. Later stages in the pipeline use this.

```bash
curl http://localhost:8001/v1/stage/run \
  -H "Content-Type: application/json" \
  -d @talker_output.json -o output.wav
# → audio/wav binary
```

An external coordinator chains stages by calling the first stage and
forwarding its `stage_output` to the next. To debug, save each stage's
response to a file and inspect the JSON — the `stage_output` structure
matches what the next stage expects.

## Reference coordinator

A minimal Python script that chains standalone stages over HTTP:

```python
import httpx

with httpx.Client(timeout=120) as client:
    # Step 1: talker
    resp = client.post("http://localhost:8000/v1/stage/run", json={
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "input": "Hello", "voice": "vivian",
    })
    talker_result = resp.json()

    # Step 2: code2wav
    resp = client.post("http://localhost:8001/v1/stage/run", json={
        "stage_output": talker_result["stage_output"],
        "request_id": talker_result["request_id"],
    })

    with open("output.wav", "wb") as f:
        f.write(resp.content)
```

A complete reference coordinator with error handling and timing is available at
`examples/online_serving/text_to_speech/standalone_disagg_client.py`.

## Comparison with headless mode

| | `--headless` | `--standalone` |
|---|---|---|
| Transport | ZMQ (msgpack) | HTTP |
| Exposes HTTP endpoint | No | Yes |
| Requires head process | Yes | No |
| Scheduling | Orchestrator | External infrastructure |
| Failure domain | Shares orchestrator state | Fully isolated |
| Use case | Data-parallel replicas | Stage-level disaggregation |

Both can run on separate nodes. The difference is ownership: headless workers
are orchestrator-coupled, standalone stages are orchestrator-independent.

## Limitations

Voice cloning is not supported. Requests with explicit `ref_audio`, uploaded
voices, and precomputed ICL profiles are all rejected. Use normal multi-stage mode (`vllm serve --omni`)
for voice cloning.

Async-chunk streaming is not supported (full-payload transfer only). This means
higher time-to-first-audio compared to normal multi-stage mode (`vllm serve --omni`).

Standalone entry stages apply model-agnostic codec hygiene only (dropping
negative-padded and all-zero frames). Model-specific transfer details from
each model's stage input processor — codebook-range checks, sequence-length
cropping, ref-code prepending, trim metadata — are not re-applied across the
HTTP boundary. For supported voices on validated models this matches
co-located output; unvalidated models may diverge on edge-case frames. Voice
cloning stays rejected for this reason.

Currently validated with Qwen3-TTS only. Other TTS models sharing the same
pipeline structure should work but are not yet tested. Omni (Qwen3-Omni) and image gen
(BAGEL, HunyuanImage3) require additional work for connector-based transfer and
CFG fan-out respectively.
