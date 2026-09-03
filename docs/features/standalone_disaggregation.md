# Standalone Stage Disaggregation

Standalone mode boots any single pipeline stage as an independent HTTP server.
Each stage can be deployed, scaled, and managed independently by external
infrastructure. The existing orchestrator is unchanged — standalone is opt-in.

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

## `/v1/stage/run` endpoint

Standalone stages communicate via `/v1/stage/run`. The endpoint has two modes
based on the request body:

**Entry mode** (no `stage_output` in body): runs the model and returns raw
multimodal output as serialized JSON. Currently speech-only.

```bash
curl http://localhost:8000/v1/stage/run \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "input": "Hello", "voice": "vivian"}'
# → {"stage_output": {"codes": {"audio": [[...], ...]}, ...}, "request_id": "..."}
```

**Downstream mode** (has `stage_output` in body): accepts upstream stage output,
runs the engine, and returns the final result.

```bash
curl http://localhost:8001/v1/stage/run \
  -H "Content-Type: application/json" \
  -d @talker_output.json -o output.wav
# → audio/wav binary
```

An external coordinator chains the stages by calling entry mode on the first
stage and forwarding the response to downstream mode on the next.

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
`examples/online_serving/text_to_speech/qwen3_tts/standalone_disagg_client.py`.

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

Async-chunk streaming is not supported (full-payload transfer only). This means
higher time-to-first-audio compared to co-located mode.

Currently validated with TTS pipelines only. Omni (Qwen3-Omni) and image gen
(BAGEL, HunyuanImage3) require additional work for connector-based transfer and
CFG fan-out respectively.
