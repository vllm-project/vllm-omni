# Voxtream2 Online Serving

Voxtream2 uses the OpenAI-compatible `/v1/audio/speech` endpoint and requires
reference audio for voice cloning.

## Launch Server

Run from the repository root:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VOXTREAM2_ROOT=voxtream \
  vllm serve herimor/voxtream2 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxtream2_1stage.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

Or use the example script:

```bash
./examples/online_serving/voxtream2/run_server.sh
```

`VOXTREAM2_ROOT` should point to the Voxtream checkout or installed source tree
that contains `configs/generator.json` and `configs/speaking_rate.json`.

Local `file://` reference audio is accepted under `VOXTREAM2_ROOT`. To allow
reference audio outside that tree, set `VOXTREAM2_ALLOWED_LOCAL_MEDIA_PATH`:

```bash
VOXTREAM2_ALLOWED_LOCAL_MEDIA_PATH=/absolute/path/to/reference/audio/root \
  ./examples/online_serving/voxtream2/run_server.sh
```

## Send Request

```bash
python examples/online_serving/voxtream2/speech_client.py \
  --text "This is a Voxtream2 online serving example." \
  --ref-audio voxtream/assets/audio/english_female.wav \
  --output voxtream2.wav
```

The Python client sends local paths as base64 data URLs by default. Add
`--file-uri` to send the local path to the server as a `file://` URI.

Equivalent curl request:

```bash
REF_AUDIO="$(pwd)/voxtream/assets/audio/english_female.wav"
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"This is a Voxtream2 online serving example.\",
    \"ref_audio\": \"file://${REF_AUDIO}\"
  }" --output voxtream2.wav
```

HTTP URLs and base64 data URLs are also accepted for `ref_audio`. Streaming
responses are not supported for Voxtream2 yet.
