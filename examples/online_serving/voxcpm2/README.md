# VoxCPM2 Online Serving

Serve VoxCPM2 TTS via the OpenAI-compatible `/v1/audio/speech` endpoint.

## Start the Server

```bash
python -m vllm_omni.entrypoints.openai.api_server \
    --model openbmb/VoxCPM2 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm2.yaml \
    --host 0.0.0.0 --port 8000
```

## Zero-shot Synthesis

```bash
python openai_speech_client.py --text "Hello, this is VoxCPM2."
```

Or with curl:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "voxcpm2", "input": "Hello, this is VoxCPM2.", "voice": "default"}' \
  --output output.wav
```

## Voice Cloning

Clone a speaker's voice using a reference audio file:

```bash
python openai_speech_client.py \
    --text "This should sound like the reference speaker." \
    --ref-audio /path/to/reference.wav
```

The `--ref-audio` parameter accepts:
- Local file path (auto-encoded to base64)
- URL (`https://...`)
- Base64 data URI (`data:audio/wav;base64,...`)
