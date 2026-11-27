# Speech streaming player

A browser player for any model served on `/v1/audio/speech` that streams
16-bit PCM: it plays each chunk as it arrives instead of replaying the
stream, and an uploaded recording is sent as `ref_audio` for voice cloning.

```bash
uv pip install -e '.[demo]'
vllm serve ResembleAI/chatterbox-turbo --omni --port 8091
uv run apps/tts_streaming_player/gradio_demo.py --api-base http://127.0.0.1:8091 --sample-rate 24000
```

Pass `--model` for a server that serves several models, `--voice` for a
voice name other than `default`, and `--sample-rate` for the served model's
PCM rate. The player does not sentence-segment long inputs; use the
[WebSocket speech frontend](../../docs/serving/speech_api.md#streaming-text-input-websocket)
for long narration.
