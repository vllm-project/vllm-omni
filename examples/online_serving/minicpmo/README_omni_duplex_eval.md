# Omni-DuplexEval

This runner keeps generation and scoring separate: generation uses the native
MiniCPM-o duplex WebSocket, while evaluation sends judge prompts to a local
OpenAI-compatible vLLM-Omni chat server.  It does not change serving defaults
or require the upstream benchmark checkout at runtime.

```bash
vllm-omni bench omni-duplex-eval --omni generate \
  --url ws://127.0.0.1:8099/v1/realtime?duplex=1 \
  --model /models/MiniCPM-o-4_5 --ref-audio /data/ref.wav \
  --dataset Hothan/Omni-DuplexEval --family all \
  --response-root /data/duplex/responses

vllm-omni bench omni-duplex-eval --omni evaluate \
  --dataset Hothan/Omni-DuplexEval --response-root /data/duplex/responses \
  --score-root /data/duplex/scores --judge-base-url http://127.0.0.1:8000 \
  --judge-model /models/Qwen2.5-VL-7B-Instruct --judge-video-mode video_url

vllm-omni bench omni-duplex-eval --omni summarize \
  --score-root /data/duplex/scores
```

`--pace as-fast-as-possible` writes `clock=invalid`; evaluation rejects those
artifacts unless `--allow-invalid-clock` is explicit.  Use `--limit 1` for a
smoke test.  The protocol pin is recorded in every score artifact.
