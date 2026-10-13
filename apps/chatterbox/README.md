# Chatterbox streaming player

This browser application plays PCM streaming from an existing Chatterbox Turbo
or Original speech server and accepts an optional reference recording.

From the repository root, install the demo and conditioning dependencies:

```bash
uv pip install -e '.[demo,chatterbox]'
vllm-omni serve ResembleAI/chatterbox-turbo --omni --port 8091
uv run apps/chatterbox/gradio_demo.py --api-base http://127.0.0.1:8091
```

Use `ResembleAI/chatterbox` for Original. Output is English, mono 24 kHz audio.
The player does not sentence-segment long inputs; use the shared WebSocket
speech frontend for long narration. See the
[speech API](../../docs/serving/speech_api.md#chatterbox-turbo-and-original)
and [H100 qualification](../../recipes/ResembleAI/Chatterbox-H100.md).
