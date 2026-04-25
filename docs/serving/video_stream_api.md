# Streaming Video Input API

vLLM-Omni provides a WebSocket API for streaming video frames and optional audio chunks into Qwen3-Omni, then asking questions over the buffered session context.

Each server instance runs a single model specified at startup with `vllm serve <model> --omni`.

## Quick Start

### Start the Server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --deploy-config vllm_omni/deploy/qwen3_omni.yaml \
    --omni \
    --port 8000 \
    --trust-remote-code
```

### Run the Example Client

```bash
python examples/online_serving/qwen3_omni/streaming_video_client.py \
    --url ws://localhost:8000/v1/video/chat/stream \
    --video /path/to/video.mp4 \
    --query "Describe what is happening in the video."
```

## API Reference

### Endpoint

```text
WebSocket /v1/video/chat/stream
```

### Protocol

| Direction | Type | Required fields | Description |
|-----------|------|-----------------|-------------|
| Client -> Server | `session.config` | none | First message. Configures output modalities, frame sampling, EVS, and prompts. |
| Client -> Server | `video.frame` | `data` | Base64 JPEG/PNG frame. |
| Client -> Server | `audio.chunk` | `data` | Base64 PCM16 16 kHz mono audio bytes. |
| Client -> Server | `video.query` | `text` | Ask a question over the buffered frames and audio. |
| Client -> Server | `video.done` | none | End the WebSocket session. |
| Server -> Client | `response.start` | none | Query generation started. |
| Server -> Client | `response.text.delta` | `delta` | Incremental text output. |
| Server -> Client | `response.text.done` | `text` | Final text output for the query. |
| Server -> Client | `response.audio.delta` | `data`, `format` | Incremental generated audio, base64 WAV. |
| Server -> Client | `response.audio.done` | none | Audio output finished. |
| Server -> Client | `session.done` | none | Session closed. |
| Server -> Client | `error` | `message` | Recoverable protocol or generation error. |

### `session.config` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string or null | null | Optional model name. Usually omitted because the server hosts one model. |
| `modalities` | list[string] | `["text", "audio"]` | Output modalities. Use `["text"]`, `["audio"]`, or both. |
| `num_frames` | integer, 1-128 | `4` | Number of buffered frames sampled for each query. |
| `max_frames` | integer, 1-256 | `50` | Maximum retained frame buffer size. Oldest frames are evicted first. |
| `system_prompt` | string or null | null | Optional custom system prompt. |
| `use_audio_in_video` | bool | `true` | Include streamed audio chunks in multimodal video understanding when audio is present. |
| `sampling_params_list` | list or null | null | Optional per-stage sampling parameter overrides. |
| `enable_frame_filter` | bool | `true` | Enable EVS near-duplicate frame filtering. |
| `frame_filter_threshold` | float, 0.0-1.0 | `0.95` | EVS similarity threshold. Higher keeps more frames; lower drops more near-duplicates. |

### Legacy Aliases

The server accepts these legacy field names and rewrites them before validation. New clients should send the canonical names above.

| Legacy field | Canonical field |
|--------------|-----------------|
| `num_sample_frames` | `num_frames` |
| `evs_enabled` | `enable_frame_filter` |
| `evs_threshold` | `frame_filter_threshold` |

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `VLLM_VIDEO_ASYNC_CHUNK` | `on`, `off` | `on` | Wire-level streaming switch. `off` buffers server-side deltas and emits coalesced outputs at the end of a query. |
| `VLLM_VIDEO_AUDIO_DELTA_MODE` | `fast`, `slow` | `fast` | Audio delta extraction strategy. `fast` emits only newly produced chunks; `slow` recomputes from accumulated audio and exists for A/B verification. |

## EVS Semantics

EVS compares downsampled frames and drops near-duplicate frames before they enter the session frame buffer. `frame_filter_threshold` controls retention: higher values are more permissive and keep more frames; lower values are more aggressive and drop more similar frames.

## Known Limitations

- Session KV reuse and incremental prefill are not implemented in this PR. Each `video.query` rebuilds the model prompt from the retained frame and audio buffers.
- Back-to-back short replies can still expose an engine-layer scheduler race. The PR notes an observed workaround of at least 200 ms idle between turns when clients repeatedly see idle timeouts.
- If the audio buffer exceeds the server limit, the server emits `Audio buffer overflow` and clears the currently buffered audio for the session.
- The API is intended for Qwen3-Omni streaming video understanding; other models may not support the same multimodal processor arguments.
