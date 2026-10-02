# Streaming Video Input API

vLLM-Omni provides a WebSocket API for streaming video frames and optional audio chunks into Qwen3-Omni, then asking questions over the buffered session context.

Each server instance runs a single model specified at startup with `vllm serve <model> --omni`.

## Quick Start

### Start the Server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml \
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
| ----------- | ------ | ----------------- | ------------- |
| Client -> Server | `session.config` | none | First message. Configures output modalities, frame sampling, EVS, and prompts. |
| Client -> Server | `video.frame` | `data` | Base64 JPEG/PNG frame. |
| Client -> Server | `audio.chunk` | `data` | Base64 PCM16 16 kHz mono audio bytes. |
| Client -> Server | `video.query` | `text` | Ask a question over the buffered frames and audio. |
| Client -> Server | `video.done` | none | End the WebSocket session. |
| Server -> Client | `response.start` | none | Query generation started. |
| Server -> Client | `response.text.delta` | `delta` | Incremental text output. |
| Server -> Client | `response.text.done` | `text` | Final text output for the query. |
| Server -> Client | `response.output_audio.delta` | `data`, `format` | Incremental generated audio, base64 WAV. |
| Server -> Client | `response.output_audio.done` | none | Audio output finished. |
| Server -> Client | `session.done` | none | Session closed. |
| Server -> Client | `error` | `message` | Recoverable protocol or generation error. |

### Frame Consumption Reporting

When buffered frames include client-supplied `frame_id` values, the server emits `video.frames.consumed` after the first engine output. Its `frame_ids`, `frames`, and `latest_pts_ms` describe the images included in that query's prompt.

Queries stride-sample the buffered frames, keeping the last frame, then exclude frames already known to have failed decoding. Excluded frames are not replaced with other buffered frames. Frames whose background decoding has not finished remain eligible through the image URL path.

**Bugfix compatibility note:** `video.frames.consumed` now excludes known decode failures that were already excluded from the prompt. Older versions could report those frames and their timestamps as consumed. The event name and fields are unchanged; an empty selection reports empty lists and `latest_pts_ms: null`.

### `session.config` Fields

| Field | Type | Default | Description |
| ------- | ------ | --------- | ------------- |
| `model` | string or null | null | Optional model name. Usually omitted because the server hosts one model. |
| `modalities` | list[string] | `["text", "audio"]` | Output modalities. Use `["text"]`, `["audio"]`, or both. |
| `num_frames` | integer, 1-128 | `4` | Number of buffered frames sampled for each query. |
| `max_frames` | integer, 1-256 | `50` | Maximum retained frame buffer size. Oldest frames are evicted first. |
| `system_prompt` | string or null | null | Optional custom system prompt. |
| `max_history_turns` | integer >= 1 or null | `1` | Completed user/assistant turns included as text in each new prompt. `null` includes all retained turns. |
| `use_audio_in_video` | bool | `true` | Include streamed audio chunks in multimodal video understanding when audio is present. |
| `sampling_params_list` | list or null | null | Optional per-stage parameter dictionaries. Each provided entry replaces that stage's deployment sampling settings. |
| `enable_frame_filter` | bool | `true` | Enable EVS near-duplicate frame filtering. |
| `frame_filter_threshold` | float, 0.0-1.0 | `0.95` | EVS similarity threshold. Higher keeps more frames; lower drops more near-duplicates. |

Sampling parameter bugfix: `sampling_params_list` is now forwarded to the engine;
earlier versions accepted this field but silently used deployment defaults instead.
Each provided entry constructs a fresh `SamplingParams` for that stage in pipeline
order; it is not merged with that stage's deployment defaults. Fields absent from a
provided entry use `SamplingParams` constructor defaults. Only omitted trailing
stages keep their deployment defaults. An omitted, `null`, or empty list keeps the
engine defaults. Invalid sampling parameters return an `error` when the query is
submitted.

For example, `[{"temperature": 0.2, "max_tokens": 64}]` configures the thinker while
the talker and code2wav retain their deployment defaults. With only
`[{"temperature": 0.2}]`, the thinker's `max_tokens` and `top_p` use constructor
defaults, not its YAML values.

### Conversation History

By default, each query includes only the most recent completed user/assistant
turn. To include the two most recent turns, set `"max_history_turns": 2` in the
initial `session.config` message. Set it to `null` to include all retained turns:

```json
{"type": "session.config", "max_history_turns": null}
```

The system prompt and current query are always included independently of this
limit. Earlier messages contribute only text; their images and audio are not
replayed. The current frame and audio buffers follow their existing rules.

This setting selects history for the model prompt. It does not limit the session's
stored history or reserve KV cache across queries. Larger histories consume more
context and remain subject to the model's context limit; the server does not
automatically trim them to a token budget. Omitting the field keeps the existing
one-turn behavior. Speech quality with longer context remains model-dependent.

### Legacy Aliases

The server accepts these legacy field names and rewrites them before validation. New clients should send the canonical names above.

| Legacy field | Canonical field |
| -------------- | ----------------- |
| `num_sample_frames` | `num_frames` |
| `evs_enabled` | `enable_frame_filter` |
| `evs_threshold` | `frame_filter_threshold` |

### Environment Variables

| Variable | Values | Default | Description |
| -------- | ------ | ------- | ----------- |
| `VLLM_VIDEO_ASYNC_CHUNK` | `on`, `off` | `on` | Wire-level streaming switch. `off` buffers server-side deltas and emits coalesced outputs at the end of a query. |
| `VLLM_VIDEO_AUDIO_DELTA_MODE` | `fast`, `slow` | `fast` | Both settings forward every fresh engine audio delta. `slow` is retained as a compatibility alias for `fast`. |

## EVS Semantics

EVS compares downsampled frames and drops near-duplicate frames before they enter the session frame buffer. `frame_filter_threshold` controls retention: higher values are more permissive and keep more frames; lower values are more aggressive and drop more similar frames.

## Known Limitations

- Session KV reuse and incremental prefill are not implemented in this PR. Each `video.query` rebuilds the model prompt from the retained frame and audio buffers.
- Back-to-back short replies can still expose an engine-layer scheduler race. The PR notes an observed workaround of at least 200 ms idle between turns when clients repeatedly see idle timeouts.
- If the audio buffer exceeds the server limit, the server emits `Audio buffer overflow` and clears the currently buffered audio for the session.
- The API is intended for Qwen3-Omni streaming video understanding; other models may not support the same multimodal processor arguments.
