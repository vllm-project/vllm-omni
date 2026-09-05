# Streaming Video Output WebSocket API

Use `WS /v1/realtime/video` to receive fragmented MP4 chunks while a compatible
diffusion video model is generating. This endpoint streams generated video
output; it is unrelated to [`WS /v1/video/chat/stream`](video_stream_api.md),
which streams live video input into a conversational model.

## Quick Start

Start a model that supports step execution and enable diffusion streaming
output:

```bash
vllm serve BestWishYsh/Helios-Distilled --omni \
  --diffusion-streaming-output \
  --port 8091
```

Install the client dependency and run the provided client:

```bash
pip install av websockets

python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --host 127.0.0.1 \
  --port 8091 \
  --model BestWishYsh/Helios-Distilled \
  --prompt "A serene lakeside sunrise with mist over the water" \
  --output output.mp4
```

## Protocol

Control events are JSON text frames. Each `video.chunk_metadata` event is
immediately followed by one binary WebSocket frame containing fragmented MP4
bytes.

| Direction | Event or frame | Purpose |
|-----------|----------------|---------|
| Client to server | `session.start` | Starts generation with a prompt and video parameters |
| Server to client | `video.start` | Confirms the request ID, format, and accepted configuration |
| Server to client | `video.chunk_metadata` | Describes the next binary media frame |
| Server to client | Binary frame | Carries one fragmented MP4 chunk |
| Client to server | `session.interaction` | Queues an optional prompt update during generation |
| Server to client | `session.interaction.queued` | Confirms that an interaction was queued |
| Client to server | `session.ping` | Refreshes the stall clock |
| Server to client | `session.pong` | Acknowledges a ping |
| Client to server | `session.stop` | Cancels the active generation |
| Server to client | `session.done` | Reports completion or cancellation and the chunk count |
| Server to client | `error` | Reports validation, generation, or timeout failures |

Start a minimal session with:

```json
{
  "type": "session.start",
  "model": "BestWishYsh/Helios-Distilled",
  "prompt": "A serene lakeside sunrise",
  "format": "m4s",
  "width": 640,
  "height": 384,
  "fps": 16,
  "num_frames": 99
}
```

Only `m4s` output is currently accepted. Concatenate the binary frames in
receive order, then remux the fragmented stream if the target player requires
a progressive MP4 file.

## Mid-Generation Interaction

Models that implement interaction updates can change the active prompt:

```json
{
  "type": "session.interaction",
  "interaction": {
    "event_id": "sunlight-update",
    "event": {"prompt": "Sunlight breaks through the mist"},
    "transition_chunks": 2
  }
}
```

This feature is pipeline-dependent. An enabled endpoint does not imply that
the loaded model supports midway prompt updates.

The default start timeout is 10 seconds and the server reports a stall after
about 60 seconds without generation progress or a `session.ping`. See the
[streaming video generation example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/streaming_video_generation)
for prompt-update scheduling and browser playback.
