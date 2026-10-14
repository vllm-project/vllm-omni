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
| ----------- | ---------------- | --------- |
| Client to server | `session.start` | Starts generation with a prompt and video parameters |
| Server to client | `video.start` | Confirms the request ID, format, and accepted configuration |
| Server to client | `video.chunk_metadata` | Describes the next binary media frame |
| Server to client | Binary frame | Carries one fragmented MP4 chunk |
| Client to server | `session.interaction` | Queues an optional interaction (e.g., prompt. camera action) during generation |
| Server to client | `session.interaction.queued` | Confirms that an interaction was queued |
| Client to server | `session.ping` | Refreshes the stall clock |
| Server to client | `session.pong` | Acknowledges a ping |
| Client to server | `session.stop` | Cancels the active generation; the resulting `session.done` reports `stop_latency_s` |
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

## Playback Backpressure

A compatible single-stage interactive pipeline can pace generation against a
client playhead. Enable step execution, `streaming_output=True` and
`max_num_seqs=1`, then include these fields in `session.start`:

```json
{
  "streaming_pacing": {
    "enabled": true,
    "mode": "client_playhead",
    "max_lead_ms": 1500,
    "max_output_bytes": 67108864,
    "idle_timeout_s": 30
  }
}
```

This is a request fragment; include the model's normal prompt, image and video
parameters. Omitting `streaming_pacing` or setting `enabled` to `false` keeps
unpaced execution. `streaming_buffer_seconds` remains a compatibility alias for
existing clients. The ordinary `/v1/videos` endpoint does not accept
playback pacing.

While playing, repeatedly send the actual cumulative media playhead:

```json
{"type":"session.playback","position_seconds":1.25}
```

The engine starts the next chunk only if **generated duration + next chunk
duration - played duration** fits the configured window. The window must fit a
whole chunk. Old feedback is ignored and future feedback is clamped to already
generated media. Paused playback blocks advancement; pings and duplicate
playback positions do not prevent the consumption idle timeout.

A paced stream retains at most one generation chunk until encoding and sending
finish. The byte limit checks decoded payloads and encoded fragments; it does
not cap model workspace, transient decode allocations or process RSS. Oversize
output, idle timeout, Stop and disconnect release request state without waiting
for another inference request. Other requests retain the single-stream queueing
behavior.

Chunk metadata includes `pacing.lead_seconds`, cumulative
`pacing.wait_seconds`, `pacing.retained_bytes` and `output_chunk_index`.
`AsyncOmni.generate` consumers release a chunk by advancing their async iterator;
close or abort an abandoned iterator. Controls and playback feedback remain
receivable while media output is blocked, through bounded control queues.

For interaction timing, queue limits and cancellation semantics, see the
[streaming video example protocol](../../examples/online_serving/streaming_video_generation/README.md).

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

Models that register the camera modality (for example LingBot-World) also
accept structural SE3 payloads on `event.multi_modal_data.camera`. Coordinates
follow Unity semantics (`+X` right, `+Y` up, `+Z` forward; same as LingBot's
native camera frame). WASD key tokens are not admitted by the engine:

```json
{
  "type": "session.interaction",
  "interaction": {
    "event_id": "forward-1",
    "event": {
      "multi_modal_data": {
        "camera": {
          "mode": "velocity",
          "data": {
            "translation": [0.0, 0.0, 0.05],
            "rotation": [0.0, 0.0, 0.0, 1.0]
          }
        }
      }
    }
  }
}
```

`mode` is `target` (lerp to an absolute pose over `transition_chunks`) or
`velocity` (hold a per-frame SE3 delta until replaced). Prompt and camera may
be combined in one interaction event.

This feature is pipeline-dependent. An enabled endpoint does not imply that
the loaded model supports midway prompt or camera updates.

The default start timeout is 10 seconds and the server reports a stall after
about 60 seconds without generation progress or a `session.ping`. See the
[streaming video generation example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/streaming_video_generation)
for prompt/camera-update scheduling and browser playback.
