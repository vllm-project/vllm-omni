# Streaming Video Generation

This example uses the custom WebSocket endpoint `WS /v1/realtime/video` to receive a video byte stream as chunks are produced.
It covers text-to-video, and image-conditioned models through `--image-reference`.

## Start The Server

Start a diffusion video model with streaming output enabled:

```bash
vllm serve BestWishYsh/Helios-Distilled \
  --omni \
  --diffusion-streaming-output \
  --port 8000
```

The `--diffusion-streaming-output` CLI flag is forwarded as `streaming_output=True` in the default diffusion stage `engine_args`, then loaded by `OmniDiffusionConfig.from_kwargs()`.

## WebSocket Protocol

| Direction | Message | Format | Description |
| --- | --- | --- | --- |
| Client to server | `session.start` | JSON text: `{"type":"session.start","model":"...","prompt":"...","format":"m4s"}` | Starts generation. `format` is optional and accepts `m4s` (default). Sampling fields such as `width`, `height`, `fps`, `num_frames`, and `extra_params` may be included. Image-conditioned models take a first frame as `image_reference`: `{"image_url":"https://..."}` or a `data:` URL. |
| Client to server | `session.interaction` | JSON text: `{"type":"session.interaction","interaction":{"event_id":"xxx","event":{"prompt":"..."},"transition_chunks":3}}` | Updates the active prompt and/or camera midway through generation. Camera at `event.multi_modal_data.camera` is structural SE3 (`mode` + `translation`/`rotation`). `event_id` is optional and `transition_chunks` defaults to the model setting. |
| Server to client | `video.start` | JSON text: `{"type":"video.start","request_id":"...","format":"m4s","config":{...}}` | Confirms the session and mirrors the accepted `format`. |
| Server to client | `video.chunk_metadata` | JSON text: `{"type":"video.chunk_metadata","request_id":"...","kind":"media","transport_chunk_index":0,"generation_chunk_index":0,"num_frames":9,"byte_length":1234,"started_event_ids":[],"active_event_ids":[],"completed_event_ids":[]}` | Precedes each binary frame and describes the immediately following payload. |
| Server to client | Video chunk | Binary WebSocket frame | Fragmented MP4 (`m4s`) video bytes. |
| Client to server | `session.stop` | JSON text: `{"type":"session.stop"}` | Requests cancellation of the active session. |
| Client to server | `session.playback` | JSON text: `{"type":"session.playback","position_seconds":1.25}` | Playback progress for sessions that opt into `streaming_pacing`. Position must be finite and non-negative. |
| Client to server | `session.ping` | JSON text: `{"type":"session.ping"}` | Optional keepalive; refreshes the server stall clock. |
| Server to client | `session.interaction.queued` | JSON text: `{"type":"session.interaction.queued","request_id":"...","event_id":"xxx"}` | Confirms the server queued the interaction for the active request. |
| Server to client | `session.done` | JSON text: `{"type":"session.done","request_id":"...","chunks":3,"stopped":false}` | Ends a completed or stopped session. Sessions stopped by `session.stop` also carry `stop_latency_s`: seconds from stop receipt to the done frame. |
| Server to client | `session.pong` | JSON text: `{"type":"session.pong"}` | Reply to `session.ping`. |
| Server to client | `error` | JSON text: `{"type":"error","message":"..."}` | Reports invalid input, unsupported formats, inactive requests, unsupported prompt updates, generation failures, control-message errors, or stall timeout. |

Backpressure is a request/session option. Set the canonical `streaming_pacing` object in `session.start`; omitting it or setting `enabled` to `false` keeps unpaced generation. The server requires `streaming_output=True`, step execution, `max_num_seqs=1`, and an interactive chunked pipeline. `streaming_buffer_seconds` remains a compatibility alias for existing clients.

After opting in, send `session.playback` repeatedly as the displayed video playhead advances. This message is runtime feedback, not the switch that enables pacing: `position_seconds` updates the generated-versus-played watermark, wakes a paused boundary, and is clamped to generated media. The engine admits a chunk only if generated media plus that chunk, minus reported playback, fits the window. The window must fit one complete chunk; a smaller value fails explicitly. No feedback pauses generation after filling the window. Controls, pings and cancellation still work while paused; releasing playback resumes generation. Backwards positions are ignored. A paced session blocked without consumption progress expires after `streaming_pacing.idle_timeout_s` (default 30 seconds). Pings and repeated playback positions do not reset this timeout. Budget enough lead for chunk generation, encoding and the playback-feedback round trip; a window smaller than this combined delay causes repeated underruns. Report the actual playhead rather than advancing it to hide stalls. A client's startup buffering threshold must stay below the generation window, with room for frames still held by the encoder.

For paced sessions, the engine opens an observation window when a chunk starts, sized by that chunk's actual media duration. The next chunk waits for the window deadline, playback credit, and delivery of reserved controls. The engine supplies both event arrival times and resume-boundary timestamps to workers; clients cannot assign these internal timestamps. This avoids mixing engine and worker clocks. If inference exceeds the media duration, no additional clock wait is imposed. Unpaced sessions keep their original boundary timing.

The generic runtime keeps interaction ingress ordered and reports engine-owned timestamps. Camera velocity events before the window deadline are integrated continuously onto latent control samples, including short presses mixed with targets. Events at or after the deadline remain queued for the next window. Controls deferred across a pause retain their relative intervals, shifted to the next window's start; fresh controls keep their window timestamps, shifted only when a backlog overlaps them. Events that do not fit the media window carry their remaining interval into a following window instead of collapsing press/release pairs. The wall-clock pause does not add camera travel. Consecutive pending targets coalesce, while intervening velocity events retain their intervals. A target arriving after the media deadline but before the resume boundary can affect the resumed chunk's last control sample. It supersedes older deferred controls so they cannot replay after that target.

`session.interaction.queued` confirms ingress, not application. Paced acknowledgements include an engine-clock `received_at`. Chunk metadata reports `started_event_ids`, `active_event_ids`, `completed_event_ids`, and, when non-empty, `coalesced_event_ids` for commands replaced before application. `generation_chunk_index` identifies their reporting/application chunk. Metadata does not establish when the browser displays camera movement. Direct AsyncOmni clients should pass `track_playback=True` to `submit_interaction_async` for paced sessions. Prompt updates reuse their existing revision and cache-invalidation path and report replaced pending IDs.

A pause with no new controls adds no camera travel. Explicit press/release intervals queued during a pause are preserved in media time and can replay across multiple resumed chunks. This keeps brief controls from disappearing; clients wanting to discard or shorten paused input must apply that policy before submitting it. Pending controls and replacement IDs count against the per-modality budget below.

Interactions are queued in order separately from playback feedback and aborts. The pending interaction queue is bounded to 64 messages; overflow reports an error and terminates the affected video request rather than silently dropping a control command. This does not shut down the model engine. Each worker modality retains at most 256 pending control records, including coalesced prompt IDs; overflow rejects the update before changing any modality. Interaction event IDs are limited to 128 characters and the duplicate-ID history to 4096 entries. Start a new session when that history fills.

During generation the client normally sends only `session.start` and then receives binary chunks; silence on the client socket is expected. The server closes the session with a stall error only when there is no engine progress and no `session.ping` for about 60 seconds.

Paced sessions retain at most one generation chunk until its consumer finishes encoding/sending and resumes iteration. `streaming_pacing.max_output_bytes` (default 64 MiB) limits each decoded payload and encoded fragment; an oversized chunk terminates the request. This bounds queued output, not model workspace, transient decode allocations or codec/transport copies. Final chunks retain the same delivery credit. Direct `AsyncOmni.generate` callers return credit by advancing their async iterator; abandoning it requires closing/aborting the request.

Chunk metadata includes `pacing` with `lead_seconds`, cumulative `wait_seconds` and `retained_bytes`. The lead includes the admitted chunk; camera IDs still describe generation-time application rather than browser display time. Cancellation, disconnect and idle expiry retire worker request state even if no new inference follows. Other requests keep the existing single-stream queueing behavior.

The WebSocket control reader uses a bounded response outbox so a blocked media send cannot stop playback/Stop reception. Output send timeouts also apply to the final `session.done`. This preserves metadata/binary ordering and bounds outstanding control replies.

## Install Client Dependency

```bash
pip install av websockets
# For the Gradio demo:
pip install 'vllm-omni[demo]' websockets
```

## Run The Client

```bash
python streaming_video_client.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model BestWishYsh/Helios-Distilled \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --width 640 \
  --height 384 \
  --fps 16 \
  --num-frames 99 \
  --guidance-scale 1.0 \
  --seed 42 \
  --output helios_stream.mp4
```

The example client sends one `session.start` message, prints each received binary video chunk with its byte size and elapsed time, and saves the received bytes to `--output` after `session.done`.
The client remuxes the gathered stream to a regular progressive MP4 file so that local playback knows the video duration.

Schedule midway prompt updates with `--prompt-updates`. None are sent unless you ask for them, and only pipelines that implement midway prompt updates accept them. Each entry uses `"at"` as seconds on the client clock after the server sends `video.start`:

```bash
python streaming_video_client.py \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --prompt-updates '[
    {"at": 2.5, "prompt": "A sea turtle glides past the reeds"},
    {"at": 5.0, "prompt": "Sunlight breaks through the morning mist", "transition_chunks": 2}
  ]'
```

Schedule midway camera updates with `--camera-updates` when the served model
registers the camera modality (e.g., LingBot World 2).
It accepts `["camera"]["data"]["translation"/"rotation"]` in the endpoint's required SE3 format.
It also accepts `["actions"]` at the update level and converts keystrokes to the SE3 format.

```bash
python streaming_video_client.py \
  --model robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --image-reference /path/to/first_frame.png \
  --camera-updates '[
    {"at": 1.0, "actions": ["w"]},
    {"at": 3.0, "camera": {"mode": "target", "data": {"translation": [0.0, 0.0, 1.0], "rotation": [0.0, 0.0, 0.0, 1.0]}}, "transition_chunks": 2}
  ]'
```

The Helios demo continues the default prompt into an underwater storm:

```bash
python streaming_video_client.py \
  --model BestWishYsh/Helios-Distilled \
  --prompt-updates '[
    {"at": 4,
     "prompt": "An underwater tornado appears and affects the ocean floor in a dramatic and chaotic scene. The water is murky, swirling violently, carrying debris and marine life into the vortex. The tropical fish on the scene all swim in panic, trying to avoid the powerful currents. The camera remains stationary, capturing the intensity of the underwater tornado as it disrupts the serene ocean floor. Close-up shot emphasizing the turbulent motion and destruction."},
    {"at": 11,
     "prompt": "The swirling underwater vortex now seizes a heavy, encrusted treasure chest, its lid flapping open as it is smashed onto the ocean floor. Gold coins and silver trinkets spill out, glittering briefly in the murky water before being swept instantly into the violent funnel. The heavy wooden box tumbles end over end, colliding with floating rocks and adding to the debris field. Swirling sediment and bubbles surround the spilling fortune, highlighting the chaotic power of the storm as it ravages the seabed. Close-up shot emphasizing the turbulent motion and destruction."}
  ]'
```

## Run The Gradio Demo

```bash
python gradio_demo.py \
  --host 127.0.0.1 \
  --port 7860
```

The Gradio demo requests fMP4 (`m4s`) chunks and appends them directly in the browser with a Media Source Extensions player.

## Model Choice

### Helios

The example uses `BestWishYsh/Helios-Distilled` model by default.

To ensure streaming-level generation speed, `pyramid_num_inference_steps_list` is suggested to be as low as `[1, 1, 1]`. Both example clients uses the following Helios-Distilled preset by default:

```json
{
  "is_enable_stage2": true,
  "pyramid_num_stages": 3,
  "pyramid_num_inference_steps_list": [1, 1, 1],
  "is_amplify_first_chunk": true
}
```

Disable it in the CLI example with `--no-helios-distilled-preset`, or override/extend it with `--extra-params`:

```bash
python streaming_video_client.py \
  --extra-params '{"pyramid_num_inference_steps_list":[2, 2, 2]}'
```

### LingBot-World 2.0

`robbyant/lingbot-world-v2-14b-causal-fast-diffusers` is image-conditioned and runs on the
AR-Diffusion engine, so it needs the stepwise deploy config and a first frame. See
[the recipe](../../../recipes/Robbyant/LingBot-World-2.0.md) for the constraints on
`width`/`height` and `num_frames`.

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni \
  --deploy-config vllm_omni/deploy/lingbot_world_v2_stepwise.yaml \
  --port 8000
```

```bash
python streaming_video_client.py \
  --model robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --prompt "The camera moves slowly forward through the scene." \
  --image-reference /path/to/first_frame.png \
  --width 832 --height 480 --num-frames 33 --fps 16 --seed 42 \
  --extra-params '{"camera_action_script":[[["w"],["w"],["w"]],[["a"],[],[]],[[],[],[]]]}' \
  --output lingbot_world_v2_stream.mp4
```

`camera_action_script` is an optional request-scoped WASD script (one
three-latent-frame action list per generated chunk). Prefer mid-session
`session.interaction` camera control via `--camera-updates` when you need
live motion; the engine wire format is structural SE3 only.
