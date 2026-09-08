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
| Client to server | `session.interaction` | JSON text: `{"type":"session.interaction","interaction":{"event_id":"xxx","event":{"prompt":"..."},"transition_chunks":3}}` | Updates the active prompt midway through generation. `event_id` is optional and `transition_chunks` defaults to the model setting. |
| Server to client | `video.start` | JSON text: `{"type":"video.start","request_id":"...","format":"m4s","config":{...}}` | Confirms the session and mirrors the accepted `format`. |
| Server to client | `video.chunk_metadata` | JSON text: `{"type":"video.chunk_metadata","request_id":"...","kind":"media","transport_chunk_index":0,"generation_chunk_index":0,"num_frames":9,"byte_length":1234,"started_event_ids":[],"active_event_ids":[],"completed_event_ids":[]}` | Precedes each binary frame and describes the immediately following payload. |
| Server to client | Video chunk | Binary WebSocket frame | Fragmented MP4 (`m4s`) video bytes. |
| Client to server | `session.stop` | JSON text: `{"type":"session.stop"}` | Requests cancellation of the active session. |
| Client to server | `session.ping` | JSON text: `{"type":"session.ping"}` | Optional keepalive; refreshes the server stall clock. |
| Server to client | `session.interaction.queued` | JSON text: `{"type":"session.interaction.queued","request_id":"...","event_id":"xxx"}` | Confirms the server queued the interaction for the active request. |
| Server to client | `session.done` | JSON text: `{"type":"session.done","request_id":"...","chunks":3,"stopped":false}` | Ends a completed or stopped session. |
| Server to client | `session.pong` | JSON text: `{"type":"session.pong"}` | Reply to `session.ping`. |
| Server to client | `error` | JSON text: `{"type":"error","message":"..."}` | Reports invalid input, unsupported formats, inactive requests, unsupported prompt updates, generation failures, control-message errors, or stall timeout. |

During generation the client normally sends only `session.start` and then receives binary chunks; silence on the client socket is expected. The server closes the session with a stall error only when there is no engine progress and no `session.ping` for about 60 seconds.

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

The client sends one `session.start` message, prints each received binary video chunk with its byte size and elapsed time, and saves the received bytes to `--output` after `session.done`.
The client remuxes the gathered stream to a regular progressive MP4 file so that local playback knows the video duration.

Schedule midway prompt updates with `--prompt-updates`. None are sent unless you ask for them, and only pipelines that implement midway prompt updates accept them (LingBot-World, for one, does not, and rejects the whole session). Each entry uses `"at"` as seconds on the client clock after the server sends `video.start`:

```bash
python streaming_video_client.py \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --prompt-updates '[
    {"at": 2.5, "prompt": "A sea turtle glides past the reeds"},
    {"at": 5.0, "prompt": "Sunlight breaks through the morning mist", "transition_chunks": 2}
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

`camera_action_script` carries one three-latent-frame WASD action list per generated
chunk. Mid-session `session.interaction` only updates the prompt, so a LingBot rollout
follows the camera script it started with.
