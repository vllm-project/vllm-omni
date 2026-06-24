# Streaming Video Generation

This example uses the custom WebSocket endpoint `WS /v1/realtime/video` to receive a video byte stream as chunks are produced.
It covers text-only video generation. Image/reference input is intentionally not included for now.

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
| Client to server | `session.start` | JSON text: `{"type":"session.start","model":"...","prompt":"...","format":"m4s"}` | Starts generation. `format` is optional and accepts `m4s` (default). Sampling fields such as `width`, `height`, `fps`, `num_frames`, and `extra_params` may be included. |
| Server to client | `video.start` | JSON text: `{"type":"video.start","request_id":"...","format":"m4s","config":{...}}` | Confirms the session and mirrors the accepted `format`. |
| Server to client | Video chunk | Binary WebSocket frame | Fragmented MP4 (`m4s`) video bytes. |
| Client to server | `session.stop` | JSON text: `{"type":"session.stop"}` | Requests cancellation of the active session. |
| Client to server | `session.ping` | JSON text: `{"type":"session.ping"}` | Optional keepalive; refreshes the server stall clock. |
| Server to client | `session.done` | JSON text: `{"type":"session.done","request_id":"...","chunks":3,"stopped":false}` | Ends a completed or stopped session. |
| Server to client | `session.pong` | JSON text: `{"type":"session.pong"}` | Reply to `session.ping`. |
| Server to client | `error` | JSON text: `{"type":"error","message":"..."}` | Reports invalid input, unsupported formats, generation failures, control-message errors, or stall timeout. |

During generation the client normally sends only `session.start` and then receives binary chunks; silence on the client socket is expected. The server closes the session with a stall error only when there is no engine progress and no `session.ping` for about 60 seconds.

## Install Client Dependency

```bash
pip install websockets
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
