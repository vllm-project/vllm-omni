# Shared Realtime Web UI

One browser shell provides microphone capture, speaker playback, a level meter,
conversation text, and an event log. Model profiles own the wire protocol.
This implements [RFC #7222](https://github.com/vllm-project/vllm-omni/issues/7222).

| Profile | Turn control | Camera | Playback acknowledgement |
| --- | --- | --- | --- |
| `minicpm-native` | Model-controlled listen/speak, continuous audio input | Frames accompany audio | Yes |
| `qwen3-turn --stt` (default) | User presses **Send turn** | No | No |
| `qwen3-turn --vad` | Server detects trailing silence; speech interrupts replies | Sampled frames with each spoken turn | Yes |

Qwen3 VAD uses an engine-owned duplex plugin: microphone upload continues
while replies stream, and speech can interrupt generation and playback. The
checkpoint still generates committed turns; this is not native streaming-input
KV decoding. STT remains explicit-turn and audio-only.

## MiniCPM: existing command stays valid

Start your MiniCPM backend using the [existing deployment instructions](../minicpmo/README.md), then:

```bash
python -m examples.online_serving.minicpmo.realtime_web \
    --ws-backend ws://127.0.0.1:8099 \
    --ref-audio /path/to/ref_minicpm_signature.wav
```

The compatibility wrapper serves the shared assets with `minicpm-native`.
The current native query, `extra_body`, reference voice, continuous microphone
upload, camera frames and playback acknowledgements are retained.

## Qwen3: explicit-turn STT adapter

Start the backend, then the UI in a second terminal:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091

python -m examples.online_serving.qwen3_omni.realtime_web \
    --backend ws://127.0.0.1:8091 --stt --port 7863
```

Open `http://localhost:7863`, start a session, speak, and press **Send turn**.
Wait for the answer to finish playing before speaking again. The UI opens a
fresh STT connection for each turn; the displayed conversation is a local log,
and **previous turns are not provided as model history**. The STT endpoint
accepts only model selection, so system-prompt controls are hidden in this mode.

The adapter follows the shipped `qwen3_omni/openai_realtime_client.py` and
`vllm_omni/entrypoints/openai/realtime_connection.py`:

1. Explicit `duplex=0` selects the legacy STT handler on a turn deployment.
   It does not enable STT on a duplex deployment.
2. Send `{type: "session.update", model: ...}` and `commit(final=false)`.
3. Stream mono PCM16 at 16 kHz; **Send turn** flushes buffered audio, then sends `commit(final=true)`.
4. `response.output_audio.delta.audio` contains PCM; `response.output_audio.done`
   terminates this adapter, including a response with no audio.
5. The local implementation emits **model output** on `transcription.*`, so the
   UI displays it as assistant text. This differs from the early RFC's suggested
   user-transcript mapping. The VAD adapter also supports `response.output_text.*`
   and audio-transcript events without duplicating the answer.

## Qwen3 with Server VAD (automatic turns)

Run commands from the repository root with the vLLM-Omni environment activated.
Prepare the pinned Silero v6.2 ONNX artifact as described in the
[Qwen3 Server VAD instructions](../qwen3_omni/README.md#realtime-websocket-client-openai_realtime_clientpy).
The backend requires ONNX Runtime and a compatible local artifact; setting
`--vad` on the UI alone does not enable VAD on the backend.

Create an overlay named `qwen3_vad.yaml`, replacing both paths with absolute paths
on the backend host:

```yaml
base_config: /path/to/vllm-omni/vllm_omni/deploy/qwen3_omni_duplex.yaml
duplex_session:
  server_vad_model_path: /path/to/silero_vad.onnx
```

The base configuration enables duplex mode and sets the per-prompt audio and image limits.

Start the backend:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config qwen3_vad.yaml
```

In a second terminal, wait until the health check succeeds, then start the UI:

```bash
curl --fail http://127.0.0.1:8091/health
python -m examples.online_serving.qwen3_omni.realtime_web \
    --backend ws://127.0.0.1:8091 --vad --port 7863
```

Open `http://localhost:7863` and start a session. Speak and pause: the server
commits the turn automatically after 500 ms of silence. No **Send turn** action
is needed. Defaults are a speech threshold of 0.5 and 300 ms of prefix padding.
Speak again to interrupt an answer. The engine owns input buffering, VAD,
request cancellation, playback acknowledgements, and conversation history.
Each committed utterance starts a normal Thinker → Talker → Code2Wav request;
it does not append input to a live Qwen KV cache. The capability response reports
`implementation_level: turn_based_duplex` and `supports_core_resumable_request: false`.

The plugin retains up to four audio inputs (including the current turn), pruning
older audio at an 8 MiB base64 budget; one current utterance is bounded by the
engine's pending-input limit. The UI reports playback progress both on normal
drain and on interruption. Qwen does not provide text/audio alignment, so assistant
text enters history only after the complete audio response has been generated and
acknowledged as played. A partially played answer contributes no guessed text
prefix. Image conversation items remain available until deleted. Model context limits still apply.
The bundled duplex deploy allows four audio inputs and eight images per prompt.
Use headphones to avoid speaker audio triggering VAD interruption. Reference
voices and tool calls are not supported by this plugin.

### Switching between VAD and manual turns

| Mode | Backend | UI flag | Submit a turn |
| --- | --- | --- | --- |
| Without VAD | Default Qwen turn deployment | `--stt` (default) | Press **Send turn** |
| With VAD | Server VAD configuration and Silero artifact | `--vad` | Pause after speaking |

Stop the existing UI process before starting another on the same port, then
refresh the browser and reconnect. To use STT, start a backend with the default
turn deployment and restart the UI with `--stt`. The duplex backend exposes
Realtime duplex and ChatCompletion; it does not expose the legacy STT handler.
To enable VAD on a default backend, first restart that backend with the overlay.
Make `--backend` match the actual backend port; the UI and backend use separate
ports. `Connection refused` means the target backend is unavailable: check its
startup log and `/health` before connecting.

VAD connects with `duplex=1` and sends nested `session.audio.input` configuration
with `create_response: true` and `interrupt_response: true`. It sends playback
ACKs when speaker audio drains. `response.audio.done` / `response.output_audio.done`
only drain playback; `response.done` terminates a turn. Backpressure displays a
notice without pausing the microphone. Failed handshakes time out after 15 seconds;
Silero initialization failures display a specific error.

After updating the backend Python code, restart the backend, refresh the UI, and
start a new session so the plugin and browser assets use the updated code. A
successful `/health` check confirms readiness, not which deployment mode is active.

## Qwen VAD camera input

After starting a session with `--vad`, click **Camera** and allow access. The UI
sends one JPEG per second, resized to a longest side of 448 pixels, independently
of microphone uploads (including while muted). Each frame is a user conversation
item using the [OpenAI Realtime image input](https://developers.openai.com/api/docs/guides/realtime-conversations#image-inputs) shape:

```json
{"type":"conversation.item.create","item":{"id":"camera_1","type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/jpeg;base64,..."}]}}
```

Images become conversation context immediately; they do not start a response.
Speak and let VAD commit your audio, or send `response.create` after adding an
image/text item. A manual `input_audio_buffer.commit` only commits audio; follow
it with `response.create` to generate. Images persist across audio commits,
interruptions, and audio-buffer clears. Remove them with `conversation.item.delete`.
The browser retains eight camera items and deletes the oldest before adding a
ninth. Camera-off stops capture but leaves those items in the conversation.

This deployment accepts JPEG/PNG Base64 data URLs with a total image-context
budget of eight images and 4 MiB; external image URLs are not supported. Clients
must delete old image items when reaching the limit. There is no precise
video/audio timestamp alignment or native streaming-video KV update.

### Protocol compatibility boundary

Qwen VAD uses `session.update`, 24 kHz PCM16 `input_audio_buffer.append`,
`conversation.item.create` with `input_image`/`input_text`, and `response.create`.
The engine resamples input to Qwen's 16 kHz processing rate. Qwen rejects the
custom `video_frames` field on an append: it answers whole turns, so it has no
audio unit for a frame track to align to. MiniCPM keeps that field for its
native camera protocol, where Stage 0 interleaves frames at unit boundaries.

This is a supported subset, not full OpenAI Realtime API compatibility.
`duplex=1`, `overlap_policy`, `playback.ack`, and `session.close` remain vLLM-Omni
extensions. The legacy `--stt` mode is unchanged and is not the standard Realtime
conversation API. This demo does not add WebRTC, semantic VAD, or tool calling.

## Shared host options

The shared entry point can select either profile explicitly:

```bash
python -m examples.online_serving.realtime_web --profile qwen3-turn \
    --backend ws://127.0.0.1:8091 --stt --port 7863
```

- `--backend` / `--ws-backend`: backend WebSocket origin (not the full `/v1/realtime` URL).
- `--model`: override the wrapper's model name.
- `--public-realtime-url`: optional browser-visible WebSocket URL; otherwise the
  static host proxies `/v1/realtime` on the same origin.
- `--host`, `--port`: UI bind address and port (default port 7862).
- `--ref-audio`: required for MiniCPM; rejected for Qwen3.

Microphone and camera access require `localhost` or HTTPS. For a remote backend,
use an SSH tunnel to the UI host or an HTTPS reverse proxy with WebSocket support.
An explicit `wss://` public URL retains its scheme.

## Validation

```bash
node --test tests/examples/test_realtime_web_profiles.cjs
pytest tests/examples/test_minicpmo_realtime_web_server.py \
       tests/examples/test_minicpmo_realtime_web_static.py \
       tests/examples/test_realtime_web_profiles.py
pytest -m cpu tests/model_executor/models/qwen3_omni/test_duplex_plugin.py \
       tests/engine/duplex/test_session_runner.py \
       tests/engine/test_duplex_orchestrator.py
```

These CPU tests cover profile messages, two STT turns, VAD terminal/drain ordering,
backpressure, native capture during playback, server config injection, and shared
worklets. Backend regressions cover current audio inclusion in history-based
prompts, delayed playback ACK ordering, and no extra generation during silence
after a completed VAD turn. Hardware acceptance still requires a MiniCPM call
with camera/barge-in and two Qwen turns against each enabled backend, including
speech interruption in VAD mode; simulated tests do not
establish model quality or hardware latency.
