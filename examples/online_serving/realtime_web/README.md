# Shared Realtime Web UI

One browser shell provides microphone capture, speaker playback, a level meter,
conversation text, and an event log. Model profiles own the wire protocol.
This implements [RFC #7222](https://github.com/vllm-project/vllm-omni/issues/7222).

| Profile | Turn control | Camera | Playback acknowledgement |
| --- | --- | --- | --- |
| `minicpm-native` | Model-controlled listen/speak, continuous audio input | Frames accompany audio | Yes |
| `qwen3-turn --stt` (default) | User presses **Send turn** | No | No |
| `qwen3-turn --vad` | Server detects trailing silence | Sampled frames with each spoken turn | No |

Qwen3 is **turn-based, not full-duplex**: microphone upload pauses while an
answer is generated and played, followed by a 300 ms echo guard. There is no
barge-in. Camera input is available in Qwen VAD mode; STT remains audio-only.
Gradio remains available for upload/chat use cases.

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

1. Explicit `duplex=0` selects STT even on deployments with automatic duplex routing.
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
base_config: /path/to/vllm-omni/vllm_omni/deploy/qwen3_omni_moe.yaml
session_mode: duplex
duplex_session:
  server_vad_model_path: /path/to/silero_vad.onnx
```

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
Wait for the reply to finish before speaking again; voice interruption is disabled.
The `duplex` routing setting does not make this Qwen UI a full-duplex voice call.

### Switching between VAD and manual turns

| Mode | Backend | UI flag | Submit a turn |
| --- | --- | --- | --- |
| Without VAD | Default Qwen deployment or the VAD-enabled deployment | `--stt` (default) | Press **Send turn** |
| With VAD | Server VAD configuration and Silero artifact | `--vad` | Pause after speaking |

Stop the existing UI process before starting another on the same port, then
refresh the browser and reconnect. To disable VAD, restart the UI with `--stt`;
the VAD-enabled backend can remain running because STT explicitly uses `duplex=0`.
To enable VAD on a default backend, first restart that backend with the overlay.
Make `--backend` match the actual backend port; the UI and backend use separate
ports. `Connection refused` means the target backend is unavailable: check its
startup log and `/health` before connecting.

VAD connects with `duplex=1` and sends nested `session.audio.input` configuration
with `create_response: true` and **`interrupt_response: false`**. It sends no
manual commits, native flags, reference voice or playback ACKs.
It explicitly selects `playback_commit_policy: commit_all_on_done` so completed
assistant replies enter session history without playback ACKs. User audio and
assistant text are retained for subsequent VAD turns on the same connection.
`response.audio.done` / `response.output_audio.done` only drain playback;
`response.done` terminates a turn. Backpressure or input-cleared notifications
pause microphone upload and display a notice; repeat discarded speech after the
answer. Failed handshakes time out after 15 seconds with a deployment hint;
Silero initialization failures display a specific error.

## Qwen VAD camera input

After starting a session with `--vad`, click **Camera**, allow access, and ask
about what the camera sees. The UI uploads one JPEG per second alongside audio,
resized to at most 448 pixels on the longest side. The backend attaches sampled
frames as ordered `image_url` parts alongside the utterance's audio when VAD
ends the turn. This requires the updated backend as well as the updated UI;
restart the backend after installing this change.

During silence only the latest frame is retained. During speech the most recent
8 frames are kept, with a 4 MiB encoded-image budget per buffered window. Frames
are cleared after commit, cancellation, or input reset. Committed frames remain
in conversation history, so long visual sessions consume additional context.
Camera-off stops new uploads; already uploaded frames can still accompany the
current turn. Speech triggers the response; video alone does not trigger one.
This is sampled visual context, without precise audio/video timestamp alignment.
Microphone and frame upload pause while a response is generated and played.

## Shared host options

The shared entry point can select either profile explicitly:

```bash
python -m examples.online_serving.realtime_web --profile qwen3-turn --stt --port 7863
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
```

These CPU tests cover profile messages, two STT turns, VAD terminal/drain ordering,
backpressure, native capture during playback, server config injection, and shared
worklets. Hardware acceptance still requires a MiniCPM call with camera/barge-in
and two Qwen turns against each enabled backend; simulated browser tests do not
establish model quality or hardware latency.
