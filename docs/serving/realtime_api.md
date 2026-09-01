# Realtime Audio WebSocket API

Use `WS /v1/realtime` to stream PCM audio into a compatible omni model and
receive incremental transcription and synthesized response audio. This is a
turn-based streaming API: it starts one generation over an audio stream and
finishes after the client closes that input stream.

For a model that listens while it is already speaking, use the experimental
[Full-Duplex API](full_duplex_api.md) instead.

## Quick Start

Start a compatible model. The currently documented example uses Qwen3-Omni:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

Run the provided client with a mono, 16-bit PCM, 16 kHz WAV file:

```bash
python examples/online_serving/qwen3_omni/openai_realtime_client.py \
  --url ws://localhost:8091/v1/realtime \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input-wav input_16k_mono.wav \
  --output-wav response.wav \
  --output-text response.txt
```

The client dependency is `websockets`.

## Protocol

Messages in both directions are JSON text frames. Audio carried inside an
event is base64-encoded raw PCM16.

| Direction | Event | Purpose |
|-----------|-------|---------|
| Server to client | `session.created` | Confirms that the WebSocket connection is ready |
| Client to server | `session.update` | Selects and validates the served model |
| Client to server | `input_audio_buffer.commit` with `final: false` | Starts generation over the incoming stream |
| Client to server | `input_audio_buffer.append` | Appends base64 PCM16 audio |
| Client to server | `input_audio_buffer.commit` with `final: true` | Marks the end of input |
| Server to client | `transcription.delta` | Carries incremental response text |
| Server to client | `transcription.done` | Carries final text and token usage |
| Server to client | `response.audio.delta` | Carries incremental PCM16 response audio |
| Server to client | `response.audio.done` | Marks the end of response audio |
| Server to client | `error` | Reports an invalid event, model, or audio payload |

A minimal client sends events in this order:

```json
{"type":"session.update","model":"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
{"type":"input_audio_buffer.commit","final":false}
{"type":"input_audio_buffer.append","audio":"<base64-pcm16>"}
{"type":"input_audio_buffer.commit","final":true}
```

The initial non-final commit intentionally starts generation before all audio
has arrived. Continue sending `input_audio_buffer.append` events while the
engine is consuming the stream.

## Audio Handling

- Input is mono PCM16 at 16 kHz for the Qwen3-Omni example.
- `response.audio.delta.audio` contains base64-encoded PCM16 bytes.
- Read `sample_rate_hz` from each audio event instead of assuming an output
  rate. Qwen3-Omni output is typically 24 kHz.
- Concatenate audio deltas in receive order to construct the output waveform.
- Set `input_audio_buffer.commit.final` to `true` after the last input chunk so
  the server can terminate the streaming request.

## Availability and Limitations

The path is registered on the unified API server, but it is usable only when
the loaded pipeline implements realtime audio input and produces compatible
audio output. Unsupported deployments return an `error` event. The endpoint
does not provide duplex session resume, playback acknowledgement, overlap
policy, or barge-in controls.

See the [Qwen3-Omni online serving example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen3_omni)
for concurrency options, chunk pacing, and per-delta audio debugging.
