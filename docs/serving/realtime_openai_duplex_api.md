# Realtime Audio WebSocket API

As a companion to the vLLM-Omni-proprietary [fullduplex `/v1/realtime`](full_duplex_api.md),
supported models also provide a `/v1/realtime` handler supporting
a substantial subset of the OpenAI `/v1/realtime` API supporting
audio streaming, client-driven VAD, and tool calling

## Quick Start

The endpoint is currently only supported for Qwen3-Omni. To start:

Start vLLM-Omni with port `8000` bound to an address reachable from Docker:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8000 --host 0.0.0.0
```

Then start the docker compose:

```bash
cd examples/online_serving/qwen3_omni/fullduplex-livekit-frontend
VLLM_OMNI_HOST=$VLLM_OMNI_HOST docker compose up --build
```

Where `VLLM_OMNI_HOST` is a network address. If vLLM-Omni is running on localhost, use
`host.docker.internal`

Open [http://localhost:3000](http://localhost:3000) and begin chatting.

## Protocol

Messages in both directions are JSON text frames. Audio carried inside an
event is base64-encoded raw PCM16.

| Direction | Event | Purpose |
| ----------- | ------- | --------- |
| Server to client | `session.created` | Confirms that the WebSocket connection is ready |
| Client to server | `session.update` | Selects and validates the served model. Configures system prompt and tools |
| Client to server | `input_audio_buffer.append` | Appends base64 PCM16 audio |
| Client to server | `input_audio_buffer.commit` | Starts generation over the incoming stream |
| Server to client | `transcription.delta` | Carries incremental response text |
| Server to client | `transcription.done` | Carries final text and token usage |
| Server to client | `response.output_audio.delta` | Carries incremental PCM16 response audio |
| Server to client | `response.output_audio.done` | Marks the end of response audio |
| Server to client | `error` | Reports an invalid event, model, or audio payload |

A minimal client sends events in this order:

```json
{"type":"input_audio_buffer.append","audio":"<base64-pcm16>"},
{"type":"input_audio_buffer.commit"}
```

This OpenAI-compatible streaming path is distinct from the vLLM-Omni-proprietary
lane and as such is reached via different configuration. The vLLM-Omni-proprietary
path must be specified explicitly via "duplex=1|true|on" as the WS query parameter
or via an explicit configuration in the `deploy_config`. The OpenAI path simply replaces
the default `/v1/realtime` handler in all supported models.

## Audio Handling

- Input is mono PCM16 at 16 kHz for the Qwen3-Omni example.
- `response.output_audio.delta.audio` contains base64-encoded PCM16 bytes.
- Read `sample_rate_hz` from each audio event instead of assuming an output
  rate. Qwen3-Omni output is typically 24 kHz.
- Concatenate audio deltas in receive order to construct the output waveform.

## Availability and Limitations

The path is registered on the unified API server, but it is usable only when
the loaded pipeline implements realtime audio input and produces compatible
audio output. Unsupported deployments return an `error` event. The endpoint
does not provide duplex session resume, playback acknowledgement, overlap
policy, or barge-in controls internally. Those must be driven by the client application
via the relevant WS events (e.g. `response.cancel`, `conversation.item.truncate`)

See the [Qwen3-Omni online serving example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen3_omni)
for concurrency options, chunk pacing, and per-delta audio debugging.
