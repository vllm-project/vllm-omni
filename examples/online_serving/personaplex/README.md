# PersonaPlex full-duplex online serving

Serve [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
(a Moshi-based full-duplex speech-to-speech model) with the native vLLM-Omni engine
through the unified full-duplex framework (`/v1/realtime?duplex=1`, alias `/v1/duplex`).

> Requires a GPU and Hugging Face access to the gated repo
> (`HF_TOKEN` with access to `nvidia/personaplex-7b-v1`).

## Start the server

The default `vllm_omni/deploy/personaplex.yaml` is a duplex deployment
(`session_mode: duplex`, two sessions per replica):

```bash
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
  /path/to/personaplex-7b-v1 \
  --omni \
  --deploy-config vllm_omni/deploy/personaplex.yaml
```

This exposes `WS /v1/realtime?duplex=1` (alias `WS /v1/duplex`): the OpenAI
Realtime session protocol projected onto vLLM-Omni duplex sessions (client API and
wire protocol: [`docs/serving/realtime_duplex_api.md`](../../../docs/serving/realtime_duplex_api.md)).
There is no `/v1/chat/completions` route: PersonaPlex answers speech only.

PersonaPlex is a pure-lockstep model: every session is native duplex, audio flows
continuously in both directions in 80 ms frames, the model decides when to speak,
and there are no client commits or external turn signals
(`supports_client_commit=false`, `supports_external_turn_signal=false`). A session
therefore auto-responds without any vendor flag.

## Talk to it

With the client library and the PersonaPlex preset (24 kHz `pcm_f32le` in, bundled
voice prompt, persona text):

```python
from vllm_omni.clients.duplex import DuplexClient
from vllm_omni.clients.personaplex import create_duplex_session_config

cfg = create_duplex_session_config(voice="NATF2.pt", persona="You are a concise assistant.")
async with DuplexClient("ws://127.0.0.1:8000/v1/realtime?duplex=1", model="/path/to/personaplex-7b-v1", config=cfg) as c:
    await c.stream_pcm(pcm_f32le_24k)          # keep streaming; the model speaks while it listens
```

Voice and persona are fixed for the session (`session.update` cannot change them).

## Validate the serving path

Validate the `/v1/realtime?duplex=1` scheduler path with paced 24 kHz
PCM, two concurrent sessions, overflow admission, per-session slot recycling,
and non-silent output:

```bash
python tests/e2e/online_serving/personaplex_realtime_duplex.py \
  --model /path/to/personaplex-7b-v1 \
  --input-wav /path/to/speech.wav \
  --output-dir /tmp/personaplex-realtime-duplex
```

The endpoint advertises `supports_barge_in=false`: overlapping speech is native
model behaviour, but destructive output interruption and model-state rewind have
not been validated for PersonaPlex. `response.cancel` and
`output_audio_buffer.clear` restart the model's conversation context (a fresh
Stage 0 request replays the voice/persona prefill).

## Notes

- **Run the client near the server.** Real-time 80 ms frame audio is sensitive to
  network latency/jitter; over a high-latency remote link playback can stutter
  regardless of engine speed. On localhost it is smooth.
- The earlier standalone Moshi-web compatibility server (browser client at `/`,
  binary WS protocol at `/api/chat`, raw-PCM `/v1/audio/duplex`) was demo-only
  and has been removed; use the unified endpoint above.
- The model plugin, worker-side lockstep runtime and input framing live in
  `vllm_omni/model_executor/models/personaplex/duplex/`; design notes in
  [`docs/design/fullduplex-personaplex.md`](../../../docs/design/fullduplex-personaplex.md).
- Full runbook: [`recipes/NVIDIA/PersonaPlex.md`](../../../recipes/NVIDIA/PersonaPlex.md).
