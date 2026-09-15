# PersonaPlex full-duplex online serving

Serve [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
(a Moshi-based full-duplex speech-to-speech model) with the native vLLM-Omni engine
through the unified duplex serving stack (`/v1/duplex` and `/v1/realtime?duplex=1`).

> Requires a GPU and Hugging Face access to the gated repo
> (`HF_TOKEN` with access to `nvidia/personaplex-7b-v1`).

## Start the server

The default `vllm_omni/deploy/personaplex.yaml` enables the engine-owned
full-duplex control plane (`session_mode: duplex`):

```bash
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
  /path/to/personaplex-7b-v1 \
  --omni \
  --deploy-config vllm_omni/deploy/personaplex.yaml
```

This exposes:

- `WS /v1/duplex` — the native duplex session dialect
  (`session.create` / `input_audio_buffer.append` / `response.output_audio.delta` ...);
- `WS /v1/realtime?duplex=1` — the same sessions projected onto the
  OpenAI Realtime event vocabulary (client API and wire protocol:
  [`docs/serving/realtime_duplex_api.md`](../../../docs/serving/realtime_duplex_api.md)).

PersonaPlex is a pure-lockstep model: every session on a PersonaPlex deployment is
native duplex (`is_enabled()` is unconditionally true), audio flows continuously in
both directions, and there are no client commits or external turn signals
(`supports_client_commit=false`, `supports_external_turn_signal=false`).

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

The unified endpoint advertises `supports_barge_in=false`: overlapping speech
is native model behavior, but destructive output interruption and model-state
rewind have not been validated for PersonaPlex.

## Notes

- **Run the client near the server.** Real-time 80 ms frame audio is sensitive to
  network latency/jitter; over a high-latency remote link playback can stutter
  regardless of engine speed. On localhost it is smooth.
- The earlier standalone Moshi-web compatibility server (browser client at `/`,
  binary WS protocol at `/api/chat`, raw-PCM `/v1/audio/duplex`) was demo-only
  and has been removed; use the unified endpoints above.
- Session config (voice / persona / sampling) is passed per session via
  `extra_body`; see
  `vllm_omni/model_executor/models/personaplex/duplex/serving_adapter.py`.
- Full runbook: [`recipes/NVIDIA/PersonaPlex.md`](../../../recipes/NVIDIA/PersonaPlex.md).
