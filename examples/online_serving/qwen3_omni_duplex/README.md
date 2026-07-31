# Qwen3-Omni full-duplex browser demo

Talk to Qwen3-Omni over the duplex WebSocket and watch the three stages light
up as your turn moves through them.

Status: experimental. See
`vllm_omni/experimental/fullduplex/qwen3omni/` and
`docs/design/vllm_omni_architecture_primer.md`.

## 1. Start the server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --deploy-config <your duplex deploy yaml> \
  --host 0.0.0.0 --port 8099
```

The deploy config must set `session_mode: duplex` at the top level — that is
what registers the duplex routes. See
`docs/design/qwen3_omni_duplex_deploy.example.yaml`.

## 2. Serve this page

```bash
python -m http.server 8100 --directory examples/online_serving/qwen3_omni_duplex
```

## 3. Open it

**Local machine:** <http://localhost:8100/app.html>

**Remote server:** forward both ports, then open the same URL locally.

```bash
ssh -L 8100:localhost:8100 -L 8099:localhost:8099 user@host
```

Forwarding matters for more than reach: browsers only grant microphone access
on `localhost` or HTTPS, so tunnelling to `localhost` is what makes the mic
work without setting up TLS.

## Using it

1. **Connect** — opens the WebSocket and sends the `session.update` handshake.
2. **Hold to talk** — streams 200 ms `input_audio_buffer.append` frames while
   held; release sends `input_audio_buffer.commit`.
3. **Barge in** — sends `input.cancel` mid-reply, bumping the session epoch.

The reply plays automatically and the transcript appears as it arrives.

## What the diagram shows

```
mic ──ws append──▶ Stage 0 ──hidden──▶ Stage 1 ──shm──▶ Stage 2 ──ws audio──▶ playback
                   Thinker             Talker           Code2Wav
                   audio→text          text→codec       codec→audio
```

Each stage is a separate OS process with its own GPU allocation and scheduler.
A card turns blue while its stage is active and green when it finishes; the
counters are driven by real events on the wire.

Stage 1 shows `flowing` rather than a count. Its codec tokens are internal to
the pipeline and never surface as client events — you only see the effect when
Stage 2 starts emitting audio.

## Notes on this adapter

**Audio is held until commit.** The framework can stream each appended chunk
to the thinker as it arrives, which is right for a model with learned
listen/speak tokens. Qwen3-Omni has none, so an intermediate append asks it to
continue a user turn that has no `<|audio_end|>` yet, and it produces garbage.
Holding until commit gives it one well-formed turn. The cost is that
time-to-first-token starts at release rather than during speech.

**Barge-in works on the session epoch,** not on append cadence, so holding
audio does not affect it. The session and its KV survive a cancel.

**`extra_body.auto_response: true` is required.** Without it the server never
starts a response on commit. Do not send `response.create` — on this path it
routes to the chat fallback rather than the native duplex runtime.

## Multi-turn

Multi-turn works. Verified: two spoken turns in one session produce two
independent replies, 9 audio deltas total.

This needed a framework fix. Downstream async-chunk stages stop polling the
connector once their segment finishes, and only resume when they receive a
streaming update. The orchestrator prewarmed them on the *initial* stage-0
submit but not on updates, so a second turn deadlocked: stage 0 generated and
marked its segment finished while stage 1 was still parked from the previous
turn and never read the new chunks. Stages 1 and 2 sat idle and the client saw
nothing.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Connect fails | Server not up, or port 8099 not forwarded. Check `curl localhost:8099/health`. |
| No mic prompt | Page not on `localhost`/HTTPS. Use the SSH tunnel above. |
| Connects, no reply | `auto_response` missing, or the deploy config lacks `session_mode: duplex`. |
| Reply is garbled | Server is running an older build without the hold-until-commit fix. |
