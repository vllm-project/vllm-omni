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

## Without a browser

`headless_client.py` speaks the same protocol this page does, reading
`sample_16k.wav` instead of a microphone, so you can exercise the pipeline over
SSH with no tunnel and no mic — and repeat it enough times to turn "it goes
quiet sometimes" into a number:

```bash
python examples/online_serving/qwen3_omni_duplex/headless_client.py \
  --sessions 5 --turns 3
```

It exits non-zero unless every turn of every session produced audio.

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

## Multi-turn and multi-session

Both work. Measured on one boot with a scripted client replaying this page's
exact wire protocol: 5 consecutive sessions all answered, and 4 sessions × 3
turns produced audio on all 12 turns.

Two framework fixes got it there.

**One session per boot** — `num_waiting_for_streaming_input` leaked one per
session. `EngineCore.has_work()` reads through that counter, so a single
phantom made stage 1 conclude it had no work while a live request sat in its
`waiting` queue; the engine parked in `input_queue.get()` and never scheduled
again. The talker never ran, so no codec tokens reached Code2Wav and the client
waited forever on a healthy-looking server that kept accepting audio. Fixed in
`OmniSchedulerMixin._resync_streaming_input_counter` — full mechanism in that
docstring and in `docs/design/qwen3_omni_duplex_pr_map.md`.

**One turn per session** — downstream async-chunk stages stop polling the
connector once their segment finishes, and only resume when they receive a
streaming update. The orchestrator prewarmed them on the *initial* stage-0
submit but not on updates, so a second turn deadlocked: stage 0 generated and
marked its segment finished while stage 1 was still parked from the previous
turn and never read the new chunks.

If you are debugging something new here, still **reset before each experiment**
-- a wedged server accepts audio and frames it correctly while producing
nothing, so measurements taken after a wedge look like data and are not. Two of
ours were contaminated that way.

## Solved: later turns answered the first question

Symptoms, all one bug: later turns affirming turn 1's topic, generic greetings
("I'm doing well, how can I assist you today"), and finally the model saying
outright that it was being sent *a lot of empty messages*. It was: from turn 2
on, the thinker received a span of unfilled `<|audio_pad|>` placeholders where
the audio should have been.

Two coordinate systems were compared directly. `audio_offset` is `len(prefix)`,
relative to **this append's** token span; `duplex_token_offset` is
`num_computed_tokens`, absolute within the **session**. They coincide on the
first append (both anchored at 0), which is why turn 1 always worked. On turn 2
the absolute offset has advanced past the whole conversation so far, so
`take_from` became 140 into a 78-row tensor, the window came out empty, and the
guard returned without splicing -- silently, because the model runner absorbs a
reservation/embedding mismatch rather than raising.

Verified by sending *different* audio on each turn (the sample clip split in
half) and transcribing the replies with a separate Qwen3-Omni:

| turn | audio | before | after |
|---|---|---|---|
| 1 | 他当时还跟线下其他的站姐吵 | 他当时还跟线下其他站姐一起吗？ | *(identical -- control)* |
| 2 | 吵架，然后打架进局子了 | 是的，她当时还和其他站姐一起线下活动。 | 真的假的啊？打架还进去了？现在人怎么样了？ |

Turn 2's reply now uses 打架 and 进去了, which appear only in the second half.

**Two lessons worth keeping.**

The old ruled-out list here asserted "turn 2's audio *does* reach the model --
instrumenting the splice showed `embeds=(52, 2048)` spliced at the right
offset". That was false, and it cost the most time of anything in this
investigation. The `audio embeds=` log line reports that the embeddings were
*built*; the separate `[splice]` line reports that they were *installed*. Turn 2
logged the first and not the second. Reading the wrong line as proof turned the
real cause into a ruled-out branch, and sent the next three experiments at KV
and position offsets instead.

The proposed next step here -- fingerprint the spliced span's mean and norm per
turn -- would not have found it either. The embeddings were fine and identical
to turn 1's; they simply never landed. When output looks like the input was
empty, check that the input was *installed* before checking whether it was
*correct*.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Connect fails | Server not up, or port 8099 not forwarded. Check `curl localhost:8099/health`. |
| No mic prompt | Page not on `localhost`/HTTPS. Use the SSH tunnel above. |
| Connects, no reply | `auto_response` missing, or the deploy config lacks `session_mode: duplex`. |
| Reply is garbled | Server is running an older build without the hold-until-commit fix. |
