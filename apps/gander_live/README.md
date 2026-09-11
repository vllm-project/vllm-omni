# Gander Live

A local browser app for the experimental Gander full-duplex service. The audio
capture and protocol client build on the existing MiniCPM Realtime web demo;
this app adds a neon interface, visible tool activity, per-response playback
cancellation, and optional playback auditing. It does not load model weights.

## Start

First compose the weights and start the engine using the
[Gander serving guide](../../docs/serving/gander.md). On the browser machine:

```bash
python3 -m venv apps/gander_live/.venv
apps/gander_live/.venv/bin/pip install -r apps/gander_live/requirements.txt
apps/gander_live/.venv/bin/python apps/gander_live/server.py \
  --host 127.0.0.1 --port 7865 --ws-backend ws://127.0.0.1:8091 \
  --model /path/to/new/gander-model --ref-audio /local/path/to/ref_audio.wav
```

The model argument must match the name served by the engine. The reference WAV
must exist on the browser/proxy machine. For a remote engine, forward its port:

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -L 127.0.0.1:8091:127.0.0.1:8091 gpu-host
```

Open <http://localhost:7865> in Chrome, use headphones, and allow the microphone.
Start a new conversation after changing the tool toggle. The tested deployment
admits four sessions. There is no client VAD gate: capture continues in 200 ms
packets during playback, and mute sends silence to preserve the input timeline.
The listening display does not cancel audio. End the conversation to release
the engine session. This local demo has no authentication; use the loopback bind
shown above. A remote browser needs HTTPS for microphone access.

## Verify a real tool call

Enable the optional lookup tool (disabled by default), start a new session, and say:

> 帮我查询一下取货暗号，查到之后告诉我。

The right-hand panel must show both the model's `task_start` function call
(with its call ID and arguments) and the local execution result. The proxy reads
`pickup.json`, returns `function_call_output`, and the model should say
`蓝鲸四七二`. Saying “I will look it up” alone is not a successful tool call.

The proxy executes only after a model-generated structured `function_call`;
it does not detect spoken keywords or inject a forced call. The answer is not
included in model instructions. Edit `pickup.json`, start a fresh conversation,
and query again to check whether the model uses the changed external result.
The file is reread per invocation, so no proxy restart is needed.

`task_start` is a deliberately bounded demonstration: its task name is for
display, and every accepted call performs the same local lookup. This app does
not implement arbitrary business tools, a persistent task ledger, task revision,
cancellation/retry semantics, trusted ASR binding, Gateway, or Brain/GPT.
The “view context” button only inspects unit IDs and KV reconstruction metadata.

## Playback and diagnosis

The client cancels by response ID, so cancelling an old reply cannot clear a
new reply's queued audio. Playback activity follows the queued response rather
than instantaneous amplitude. The worklet measures internal audio starvation;
trailing silence while waiting for a terminal event is not counted as an
internal gap. Optional browser audit messages capture actual rendered samples.

The proxy records model output audio and events under `recordings/`; it does
not record the uploaded microphone audio. These files stay local, are ignored
by Git, and can be deleted when no longer needed. They may include conversation
text and tool results. No sample recordings or reference voice are bundled.

With Node.js installed, run the deterministic playback regression:

```bash
node apps/gander_live/test_playback.cjs
```

It covers PCM scaling/duration, clearing playback, retaining a new response
when an old response is cancelled, and measuring an 80 ms internal gap without
counting the final wait. See the [validation record](../../docs/design/gander_validation.md)
for actual H200/browser checks. Those checks reach AudioWorklet output, not
physical speaker/headphone output.
