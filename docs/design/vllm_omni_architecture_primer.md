# vLLM-Omni architecture primer

Written to orient someone joining the Qwen3-Omni full-duplex work. Everything
here was verified against the code or observed on a live server during
bring-up; where something is inferred rather than observed it says so.

---

## 1. vLLM-Omni in general

vLLM serves one model. vLLM-Omni serves a **pipeline of models** — a
"multi-stage" model where each stage is its own vLLM engine in its own OS
process, and an orchestrator moves data between them.

```
                 ┌──────────────────────────────────────────┐
   HTTP/WS  ───▶ │  API server  (entrypoints/openai/)        │
                 └───────────────────┬──────────────────────┘
                                     │
                 ┌───────────────────▼──────────────────────┐
                 │  Orchestrator  (engine/orchestrator.py)   │
                 │  • owns per-request state                 │
                 │  • polls every stage, routes outputs      │
                 │  • forwards stage N output -> stage N+1   │
                 └────┬───────────────┬───────────────┬──────┘
                      │ ZMQ           │ ZMQ           │ ZMQ
                 ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
                 │ Stage 0 │     │ Stage 1 │     │ Stage 2 │   each = its own
                 │ process │     │ process │     │ process │   StageEngineCoreProc
                 └─────────┘     └─────────┘     └─────────┘   subprocess + GPU
                       └──── SharedMemoryConnector ────┘
                            (bulk tensors/codec chunks)
```

### The pieces worth knowing

| Concept | Where | What it is |
|---|---|---|
| **PipelineConfig** | `config/stage_config.py`, per-model `pipeline.py` | Frozen topology: how many stages, what each one is, how they connect. Registered per model architecture. |
| **StagePipelineConfig** | same | One stage: `stage_id`, `model_stage` name, execution type, `input_sources`, whether its output is client-visible (`final_output`). |
| **Deploy YAML** | `vllm_omni/deploy/*.yaml` | Runtime knobs per stage — GPU placement, memory, batching, samplers. Can overlay another via `base_config:`. |
| **Orchestrator** | `engine/orchestrator.py` | The hub. Polls all stages in one loop, routes each output, decides when the whole request is done. |
| **StagePool** | `engine/stage_pool.py` | One per stage; holds N replica clients. |
| **Connectors** | `distributed/omni_connectors/` | Move bulk data stage→stage. Default `SharedMemoryConnector`. ZMQ carries control; SHM carries tensors. |

### Two things that surprise people

**Control and data travel differently.** Request submission and output polling
go over ZMQ to each stage subprocess. The heavy payloads (hidden states, codec
frames) go through shared memory. So "it's all ZMQ" is wrong.

**One request ID spans all stages.** The same `req_id` is reused as a key in
three disjoint scheduler namespaces. The orchestrator tracks
`final_output_stage_ids` (a *set* — text and audio can both be client-visible)
and only finishes the logical request when all of them have finished
(`orchestrator.py:1233-1241`).

---

## 2. Full duplex

### The problem it solves

Normal serving is request-shaped: prompt in → output out → KV freed → done.
A live voice assistant is stream-shaped: audio flows in continuously *while*
audio flows out, the user can interrupt, and one conversation lives for
minutes. RFC #3745 calls this the impedance mismatch.

Concretely, without duplex you get one WebSocket connection per utterance and
no way to interrupt without killing the connection.

### How the framework does it

Everything lives under `vllm_omni/experimental/fullduplex/`. It was built for
MiniCPM-o 4.5 in PR #3907 and is targeted to leave `experimental/` in v0.28.

```
  WebSocket  /v1/realtime?duplex=1
       │
       ▼
  OmniDuplexSessionHandler        openai/serving.py
       │   session lifetime, admission, capabilities
       ▼
  DuplexSession  (+ per-model serving state)
       │   epoch, turn, playback cursor, pending input
       ▼
  DuplexControlPlane              engine/duplex_control_plane.py
       │   open / append / signal / close, all fenced
       ▼
  DuplexRuntimeExtension          ← THE MODEL PLUGS IN HERE
       │
       ▼
  StagePool  ->  resumable scheduler request  ->  stages 0/1/2
```

### The three ideas that make it work

**1. The session is the unit, not the request.** A stage-0 request is
*resumable*: at a segment boundary it parks in `WAITING_FOR_STREAMING_REQ`
with its KV intact instead of finishing, then accepts more input. KV is freed
only at session close.

**2. Fencing.** Every operation carries a `DuplexFence`
(`session_id, epoch, turn_id, response_seq, incarnation`). Barge-in bumps the
epoch; anything tagged with a stale epoch is dropped. This is how "stop
talking, the user interrupted" works without tearing down the session.

**3. A narrow model seam.** A model supplies exactly two objects:

| Protocol | File | Methods |
|---|---|---|
| `DuplexRuntimeExtension` | `engine/contracts.py:63-97` | `configure_sampling_params`, `plan_append`, `decide_output` |
| `ServingRuntimeAdapter` | `openai/runtime_adapter.py:107-146` | 9 methods + a 6-method `data_plane` |

Registered by dotted path on the model's `PipelineConfig`
(`duplex_runtime_extension`, `duplex_serving_adapter`,
`duplex_control_enabled`).

### The wire protocol that actually works

This cost real time to discover — the working path is not the obvious one.

- **Endpoint:** `ws://host/v1/realtime?duplex=1` — **not** `/v1/duplex`.
  `/v1/duplex` exists but is a separate, far less exercised entry point.
- Query params: `duplex=1`, `model=...`, `<model>_native_duplex=1`,
  `autostart=0`.
- Handshake: `session.update` (not `session.create`); server replies
  `session.created`.
- **`extra_body.auto_response: true` is the master switch.** Without it the
  server never starts a response on commit, and the session silently produces
  nothing.
- Audio in: `input_audio_buffer.append`, base64 **pcm16** @ 16 kHz, ~200 ms
  chunks.
- Turn end: `input_audio_buffer.commit` with `final: true`. **Never send
  `response.create`** — it routes to the chat fallback.
- `turn_detection: null` — for a model-driven duplex model, the model owns
  turn-taking, not server VAD.
- Audio out: `response.audio.delta` (base64 pcm16 @ 24 kHz), bracketed by
  `response.speak` and `response.audio.done`.

### What the framework does *not* claim (`DESIGN.md:63-70`)

- scheduler-native KV append
- deterministic VAD-triggered interruption
- production multi-session admission / fairness / failure recovery
- bounded long-session KV
- video input

---

## 3. Qwen3-Omni specifically

### The model

```
Stage 0  Thinker    audio+text+video in -> text out        (LLM_AR,        final_output: text)
Stage 1  Talker     text/hidden -> RVQ codec codes         (LLM_AR)
Stage 2  Code2Wav   codec codes -> waveform                (LLM_GENERATION, final_output: audio)
```

Defined in `model_executor/models/qwen3_omni/pipeline.py:34-69`. Structurally
near-identical to MiniCPM-o 4.5's topology, which is why the port is viable at
all.

### Where Qwen3-Omni differs from MiniCPM — and why it matters

| | MiniCPM-o 4.5 | Qwen3-Omni |
|---|---|---|
| Turn-taking | **Learned `<\|listen\|>` / `<\|speak\|>` tokens** — the model decides | **None.** Standard instruct LLM. Turn boundaries must come from the client |
| Audio encode | Streaming encoder with KV carried across chunks | No incremental encode — but its conv stack is chunked at exactly 1 s, so per-chunk encoding is lossless at the conv level |
| Prompt | Model-native unit framing | Needs explicit chat-template scaffolding to reply at all |
| `/v1/realtime` | Free | **Already taken** by the existing half-duplex handler |

The first row is the deep one: MiniCPM's whole duplex design assumes the model
owns listen/speak. Qwen3-Omni can't, so this port advertises
`supports_model_native_turn_policy=False` and relies on client-signalled turns.

### Audio geometry (load-bearing, and easy to get wrong)

The thinker uses a `WhisperFeatureExtractor`, `hop_length=160` @ 16 kHz →
100 mel frames per second. Token count follows vLLM's
`_get_feat_extract_output_lengths`: **13 tokens per second**, not a linear
ratio.

Two ways this bites:
- A linear `samples/token` approximation gives 10 tokens/s — 30% under.
- Whisper's extractor pads to 30 s by default → 3000 mel frames for a 1 s
  chunk instead of 100.

Both are **silent**: the model runner absorbs an embedding/reservation
mismatch by truncating or padding without raising.

### Why 1-second chunks

`Qwen3OmniMoeAudioEncoder.forward` splits its conv input into `n_window * 2 ==
100` mel frames = exactly 1.0 s. A 1 s duplex chunk lands on the model's own
conv boundary, so per-chunk encoding has **zero** convolutional boundary error.
Attention still spans 8 such chunks, so streaming remains an approximation at
the attention level only.

### Current state of the port

Branch `feat/qwen3-omni-fullduplex`, worktree `~/vllm-omni-duplex`.

**Working, observed on a live 3-stage server:**
- session handshake, audio ingest through the real audio tower
- chat scaffolding and turn closure
- thinker generation
- **speech output** — 20 `response.audio.delta`, 24 kHz, verified speech-like
  by envelope analysis (33% silent / 42% loud 100 ms frames), not just
  "bytes arrived"

**Not yet — and the first item is serious:**
- **The reply does not terminate sensibly.** A 4 s input produces ~385 s of
  audio, byte-identical across runs, which is the `max_tokens` ceiling rather
  than an end-of-speech decision. Speech comes out, but the turn never ends on
  its own. Until this is fixed the endpoint is not usable.
- transcript deltas arrive but carry no text (extraction or projection issue)
- barge-in still unproven: in the one attempt, zero audio deltas had arrived
  before the interrupt, so no true overlap was exercised
- stage 1 crashes on abort: `RuntimeError: Invalid request status:
  FINISHED_ABORTED` (framework, defect 7)

### Defects found during bring-up

| # | Defect | Origin |
|---|---|---|
| 1 | Audio token geometry 10 vs 13 per second | this port |
| 2 | Whisper 30 s default padding (230x over-reservation) | this port |
| 3 | `DuplexFence` not JSON serializable in `runtime_control` | **framework** — issue #5612, PR #5613 |
| 4 | `mtp_inputs` demanded on thinker decode | this port |
| 5 | Data plane dropped every output (objects vs Mappings) | this port |
| 6 | `chat_fallback.py:38` crashes reporting an error (`result.message`) | **framework**, unfiled |
| 7 | Stage 1 abort → `FINISHED_ABORTED` in waiting queue | **framework**, under investigation |

Every one presented identically: **nothing happens, no error**. The recurring
shape is a silent drop at a boundary.

---

## 4. Where help is most useful

**Highest leverage, no context needed**
- Listen to a captured reply and judge whether it's a sensible answer to the
  input, not just speech-shaped audio. Automated checks can only prove
  "speech-like".
- Decide the product answer for turn-taking. Qwen3-Omni has no learned
  listen/speak, so *something* must decide when the user stopped: client
  push-to-talk, client VAD, or a server VAD stage. This is a product call, not
  a code one, and it shapes the design.

**Needs repo context**
- Defect 7 (stage-1 abort) — a framework bug that blocks barge-in.
- Whether `response.audio_transcript.delta` is meant to carry text for a model
  whose transcript comes from a different stage than its audio.

**Worth knowing before deep investment**
- A maintainer opened an architecture discussion for full duplex on #3745 and
  targets v0.28 (~4 weeks) for moving it out of `experimental/`. The
  model-facing seams are the likely survivors; packaging is the likely churn.
