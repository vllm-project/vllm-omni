# Qwen3-Omni Full-Duplex Feasibility Assessment

Status: assessment only. No implementation, no hardware validation.
Author: TheCodeWrangler (nathan@abridge.com)
Date: 2026-07-29
Scope: can `vllm_omni/experimental/fullduplex/` (merged in #3907 for MiniCPM-o 4.5)
be extended to Qwen3-Omni-30B-A3B-Instruct (thinker → talker → code2wav)?

Related: RFC #3745, PR #3907 (`05b794f2`), closed PR #5328, half-duplex
`/v1/realtime` work in #5555 / #5565 / #5566.

---

## 1. Summary verdict

**Tractable as a follow-on PR, but not as a pure adapter. The engine and
serving layers port cleanly; the worker-side audio ingress does not, and that
is the binding constraint. It does *not* need the scheduler work the RFC
implies.**

> **Update 1 (2026-07-29, after implementing the engine/serving layers on
> `feat/qwen3-omni-fullduplex`):** two further blockers were found that are
> not visible from reading the adapter contracts alone. See §2.10.
>
> **Update 2 (2026-07-30, after implementing stage-0 audio ingress):** both
> §2.10 blockers are now **resolved**, and one of them turned out to be much
> smaller than this document originally claimed. See §2.11. Current state:
> engine, serving and worker-side audio ingress are all implemented; the
> audio tower has been driven with real checkpoint weights on a GPU and
> produces exactly the reserved embedding count. What remains unvalidated is
> the full 3-stage duplex serving path.

Three things drive this verdict:

1. **The multi-process worry is unfounded.** MiniCPM-o 4.5 is *already* a
   genuine three-stage, three-process pipeline
   (`vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py:42-79`),
   structurally the same shape as Qwen3-Omni. #3907 did not solve duplex for a
   single-process model and then leave multi-process as an open problem —
   its topology precedent already matches ours.

2. **The scheduler blocker in the RFC is already closed for our path.**
   Resumable, KV-preserving request re-entry exists today
   (`vllm_omni/core/sched/omni_generation_scheduler.py:59-72`) and
   Qwen3-Omni's shipped `async_chunk: true` deploy already rides it. The
   RFC's blocker #1/#2 ("every turn re-prefills") does not describe the
   current Qwen3-Omni async-chunk path.

3. **The real blockers are elsewhere, and one of them is a framework defect.**
   The orchestrator's per-request streaming segment buffer is *not keyed by
   stage*, so `decide_output(stage_id=...)` receives a last-writer-wins
   snapshot across all three stages (§2.3). This is latent-benign for MiniCPM
   and actively wrong for any model that must read talker/code2wav metadata.
   Separately, Qwen3-Omni has **no learned listen/speak control token** —
   the single affordance MiniCPM's entire turn-taking policy rests on (§2.5).

So the honest framing is: the plumbing generalizes better than expected, the
turn-taking *policy* has no model-level foundation to stand on, and there is a
stage-aliasing bug to fix before a 3-stage adapter can be written correctly.

---

## 2. Architectural compatibility findings

### 2.1 MiniCPM-o 4.5 is genuinely 3 processes — the topology precedent holds

The brief's premise (that MiniCPM might be effectively single-process, with
`stage_id` only nominally present) is **incorrect**. `MINICPMO_4_5_PIPELINE`
declares three `StagePipelineConfig` entries:

- `stage_id=0`, `model_stage="llm"` — thinker
  (`minicpmo_4_5/pipeline.py:44-55`)
- `stage_id=1`, `model_stage="tts"`, `input_sources=(0,)` — talker
  (`pipeline.py:56-66`)
- `stage_id=2`, `model_stage="code2wav"`, `input_sources=(1,)`,
  `model_arch="MiniCPMO45Code2Wav"` (`pipeline.py:67-77`)

The deploy YAML confirms three independently-configured stages with a
shared-memory connector on the stage-1 → stage-2 edge
(`vllm_omni/deploy/minicpmo_4_5.yaml:18-60`, `output_connectors: to_stage_2`).
`_OrchestratorDuplexStagePort.stage_count` returns `len(self._stage_pools)`
(`vllm_omni/engine/orchestrator.py:235-237`), so `final_stage_id` is genuinely
`2` for MiniCPM, not `0`.

Compare Qwen3-Omni (`vllm_omni/model_executor/models/qwen3_omni/pipeline.py:34-69`):
`stage_id=0` thinker (`LLM_AR`, `final_output=True`, text), `stage_id=1` talker
(`LLM_AR`), `stage_id=2` code2wav (`LLM_GENERATION`, `final_output=True`,
audio). **The two topologies are near-identical.** Qwen3-Omni's only
structural difference is that it declares *two* final-output stages (text from
thinker, audio from code2wav) where MiniCPM also declares two
(`pipeline.py:49,72`). This is the same shape.

> Note this contradicts a plausible reading of `DESIGN.md`, which never
> mentions `StagePool` beyond one box in a diagram (`DESIGN.md:85`), never
> mentions ZMQ, and never defines `stage_id`/`final_stage_id` at all. The doc's
> silence is not evidence of single-process design — the pipeline config is.

### 2.2 …but the duplex control plane only ever drives stage 0

Topology support is not the same as topology *use*. Every write path in the
control plane is hard-coded to stage 0:

- `handle_open` reserves only stage 0:
  `ensure_stage_request(session, stage_id=0)`
  (`engine/duplex_control_plane.py:289`)
- `append_via_data_plane` opens with a literal `stage_id = 0`
  (`duplex_control_plane.py:444`), builds one `DuplexStageSubmission`, and
  makes exactly one `await self._stage_port.submit(...)` call — no loop, no
  `gather` over stages.
- `handle_signal` / `handle_close` / `handle_touch` / `handle_resume` all emit
  a sentinel `{"stage_id": -1, "replica_id": -1}` rather than per-stage results
  (e.g. `duplex_control_plane.py:551-563`).

`plan_append` in the protocol has **no `stage_id` parameter at all**
(`engine/contracts.py:73-86`) — appends are structurally stage-0-only by
design. That is defensible (audio enters at the thinker) and Qwen3-Omni would
inherit the same assumption harmlessly.

`decide_output` *is* invoked per-stage: `_route_output` calls
`self._duplex_output_decision(stage_id, output, req_state)` for every stage's
output (`orchestrator.py:1284`, inside the loop over
`for stage_id in range(self.num_stages)` at `orchestrator.py:896`). MiniCPM's
implementation then discards all but the pre-final stages via
`if stage_id >= final_stage_id or not segment_finished: return None`
(`minicpmo45/runtime.py:314-315`), and in practice only stage 0 carries the
`listen_token_id` metadata it looks for (`runtime.py:317-323`).

**So: the framework is multi-stage-aware in its invocation, and
single-stage in every decision it actually makes.**

### 2.3 Concrete defect: segment metadata is aliased across stages

This is the finding that most affects a Qwen3-Omni port, and it is not
documented anywhere in `DESIGN.md`.

The orchestration loop iterates all stages
(`orchestrator.py:896` `for stage_id in range(self.num_stages)`), and for each
LLM-type stage's raw output writes into a **single, non-stage-keyed** buffer on
the request state (`orchestrator.py:918-932`):

```python
req_state.streaming.segment_finished = bool(getattr(eco, "is_segment_finished", False))
req_state.streaming.segment_token_ids = (...)
req_state.streaming.segment_output_metadata = (dict(raw_mm) if ... else {})
```

Those fields are declared once per request, not per stage
(`orchestrator.py:196-201`).

`_duplex_output_context` then reads that same buffer regardless of which
`stage_id` it is building context for (`orchestrator.py:1444-1447`):

```python
segment_finished=req_state.streaming.enabled and req_state.streaming.segment_finished,
segment_token_ids=tuple(req_state.streaming.segment_token_ids),
segment_output_metadata=req_state.streaming.segment_output_metadata,
```

**Consequence:** when `decide_output(stage_id=1, ...)` is called, the
`segment_token_ids` / `segment_output_metadata` it receives belong to whichever
stage's raw output landed most recently — which, given that the three stages
are independently scheduled and explicitly designed to run at different paces
(§2.4), is frequently *not* stage 1.

For MiniCPM this is latent-benign: its `decide_output` only ever reads
`listen_token_id`, which only stage 0 emits, so a stale stage-1 snapshot yields
`listen_id is None → return None` (`runtime.py:321-322`). The bug is masked by
the policy being stage-0-only.

For Qwen3-Omni, any policy that needs to read talker or code2wav segment
metadata (e.g. "how much audio has actually been committed to the vocoder")
would read cross-stage-contaminated data. **This must be fixed — segment state
keyed by `stage_id` — before a 3-stage-aware `decide_output` can be written
correctly.** It is a small, self-contained orchestrator change, but it is a
change to shared non-experimental code, not to `experimental/fullduplex/`.

### 2.4 Fencing and leasing are session-scoped singletons, not per-stage

Nothing in the session/control-plane layer coordinates N processes:

- `DuplexSessionRuntimeState` holds exactly one `fence: DuplexFence` and one
  `lease: DuplexLeaseState` (`engine/duplex_session.py:80-81`).
- `DuplexFence` has fields `session_id, epoch, turn_id, response_seq,
  incarnation` (`engine/messages.py:15-22`) — **no `stage_id`**. It is a single
  logical clock, not a vector clock over stages.
- `stage_bindings: dict[int, DuplexStageBinding]` and
  `request_resources: dict[tuple[int, str], ...]`
  (`duplex_session.py:87-88`) are *shaped* for N stages and `release_fence`
  iterates them (`duplex_session.py:313-323`) — but only `stage_id=0` is ever
  written (§2.2), so they hold at most one entry today.
- Barge-in fan-out is delegated wholesale to one opaque call:
  `await self._stage_port.cleanup(list(pending.submitted_request_ids), abort=True)`
  (`duplex_control_plane.py:878-881`). There is no per-stage acknowledgment,
  no two-phase commit, no "commit only if all three stages accepted the new
  epoch."

The module docstring is explicit that this is intentional: *"The module owns
duplex session/control algorithms. It deliberately does not own stage pools,
request queues, or OpenAI Realtime protocol state."*
(`duplex_control_plane.py:12-14`).

**Assessment:** this is *adequate but unproven* for Qwen3-Omni. Because the
whole pipeline shares one external `req_id` across all three stages
(`orchestrator.py:109-158`, `external_req_id = request_id` at
`orchestrator.py:1408`), a single `cleanup([req_id], abort=True)` does fan out
to all three stage pools via `_cleanup_request_ids` /`_abort_request_ids`
(`orchestrator.py:819-830`). So barge-in *can* reach all three processes.
What does not exist is any guarantee about **atomicity or ordering** — stage 2
may emit one more audio chunk after stage 0 has accepted the new epoch. MiniCPM
tolerates this because its barge-in granularity is a full 1-second chunk
anyway. Qwen3-Omni would inherit the same "in-flight audio plays out" behavior,
which is acceptable for a first cut but should be stated, not hidden.

### 2.5 The real gap: Qwen3-Omni has no learned listen/speak token

This is the blocker that no amount of framework work removes.

MiniCPM's entire turn-taking policy is: the model itself emits `<|listen|>` /
`<|speak|>` control tokens at chunk boundaries, and `decide_output` simply
detects `<|listen|>` and reports it
(`minicpmo45/runtime.py:317-328`; token vocabulary at
`minicpmo45/policy.py:58-74`). `DESIGN.md:66-67` is explicit that this is the
*only* interruption mechanism the checkpoint claims: *"deterministic
VAD-triggered interruption (the browser intentionally does not run VAD;
MiniCPM owns listen/speak decisions at model-unit boundaries)"* is listed
under **"The checkpoint does not claim"**.

Qwen3-Omni has no equivalent. A grep for listen/speak control tokens across
`vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py` returns only
`speaker`-as-voice-selection hits (lines 231, 295-297, 715-732, 854-922) —
i.e. the voice-ID feature added in #5565, semantically unrelated. Qwen3-Omni's
thinker is a standard instruct LLM that emits text and stops; it has no trained
notion of "the user is still talking, stay quiet."

MiniCPM's per-unit turn state machine also depends on **re-injecting the
model's own previous decision token into context** each unit, precisely because
"the model's listen/speak policy depends on seeing its own past decisions in
context" (`minicpmo45/stage0.py:231-244`). There is nothing to re-inject for
Qwen3-Omni.

**Therefore Qwen3-Omni cannot use the model-owned turn-taking path at all.**
Its options are:

- **(a) External VAD** gating the thinker. This is exactly the path
  `DESIGN.md:66-67` disclaims, `#3745`'s Phase 2 defers, and `tc-mb` warns is
  "structurally much heavier" than end-to-end duplex
  (issue #3745 comment, 2026-05-21). It needs the `DUPLEX_VAD` stage type that
  does not exist yet (`StageExecutionType` has only `LLM_AR`,
  `LLM_GENERATION`, `DIFFUSION` — RFC blocker #6).
- **(b) Client/server-side barge-in only** — no model turn-taking; the client
  signals `input.cancel`, the server bumps the epoch and flushes. This is
  achievable with the existing framework and is the honest first cut, but it
  is *interruption*, not *full duplex*: the model never decides to stay silent
  because the user is mid-sentence.

Option (b) is the tractable follow-on PR. Option (a) is a separate,
larger project.

### 2.6 Resumable append already works — the RFC's scheduler blocker does not bind here

Good news that changes the recommendation materially.

`DESIGN.md:527-544` ("Why Scheduler Changes Remain") describes the required
contract as one resumable request parked in `WAITING_FOR_STREAMING_REQ` with KV
retained between segments, released only at session close. That machinery
**exists and is in use**:

- `OmniGenerationScheduler._handle_stopped_request` re-enqueues a `resumable`
  request as `WAITING` instead of finishing it
  (`vllm_omni/core/sched/omni_generation_scheduler.py:59-72`).
- KV is only freed on `finished=True`
  (`vllm_omni/core/sched/omni_ar_scheduler.py:507-520`), so a parked resumable
  request keeps its blocks.
- Qwen3-Omni's shipped deploy sets `async_chunk: true`
  (`vllm_omni/deploy/qwen3_omni_moe.yaml:15`) and its stage input processors
  have an explicit resumable branch
  (`vllm_omni/model_executor/stage_input_processors/qwen3_omni.py:506-509`
  dispatching to `_construct_thinker2talker_streaming_input_async_chunk`).
- `talker_preprocess` already threads `meta.get('resumable', False)` through
  its token bookkeeping (`qwen3_omni.py:743-789`).

So RFC blockers #1 and #2 ("KV freed on finish", "streaming update sets
`num_computed_tokens = 0`") **do not describe the Qwen3-Omni async-chunk
path**. `DESIGN.md:65` still lists "scheduler-native KV append" as not
claimed — that refers to a deeper scheduler-native primitive, not to the
resumable-request workaround, which is what actually carries #3907 and would
carry us.

**This is the single biggest de-risking finding: we do not need scheduler
changes.**

### 2.7 Stage skew is designed-for, not a hazard

Qwen3-Omni's three stages are explicitly intended to run at different
wall-clock paces: `async_chunk` exists so "each stage forwards chunks so the
next stage can start as soon as the first chunk is ready"
(`docs/design/qwen3_omni_tts_performance_optimization.md:228`), stages use
different batching policies (thinker/talker continuous, code2wav static, same
doc), and the orchestration loop polls every stage non-blockingly with no
inter-stage barrier (`orchestrator.py:892-1037`, `timeout_s=0.001` at
`orchestrator.py:912`). Failure handling marks individual replicas unavailable
rather than failing the pipeline (`orchestrator.py:960-1019`).

No explicit skew bound exists. This is fine for duplex — it is the same
property MiniCPM relies on — but it means barge-in latency is bounded by
in-flight code2wav buffer depth, not by the control plane.

### 2.8 Volume of model-specific work required

MiniCPM's three protocol methods (`runtime.py`, 354 lines) are backed by
~1,650 lines of supporting state in `stage0.py` (816) and `data_plane.py`
(837). Reviewing both: roughly two-thirds is model-specific and would need a
from-scratch rewrite for a different audio front-end.

`stage0.py` is almost entirely MiniCPM-coupled — the `<unit>`/`</unit>` framing
grammar, 1000 ms/1035 ms chunk cadence with CNN warm-up padding
(`stage0.py:359-425`), the stateful streaming audio-encoder KV carried across
chunks (`stage0.py:443-489`), and the decision-token re-injection loop
(`stage0.py:231-244`). Qwen3-Omni's audio path is architecturally different
(the thinker consumes a processor-produced audio feature via
`buffer_realtime_audio`, `qwen3_omni.py:224-324`, with a 5.0 s segment default
at line 244 — five times MiniCPM's cadence).

`data_plane.py` is roughly half generic metadata-coercion scaffolding and half
MiniCPM-specific fusion logic. Critically, its core `project_output`
(`data_plane.py:114-454`) assumes **one stage output carries text + audio +
turn decision together** — and it contains a hard-coded Qwen tokenizer constant
`151645` as an EOS fallback (`data_plane.py:227-234`). For Qwen3-Omni, where
text comes from stage 0 and audio from stage 2 as separate outputs at
different times, that fusion logic needs replacing with genuine cross-stage
correlation.

Realistic estimate: **~1,000-1,500 new lines** of Qwen3-Omni-specific runtime
state, plus the orchestrator fix in §2.3, plus a serving adapter (~100 lines,
`serving_adapter.py` is a thin delegating wrapper and ports easily).

### 2.10 The binding constraint: audio cannot reach the thinker

Found while implementing, not while reading contracts. This is the reason the
port is not adapter-only.

**Audio has no route into the thinker except the model's own `preprocess`
hook.** In the duplex append path:

- `build_engine_core_request_from_tokens` forwards only `prompt_token_ids`,
  `prompt_embeds`, `cache_salt`, `additional_information` and
  `model_intermediate_buffer` (`orchestrator.py:118-158`). Every other prompt
  key — **including `multi_modal_data`** — is dropped silently.
- `_OrchestratorDuplexStagePort.submit` passes no `mm_features`
  (`orchestrator.py:289-302`).

So Qwen3-Omni's normal multimodal audio route is unavailable in duplex mode.
Audio must ride as base64 PCM inside
`model_intermediate_buffer["duplex"]["payload"]` and be converted to
embeddings by `model.preprocess`, dispatched at `gpu_model_runner.py:1685`
under `model.has_preprocess`. Two things block that:

**(a) The Qwen3-Omni thinker has no `preprocess` hook.**
`Qwen3OmniMoeForConditionalGeneration` sets `has_preprocess = False`
(`qwen3_omni.py:107`) and enables it only for the talker
(`qwen3_omni.py:156`). MiniCPM enables it for both its LM and TTS stages
(`minicpmo_4_5_omni.py:140`). Adding one is core model-code work, outside
`experimental/`.

**(b) Qwen3-Omni has no incremental audio encode.** MiniCPM carries an
audio-encoder KV cache across chunks — `get_audio_embedding_streaming` plus
`audio_past_key_values`, with explicit prefix/suffix context frames
(`minicpmo45/stage0.py:443-489`). Qwen3-Omni exposes no equivalent; its only
streaming affordance is `code2wav.chunked_decode_streaming`
(`qwen3_omni.py:593`), which is on the **output** side. Without incremental
encode the options are: re-encode the whole buffer each append (correct but
quadratic in session length, defeating the persistent session); encode each
chunk in isolation (cheap but wrong at boundaries, since the conv front end
loses left context); or encode a bounded sliding window and keep only the new
chunk's embeddings (bounded and approximately correct — recommended, and what
MiniCPM achieves via `cnn_redundancy_ms`).

Option (c) is implementable, but the frame arithmetic (mel hop → conv stride →
pooling ratio → embeddings per chunk) must be derived from the checkpoint.
`Qwen3OmniDuplexPolicy.SAMPLES_PER_AUDIO_TOKEN` is currently an **unverified
placeholder**, and it is load-bearing: the engine reserves scheduler slots
from it and the worker must produce exactly that many embeddings, or the model
runner truncates/pads with no error.

### 2.11 Resolution of §2.10 (2026-07-30)

Both blockers are closed. One was real but small; the other was overstated in
§2.10 and is worth correcting explicitly.

**Blocker (a) — no thinker `preprocess` hook: real, and fixed.**
`qwen3_omni.py` now sets `has_preprocess = True` on the thinker stage and
registers `thinker_duplex_preprocess` via `set_custom_preprocess`. This is
behaviour-neutral for ordinary serving: the thinker is a multimodal stage
(`requires_multimodal_data=True`, `pipeline.py:57`), so the runner already
routed it through `inputs_embeds` (`gpu_model_runner.py:1569`) rather than
the token-id fast path. `has_preprocess` only adds the dispatch at
`gpu_model_runner.py:1685`, and the hook returns the ordinary embeddings
when no duplex payload is present.

**Blocker (b) — "no incremental audio encode": substantially overstated.**
§2.10 assumed Qwen3-Omni needs MiniCPM-style encoder-state carryover. It does
not. `Qwen3OmniMoeAudioEncoder.forward` already splits its conv input into
`n_window * 2 == 100` mel-frame chunks and runs the conv stack on each
independently. At `hop_length=160` / 16 kHz that is exactly 1.0 s. So a 1 s
duplex chunk lands on the model's own conv boundary and per-chunk encoding
carries **no convolutional boundary error at all** — the sliding-window
scheme §2.10 called for is unnecessary.

The residual approximation is at the *attention* level only: the tower's
attention spans up to `n_window_infer // (n_window * 2) == 8` chunks, so
encoding one chunk at a time is not bit-identical to encoding 8 s at once.
That is a real but bounded inaccuracy, and it is not corrected.

**The token geometry was the actual hazard, and it was wrong.** The original
`SAMPLES_PER_AUDIO_TOKEN = 1600` placeholder implied 10 tokens/s; the model
produces **13**. Two independent ways to get this wrong were found and fixed:

1. The linear ratio itself (10 vs 13 — a 30% under-reservation).
2. Whisper's feature extractor pads to its 30 s `n_samples` by default,
   yielding **3000** mel frames for a 1 s chunk instead of 100 — a 230×
   over-run of the reservation. `stage0._encode_chunk` passes
   `padding="longest", truncation=False`.

Both are silent failures: the model runner absorbs an embedding/reservation
mismatch by truncating or padding without raising. `_encode_chunk` now
asserts the counts agree.

**What was actually executed** (see commit `bc9648e8`):

- Real checkpoint weights on a GPU: 525 `thinker.audio_tower` tensors loaded
  from Qwen3-Omni-30B-A3B-Instruct; 1 s of audio → mel `(128, 100)` → tower →
  `(13, 2048)` bf16, rows equal to the reservation, all finite.
- Real feature extractor with a stubbed tower: chunk accumulation, 3 s → 39
  embeddings via 3 per-chunk calls, `(epoch, seq)` replay memoized without
  re-encoding, per-session buffer isolation, mismatch raises, bad format
  rejected.
- `validate_duplex_runtime_extension` and `validate_serving_runtime_adapter`
  — the engine's and API server's own startup gates — pass.
- Our token-geometry helper equals vLLM's `_get_feat_extract_output_lengths`
  for mel-frame counts 1–1000.

**Still unvalidated:** the full 3-stage duplex serving path. That requires an
environment matching upstream/main's vLLM (`v0.26.0`, per
`docker/Dockerfile.cuda:1`); the available image ships vLLM 0.24.0, which
lacks `vllm.entrypoints.scale_out` and cannot import `api_server`.

### 2.9 Comparable PRs offer no better template

Checked the three PRs #3745 cites as having hit the same wall:

- **#3642** (MiniCPM-o 4.5) — merged; became #3907. Same 3-stage shape as
  us. This *is* the best template.
- **#3512** (Nemotron VoiceChat) — still open/draft with failing CI gates.
  Its vllm-omni-side diff is generic plumbing only; the actual pipeline lives
  in an unmerged external NeMo plugin and defines **2** stages
  (thinker + EarTTS talker, no separate vocoder stage).
- **#1967** (SoulX-Duplug) — **not a PR at all**; it is an open New Model
  Request issue with no code. A maintainer explicitly steered it toward
  request/response rather than duplex (linyueqian, 2026-03-21).

No alternative multi-stage precedent exists. #3907 is the only template.

---

## 3. `Qwen3OmniDuplexRuntimeExtension` skeleton

Method signatures match `DuplexRuntimeExtension`
(`engine/contracts.py:63-97`) exactly. Bodies are deliberately unimplemented.

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SKELETON — not implemented, not validated on hardware.

Sketch of a Qwen3-Omni duplex runtime extension for
``vllm_omni/experimental/fullduplex/qwen3omni/runtime.py``.

Assumes the client-signalled barge-in model (assessment option (b), §2.5):
Qwen3-Omni has no learned listen/speak control token, so this extension
CANNOT implement model-owned turn-taking the way MiniCPM's does.
"""

from typing import Any

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence


class Qwen3OmniDuplexRuntimeExtension:
    """Pure model policy for Qwen3-Omni thinker -> talker -> code2wav."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Return per-stage SamplingParams, one entry per pipeline stage.

        ``defaults`` has length 3 for Qwen3-Omni (thinker, talker, code2wav);
        the return tuple MUST be the same length and order.

        Qwen3-Omni-specific work:

        TODO(stage 0 / thinker): clone the default and override ``max_tokens``
          from ``runtime_config`` so a duplex segment is bounded. Unlike
          MiniCPM there is no ``<|chunk_eos|>`` to terminate a unit, so the
          per-segment token budget is the ONLY stop condition. Getting this
          wrong means the thinker runs away past the user's next utterance.
        TODO(stage 0): preserve ``detokenize: True`` — the thinker is a
          ``final_output`` text stage (pipeline.py:38-39) and the Realtime
          projection needs text deltas.
        TODO(stage 1 / talker): keep ``detokenize: False`` and the
          checkpoint's codec ``stop_token_ids``; do NOT let runtime_config
          override codec sampling, which comes from ``talker_config``.
        TODO(stage 2 / code2wav): almost certainly pass ``defaults[2]``
          through untouched — it is a vocoder, not a sampler.
        TODO: reuse MiniCPM's ``_stage_config_value`` indexing helper
          (minicpmo45/runtime.py:236-243) so per-stage overrides can be
          addressed as ``runtime_config[key][stage_id]``.
        TODO: thread the ``speaker`` / voice selection added in #5565 through
          here rather than through ``additional_information``, so voice is
          session-scoped rather than per-utterance.
        """
        raise NotImplementedError

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        """Build the stage-0 prompt for one appended input chunk.

        Note there is no ``stage_id`` parameter: appends are structurally
        stage-0-only (control plane hard-codes ``stage_id = 0`` at
        duplex_control_plane.py:444). Audio enters at the thinker; the talker
        and code2wav are fed by the orchestrator's async-chunk forwarding,
        not by this method.

        Qwen3-Omni-specific work:

        TODO: accept only ``DuplexInputMode.APPEND_AUDIO_CHUNK`` and
          ``TURN_COMMIT_ONLY``; reject ROLLBACK_TO_CHECKPOINT and
          REPLACE_LATEST_CHUNK explicitly (Qwen3-Omni has no ASR-correction
          rollback affordance). Fail closed rather than silently degrading.
        TODO: decode ``payload`` as base64 pcm_f32le and accumulate into
          per-session state. Qwen3-Omni's realtime buffer currently uses a
          5.0 s segment (qwen3_omni.py:240) vs MiniCPM's 1.0 s unit — this
          MUST be reduced for usable barge-in latency, and reducing it is a
          model-behavior change that needs its own evaluation, not a config
          tweak. Segment length directly sets the interruption floor.
        TODO: build ``prompt_token_ids`` via the chat template, reusing the
          ``safe_apply_chat_template`` path already established for tools and
          instructions (qwen3_omni.py:274-291 from #5555/#5566), so a duplex
          session keeps tool-calling and system-prompt support.
        TODO: carry the fence (epoch/turn_id/response_seq) into the prompt's
          intermediate-buffer payload so stale-epoch outputs can be dropped
          downstream — mirror build_duplex_data_plane_prompt
          (minicpmo45/runtime.py:99-162).
        TODO(VERIFY BY TRACING, DO NOT ASSUME): confirm every key placed in
          the returned prompt dict actually survives to the worker. The
          render pipeline silently drops unrecognized prompt-dict keys — this
          exact class of bug already shipped once in this project (fixed in
          54d71650, voice selection dropped by render_cmpl_async). Assert on
          observed worker-side keys, not on a mocked unit test.
        TODO: size ``prompt_token_ids`` against the scheduler token budget so
          the resumable request is not starved.
        """
        raise NotImplementedError

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        """Optionally short-circuit a stage output into a direct response.

        Invoked once per stage per output (orchestrator.py:1284, inside the
        loop over all stages). ``final_stage_id`` is 2 for Qwen3-Omni.

        !! BLOCKED ON THE §2.3 DEFECT !!
        ``segment_token_ids`` and ``segment_output_metadata`` are read from a
        per-request, NOT per-stage buffer (orchestrator.py:918-932 writes,
        orchestrator.py:1444-1447 reads). For a 3-stage model whose stages run
        at different paces by design, the snapshot passed alongside
        ``stage_id=N`` may belong to a different stage. Any implementation
        here that reads stage-1 or stage-2 metadata is unsound until the
        orchestrator keys that state by stage_id. MiniCPM does not hit this
        because it only ever reads stage-0 metadata.

        Qwen3-Omni-specific work:

        TODO: return None for ``stage_id >= final_stage_id`` and for
          ``not segment_finished``, matching minicpmo45/runtime.py:314-315.
        TODO: DO NOT port MiniCPM's listen-token detection
          (runtime.py:317-328). Qwen3-Omni emits no ``<|listen|>``; there is
          no model-native turn decision to detect (§2.5). Attempting to
          synthesize one from text EOS would conflate "finished this reply"
          with "the user should speak now" — those are different events and
          the model was not trained to distinguish them.
        TODO: decide what, if anything, this method is FOR under the
          client-signalled barge-in model. Plausibly nothing: if turn
          boundaries come from the client and audio flows through the normal
          stage-2 final_output path, returning None unconditionally may be
          correct, and the class exists only for
          configure_sampling_params + plan_append. Resist inventing a
          decision that has no model signal behind it.
        TODO(IF a VAD stage is added later): this becomes the seam where a
          DUPLEX_VAD stage's turn verdict is translated into a
          DIRECT_RESPONSE — but that stage type does not exist
          (StageExecutionType has only LLM_AR / LLM_GENERATION / DIFFUSION)
          and is out of scope here.
        """
        raise NotImplementedError
```

A `Qwen3OmniServingRuntimeAdapter` would also be needed, implementing
`ServingRuntimeAdapter` (`experimental/fullduplex/openai/runtime_adapter.py:107-146`),
wired via `PipelineConfig.duplex_serving_adapter`
(`vllm_omni/config/stage_config.py:283`). MiniCPM's is a thin delegating
wrapper (`minicpmo45/serving_adapter.py:21-96`) and ports with low risk.
`vllm_omni/model_executor/models/qwen3_omni/pipeline.py` currently declares
**no** duplex fields at all — `duplex_runtime_extension`,
`duplex_serving_adapter`, and `duplex_control_enabled=True` would all need
adding, mirroring `minicpmo_4_5/pipeline.py:26-30`.

---

## 4. What this assessment does not claim to solve

Mirroring the pattern at `DESIGN.md:63-70`. This assessment does not claim:

- **any working code.** Nothing here has been executed. The skeleton in §3
  does not compile as a functioning extension and has never been run.
- **any hardware validation.** #3907 required a dedicated H20 host to
  validate; no GPU was available for this assessment. Every finding above is
  from code reading, not from observed runtime behavior.
- **that the §2.3 aliasing defect is a live production bug.** It is
  *latent-benign* for MiniCPM by the reasoning given, and I have not
  reproduced it. It is a hazard for a 3-stage-aware policy, established by
  reading, not by a failing test.
- **model-owned turn-taking for Qwen3-Omni.** §2.5 concludes the checkpoint
  has no affordance for it. Only client-signalled barge-in is in scope.
- **deterministic VAD-triggered interruption** — same disclaimer as
  `DESIGN.md:66-67`, and additionally the `DUPLEX_VAD` stage type does not
  exist.
- **any barge-in latency figure.** In-flight code2wav audio will still play
  out after an interrupt (§2.4, §2.7). The floor is set by buffer depth and
  by the input segment length (currently 5.0 s, `qwen3_omni.py:240`), and I
  have measured neither.
- **bounded long-session KV** — unsolved for MiniCPM too (`DESIGN.md:69`);
  nothing here improves it.
- **multi-session admission, fairness, or capacity** for Qwen3-Omni.
  `DESIGN.md:774-788` limits even MiniCPM's claims to a validated
  two-session shape.
- **that the estimate in §2.8 is reliable.** ~1,000-1,500 lines is a
  reading-based guess, not a plan.
- **the plugin-descriptor prerequisite is met.** `DESIGN.md:718-719` states
  that *"Adding a second native model should first introduce that descriptor
  and reject serving/engine plugin mismatches explicitly."* Qwen3-Omni would
  be that second native model, and no such descriptor exists. This is an
  explicit, documented precondition that this assessment does not satisfy.

---

## 5. Coordination status on #3745

I posted a comment on issue #3745 asking whether the Qwen3-Omni duplex adapter
slot is claimed
(https://github.com/vllm-project/vllm-omni/issues/3745#issuecomment-5122402496,
2026-07-29T19:13:18Z).

**As of writing, there are no replies.** It is the most recent comment on the
thread. No adjustment to the recommendation is possible yet; the cc list
(@linyueqian @yinpeiqi @Sy0307 @hsliuustc0106 @tc-mb @vklimkov-nvidia) has not
responded, and the slot should be treated as *unconfirmed* rather than open.

Relevant prior maintainer positions from the thread that do bear on the
recommendation:

- **tc-mb (2026-05-21):** end-to-end duplex models are "much simpler at the
  engine level"; a half-duplex + external-VAD pipeline is "structurally much
  heavier." Qwen3-Omni falls in the *heavier* category per §2.5 — this is
  direct maintainer support for keeping the first cut narrow.
- **Sy0307 (2026-05-20):** append mode "should be a model capability, not the
  default input semantics for all duplex models." Consistent with §3's
  recommendation to explicitly reject unsupported `DuplexInputMode` values.
- **Sy0307 (2026-05-20), point 4:** barge-in "does not define history commit…
  server generated/sent does not mean the user actually played/heard it."
  Directly relevant to §2.4/§2.7 — unresolved for Qwen3-Omni as well.
- **linyueqian (2026-05-20):** puts the one-chunk barge-in floor at ~1 s for
  chunk-based models and notes sub-300 ms "needs new engine work." Qwen3-Omni
  starts from a 5.0 s segment, i.e. worse than MiniCPM's floor until the
  segment size is addressed.

---

## 6. Recommendation

**Proceed, but in three separately-reviewable pieces, and do not describe any
of them as "full duplex for Qwen3-Omni."**

1. **Fix the stage-aliasing defect (§2.3) first, standalone.** Key
   `segment_finished` / `segment_token_ids` / `segment_output_metadata` by
   `stage_id` in `OrchestratorRequestState.streaming`. This is small, is
   independently justifiable as a correctness fix, touches shared code rather
   than `experimental/`, and is a hard prerequisite for any stage-aware
   policy. It is also the piece most likely to be accepted quickly on its own
   merits. Needs a regression test demonstrating cross-stage contamination.

2. **Then reduce and parameterize the realtime input segment length**
   (currently 5.0 s, `qwen3_omni.py:240`). This is a model-behavior change
   requiring quality evaluation — MiniCPM's own paper reportedly shows quality
   collapse at very short chunks, so this cannot be assumed safe. Without it,
   barge-in latency is bounded at ~5 s regardless of framework work, which is
   not a usable voice assistant. **This, not the duplex framework, is the
   critical path to the user-visible outcome.**

3. **Only then** add the `Qwen3OmniDuplexRuntimeExtension` +
   `Qwen3OmniServingRuntimeAdapter` + pipeline wiring, scoped explicitly to
   *client-signalled barge-in over a persistent session*, and named as such.
   Raise the plugin-descriptor question from `DESIGN.md:718-719` with
   maintainers before writing it, since Qwen3-Omni is precisely the "second
   native model" that precondition names.

**Does it need scheduler changes first? No** — §2.6 establishes the resumable
KV-preserving path already exists and Qwen3-Omni already uses it. This is the
main way the task is smaller than the RFC's "Concrete blockers" table implies.

**Does it need Qwen3-Omni model-level changes first? Partly** — not to the
network, but the input segmentation (item 2) is a model-behavior change and is
the real latency determinant. And no amount of engine work gives Qwen3-Omni
model-owned turn-taking; that would require a differently-trained checkpoint.

**Suggested next action:** wait for a reply on #3745 before writing code
beyond item 1. Item 1 is worth opening regardless, as it stands on its own as
a correctness fix.
