# Qwen3-Omni duplex: PR decomposition map

This branch was built whole-first to find the hairy cases, then sliced. This
is the slicing map — the natural fault lines, recorded while they were fresh so
carving PRs later is a jigsaw puzzle rather than an archaeology dig.

Read with `git log --oneline upstream/main..HEAD`. Every defect is its own
commit with its evidence in the message, so most commits are already PR-shaped.

---

## The organising principle

**Framework fixes land independently of whether the adapter ever ships.**

That is the cleanest first cut, and it is not a technicality: each of these is
a bug the *next* native duplex model hits too. They are worth landing on their
own merits even if Qwen3-Omni duplex is abandoned tomorrow.

So the slice order is: framework fixes → model-code fixes → the adapter itself
→ the demo. Value front-loaded, risk back-loaded.

---

## Tier 1 — framework fixes, no dependency on this port

Land these first. Each is small, has a reproduction that needs no GPU, and is
defensible without reference to Qwen3-Omni.

| # | Defect | State |
|---|---|---|
| 1 | `DuplexFence` not JSON serializable in `runtime_control` — kills the session at handshake | issue #5612, **PR #5613** |
| 2 | `chat_fallback.py` raises `AttributeError` while reporting an error, hiding the real cause | draft **PR #5626** |
| 3 | Abort raised inside the AR scheduler's dequeue kills the engine core | commit `fb8c46fc`, unfiled |
| 6 | `num_waiting_for_streaming_input` leaks on abort, parking the stage forever | commit `97cefb59`, unfiled |

Defect 6 is the one-session-per-boot wedge, and it is the strongest Tier 1
candidate after #5613: upstream's counter, omni's status rewrites, no
Qwen3-Omni in the reproduction. Full mechanism in
`OmniSchedulerMixin._resync_streaming_input_counter`; regression test in
`tests/core/sched/test_omni_scheduler_streaming_input_counter.py`.

Defect 3 is worth adding to #5626 or filing separately. Note the mechanism:
upstream pops from `waiting`, appends to `running`, *then* checks status, so
sweeping the queues beforehand cannot prevent it — the fix recovers and retries
rather than preventing.

**Also unfiled and worth a look:** twelve abandoned browser sessions produced
zero reaping despite `disconnect_grace_s` and `reaper_interval_s` being
configured. If that reproduces deliberately it is a real orphan-reaping bug,
and the framework claims that mechanism as implemented.

## Tier 2 — Qwen3-Omni model code, small and self-contained

| # | Change | Commit |
|---|---|---|
| 4 | Do not advertise talker MTP on the thinker stage (`KeyError: 'mtp_inputs'`) | in **PR #5626** |
| 5 | Thinker `preprocess` hook + `has_preprocess=True` for duplex audio ingest | `bc9648e8` |

Item 5 only makes sense alongside the adapter, so it rides with Tier 3.

## Tier 3 — the adapter

`vllm_omni/experimental/fullduplex/qwen3omni/` plus pipeline registration.
Roughly 1,900 lines. Natural sub-seams if it needs splitting further:

1. **policy + audio geometry** — constants, the 13-tokens-per-second formula
   verified against vLLM's own `_get_feat_extract_output_lengths`. Standalone
   and unit-testable.
2. **serving adapter + session state + PCM buffer** — satisfies
   `ServingRuntimeAdapter`. No engine dependency.
3. **runtime extension** — `configure_sampling_params` / `plan_append` /
   `decide_output`, plus prompt assembly.
4. **stage-0 audio ingest** — the worker-side encode. Needs item 5.
5. **data plane** — projection of stage outputs into Realtime events.

## Tier 4 — demo and docs

`examples/online_serving/qwen3_omni_duplex/` and `docs/design/*`. Independent
of everything above; useful to land late so it documents what actually shipped.

---

## Seams worth knowing before slicing

These are the boundaries that turned out to matter. Getting them wrong is what
cost the most time.

- **Only five prompt keys survive** `build_engine_core_request_from_tokens`
  (`orchestrator.py:118-158`). `multi_modal_data` is *not* one of them, which
  is why duplex audio must ride in `model_intermediate_buffer` and become
  embeddings in the model's own `preprocess`.
- **`duplex.data_plane = True` is a mandatory gate.** Absent it, every
  worker-side duplex branch silently no-ops.
- **Reservation must equal produced embeddings.** The model runner absorbs a
  mismatch by truncating or padding without raising. This invariant caused
  three separate defects here.
- **Direct response and stage advancement are mutually exclusive.** Returning a
  decision from `decide_output` short-circuits forwarding
  (`orchestrator.py:1284-1295`), so you can surface stage-0 text *or* get audio,
  not both. That is why the transcript is empty.
- **Stage 0 emits a raw vLLM `RequestOutput`;** other stages wrap it in
  `OmniRequestOutput`. Reading only the wrapped form sees empty text
  everywhere.
- **Code2Wav output is cumulative** and shaped `[1, samples]`, so `len()` gives
  the batch dimension.

## The one-session-per-boot bug: solved

**Root cause: upstream's `num_waiting_for_streaming_input` counter leaked one
per session, and `EngineCore.has_work()` reads through it.**

`get_num_unfinished_requests` computes
`len(waiting) + len(skipped_waiting) - num_waiting_for_streaming_input + len(running)`.
Upstream keeps the counter incrementally — `+1` when a resumable request parks
as `WAITING_FOR_STREAMING_REQ`, `-1` in `_update_request_as_session` or in
`finish_requests` when the request is *still in that status*. Omni rewrites
`request.status` outside both hooks (the chunk-transfer adapter's park/restore,
and `_realign_request_status_to_queues`), so at session close the talker's
request was aborted while counted but with status already stomped to `RUNNING`.
Upstream took the running branch, never decremented, and the counter stayed at
1 for the life of the process.

One phantom is enough. On the next session stage 1 had exactly one request in
`waiting`, so `(1 + 0 - 1) + 0 == 0`, `has_work()` was false, and the engine
blocked in `input_queue.get()` — it never called `schedule()` again. The talker
never ran, no codec tokens reached Code2Wav, and the client waited forever on a
server that stayed healthy and kept accepting audio.

Fixed by restating the invariant instead of chasing the writes:
`_resync_streaming_input_counter` re-derives the counter from the waiting
queues (a no-op when upstream's arithmetic is right, self-healing when it
isn't), called from both schedulers' `finish_requests` and from
`OmniARScheduler.schedule`. Measured on a freshly booted server, scripted
client replaying the browser's exact wire protocol: **before 1/2 sessions
answered, after 5/5, and 4 sessions × 3 turns = 12/12 turns.**

How it was found, since the log evidence was misleading three times over: the
API server, orchestrator and stage 0 all looked healthy, and stage 1 logged
*nothing*. `py-spy dump` on the stage-1 process was the turn — it showed the
main loop parked in `_process_input_queue`, which is only reachable when
`has_work()` is false, i.e. the engine believed it had no work while holding a
queued request. Everything after that was arithmetic.

## Ruled out for the one-session-per-boot bug

Kept because a negative result is worth as much as a positive one, and because
each of these cost a measurement:

- not session capacity (reproduces at `max_sessions: 4` with matching stage
  slots)
- not the client (browser and scripted client fail identically)
- not `session.close` (a session that never closes still stalls)
- not stage-0 admission (request created, first token generated)
- not the stop token (removing `stop_token_ids` changes nothing)
- not a race (deterministic on a freshly booted server)
- not the orchestrator's submit path: stage 1 *was* submitted every session,
  the request reached its engine core, and `scheduler.add_request` ran. Chasing
  the submit path cost the most time of anything here — the request arrives
  fine, it is the engine's decision to sleep that is wrong.

## Fixes that were wrong

Left in history deliberately, so nobody re-runs the experiment:

- Per-chunk audio encoding was claimed lossless because 1 s aligns with the
  tower's conv chunk. The conv part is true; the 8-chunk *attention* window is
  not, and it dominates. Cumulative re-encode improved cosine 0.844 → 0.949 and
  changed the output not at all.
- Marking thinker output as a direct response did surface the transcript, and
  cost all the audio.
