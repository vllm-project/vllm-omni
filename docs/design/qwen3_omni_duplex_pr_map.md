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

## Ruled out for the one-turn-per-boot bug

Kept because a negative result is worth as much as a positive one when someone
picks this up:

- not session capacity (reproduces at `max_sessions: 4` with matching stage
  slots)
- not the client (browser and scripted client fail identically)
- not `session.close` (a session that never closes still stalls)
- not stage-0 admission (request created, first token generated)
- not the stop token (removing `stop_token_ids` changes nothing)
- not a race (deterministic on a freshly booted server)

Next place to look: whether stage 1 receives anything at all on turn 2.

## Fixes that were wrong

Left in history deliberately, so nobody re-runs the experiment:

- Per-chunk audio encoding was claimed lossless because 1 s aligns with the
  tower's conv chunk. The conv part is true; the 8-chunk *attention* window is
  not, and it dominates. Cumulative re-encode improved cosine 0.844 → 0.949 and
  changed the output not at all.
- Marking thinker output as a direct response did surface the transcript, and
  cost all the audio.
