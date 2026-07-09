---
name: review-worker-runner
description: Review or audit vLLM-Omni worker / model-runner code (vllm_omni/worker/*, platform runner variants). Applies a fixed set of review lenses distilled from prior audits — upstream-vs-omni fork drift, dead/deprecated code, base↔AR divergent duplicates, model-specific logic that belongs in OmniModelState, wrapper-vs-raw model access, silent failures, async/threading correctness, rebase fork-fragility. Use when the user says "review the model runner", "audit gpu_model_runner / gpu_ar_model_runner / the worker", "explain this runner function and flag issues", or asks to mark review findings inline or produce a runner audit.
---

# Worker / ModelRunner Code Review

Structured review of `vllm_omni/worker/*` (the GPU worker + AR / generation model runners, shared
`OmniGPUModelRunner` base, connector mixin, memory/payload helpers) and the platform runner variants.
This skill encodes the recurring findings so every review applies the same lenses and produces a
consistent, actionable output.

**Three outputs the user typically wants:**
1. **Inline marks** — annotate findings in-place as `# ISSUE(review): …` (and `# OMNI:` for unmarked
   fork deltas), keeping the file `py_compile`-clean.
2. **Rollup audit** — a single markdown file, **type-first**: Level 1 = error type (model-specific
   issues listed individually), Level 2 = `file:line — description`.
3. **Refactor design doc** — when a subsystem is tangled enough that the audit isn't the deliverable
   (e.g. the async-omni-output path), a design that makes it maintainable. Principles that recur:
   *one behavior, two execution modes* (same builder inline or on a thread); *the boundary is a type,
   not a convention* (a frozen snapshot dataclass, not a closure); *models declare, the runner
   decides* (a capability spec + one policy gate); *mechanical moves first, semantic fixes second*,
   each step shippable and output-hash-verified.

Confirm which (or both) before starting. Default to inline marks when reviewing one function, rollup
when auditing a whole file/area, a design doc when asked to *fix/refactor* a tangled subsystem.

**Prior-art artifacts** (sample outputs of this skill; keep in sync when re-auditing):
`worker_runner_audit.md` (type-first rollup) and `async_omni_output_refactor_design.md` (refactor
design) on branch `dev/worker-runner-audit-comments` of the `tzhouam/vllm-omni` fork.

## Workflow

1. **Read the whole file — do not keyword-grep.** Keyword filters miss inline reviewer comments and
   real issues. First build the map:
   - `grep -nE "^    def |^def "` → the function inventory.
   - `grep -nE "^\s*#"` → the *complete* comment inventory (existing reviewer notes are findings).
   - Then read each function body.
2. **Per function, answer four things:** (a) what it does; (b) **upstream fork body or OMNI delta?**;
   (c) **which model(s) require it?** (text-only pays nothing / it's for talker-MTP / Fish-KV / …);
   (d) run the lenses below.
3. **Verify empirically when cheap.** Don't assert "dead"/"over-provisioned"/"never fires" from
   reading — confirm it: `grep` the field/flag/def across the repo (dead code, dead extension
   points), or run a small probe (e.g. the FA3 `scheduler_metadata` size claim was proven with a real
   run + kernel instrumentation). Cite the evidence.
4. **Mark inline** as `# ISSUE(review): <what> — <why> — <fix>`. Use `# OMNI:` on unmarked fork
   deltas. After edits, run `<venv>/bin/python -m py_compile <file>` — must stay clean.
5. **Roll up** into the type-first audit (see `references/reference.md` for the template + the
   suggested-order footer). Line numbers must match the branch you annotated.

## Priority 0 — correctness bugs outrank cleanups

Real bugs found by these lenses go **first** in any rollup (own section, severity-tagged, "fix + test
before the cleanup passes"). Patterns that have produced confirmed bugs here:
- **Off-by-one on a size used as an index**: `x.clamp(max=size)` permits index `size` — OOB for a
  `size`-wide tensor (valid `0..size-1`). Almost always wants `size - 1`.
- **Early return shadowing a careful guard**: an unconditional `if not tokens: return` above a
  nuanced no-tokens block makes the latter unreachable — skipping the DP `_dummy_run(1)` that
  prevents a `coordinate_batch_across_dp` out-of-sync **hang**, and the `kv_connector_no_forward` path.
- **Silent wrong-data fallback**: on index-out-of-range return `v[0]` (request 0's payload) instead
  of raising — ships the wrong request's output with no signal.
- **In-place mutation of engine-shared state**: mutating `scheduler_output` (e.g.
  `custom_metadata`) without the `replace()`-copy the ngram path uses — contaminates the
  engine-core process's copy.
- **`bool(value)` on a maybe-tensor**: raises "Boolean value of Tensor with more than one element is
  ambiguous" when a marker is a multi-element tensor/ndarray.
- **Jointly-constraining asserts**: `assert shape[0]==1` + `assert shape[0]==num_reqs` silently
  forces `num_reqs==1` — batched requests unsupported on that path with no error; the sibling branch
  asserting only `len(list)==1` silently **misaligns** payloads for batches.
- **capture≠replay**: a `_dummy_run` copy missing the base's `has_preprocess` input branch —
  captures on `input_ids` while runtime feeds `inputs_embeds`.

## The review lenses

Apply all of these to each function. Each = *the tell* → *why it matters* → *fix direction*.

1. **Provenance — upstream fork vs OMNI delta.** Is this block upstream `GPUModelRunner` body or an
   omni addition? Omni tells: `model_intermediate_buffer`, `make_omni_output`, `talker_mtp`,
   connector calls, `additional_information`, per-request `model.preprocess`/`postprocess`, hardcoded
   model names. Upstream tells: `query_start_loc`, `CpuGpuBuffer`, `num_scheduled_tokens`,
   `cudagraph`, `spec_decode`. **Unmarked deltas are silent fork drift** → mark `# OMNI:` so rebases
   notice; large fork bodies (`_update_states`, `_dummy_run`, `_preprocess`) need per-delta markers.

2. **Dead code (verify with grep).** Write-only fields (assigned, never read — e.g.
   `_omni_last_model_output`); uncalled helpers (only the `def` matches — e.g.
   `_decode_and_store_request_payloads`); **dead extension points** — a `getattr(model, "supports_X",
   False)` flag *no model declares* (grep the flag across `model_executor/models/` — if absent, the
   branch is dead, e.g. `_sampled_token_ids_cpu_override`); dead defensive guards (`if callable(x)`
   where `x` is always a tensor attr — `CpuGpuBuffer.cpu`); unused params.

3. **Deprecated / legacy remnants.** Back-compat aliases (`runtime_additional_information`),
   deprecated methods still called (`_update_additional_information`,
   `_merge_additional_information_update`), no-longer-used wire fields (`pooler_output`). The
   `additional_information` naming is the canonical case — 3 names for one concept; converge on
   `model_intermediate_buffer` and retire aliases.

4. **Divergent duplicates (base ↔ AR ↔ generation).** Match method names across the 3 runner files;
   the dangerous copies **differ subtly** (AR's trailing `-1` truncation in
   `_build_model_sampler_output_token_ids`; AR's `skips_...` short-circuit in
   `_sampling_metadata_for_model_sampler`; `_dummy_run` ~90% dup). **NamedTuple forks**
   (`ExecuteModelState`) can't be fixed by inheritance (you cannot extend a `typing.NamedTuple`) — a
   full copy is unavoidable, so guard it with a field-parity test + `# OMNI:` on added fields, and
   watch for silent field-order/type drift. Merge target: the `OmniModelState` hook (B-align).

5. **Model-specific logic in the generic runner (the biggest theme — list each individually).**
   Tells: hardcoded arch strings / allowlists (`_OMNI_CONNECTOR_INIT_ARCHS`); `__class__.__name__ ==
   "…"` checks (MiMoAudio); model-name-keyed behavior (`qwen3_tts_request_seed`, MammothModa2
   `generated_len`, `thinker_reply_part_per_request`); the ~30 `getattr(self.model, "<cap>", …)`
   probes; "Only required by X" comments. Fix: move behind a **model-declared capability on
   `OmniModelState`** (prefer `get_model_state_cls()` over any arch allowlist). See the model→feature
   map in `references/reference.md`.

6. **Wrapper vs raw model.** `self.model` may be a `CUDAGraphWrapper`/`UBatchWrapper`.
   `isinstance` / `supports_X(...)` **must** use `self.get_model()` (unwraps); attr/method access
   works via `__getattr__` delegation but is inconsistent. Rule: bind `model = self.get_model()` once
   per function and use it for both the check and the calls.

7. **Silent failures.** `try/except Exception` that logs + drops (prompt_embeds decode swallows a
   native tensor); `traceback.print_exc()` to stdout (use `logger.exception`); `getattr` fallbacks
   that silently skip real work. Ask: *does swallowing this hide a bug?* If yes → narrow the except /
   raise.

8. **Fork-fragility (rebase hazards).** Code that silently no-ops or breaks when upstream shifts:
   `getattr` on `input_batch`/`model_config` internals (renamed attr → silent skip of backfill,
   etc.); `dataclasses.replace` assuming a type stays a dataclass; `self.pin_memory` coupling
   assuming base reads it; imports of upstream modules that may not exist
   (`vllm.compilation.breakable_cudagraph`); **stale over-provisioning** from an old upstream formula
   (FA3 `scheduler_metadata` resize) — verify empirically. Mark `# OMNI:`.

9. **Async / threading correctness.** Event recorded but never waited
   (`async_copy_ready_event.record()` with no reader `.synchronize()`); daemon thread started in
   `__init__` (a background exception only surfaces if the result is awaited — dropped object
   swallows it; at minimum `logger.exception` in the thread); `non_blocking=True` D2H copies read
   before completion; `torch.Event()` vs `torch.cuda.Event()` device backing. Ask: *who
   synchronizes, and does every reader wait first?*
   **Deferred-builder anti-patterns** (the async-omni-output cluster; full treatment in the
   `async_omni_output_refactor_design.md` prior-art doc):
   - *Implicit snapshot boundary* — a closure capturing N locals encodes "everything the background
     thread reads must be frozen" as convention; a missed copy compiles fine and races silently.
     Fix: a frozen snapshot dataclass + one `capture()`.
   - *Cross-thread live-state access* — trace the whole deferred call graph for reads/writes of
     `self.requests` / `input_batch` / `model_intermediate_buffer` / connector state
     (`accumulate_full_payload_output`). Verify per site whether it's main-thread-only (check actual
     callers — don't assume) or a real race; resolve at capture time or assert unreachable.
   - *Dual-mode function with an implicit contract* — same body runs inline (sync) and on a thread
     (async) with mode differences as scattered `if`s; nothing enforces "async reads only frozen
     args". One builder + typed snapshot + mode asserts (e.g. `async ⇒ no prefix cache`).
   - *Clone-before-copy* — CUDA-graph output buffers are reused by step N+1; payloads must be
     `clone()`d on the producing stream before the copy stream D2H-copies them.
   - *Eager side-effects can't defer* — postprocess that writes cross-step state
     (`hidden_states['last']`) must run on the hot path before snapshotting; only pure output
     assembly may defer (`postprocess_already_applied` travels in the snapshot).

10. **Implicit cross-phase state.** Instance attrs written in `_preprocess` and read later in
    `sample_tokens`/`_build_model_kwargs_extra` (`_omni_num_scheduled_tokens_np`) → should be
    `ExecuteModelState` fields, not mutable attrs. Base methods reaching **subclass-only** attrs via
    `hasattr` (`_downstream_payload_cache`, "only appears on the AR runner") → layering smell; own the
    lifecycle in one place (`OmniModelState.remove_request`).

11. **Naming / docstring / type-hint drift.** Stale docstrings ("Align with v0.14.0"); wrong return
    annotations (`-> dict` returns a tuple; `-> dict[str,dict]` returns `None`); `object` where
    `torch.Tensor` is meant; misleading names (`_collect_additional_information_for_prefill` that only
    overlays prompt_embeds).

12. **Style / structure.** `*args/**kwargs` + `kwargs.pop(...)` + `if kwargs: raise` (make the
    signature explicit — this also restores type-checking); imports-inside-functions — **distinguish**
    a deliberate lazy-dep (keep: `fish_kvcache_backend`, `breakable_cudagraph`) from a hoistable
    import-safe one (hoist: `ray_utils`, `RoutedExpertsLists`); over-long functions (`_update_states`,
    `_dummy_run` — split); inner-helper closures ("no inner helper" — make a method); fragile
    `**kwargs` unpack with key-exclusion hacks.

13. **Test coverage.** Flag untested modules (`base.py`, `memory_utils.py`, `payload_span.py` have no
    direct unit tests) — "add L1 CPU characterization tests before any refactor that moves them."
    Otherwise everything in `tests/worker/` is CPU-only L1/L2.

14. **CUDA-graph capture == replay contract.** The `has_preprocess` buffer path in `_dummy_run` must
    mirror `_preprocess` (same fixed persistent buffers, sized to `max(max_num_reqs,
    max_cudagraph_capture_size)`) — capture and replay must read the same addresses. A change to one
    without the other is a silent capture≠runtime bug.

15. **Subsystem-level redundancy.** Beyond dead functions, ask whether a *whole subsystem's premise*
    still holds. Method: find the original justification (git log / PR / rebase notes) → check whether
    upstream or the orchestrator has since absorbed it → if plausible, list the investigation steps
    and the full removal cluster rather than keeping it by inertia. Known candidates: cross-stage
    `prompt_embeds` decode+overlay (upstream `EngineCoreRequest` now carries `prompt_embeds`
    natively); per-process NVML memory estimation (only earns its keep if stages still init in
    parallel on one device — check orchestrator + measure init cost). File these as
    "Possibly-redundant subsystem: <name>" with numbered investigation questions.

16. **Misplaced code units.** Free functions and dataclasses defined inside a runner module
    (payload helpers `_to_cpu_contiguous`/`_ensure_tensor_values`/…, `ExecuteModelState`,
    `OmniAsyncGPUModelRunnerOutput`, snapshot types) → move to `utils` / a neutral data module.
    This also fixes backwards imports (generation runner importing from the AR runner).

## Output conventions

- **Inline:** `# ISSUE(review): <defect> — <why it matters> — <fix direction>`. For fork deltas that
  should survive rebases: `# OMNI: <what and why>`. One concern per comment; keep the file parsing.
- **Rollup audit:** type-first (see `references/reference.md`). Level 1 = error type; **model-specific
  types listed individually** (talker-MTP, Fish-KV, GLM-Image, Higgs, MammothModa2, MiMoAudio,
  connector-allowlist, capability-probes — each its own `##`). Level 2 = `file:line — description`.
  End with a suggested order (quick wins now → investigate → B-align structural).
- **The B-align through-line:** most structural findings (duplicates + model-specific) converge on
  extracting model logic into a v1 `OmniModelState` that mirrors `worker_v2/OmniModelState` — call
  that out as the destination rather than proposing ad-hoc fixes.

See `references/reference.md` for: the model→feature map, the capability-probe catalogue, the key-file
roles, upstream-vs-omni tells, known gotchas, and the audit template.
