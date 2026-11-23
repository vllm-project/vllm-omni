# Maintainer Review Style

Use this reference when turning proved findings or specific evidence gaps into
comments. The baseline maintainer samples favor short, direct comments and
reserve long explanations for architecture or non-obvious failure paths.

Official docs: [contributing and review process](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/).

## Evidence basis

This calibration used 928 PRs merged from 2026-05-10 through 2026-08-08.
Candidates were ranked by substantive human inline-review threads, then by
distinct reviewers and review rounds. Author self-comments, bots and AI
reviewers, and bodyless approvals were excluded. Counts selected the sample;
the practices below came from reading root comments and follow-up replies.

The main set contains the six leaders after filtering; three other high-ranked
PRs add core-runtime and benchmark coverage. All were merged.

| PR | Review boundary |
| --- | --- |
| [#2906](https://github.com/vllm-project/vllm-omni/pull/2906) | New TTS model, shared utilities, recipes, and request state. |
| [#3907](https://github.com/vllm-project/vllm-omni/pull/3907) | Full-duplex API, session lifecycle, concurrency, and realtime evidence. |
| [#2162](https://github.com/vllm-project/vllm-omni/pull/2162) | Diffusion world model, OpenPI serving, tests, examples, and docs. |
| [#3454](https://github.com/vllm-project/vllm-omni/pull/3454) | Diffusion model, generic CLI surface, loaders, and configuration scope. |
| [#4425](https://github.com/vllm-project/vllm-omni/pull/4425) | Structured configuration, typing, defaults, and compatibility. |
| [#5397](https://github.com/vllm-project/vllm-omni/pull/5397) | Distributed offload, collectives, failure handling, and load semantics. |
| [#4281](https://github.com/vllm-project/vllm-omni/pull/4281) | Composable parallelism, configuration contracts, and examples. |
| [#3628](https://github.com/vllm-project/vllm-omni/pull/3628) | Stage metrics, benchmark definitions, opt-in API behavior, and cleanup. |
| [#3855](https://github.com/vllm-project/vllm-omni/pull/3855) | Stage runtime, thread safety, startup, and compatibility. |

Historical comments calibrate delivery; they do not create merge policy. Live
code, tests, repository docs, and the current `CODEOWNERS` file remain
authoritative. When refreshing this study, promote a practice only when it
recurs across at least three PRs or is supported by current repository guidance.

## Recurring review behavior

- Start from a changed contract, not a style preference. State the exact
  request, configuration, topology, or failure path that triggers the issue.
- Protect repository boundaries. Keep model-specific behavior in its domain,
  move genuinely reusable logic to a shared module, and reuse existing loaders,
  protocols, definitions, and config sources instead of bypassing them.
- Challenge unrelated scope and abstractions. Ask why a new flag, config, class,
  helper, fallback, or file is needed when an existing path appears sufficient.
- Trace concurrency and terminal paths: overlapping requests, disconnect,
  cancellation, partial failure, stale replies, collective schedule mismatch,
  cleanup, and retry. Prefer an explicit startup error for unsupported
  combinations over silently running with incomplete state or wrong weights.
- Preserve user-visible behavior across variants. An accepted CLI/API/config
  field must be applied, explicitly rejected, or visibly warned about; compare
  normal and headless, offline and online, and streaming and non-streaming paths.
- Ask for evidence at the affected boundary: a regression test using the real
  tensor or protocol shape, correct pytest markers so CI runs it, comparable A/B
  performance results, or an SLO-oriented latency/quality measurement.
- Keep interfaces explicit: avoid `Any` and opaque strings where a precise
  union, enum, or typed envelope exists; remove dead code and duplicated
  constants; validate requests before expensive execution.
- Document module scope and non-obvious invariants. Update examples, recipes,
  supported-model/feature docs, and hardware assumptions when users rely on
  them.
- When expertise crosses an ownership boundary, ask a named owner to inspect a
  specific file or contract and explain why. Do not send a generic `PTAL` list.
- Keep non-blocking cleanup in a linked follow-up only after the current PR is
  safe. If a revision fixes only part of a failure chain, continue the existing
  thread with the remaining trigger instead of opening a duplicate finding.

## Consolidated replies with multiple actions

Use a short action checklist for several independent next steps. Aim for 2-4
items when that fits the evidence; do not invent work or omit a real blocker
to meet a count. A single action can remain one sentence; zero findings is valid.

- Lead with the useful request. Keep any scope/head line short; link a short SHA
  to the full commit when a revision is relevant.
- Write each `- [ ]` item as a verb plus a concrete contributor-owned deliverable
  or completion condition. Put only necessary reasons or evidence underneath.
- Keep verified defects, evidence gaps, and optional suggestions distinct.
  Preserve each defect's source location, trigger, impact, and fix direction.
- Read existing discussion and replies first. Include only remaining relevant
  actions; do not repeat resolved requests or unchanged nudges.
- Do not add routine maintainer/CI status sections, workflow explanations, or
  edit-history boilerplate. Mention a testing limitation briefly only when it
  materially affects the request or verdict. Never imply a completed review or
  validation that did not occur.
- To withdraw a mistaken contributor-directed L2 request, one line is enough:
  "Please disregard my earlier request for CUDA L2 results."

### Reviewer CI responsibilities

Never ask contributors to provide the project's CUDA L2/Buildkite result.
Maintainers handle CI admission and inspect matching-head results. Keep these
checks in the review record rather than routinely narrating them to contributors.
Check actual labels and runs; distinguish missing admission, queued/running,
failed, passed, and unknown. Do not require an L2 result before its trigger label.
Applicable current-head L2 remains required at the existing approval/merge gate,
subject to the documented docs-only skip. This does not replace justified local
regression, reproduction, or matched A/B evidence and grants no authority to apply
labels, rerun CI, or launch compute.

## Write the finding

- Lead with the issue or decisive question; omit a review preamble.
- Name the concrete trigger and impact, then the smallest fix direction.
- Keep obvious fixes to one sentence. Use more detail only when the call path or
  ownership boundary is otherwise unclear.
- Prefer one root-cause comment over repeated symptom comments.
- Hedge only when evidence is genuinely incomplete; mark that item as a
  question or validation gap rather than overstating severity.
- Say briefly that no findings were found when the review is clean; zero
  comments is a valid outcome when posting is authorized.

Useful forms:

```text
This leaves <state> allocated when <failure/cancellation path>. Release it in
the shared terminal cleanup path.
```

```text
Does this producer also update <consumer>? It still reads the old field here.
```

```text
With <configuration>, this bypasses <existing loader/validator>, so <impact>.
Reuse that path or reject this combination before startup.
```

```text
@<owner>, could you check <specific boundary>? This changes <contract> used by
<consumer>.
```

Use a priority label only for a proved high-impact defect. Avoid generic praise,
routine “Nit:” prefixes, decorative severity labels, audit templates, rule IDs,
comment-count narration, and “I left a few comments.” Keep approvals, review
events, reviewer requests, mentions, and GitHub replies subject to the user's
explicit authorization.
