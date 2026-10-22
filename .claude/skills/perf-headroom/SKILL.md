---
name: perf-headroom
description: Assess vLLM-Omni performance headroom from its resolved stage topology and executed AR, diffusion, transport, and output paths. Use for component-level performance reports and RSI continue/stop evidence, not automatic optimization implementation.
---

# Omni Performance Headroom

Answer, for a declared workload: **which Omni components are already efficient,
which have worthwhile candidates, and which cannot yet be judged?** This is our
repository-specific assessment workflow. L0-L6 are coverage dimensions, not
runtime stages or the repository's CI test levels.

## When to use

Use for a manual or agent-assisted performance diagnosis, a component-state
report, or evidence about remaining optimization space. Deliver analysis and
recommendations, not an optimization implementation. The L0-L6 taxonomy is local
to this skill; it is not a standard from a paper or a hardware-utilization score.

## Prerequisites and authority

Identify the target revision, workload, completion boundary, quality contract,
objective and permitted environment. Reuse the user's explicit choices. A source
review can proceed without hardware; measured claims require matching runtime
artifacts. Diagnostic runs and matched references must stay inside the approved
resource budget. Installing a backend, changing deployed configuration, rewriting
kernels, or testing a different quality regime is not implied by this skill.

## One entry, two evidence modes

Invoke `$perf-headroom` with a workload and an approved environment or existing
artifacts. Start with [the Omni execution map](references/omni-execution.md), then
use [the assessment method](references/methodology.md). Read
[the evidence interface](references/evidence.md) when preparing the report input.
For a complete analysis use `analysis_depth: detailed`, visit all seven layer
guides below, and read [diagnosis and recommendation rules](references/diagnostics.md).
Follow each guide's symptom-specific references only when applicable. A quick triage
may use `triage`; it must not be presented as a completed deep analysis.

```bash
python .claude/skills/perf-headroom/scripts/analyze.py assess \
  --evidence /absolute/path/evidence.json --output /absolute/path/assessment
```

The default `measured` mode assesses runtime evidence. Use
`assessment_mode: source_only` when weights, hardware or execution are unavailable:
the same command reports source hypotheses and missing evidence, without invented
timings or a saturation verdict. A partial runtime measurement stays `measured`
with unmeasured components; it is not a source-only run.

`--layers L3,L4` narrows the report; L0 and L6 remain mandatory. `trace` accounts
for one device's recorded intervals. `replay` evaluates an explicitly supplied
fixed-resource-order DAG. Neither discovers a model graph or proves causality.
The numeric decision helper currently screens latency, not throughput or SLO
optimality; use native load metrics and a separately justified analysis for those.

## Layer navigation and deeper analysis

| Layer guide | Start here to answer | Deeper references selected by the guide |
| --- | --- | --- |
| [L0 measurement](references/l0-measurement/guide.md) | Can these observations support the intended claim? | Provenance, baseline/noise, quality, capture separation |
| [L1 pipeline](references/l1-pipeline/guide.md) | Where does request completion time go? | Wall time vs summed work, overlap, serving/streaming |
| [L2 host and scheduling](references/l2-host/guide.md) | Why was useful work not progressing? | Host root causes, synchronization, CUDA Graphs, Nsys collection |
| [L3 work graph](references/l3-work-graph/guide.md) | Which executed work is unnecessary or poorly composed? | Invariants, exact-cache validity, fusion, layout, lifetimes |
| [L4 kernels](references/l4-kernels/guide.md) | What limits each important executed shape? | NCU collection, memory, compute/scheduling, roofline, operator-specific diagnosis |
| [L5 communication](references/l5-communication/guide.md) | Which rank/transfer/residency cost delays the consumer? | Topology, readiness, exposed communication, offload |
| [L6 assessment](references/l6-assessment/guide.md) | What is mature, actionable, uncertain, or outside scope? | Bounds, interactions, reporting, recommendation cards |

The guides contain detection, distinguishing checks, interpretation and handoff
criteria. They are not independent timing buckets. In particular, read
[timing and overlap](references/l1-pipeline/timing-and-overlap.md) before converting
any kernel table or overlapping phase spans into an E2E share. Revisit L3 after
L4 finds unnecessary traffic; revisit L2/L5 when faster compute would expose waits.
For reusable precedents, use [the casebook protocol](references/casebook.md), not
historical model measurements as evidence about the current run.

## Workflow: request path before counters

1. **L0 — Freeze the experiment.** Pin source/weights/dependencies and verify the
   code actually imported on the execution host. Reuse the user's quality and
   workload choices. Record endpoint/task, input sizes, steps or output lengths,
   batch/concurrency, cache treatment, actual backend, dtype, graph/compile mode,
   placement and metric boundary. Fetching main does not prove that a running
   worker uses it. Keep native defaults separate from experimental overrides.
2. **L1 — Resolve the Omni path.** Read the effective `PipelineConfig`, deployment
   and runtime configuration. Map request entry, stage/replica execution, handoffs
   and final output. A text encoder, DiT and VAE inside one diffusion pipeline
   are semantic phases, not necessarily three engine stages. Measure the chosen
   objective through its real completion boundary, including media materialization.
3. **L2 — Account for the timeline.** Separate admission/queueing, device work,
   internal gaps, transfer and output tail. Audit trace coverage before calling
   anything a bubble, especially CUDA Graph nodes versus graph envelopes. Check
   whether a delay is exposed to this request or hidden by another stage/request.
4. **L3 — Audit executed work.** Follow the selected request/step execution path.
   Separate AR prefill/decode or diffusion condition/prefill/denoise/decode phases.
   Check existing fusion, exact cache hits, repeated invariants, materialization,
   graph capture/replay and offload. Prove that a source hypothesis reaches the
   measured path; graph and eager paths can differ.
5. **L4/L5 — Deepen the expensive components.** Inventory shapes and cost before
   choosing targets; include GEMM, attention, norm/modulation/layout, encoders and
   VAE when present. Report sampled and unsampled populations. Use matching-shape kernel
   references and targeted counters for GEMM, attention, norm/layout and media
   decoders. Analyze stage transport separately from intra-stage TP/SP/CFG/VAE
   communication. Multi-rank claims require the relevant ranks and consumer
   completion; single-rank traces cannot establish distributed saturation.
6. **L6 — Return component state and a proposed next experiment.** For each applicable
   component report execution efficiency AND structural work state, evidence,
   scoped reference, local gap, E2E relevance and missing validation. Give a
   falsifiable next action for candidates, not a generic optimization menu.
   Rank evidence-backed recommendation cards, including counter-signals, exact
   applicability, source owners, alternatives, interactions and acceptance gates.
   Include every layer as assessed, UNKNOWN or justified N/A. Run the helper,
   inspect both outputs, then explain what the evidence actually supports.

Freeze the evaluator before collection; record its hash separately from the model
revision. When asked to restart from zero, create a fresh campaign, collect ALL
runtime evidence and comparator measurements again, and set
`campaign.reuse_policy: fresh_only`. Old collectors may be reviewed and reused;
old traces, timings, counters, references and derived measurements may not. No
renaming or rehashing makes a historical measurement fresh. Otherwise explicitly
label any reused evidence and recheck comparability. Do not silently mix campaigns.

## Required handoff

Lead with the workload boundary and decision, then present **L0 through L6 in
order**. Every layer states verdict/confidence, metrics and denominators, evidence,
what is already good, opportunities, unknowns and its next discriminating
experiment. Put the component roll-up and experiment queue after these views:

| Omni component / phase | Time and denominator | Execution state | Work-volume state | Reference / evidence | Next action |
| --- | --- | --- | --- | --- | --- |

Populate rows from the resolved path, not a fixed list of model names. Include
queueing/handoffs/output and unclassified work where applicable. Distinguish:

- **Near reference:** close to a matched comparator, not globally optimal.
  Limited comparability stays in a separate summary group and blocks a stop
  proposal; a representative kernel does not certify its whole operator family.
- **Low observed leverage:** a small exclusive share in this window; not a
  throughput, memory-capacity or cross-request interaction bound.
- **Candidate:** source/trace-supported mechanism, with unvalidated E2E impact
  explicitly labeled. A highly efficient GEMM can still perform redundant work.
- **Unknown:** missing runtime, reference, trace coverage or structural review.

Give an ordered experiment queue using the recommendation contract in
`references/diagnostics.md`. A report with only counters, labels or a generic
optimization menu is incomplete. Preserve rejected ideas with their evidence.
Keep model-specific commands, shape maps and reports in run artifacts outside
the skill; do not add per-model reference pages or model-name branches here.

## Decision boundaries

Unprofiled baseline, diagnostic traces and NCU runs serve different purposes.
Reused evidence stays labeled historical. Source-only hypotheses are not measured
opportunities. Low gap share, high MFU or no discovered candidate cannot alone
establish completion. Reference gaps, sensitivity scenarios, predictions and
validated gains remain separate; do not sum overlapping opportunities.

`STOP_REVIEW_REQUIRED` only proposes a scoped review after coverage, precision
and a residual argument pass. It never authorizes RSI to stop or certifies global
optimality. Defaults are exploratory, not user-approved stopping thresholds.
An arithmetically closed ledger is not collection completeness: submit a
`diagnostic.timeline_coverage` audit. Missing or partial event/window coverage
blocks a stop proposal even when every recorded bucket has a module label.
An evaluator/controller independent of the candidate must approve scope, quality
and stopping; the candidate must not rewrite its evaluator or acceptance policy.
Budget exhaustion is `STOP_BUDGET`, not saturation.

Assessment does not authorize implementing candidates, installing dependencies,
downloading/substituting checkpoints or changing shared deployments. Use the
approved remote environment's scheduling rules; keep host details and credentials
out of this skill. The scripts check arithmetic and artifact integrity, not the
scientific truth of submitted annotations or independent approval.

## Skill maintenance and self-review

Run the CPU commands in [evidence.md](references/evidence.md#validation-commands)
and check local links after changes. Exercise the synthetic cases in
[review scenarios](references/review-scenarios.md): correct JSON alone is not a
behavioral pass. Never turn those synthetic cases into measured model results.
Preserve the source/evidence interface and document new helper outputs; do not
silently make existing assessment artifacts claim a deeper analysis.
