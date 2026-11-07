# Executable evidence format (schema_version 1)

The entry point consumes facts and source-backed hypotheses, not prewritten module
verdicts. All durations are milliseconds. Paths are relative to the evidence JSON
unless absolute. Artifacts may remain outside the repository; do not commit raw
traces, weights, private prompts or machine connection details into the skill.

## Required top-level fields

| Field | Shape and meaning |
| --- | --- |
| `schema_version` | `1` |
| `assessment_mode` | Optional `measured` (default) or `source_only`; see below |
| `scope` | Nonempty `model`, `weight_revision`, `code_sha`, `hardware`, `workload`, `runtime`, `boundary`, `quality_contract`; values may be strings or structured objects |
| `artifacts` | List of `{id, path, sha256}`; optional collection date/kind/command metadata is preserved |
| `provenance` | `{scope_verified: true/false, origin: string, evidence: [artifact ids]}`; verification is the analyst's explicit scope/source audit, not inferred from matching numbers |
| `benchmark` | `{profiled: false, cohorts_ms: [[...], [...]], evidence: [...]}`; each cohort is a series of warmed unprofiled requests under the same contract |
| `diagnostic` | `{window_ms, evidence: [...], exclusive_ledger_ms: {label: duration}}`; ledger is optional for triage, required for a stop proposal, and must close to the window |
| `modules` | Module records below, with separate layers L1-L5; missing layers become UNKNOWN |

### Source-only assessment

Use the same schema and command when the requested checkpoint or runtime is
unavailable. Scope still identifies the intended workload and reviewed code;
unknown fields use explicit objects such as
`{"status": "unavailable", "reason": "Target snapshot not present"}` rather than
claiming an inspected revision. Hash source snapshots and discovery/failure logs
as artifacts. `scope_verified` attests that this is an honest source-only scope,
not that the model ran.

Set `assessment_mode` to `source_only`. Omit `benchmark`, `diagnostic`, `coverage`
and `scoped_stop` or set them to null. All modules must have `measured: false`;
no runtime `metrics`, non-null `exclusive_ms` or `reference` are accepted. Source
reviews and candidate hypotheses may carry evidence, next_action and validation,
but not quantified impact. The report emits UNKNOWN execution, null noise and
uncovered time, `source_hypotheses`, and NEED_EVIDENCE. It cannot recommend stopping
or promote a source hypothesis into a measured optimization opportunity.

For a partially profiled run keep `measured` mode and mark individual missing
components unmeasured. A source-reviewed component can be structurally REVIEWED
while its execution remains UNKNOWN. This distinction also blocks a stop proposal.

For a stop proposal, supply `coverage: {ledger_modules: {label: [module_ids]},
unclassified_labels: [labels], evidence: [artifact_ids]}`. Every positive-cost
ledger bucket must bind to measured, structurally reviewed applicable modules.
Unmapped buckets, declared unknowns and the reserved `unattributed` label block
the proposal. Each binding assigns the full bucket cost to each named module;
use finer buckets for partitions. If supplied, a module's `exclusive_ms` must
cover its assigned cost. Do not add costs across layer views. This checks
accounting completeness, not whether the submitted semantic mapping is true.

Also supply `diagnostic.timeline_coverage: {status, reason, evidence}`. Status is
`complete`, `partial` or `unknown`; omission means unknown. The reason identifies
the device/rank/request window, event classes and graph-node/dropped-event checks.
Only complete coverage supports a stop proposal. This is a reviewed collection
claim, not completeness inferred by the script. A graph-envelope-only capture is
partial for internal-gap analysis, even if its arithmetic ledger closes.

`--repo` additionally checks HEAD and clean tracked/untracked runtime paths
(`vllm_omni`, `requirements`, `pyproject.toml`). A runtime patch fails this simple
current-tree check: separately snapshot and assess the patched revision instead
of bypassing provenance. Experiment overrides outside those paths must be hashed
as artifacts and declared in `scope.runtime`. Skill-only changes are not changes
to measured model execution. Without `--repo`, results describe the supplied
snapshot and do not independently verify a currently running system.

## Module record

- `id`, `layer` (L1-L5), boolean `applicable`, boolean `measured`.
- `reason` is required for N/A. Do not label unmeasured applicable work N/A.
- `evidence` is required when measured; optional `observation` explains the facts.
- Optional `context`: descriptive `{component, stage_id, phase, source_paths}`
  from the resolved Omni execution map, preserved in JSON/text output. These are
  analyst annotations, not automatically verified routing or ownership. Include
  rank/replica details in `scope.runtime` or the linked map artifact as needed.
- Optional `exclusive_ms`: only use a cost attributed without overlap inside the
  shared diagnostic window. Do not use kernel-duration sums here. Modules can be
  views of the same work across layers; their shares are NOT summed.
- Optional `metrics`: named value/unit/window records, preserved verbatim. Null
  counters remain unknown. Metrics do not automatically become stop criteria.
  In measured mode, distinguish MFU, useful kernel TF/s, Tensor-pipe activity,
  logical bandwidth and measured DRAM bandwidth by their actual definitions.
- Optional `reference`: `kind` (`measured_reference` or
  `conditional_resource_model`), `current_ms`, `time_ms_range: [low, high]`,
  `comparability` (`matched`, `limited`, `unknown`), `evidence`, and caveats.
  Current/reference times must cover identical call counts and semantic work.
  Their ratio describes relative cost; it is not hardware utilization.
  Limited comparability is preserved but blocks a stop proposal, even for a
  near-reference ratio. Do not extend a sampled kernel's result to unmeasured
  shapes, phases or an entire operator family.
- `structural_review: {complete: boolean, evidence: [...]}`. Completion means the
  declared structural search has been performed, not mathematical optimality.
- Optional `candidates`: list of `{id, evidence, next_action, validation}`. Add
  `priority`/`mechanism` as useful; do not populate candidates from generic menus.
- A candidate's optional `impact` has `type` (`reference_gap`, `scenario`,
  `model_prediction`, `validated_gain`), `range_ms: [low, high]`, `assumptions`.
  Negative values represent regressions. A validated gain additionally requires
  its own `evidence`, `quality_passed: true`, and `paired_unprofiled: true`.

## Policy and decisions

### Detailed analysis and fresh campaigns

Use `analysis_depth: detailed` for full investigations. Omission preserves the
older `triage` interface; triage is not a deep-analysis completion claim.
`detail_checks` lists missing review/card/counter coverage and blocks a detailed
stop proposal when incomplete. It does not suppress useful partial results.

Supply `layer_reviews` keyed L1-L5, each with `verdict`, `confidence`
(`high|medium|low`), `evidence`, `mature`, `opportunities`, `unknowns` and
`next_experiment`. Explicit explanations such as "none found within this search"
or a source-backed N/A are preferable to empty fields. L0 and L6 are generated
from the contract and decision checks. Reports render L0-L6 in order.

Detailed candidates carry all fields defined in
[diagnostics.md](diagnostics.md#recommendation-card-contract). Each applicable L4
module has `kernel_audit: {basis, population, selection, limitations,
counter_findings, evidence}`. Population identifies shape/dtype/backend/phase,
sampled and total calls/cost with denominators; findings preserve NCU rules and
the analyst's applicability decision. Basis is `in_model_ncu`, `isolated_ncu`,
`timeline_only` or `not_required`. Isolated/timeline evidence leaves in-model
counter coverage unresolved. `not_required` needs a `reason` justified by observed
leverage; it is not permission to omit dominant or unknown-cost work. These are
reviewed annotations, not automatic attribution or proof of causal diagnosis.

For a new campaign supply `campaign: {id, evaluator_revision, started_at_utc,
reuse_policy}`. Policies are `fresh_only` or `declared_reuse`. All artifact records
then declare `kind: measurement|source|method`; measurements include `run_id` and
`collected_at_utc` with timezone. Baseline/diagnostic evidence must be measurement
artifacts. A fresh-only manifest with old dates or mixed run IDs invalidates
provenance. Derived facts retain the original collection campaign/time and link
raw parents in the run manifest. The helper cannot detect false timestamps or a
historical measurement disguised as source: raw lineage review remains mandatory.
New file mtimes or re-exported old profiles do not satisfy fresh-only collection.

Optional `policy` overrides the documented numeric defaults in `analyze.py`:
2% worthwhile gain, 2% cohort-median drift, 3% within-cohort range, at least two
cohorts of five samples, 0.9 near-reference ratio and 2% low-leverage share.
`max_noise_to_useful_gain_ratio` defaults to 0.25. The noise proxy is the maximum
of cohort drift and within-cohort ranges. A proposal requires that proxy below
the configured fraction of useful gain and residual upper + noise < useful gain.
These are initial triage settings, not universal bounds. `policy_predeclared`
must explicitly be true before the evaluator can support a scoped stop. They
do not substitute for confidence intervals or the user's acceptance contract.

Optional `limitations` (strings), `quality_status`, `budget_exhausted` are reported.
The performance decision does not certify deployment quality.

`component_summary` groups near-reference execution, limited/unresolved
references, low observed leverage, candidate modules, unknown execution and
unreviewed work. Groups can overlap: high efficiency and redundant work are
compatible. `MEASURED_NO_REFERENCE` means cost is measured but efficiency relative
to an appropriate ceiling is unknown. Metrics retain their value/unit/window in
both machine and text reports; these are descriptive, not universal stop gates.

Optional `scoped_stop` requires `remaining_gain_upper_pct`, `search_scope`,
`assumptions`, and `evidence`. This is a submitted bound/search argument for a
DECLARED candidate space, including interactions. The script cannot verify its
scientific truth. A caller's `reviewed` flag is not authorization. Only stable,
sufficiently precise data, closed ledger coverage, complete L1-L5/structural
coverage, resolved references, no candidates, all layers and a predeclared
threshold can yield STOP_REVIEW_REQUIRED. Independently review assumptions,
search coverage and raw evidence before any controller stops RSI. Missing
evidence yields NEED_EVIDENCE. `stop_checks` and `stop_blockers` explain the
result; `stop_authorized` is always false, and the certified global bound is null.

The numeric helper screens a scalar latency contract. Throughput, SLO/tail,
first-packet and memory-capacity assessments need native load/phase evidence and
their own acceptance reasoning; latency cohort precision does not validate those
objectives. Do not submit a latency `scoped_stop` as a whole-service stop claim.

## Trace and DAG subcommands

`trace --trace trace.json.gz --device 0 --output trace-ledger.json` accounts over
the device-event span. Supply BOTH `--start-us` and `--end-us` to select a request
window in the trace timebase. With no role rules, all device activity is retained
as unattributed. Optional `--rules rules.json` accepts a list of
`{label, contains: [literal kernel-name fragments]}`; ambiguous matches stay
unattributed. This helper does not replace semantic/graph-node attribution.
It assumes complete node-level collection for the chosen window. It does not
parse Nsight SQLite graph envelopes or detect missing graph-node coverage; apply
the gate in [omni-execution.md](omni-execution.md) before interpreting its gaps.

The output also separates `summed_event_ms` (clipped kernel/copy/set durations
added independently) from `busy_union_ms` and `window_ms`. `overlap_excess_ms`
is summed event time minus union time: duplicate-counted activity, not elapsed
overlap or removable time. See [timing and overlap](l1-pipeline/timing-and-overlap.md).

`replay --graph graph.json --output prediction.json` consumes
`{resource_order_confirmed: true, nodes: [{id, ms, deps: [node ids]}]}`. Include
stream/resource order and data dependencies explicitly. Submit a second graph
for a proposed transformation, then compare predictions. Cycles, missing nodes,
negative/non-finite costs and unconfirmed resource order are rejected. This is
fixed-DAG sensitivity, not an automatic Daydream/dPRO implementation.

## Validation commands

From the repository root, CPU only, Python 3.10+ and pytest:

```bash
python -m pytest .claude/skills/perf-headroom/scripts/test_analyze.py -q -o addopts="" -m 'core_model and cpu'
python .claude/skills/perf-headroom/scripts/analyze.py --help
```

The addopts override avoids requiring xdist for these standalone CPU tests.
These are contributor-tool tests, not GPU CI or model-quality acceptance. No
Buildkite job is installed by this experimental skill. Run the skill-creator
`quick_validate.py` against the folder when that tool is available.
