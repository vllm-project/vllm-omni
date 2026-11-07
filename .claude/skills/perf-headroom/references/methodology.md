# Omni headroom assessment method

Our unit of analysis is a component on the resolved Omni request path, identified
by stage/replica, semantic phase and owning source. L0-L6 are cross-cutting views
of that path, not seven independent time buckets. Use the
[execution map](omni-execution.md) to locate the selected implementation before
applying these tests. External work at the end supplies analytical tools, not
the task structure or thresholds of this skill.

This page retains the shared concepts and research rationale. For operational
detection and root-cause procedures, follow the L0-L6 navigation in
[SKILL.md](../SKILL.md#layer-navigation-and-deeper-analysis). The detailed
[timing guide](l1-pipeline/timing-and-overlap.md) and
[bound derivation](l6-assessment/bounds-and-impact.md) distinguish elapsed time,
work sums, conditional predictions and mathematically justified space intervals.

| View | Question about this Omni workload | Evidence needed to close the view |
| --- | --- | --- |
| L0 | Is this the intended code, task, quality and metric boundary? | Actual imported source, resolved settings, baseline repetitions, noise and provenance |
| L1 | Where does request/chunk completion time go? | Entry-to-output path, semantic phases, queue/handoff/output accounting |
| L2 | Where is progress exposed to scheduling or waiting? | Complete device timeline, CPU/dependency correlation, head/tail/internal gaps |
| L3 | Is all executed work necessary under this contract? | Source-to-trace map, repetition counts, existing fusion/cache/graph behavior |
| L4 | How efficiently is each expensive operator executed? | Matching work/shape references, resource counters and comparability limits |
| L5 | What communication or placement cost is exposed? | Participating ranks/replicas, topology, readiness, collective and transport dependencies |
| L6 | What is worth doing next, or what justifies a scoped stop review? | Non-overlapping attribution, typed impact, complete unknowns and residual argument |

Each component has two independent axes: execution efficiency and work volume.
For example, a near-reference linear can still execute too often; a low-gap
denoising loop can still contain inefficient attention; a fast DiT does not
close a pipeline whose VAE or output tail is unmeasured. Counter evidence, source
evidence and E2E impact should agree before promoting a candidate.

## L0: measurement contract and validity

Freeze the implementation and runtime treatment, including temporary backend
overrides and instrumentation. Bind results to hashes of the workload, dependency
manifest, benchmark command and selected source. Include untracked runtime code.
Keep profiling overhead out of authoritative latency/throughput samples. Use
warmups and repeated steady-state requests; compare cohorts for drift. A sample
range or cohort-median spread is descriptive, not a confidence interval.

For gain claims, use paired/rotated A/B order where practical. The noise floor
must be materially below the minimum useful gain. Repeated optional testing
requires an appropriate statistical design, not repeated use of a fixed-test CI.

If only source is available, use source-only mode. Record unavailable weight or
runtime fields explicitly as structured status/reason values. This can establish
which branches/fusions exist and suggest hypotheses, not actual backend activity,
MFU, gaps, candidate latency gains or a stop decision.

## L1: service and semantic stages

Distinguish client E2E, queue/dispatch, stage handoff, pipeline execution and
output serialization. For diffusion split text/condition encoding, initial
prefill, steady denoise steps, scheduler, VAE and postprocessing. For general
Omni include AR/DiT stages and connectors; latency and throughput need separate
load contracts. Cached-away stages are N/A within a warm-cache contract, but
UNKNOWN for a cold-cache deployment. Never silently generalize between them.

Existing integration points include serving/DFX benchmarks, engine orchestrator,
DiffusionWorker, diffusion model runner and OmniTorchProfilerWrapper. Inspect
their current signatures. DiffusionPipelineProfilerMixin may synchronize around
stage methods: check its implementation and do not infer natural overlap from
synchronized stage timings.

## L2: scheduling and exposed overhead

For a specified device/window, union kernel, memcpy and memset intervals. Separate
request head/tail from internal gaps. Overlapping device intervals count once;
kernel-duration sums are a different quantity. Retain unclassified activity.
First pass the execution map's trace coverage gate: graph envelopes without node
events cannot reveal internal gaps. A closed arithmetic ledger can still omit
activity if collection was incomplete.

Correlate CPU launch/runtime events with device events and semantic ranges. A
long host range is not necessarily exposed; overlap/causal attribution and a
counterfactual are needed. Inspect graph replay versus eager paths and sparse
regions, not only large gaps. Low gap share supports low scheduling leverage,
not high Tensor Core utilization. Removing all gaps is a sensitivity ceiling
only under the stated assumption that remaining work and dependencies stay fixed.

## L3: work volume and operator graph

Fusion is owned here: draw producer/consumer boundaries, reads/writes, reuse,
legal numerical transformation, lifetimes and existing fusion. L4 evaluates the
replacement's resources; L2 checks exposed launches; L6 validates E2E effects.
Do not count the same saved materialization in all four layers. Fewer kernels is
not sufficient: fusion can increase spills, reduce parallelism or block overlap.

Keep model-specific collectors and semantic maps with versioned run artifacts,
not model-name branches in this skill. Inspect the measured source and actual
backend. A pattern mapper must assert source signatures, shapes, graph-node
identity and repetition counts; mismatches remain UNKNOWN. Replaying old evidence
does not make it fresh; retain its collection date and instrumentation hashes.

Audit repeats and materialization: per-layer/per-step invariants, copies, layout
conversions, cache eligibility, broadcasts, precision casts and producer/consumer
interfaces. Verify a source hypothesis against executed paths. Existing fusion
must be recognized. Candidate savings need a replacement cost, not just deletion
of a useful operator. Structural opportunity is independent of kernel efficiency.

## L4: execution efficiency and reference ladder

- GEMM: actual shapes and useful 2MNK work, dtype/accumulation and layout; compare
  like-for-like implementations and cache/weight-reuse conditions.
- Attention: Q/K lengths, heads, dimensions, masks, layout and backend; FA includes
  softmax/reduction/synchronization, so GEMM's useful-FLOP ratio is not its target.
- Pointwise/layout: logical traffic and materialization, measured DRAM/L2 traffic
  when available, resource throughput and stalls. Logical bytes/time is not HBM
  utilization. More than 100% of an empirical reference signals comparability or
  drift questions, not an impossible GPU.

Use Nsys/torch traces to select representative NCU targets. NCU replay changes
execution and cache behavior; use counters diagnostically, not its wall time as
pipeline latency. Separate empirical reference, conditional resource model and
certified lower bound. Current executed FLOPs/bytes are not automatically minimum
required work. Empirical throughput ceilings are not universal hard bounds.
Follow [diagnostics.md](diagnostics.md) for population coverage, in-model NCU
selection, competing explanations and backend/fusion recommendation tests.

Report the numerator and denominator of every efficiency metric. For a dense
GEMM, useful throughput is `2*M*N*K / device_time`; a ratio to dense peak must
use the same precision, clock convention and device count, without sparse-peak
inflation. Model-level MFU instead needs useful model work over its declared
wall-time window. An NCU Tensor-pipe active percentage is neither of those.
Attention's softmax/reductions and a VAE's conv/padding/layout need their own
references; do not apply a linear's MFU target to them.

To construct a resource lower bound, state the allowed implementations first.
Derive unavoidable work/traffic, not the current kernel's executed work/traffic.
For a fixed semantic region, `max(F_min/P_max, Q_min/B_max)` is a possible resource
bound only if `F_min`/`Q_min` are valid minima and `P_max`/`B_max` are valid upper
capacities for every allowed implementation. Compute and memory may overlap;
do not add the two. Serial dependency bounds may be combined only for work that
cannot overlap, with shared-resource accounting. Empirical calibrated throughput
gives a conditional model, not a certified physical bound. If fusion, reuse,
algorithm changes or cache residency invalidate a minimum, leave the hard bound
unknown and use a scoped reference instead.

## L5: distributed communication

Collect all relevant ranks when assessing distributed completion. Align clocks,
send/receive/collective dependencies and consumer readiness. Separate wire work,
wait/straggler effects, exposed critical-path delay and resource competition with
compute. NCCL duration or overlap percentage alone cannot establish saturation.

For the device-initiated remote-store FA+A2A path studied by X-Stage, compare a
matched FA-only path, serial FA+A2A and fused path:

E_res = T_fused - T_FA

H = 1 - E_res / (T_serial - T_FA)

The incremental serial-composition cost is the denominator, not standalone A2A.
State sender-side versus remote-visible/consumer completion boundaries. Near-zero
sender residual does not prove consumer completion. Calibrate issue time, drain
rate and outstanding capacity independently when using that model; do not apply
its parameters/thresholds to generic NCCL paths or another topology.

## L6: impact and decisions

First make a closed time ledger; then model counterfactuals. Daydream/dPRO motivate
dependency-graph transforms and replay, not addition of independent savings.
Our `replay` helper is intentionally narrower: a fixed-resource-order DAG with
explicit resource edges. It ignores new contention/locality changes unless the
caller changes costs/edges. Validate the graph and duration model with held-out
measurements before using its predictions for a stop decision.

Report each impact as one of reference_gap, scenario, model_prediction or
validated_gain. Do not combine them into a certified interval. Given an actual
valid lower bound L and baseline T, 1-L/T bounds possible latency reduction;
without such a derivation leave the certified upper bound unknown.

The decision engine separates provenance, measurement stability, module execution
state, structural review, performance decision and quality status. It never uses
an 80%-SOL rule as universal completion. A scoped stop additionally needs explicit
search coverage or a valid residual bound, including interactions and unknown
work, below a predeclared worthwhile-gain threshold. No active candidates alone
is insufficient. Close the diagnostic ledger and bind each bucket to reviewed
modules, including gaps and overlap. Unknown/unmapped work blocks a stop proposal;
do not hide it in an aggregate row. Layer views may overlap and are not summed.
Use the descriptive noise proxy as a conservative screening check relative to
the useful-gain threshold and residual margin, not statistical certification.
The engine only emits STOP_REVIEW_REQUIRED: an independent reviewer/controller
must approve stopping outside the candidate's editable evidence. Budget stop and
performance stop are separate outcomes.

## Method references, not an imported workflow

Our native routing/ownership and report contract are defined above and in the
execution map. These sources motivate measurement, resource modeling or causal
analysis only; none supplies a universal "optimized" threshold for vLLM-Omni.

- [X-Stage v1, sections 3-5](https://arxiv.org/html/2607.23264v1): calibrated mechanism and residual tests with explicit timing boundaries.
- [Daydream, ATC 2020](https://www.usenix.org/conference/atc20/presentation/zhu-hongyu): dependency-graph what-if prediction.
- [dPRO, MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/file/b422680f3db0986ddd7f8f126baaf0fa-Paper.pdf): distributed graphs, critical paths and interacting transformations.
- [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html): resource ceilings conditioned on arithmetic intensity.
- [FlashAttention](https://arxiv.org/abs/2205.14135): exact attention can reduce HBM traffic by changing tiling/materialization; current traffic is not a minimum.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): work partitioning and non-matmul work affect efficiency independently of launch gaps.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608): Hopper asynchronous compute/data movement and matmul/softmax pipelining motivate matched backend experiments; published utilization is not our stopping threshold.
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): counter definitions and replay/cache/clock perturbations. Tool rules are diagnostic hypotheses, not guaranteed E2E gains.

These sources were checked on 2026-10-20. Daydream and dPRO study training;
transfer their dependency-analysis ideas, not their published error guarantees,
to Omni inference. X-Stage's calibrated remote-store model is conditional on its
mechanism and platform; it is not a bound for generic multi-GPU inference.
