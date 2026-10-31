# Deep diagnosis and actionable recommendations

Use for `analysis_depth: detailed`. The deliverable is an explanation of current
limits and a falsifiable optimization queue, not an automatic backend selector.
All conclusions are scoped to this workload, execution mode and hardware.

This page is the shared collection/recommendation contract. Use the layer guides
linked from [SKILL.md](../SKILL.md#layer-navigation-and-deeper-analysis) for the
actual investigation, including NCU memory, compute/scheduling and operator-specific
branches. Actions in the tables below are recommendations and validation plans;
they do not authorize implementing optimizations or changing a deployment.

## Collection contract

Keep four distinct evidence products: unprofiled repeated baseline, low-overhead
timeline, source/shape inventory, and targeted hardware counters. Link them with
request, process, rank, phase, kernel signature and occurrence. Never combine
durations from different products into one closed ledger. Record the overhead of
each diagnostic capture relative to the baseline, including instrumentation.

Before collecting counters, enumerate operator families and their shapes, call
counts, summed kernel durations and exclusive device-time shares. Include the
encoder, conditioning/prefill, every distinct denoise regime/CFG branch, scheduler,
VAE, output materialization and unattributed work. Explicitly distinguish calls
from kernels: an operator may launch several kernels; a fused kernel may span
operators. Cost sums, union time and E2E share have different denominators.

Select high-cost shapes, high-repeat tiny kernels and structurally suspicious
regions. State why remaining populations were omitted. A useful initial *sampling
target* is shapes explaining 90% of recorded device work plus every expensive
semantic phase; this is a collection budget, not a saturation threshold. Report
actual coverage and the unobserved cost, not just the number of NCU reports.
Do not say all linears or all attention are efficient after sampling one shape.

### In-model NCU procedure

1. Select a bounded warmed request/phase from the fresh timeline. Identify the
   worker, kernel signature and launch occurrence; verify selection counts.
2. Inspect the installed NCU version and supported sections/metrics. Start with
   SpeedOfLight, LaunchStats, Occupancy and MemoryWorkloadAnalysis; add
   SchedulerStats, WarpStateStats, InstructionStats or SourceCounters when a
   competing explanation requires them. Save `.ncu-rep`, machine export, raw
   rule text, command, filters, replay/cache/clock settings and collection errors.
3. Use the approved reservation and permissions. A failed capture is UNKNOWN,
   not zero utilization. Do not change global clocks, driver profiling policy or
   another user's process to obtain counters. Sudo, if authorized, applies only
   to the task's profiler and must preserve reservation/device restrictions.
4. Separate graph-native from eager diagnostic runs. Changing graph mode to make
   a kernel selectable changes the experiment. Check the matching shape and
   implementation; retain that limitation instead of certifying graph-native cost.
5. Keep in-model and isolated matched-shape NCU separate. The latter is useful
   for controlled alternatives but changes residency, overlap and allocator state.
   Document replay-induced serialization/cache/clock treatment. NCU durations
   are diagnostic, never the authoritative request latency.

For every metric retain exact name, unit, scope, aggregation and timebase.
SOL compute/memory percentages can summarize different subunits; inspect their
breakdown. Tensor-pipe activity is not useful-FLOP MFU. Only interpret stall
reasons alongside scheduler issue/eligible-warp evidence. High occupancy is not
an objective by itself and low occupancy is not sufficient proof of a bottleneck.

## Diagnosis routes

These are tests to choose from when evidence warrants them, not recommendations
to emit unconditionally. Every proposed action needs a concrete executed owner.

| Symptom | Distinguishing evidence | Candidate / counter-signal / acceptance |
| --- | --- | --- |
| L2 device gaps | Node-complete timeline; CPU launch, synchronization, allocator, producer readiness; first/steady/last step | Capture eligible regions or remove proven host waits. A missing producer or graph node is not a launch bubble. Accept on unprofiled latency and unchanged scheduling semantics. |
| L3 repeated work | Data dependencies across steps/CFG/requests, cache keys, invalidation, actual call counts | Hoist only invariants; reuse exact projections. Timestep-dependent refinement or request mutation refutes reuse. Test cold/hit/miss and concurrent lifecycles. |
| L3 materialization chains | Producer/consumer DAG, shapes/strides, intermediate bytes and lifetimes, already-fused calls | Fuse norm/modulation/residual/activation/layout or a GEMM epilogue when legal. Compare full subgraph cost including packing; check registers/spills, peak memory and graph compatibility. |
| L4 GEMM poor relative cost | Useful 2MNK, dense precision peak, shapes/alignment, tiles/waves, Tensor-pipe and memory limits | Match library/algorithm/layout alternatives. Tiny or skinny GEMM may be latency/wave limited; padding raises executed work. Benchmark full operator with bias/casts/epilogue. |
| L4 attention poor relative cost | Backend actually executed; Q/K/V shapes/strides, heads, mask/causal/GQA, dtype; matmul versus reduction/sync/layout cost | Compare compatible installed backends with identical semantics. A newer backend may lose on short sequences or require expensive packing. Include QKV preparation and output layout, not just the FA core. |
| L4 norm/pointwise/layout dominates | Actual materialization versus view, DRAM/L2 bytes, launch count, dependency and vectorization | Fuse adjacent work or remove conversion at its owner. Logical traffic is not DRAM traffic; warm L2 may dominate. Compare complete region and numerical tolerance. |
| L4 VAE/conv dominates | Conv shapes/dtype/layout, algorithm, tiling overlaps, padding, upsample/activation, memory capacity | Test layout/algorithm/compile or legal decoder fusion; distinguish redundant tile borders from required overlap. Check seams, temporal artifacts, peak memory and whole decode. |
| L5 offload/transport dominates | Residency and peak live memory; pinned/pageable copies; sender-ready to consumer-visible dependency | Reduce representations/copies or overlap prefetch with independent work. Memory capacity and competing bandwidth can invalidate removal/overlap. Include both endpoints. |
| L5 distributed stall | All ranks, topology, collective size, readiness skew, exposed critical-path cost | Test partition/granularity or overlap only for the active topology. NCCL duration alone is not exposed cost; faster local compute can expose more communication. |
| L1 output tail / service queue | Stage completion to final CPU media or client completion; IPC, encode/serialize, admission/backpressure | Avoid duplicate materialization or use supported async output paths. Verify the model returns the required payload type. Single-request gains need not improve loaded service. |

### Attention backend decision

Read `attention/selector.py`, the selected backend and model processor at the
frozen commit. Record supported devices, dtype, head dimensions, masks, varlen,
CFG/batching, graph/compile and TP/SP requirements. Configuration names alone do
not prove the kernel executed. Reject unsupported choices before timing them.

Partition the cost into QKV projection, norm/RoPE, packing/layout, attention core
and output projection. Check whether low useful TF/s is explained by small grids,
padding, non-matmul instructions, dependencies or memory behavior. Run the native
attention benchmark where its semantics match; otherwise use a run-local adapter.
For a separately authorized backend experiment measure core AND wrapper and then unprofiled E2E A/B.
Kernel numerical comparisons precede full image/video quality acceptance. Do not
suggest FP8 or approximate attention under a BF16 exact-semantics contract.

### Fusion decision

Record the unfused subgraph and its consumers, expected removed launches/traffic,
replacement work, intermediate residency and numerical effects. Determine whether
the change belongs in the model, common layer, compiler or backend. Existing
compiler fusion may already eliminate Python-visible operators. Test at actual
shapes and graph mode, including adverse shapes. A high-efficiency GEMM does not
close opportunities around it; a low-efficiency tiny kernel may still be irrelevant.

## Recommendation card contract

Attach each card to one owning module/layer; reference other affected layers.
The detailed evidence interface requires the following fields on each candidate:

- `id`, `evidence`, `mechanism`: observed symptom and causal explanation; separate
  observation, inference and untested conjecture.
- `source_paths`, `applicability`: exact owner/symbol and shape/dtype/backend,
  phase, graph, hardware and parallelism restrictions.
- `counter_evidence`: observed counter-signals or the test needed to discriminate
  competing explanations; “not yet collected” is honest but lowers confidence.
- `alternatives`: compared/rejected options with reasons; no generic checklist.
- `next_action`, `validation`, `quality_gate`: smallest isolating experiment,
  before/after command/config delta, numeric acceptance and rollback criterion.
- `expected_effect`: qualitative mechanism when magnitude is unknown. Quantified
  `impact` remains typed reference_gap/scenario/model_prediction/validated_gain;
  name its window, counts and assumptions. No invented percentage.
- `priority` (P0/P1/P2), `confidence` (high/medium/low), `risks`, `overlap_group`:
  rank by exposed cost, plausible improvement, confidence and validation effort.
  Explain priority; don't multiply arbitrary scores into a fake precise ranking.

P0 restores invalid evidence/correctness, P1 targets likely material exposed cost,
P2 is a lower-leverage or uncertain exploration. Source-only cards remain hypotheses.
Keep rejected ideas in the layer narrative with evidence. Do not create a card
just to fill an empty layer. Unknown is a valid result with a collection action.

## Impact, confidence and report acceptance

For a serial, exclusive fraction f and validated local speedup s, the fixed-rest
scenario is `T_new = T * (1 - f + f/s)`. This is not a bound when dependencies,
overlap, cache behavior or contention change. Use a dependency counterfactual for
interacting changes; otherwise report interactions as unknown. Do not sum backend,
fusion and graph savings that eliminate the same time.

High confidence requires concordant source, runtime and controlled comparison;
counter observations without a controlled alternative support at most a mechanism
hypothesis. Low confidence is appropriate for source-only or mismatched proxies.
Recommendations may be useful before a saturation verdict is possible.

A detailed report has seven ordered sections and, within each, an explicit verdict,
metrics/denominators, evidence/confidence, already-good work, opportunities, unknowns
and next experiment. L4 includes a population/counter coverage table and actual
NCU recommendations with applicability checks, not an unexplained metric dump.
Conclude with prioritized cards and a residual-space statement: validated gains,
reference gaps, conditional scenarios and unknown certified bound kept separate.
“Detailed contract complete” means fields/evidence routes are present; it does not
mean the system is optimal, that all hypotheses are correct or that RSI may stop.
