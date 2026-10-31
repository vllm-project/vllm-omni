# Compute, occupancy, waves and scheduler analysis

## Start from the actual work

Record useful operation count and, where available, executed instruction/work
count. Padding, masked lanes and redundant operations can make the hardware busy
without useful throughput. Match dtype/accumulation and instruction family to the
capacity being used; an SM aggregate is not a universal FLOP meter.

## Compute branch

Inspect ComputeWorkloadAnalysis and instruction breakdown to identify the active
pipeline: Tensor operations, ordinary floating point, integer/address work,
special functions or load/store issue. Verify that an expected Tensor Core path
actually executes. A kernel mixing phases can have different limits over time.

If a pipeline is highly used, ask whether the instructions are necessary and
whether a different work decomposition reduces them. If it is lightly used,
first inspect workload size and issue/latency constraints. Do not recommend more
Tensor utilization for an operation dominated by required reductions or nonlinear
functions. Architecture-specific MMA instruction counts need family-specific
interpretation before converting to logical operations.

## Occupancy and wave branch

1. Record grid/block geometry and SM count available to this launch, including
   partitions. Check whether independent blocks can populate that resource set.
2. Compare theoretical residency limits with achieved occupancy. Identify register,
   shared-memory, thread/warp or block-slot constraints from launch/resource data.
3. Inspect waves/tail imbalance: partial final waves or unequal block work can
   lower the average even when interior execution is efficient.
4. Compare the chosen tile/resource footprint against matched alternatives, not
   an arbitrary occupancy percentage. More resident warps can reduce per-block
   reuse or increase traffic; lower occupancy can be deliberate and efficient.

Low occupancy plus high useful pipeline throughput may need no occupancy change.
High occupancy plus poor issue progress suggests waits/dependencies, not shortage
of resident warps. Short or skinny work can be shape-limited; label that limit
instead of treating the dense large-matrix peak as achievable for every shape.

## Scheduler and stall branch

Use SchedulerStats to distinguish resident, eligible and issued work. Inspect
WarpStateStats and source sampling only when issue opportunities are actually
being lost. Sample percentages describe warps/instructions under a defined
normalization, not a direct percentage of removable wall time.

| Signal | Competing explanations to check | Recommendation direction |
| --- | --- | --- |
| Few eligible warps, memory-dependency waits | Grid size, memory latency, independent loads, cache treatment | Memory/latency path; not necessarily a saturated bus |
| Barrier waits | Unequal producer work, shared-memory phases, required cooperation | Work partitioning/synchronization investigation; do not delete required barriers |
| Issue/pipeline pressure | Instruction mix, non-matmul work, redundant address/convert operations | Reduce necessary issue cost or compare a compatible implementation |
| Many inactive/predicated lanes | Shape tails, masks, divergence and tile choice | Shape-matched decomposition; account for useful versus executed work |
| Achieved residency below theoretical | Tail effects, short lifetime, uneven blocks, scheduling | Distribution/launch diagnosis rather than only register reduction |

A reported long-scoreboard stall is not a guaranteed saving: other warps may
already hide the latency. Use absolute duration and useful work alongside ratios.
The [official triage guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
offers routing heuristics; its thresholds are not this skill's stop policy.

## Output

State the limiting mechanism with corroborating instruction/resource/scheduler
evidence and a counter-explanation. Recommend a bounded tile/backend/partition or
structural investigation only where applicable. List what cannot be resolved
without symbols, representative shapes or another collection mode. Send local
reference gaps to L6 rather than translating stall percentages into E2E gains.
