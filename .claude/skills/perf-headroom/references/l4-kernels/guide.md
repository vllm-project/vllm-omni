# L4: kernel population and efficiency diagnosis

## When to use

Use after L1-L3 establish expensive phases, shapes and executed work. Low gaps
do not imply efficient kernels; high kernel efficiency does not imply necessary
work. This layer diagnoses the measured implementation, not an abstract model name.

## Population before counters

Build a table keyed by phase, operator family, shape, dtype/accumulation, layout,
backend and execution mode. Include call count, kernel count, summed kernel time,
distribution and critical-path relevance where known. One operator may launch
several kernels; one kernel may implement several operators.

Choose expensive shapes, high-repeat tiny work and suspicious structural regions.
Cover initial/steady/final regimes and encoder/DiT/VAE or AR/generation paths that
materially differ. Record sampled and total calls/cost with the denominator.
Unobserved shapes remain unknown. Kernel-work coverage is not wall-time coverage.

## Diagnostic order

1. Read [NCU collection](ncu-collection.md); validate target, sampling, replay,
   cache, clocks and graph treatment before interpreting counters.
2. Compare to [a matched reference and roofline](roofline.md). Check whether the
   shape is capable of exposing enough parallelism before blaming low throughput.
3. Use [compute and scheduling](compute-and-scheduling.md) to identify instruction,
   issue, dependency, occupancy, wave or tail limitations.
4. Use [memory analysis](memory-analysis.md) for actual traffic, hierarchy,
   locality, coalescing, shared-memory conflicts and spills.
5. Interpret the mechanism with the relevant operator procedure:
   [GEMM](gemm.md), [attention](attention.md), or
   [norm/pointwise/layout and VAE](pointwise-and-vae.md).
6. Return unnecessary work to L3, exposed launch delays to L2 and remote traffic
   or rank waits to L5. Form a scoped recommendation and send its impact to L6.

## Interpretation rules

Do not classify from a single top-level percentage. Compute and memory aggregates
can hide different subunit limits; an unsaturated bus can coexist with expensive
memory latency. Small grids, masked work and tail waves can explain a low ratio
without an obviously better implementation.

Retain actual NCU rule messages and decide which apply, which are contradicted,
and which need more evidence. Rule-estimated speedups are kernel heuristics, not
measured E2E gains. Multiple rules may describe the same limitation.

## Output and exit

For each important population report measured time, current path, reference and
comparability, limiting mechanism, supporting/counter evidence, efficiency
definition, structural work state, unresolved cases and proposed experiment.
Attach the `kernel_audit` from [evidence.md](../evidence.md).

Close a shape as near-reference only after matching semantics and treatment and
explaining relevant overhead. Do not certify its family from one sample. If
counters are unavailable, retain timeline/source findings and say precisely which
hardware explanations remain unresolved. A low-cost omission needs observed cost,
not a presumption that the unprofiled phase is cheap.
