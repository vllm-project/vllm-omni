# L3: executed work and structural opportunities

## Inputs

Use the selected source/config, phase map, shape/count inventory and actual
eager/compiled/graph path. Source inspection alone can establish hypotheses, not
runtime frequency or latency. For composition details read
[fusion and layout](fusion-and-layout.md).

## Procedure

1. Follow the request through its actual dispatch branch. Map source operations
   to runtime calls/kernels; retain multi-kernel ops and fused multi-op kernels.
2. Check expected versus observed repetitions across layers, timesteps, CFG
   branches, request batches and initial/steady/final regimes. Explain discrepancies.
3. Enumerate values that might be invariant. For each record dependence on input,
   timestep, latent, branch, dtype/device and mutable request state.
4. Audit exact-cache keys, hits/misses and invalidation. Cached prompt/condition
   projections do not imply latent-dependent attention results can be reused.
5. Trace materialization, casts, packing and layout transitions with tensor
   lifetimes. Distinguish views from actual storage movement.
6. Examine producer-consumer regions for legal fusion and already-present compiler
   or backend fusion. Include the replacement's work and resource cost.
7. Review peak live memory and whether avoiding a temporary changes offload,
   tiling or feasible batching. Link those consequences to L5 and the L0 objective.

## Detection / interpretation / recommendation

| Finding | Distinguishing test | Conclusion allowed |
| --- | --- | --- |
| Same function called every step | Values and dependencies identical, mutation/invalidation absent | Exact hoisting/reuse candidate; repeated name alone is insufficient |
| Copies between consecutive ops | Actual allocation/bytes, strides and downstream consumers | Layout/materialization candidate, not proof a view is available |
| Python-visible norm/activation chain | Compiled graph and actual launch count | No fusion opportunity if already eliminated; otherwise analyze the full region |
| Large temporary | Live range, consumers, capacity effects and alternative storage | Candidate for lifetime/storage change with explicit correctness obligations |
| Near-reference GEMM called unnecessarily | Structural proof plus measured population | Execution can be mature while work volume is still actionable |

## Output

Produce a work ledger: source owner, semantic reason, calls, useful work/bytes,
observed execution and candidate region. Each candidate states validity conditions,
existing implementation, expected eliminated work, replacement cost, risks and a
discriminating test. Do not quantify savings solely by deleting useful operations.

Structural review closes a declared search, not the space of all algorithms.
Approximate caching, fewer steps, changed precision or model substitution require
a separate permitted regime and quality evaluation; they are not exact reuse.
