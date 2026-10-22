# Fusion, materialization and memory lifetimes

## Build a region before proposing fusion

Draw producers, consumers, tensor shapes/strides/dtypes, devices and live ranges.
For each edge ask whether the value is a view, register/shared-memory intermediate,
cache-resident allocation or actual device-memory traffic. Use trace/counters to
test the source model. Do not infer a DRAM write from a Python tensor variable.

Choose a semantically complete region: for example projection plus epilogue and
packing, or norm/modulation plus the consuming operation. Comparing only a faster
core while excluding a new conversion gives a misleading reference.

## Legality checks

- Is the intermediate consumed elsewhere, aliased or mutated later?
- Are reductions, accumulation precision and operation ordering allowed to change?
- Does broadcasting/gating depend on request, timestep, CFG branch or batch?
- Are stream dependencies and storage lifetime preserved?
- Does the current compiled/backend path already fuse the proposed chain?
- Can every supported target shape use the candidate, or is a fallback required?

Record proof or unresolved questions for each applicable condition. A pattern
match such as `norm -> modulation -> residual` is a search lead, not a universal
fusion recipe.

## Cost model and competing explanations

Estimate eliminated launches, reads/writes and allocations separately. Then account
for replacement arithmetic, packing, additional registers/shared memory, spills,
grid size and loss of overlap. L4 supplies evidence for those resource effects.
An increase in register pressure can make a more fused region slower; fewer kernel
names does not imply less E2E latency.

For layout changes, follow the full producer-to-consumer chain. Check whether a
faster GEMM requires a transpose/copy on entry or exit, whether weights can be
prepacked within the cache contract, and whether other consumers still need the
old layout. Include one-time versus per-request cost explicitly.

For storage reuse, distinguish allocation overhead from data-copy cost and from
capacity pressure. Reusing a buffer does not eliminate a required copy. Sharing
mutable storage across requests or graph replays can be incorrect even if one
isolated request passes.

## Recommendation handoff

Locate the correct owner: model-specific semantics, common layer, compiler or
backend. Propose one bounded region, not a whole-model rewrite. Include a diagram
or edge table only if it clarifies several consumers/dependencies.

Use one candidate ID across L2 launch savings, L3 eliminated traffic, L4 efficiency
and L6 impact. These are not additive gains. Acceptance should measure the whole
region and unprofiled request, numerical quality, adverse shapes, peak memory,
graph/eager compatibility and request concurrency when relevant.
