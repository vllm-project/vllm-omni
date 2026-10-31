# Roofline, MFU and matching references

## Define the quantities first

For a semantic region with useful FLOPs F, elapsed device time t, and bytes Q at a
specified memory level:

```text
useful throughput = F / t
arithmetic intensity at that level = F / Q
resource performance envelope = min(P, B * F / Q)
conditional resource time = max(F / P, Q / B)
```

Specify dense/sparse capability, precision/accumulation, clock convention, GPU
count, scope and byte tier. With empirical P/B and current F/Q these are
conditional models, not universal bounds. An effective bandwidth computed from
logical bytes is not a hardware bandwidth counter.

For a dense GEMM use useful `2*M*N*K` (times the number of batch instances) under
the stated multiply-add convention. Model MFU uses the defined useful model work
over the full chosen wall-time window and matching aggregate capacity; it is not
the average of per-kernel percentages. Include CFG/repetition and actual token or
frame shapes in that work model. Uncounted operators make the MFU estimate partial.

Tensor-pipe active percentage, executed hardware FLOPs and useful model FLOPs are
different quantities. Padding and redundant work can inflate executed utilization.
Attention softmax/reductions and VAE layout/conv behavior require their own models.

## Construct an empirical reference

1. Select a reference that implements the same semantics, including masks,
   accumulation, epilogue and layout conversions.
2. Match exact shapes, strides, dtype, device, graph mode and residency/cache policy.
   State differences that cannot be matched; limited comparability stays limited.
3. Benchmark with bounded repeated, warmed, unprofiled measurements. Preserve
   command/version/raw samples, not only the best result. If a needed comparator
   is unavailable, mark the reference unknown; do not install it without authority.
4. Compare complete operator/region cost, not just a core kernel with wrappers omitted.
5. Calibrate large-GEMM or streaming-memory references separately when useful for
   a resource envelope. Their peak sustained rates need not be achievable at the
   model's small or irregular shapes.

For current time tc and reference tr, `tr/tc` is relative latency efficiency
and `1-tr/tc` is the local reduction to that reference. Neither is the hardware
utilization. If the current implementation beats the reference, revisit the
reference/treatment; do not clamp the result and claim saturation.

## Interpretation branches

- Far below a matched reference: inspect dispatch, layout, tile/resource use and
  the relevant NCU mechanism before proposing a replacement.
- Near a matched reference: current execution has low relative leverage; still
  audit redundant work and region boundaries in L3.
- Near a bandwidth envelope: inspect whether Q can be reduced. A saturated bus
  does not prove the present traffic is unavoidable.
- Below a simple compute/HBM roofline: inspect other tiers, special functions,
  issue limits, dependencies, insufficient parallelism and wave quantization.
  The two-resource roofline may be missing the actual constraint.

## Hard-bound handoff

Read [L6 bounds](../l6-assessment/bounds-and-impact.md) before making any claim
about optimal latency. A valid bound needs unavoidable F_min/Q_min and genuine
capacity upper limits across the allowed implementation set. Current traffic,
best measured reference, or a published utilization number is not such a proof.

[Roofline source](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
motivates the resource view, not a universal MFU completion threshold.
