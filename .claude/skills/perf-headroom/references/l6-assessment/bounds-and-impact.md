# References, lower bounds and E2E impact

## Three reference strengths

| Type | How to obtain | Claim allowed |
| --- | --- | --- |
| Empirical matched reference | Same semantics/shape/treatment, repeated measured implementation | Relative gap to that reference, not remaining gap to the optimum |
| Conditional resource/schedule model | Current work/traffic and calibrated resources/dependencies | Prediction under named assumptions, not a universal lower bound |
| Valid lower bound | Unavoidable work/dependencies and valid capacity upper bounds across the allowed set | Bound on best possible time within that explicitly justified set |

Define the allowed implementation set first: hardware, workload, quality and
permitted algorithm/precision/storage changes. A bound for a fixed kernel graph
does not apply unchanged if graph fusion or exact work elimination is permitted.

## Constructing a resource bound

For a fixed semantic scope, justify F_min as unavoidable operations and Q_min as
unavoidable traffic through a specific memory level. Define initial residency and
final output location; compulsory HBM bytes differ when data starts in cache or
can remain on device. Use actual mathematical/algorithmic arguments for minima,
not counts copied from the current trace.

With valid capacity upper bounds P_max and B_max:

```text
T >= L_resource = max(F_min / P_max, Q_min / B_max)
```

Compute and memory may overlap, so the two are not added. Include further
dependency, communication-cut or other resource bounds only with their own
justification. Combine independent lower bounds by maximum; summing region bounds
requires those regions to be necessarily serial for every allowed implementation,
not merely serialized by today's schedule.

Empirical sustained rates create useful conditional models but are not hard
capacity ceilings. Current FLOPs/bytes can include padding, repeated work and
intermediates that another implementation removes. In those cases the rigorous
bound is unknown unless a separate proof is supplied.

## A genuine remaining-space interval

Let T0 be current time, T* the optimum over the declared allowed set, L a valid
lower bound and U a feasible, quality-accepted end-to-end implementation time:

```text
L <= T* <= U <= T0
1 - U/T0 <= possible latency-reduction fraction <= 1 - L/T0
```

Use the same objective and account for statistical uncertainty; point measurements
are not exact deterministic truths. If no faster E2E implementation has been
validated, U can be T0 and the guaranteed improvement endpoint is zero. If L is
not justified, the useful certified upper endpoint is unknown. A best isolated
kernel time cannot substitute for U for the complete pipeline.

The current assessment helper does not verify mathematical proofs or produce this
certified interval. Keep a derivation/review artifact and describe it separately;
the helper's null global-bound field must not be overwritten by a reference gap.

## Local-to-E2E translation

For a serial exclusive fraction f and a local speedup s, with fixed remaining
work and no integration overhead:

```text
T_new = T * (1 - f + f/s)
latency reduction = 1 - T_new/T
speedup ratio = T/T_new
```

A synthetic 10 s request with 2.4 s serial attention reduced to 1.5 s gives 9.1 s:
37.5% local reduction, 9% E2E reduction and about 1.10x E2E speedup. If the
attention time overlaps another limiting branch, that subtraction is invalid.
Use [timing and overlap](../l1-pipeline/timing-and-overlap.md) to check the scope.

For concurrency, create an explicit dependency/resource-order graph and compare
transformed schedules. Check the original prediction against the captured window
and, where possible, a held-out measured variant before trusting extrapolation.
The bundled `replay` only computes a fixed-order DAG makespan; it does not discover
dependencies or predict new contention, cache effects, batching or queue dynamics.

## Interaction review

Assign candidates affecting the same intervals/traffic to an overlap group.
Consider combined transformations explicitly or report their combined gain unknown:

- Fusion, epilogue and graph replay may remove the same launch/temporary.
- Faster kernels can expose host preparation or communication previously hidden.
- Parallelism changes local shapes, memory pressure and collective volume.
- Removing offload can violate memory capacity; more batching changes latency.
- A hidden component may still consume shared resources; savings in contention
  require a new model/measurement, not zero-cost or full-cost assumptions.

Use `reference_gap`, `scenario`, `model_prediction`, or `validated_gain` on each
quantified card. Prediction intervals from model assumptions are not certified
optimum intervals. Unvalidated impact can be negative; do not suppress plausible
regressions or claim that the recommendation is guaranteed to help.

## Scoped stopping

A defensible engineering stop proposal states the searched candidates, remaining
unknowns, interactions, precision and residual leverage relative to a user-approved
threshold. A mathematical near-optimality claim requires a sufficiently tight
valid bound instead. Neither high MFU nor absence of candidates replaces these
arguments. The independent evaluator/controller decides; analysis only advises.
