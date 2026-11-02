# L0: measurement validity

## When to use and inputs

Always enter here, including for a partial L3/L4 investigation. Gather the user
contract, resolved runtime configuration and available artifacts before requesting
hardware. Read [provenance and baseline](provenance-and-baseline.md) for collection.

## Procedure

1. Define the objective and completion event: warmed offline generation, final
   client receipt, first media packet, loaded-service SLO, throughput, or memory.
   Do not substitute one for another because a convenient timer exists.
2. Freeze model/weight revision, task, resolution/frames or token lengths,
   steps, CFG, batch/concurrency, input distribution and cache treatment.
   Preserve the user's quality restrictions and identify separately approved
   changes. Tiny smoke configurations do not stand in for production workloads.
3. Establish actual worker source/dependency identity, not just checkout HEAD.
   Include overrides and instrumentation in the manifest.
4. Collect repeated unprofiled baseline cohorts after the chosen warmup condition.
   Record failure counts, samples, hardware health and variation before profiling.
5. Collect a bounded diagnostic timeline; compare its timing/behavior to baseline.
   Obtain shape/stack and NCU evidence in separate targeted runs when needed.
6. Freeze the evaluator and policies independently from model-candidate changes.
   For fresh-only campaigns, check raw lineage, not artifact modification dates.

## Detection and decision branches

| Observation | Check next | Interpretation / proposed action |
| --- | --- | --- |
| Remote HEAD matches, import path differs | Worker interpreter, editable install, container mounts, loaded backend | Wrong execution identity; do not attribute results to the checkout |
| First request is much slower | Loading, compilation, graph capture, allocator/cache priming | Separate cold and warm cohorts; neither is intrinsically the correct objective |
| Later cohort drifts | Clocks, temperature, power cap, other processes, CPU contention, cache regime | Re-establish comparability before small-gain claims |
| Short Python timer, long device tail | Actual completion event and asynchronous return | Timer measured submission, not completion |
| Profile is slower or structurally different | Added synchronization, stacks/shapes, replay and graph mode | Diagnostic evidence only; disclose perturbation |
| Numerical differences exist | Allowed tolerance and modality-quality contract | Not automatically failure or acceptance; quality gate remains independent |

## Output and exit criteria

Deliver the scope manifest, raw baseline samples/statistics, measurement boundary,
provenance verdict, quality status and diagnostic limitations. Use artifact IDs
from [the interface](../evidence.md), with source and runtime artifacts distinct.

If identity or measurement is invalid, report P0 collection repair, not a model
optimization verdict. Source-only work may continue with explicit unknowns.
Stable data permits diagnosis; it does not establish enough precision to rule out
the user's minimum worthwhile gain. Use L6 to evaluate that additional condition.
