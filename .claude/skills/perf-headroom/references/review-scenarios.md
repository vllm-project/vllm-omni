# Behavioral self-review scenarios

Use after changing the skill. These are synthetic reasoning checks, not measured
model evidence. Run CPU helper tests as well; format/schema validation alone does
not prove that an analyst follows the diagnostic branches.

## Exercises and acceptance

| Supplied observation | Acceptable behavior | Failure to catch |
| --- | --- | --- |
| Concurrent A=10 ms, B=8 ms, join waits for both | Work sum 18, busy/wall 10; B->4 leaves makespan 10 under fixed durations | Claiming 18 ms wall time or 4 ms E2E saving |
| A->7 while B remains 8 | Recompute makespan 8 and identify B as limiting branch | Subtracting A's full 3 ms from the original 10 |
| Three simultaneous 10 ms events | Sum 30, union 10, overlap excess 20; not 20 ms of elapsed overlap | Calling duplicate-counted duration a removable bubble |
| Only graph envelopes, arithmetic ledger closes | Internal gaps UNKNOWN; request node coverage | Declaring zero bubbles or all envelope time busy |
| Long host synchronize waits for required compute | Trace producer/consumer and actual exposure | Equating API duration with removable CPU overhead |
| Efficient GEMM executes a proven invariant repeatedly | Mature execution plus L3 work-volume candidate | Closing all optimization because MFU is high |
| Attention backend configured but dispatch unknown | Verify actual backend, shapes and semantics | Recommending a newer backend from its name alone |
| DRAM utilization low, L2 reuse high | Inspect tier and useful work, compare matched cache treatment | Declaring bandwidth inefficiency from DRAM alone |
| Long NCCL kernel on one rank | Request all-rank readiness/consumer evidence | Diagnosing bad link bandwidth or global saturation |
| Matched isolated kernel is faster, E2E not tested | Reference gap or conditional scenario only | Reporting validated E2E gain |
| Best measured bandwidth used as a physical ceiling | Reclassify as conditional resource reference | Claiming a rigorous lower bound |
| DiT covered, VAE or output tail missing | Keep those populations UNKNOWN and visible in L1/L4/L6 | Marking whole pipeline mature |
| No candidates found, search/bounds incomplete | NEED_EVIDENCE or scoped limited finding | Autonomous RSI stop |
| Fresh report assembled from re-exported historical trace | Preserve historical lineage and reject fresh-only claim | Treating new filename/hash as fresh measurement |

## Review protocol

Read the entrypoint, relevant layer guides and raw synthetic observations. Explain
the route and evidence needed before viewing any previous diagnosis. Check that
recommendations have a real owner, counter-explanation and a bounded validation
proposal, not a generic menu. No production writes or GPU runs are needed.

Check local reference links and schema compatibility. Confirm that deeper pages
are reachable from the layer guides and previous subjects (synchronization,
roofline, exact reuse, fusion, communication, bounds, reporting) remain covered.
Record test commands, failures repaired and remaining untested hardware/tool paths
in the task's self-review report, not in a permanent generated changelog here.
