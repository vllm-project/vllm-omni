# Nsight Compute: targeted collection and validity

## Prerequisites

Start with a fresh, coverage-audited timeline and shape inventory. Identify the
actual worker, GPU/rank, phase, kernel signature, launch occurrence and graph mode.
Resolve installed tool capabilities rather than copying another GPU's metric set.

```bash
ncu --version
ncu --list-sections
ncu --query-metrics
```

Use the installed help to resolve filtering/replay syntax. Begin with a small
representative selection, not an entire multi-step model with every metric.

## Collection procedure

1. Choose the question: isolated shape efficiency, in-model launch behavior, or
   concurrent graph/range behavior. These require different replay treatments.
2. Build the exact command with the approved runner/reservation. Include child
   processes when workers execute elsewhere; filter by verified kernel/NVTX
   identity and bounded occurrence count. Verify matches after collection.
3. Start with available SpeedOfLight, LaunchStats, Occupancy and relevant workload
   sections. Add scheduler/stall, instruction, memory-table or source counters only
   when needed to distinguish a mechanism. Save the installed section definitions
   or version with any metric interpretation that depends on them.
4. Save `.ncu-rep`, machine export, rule text, selected launch details, stdout/stderr,
   filter/selection counts and replay/cache/clock settings. Record failed attempts.
5. Check target shapes/counts and compare duration to the timeline. Explain material
   discrepancies before interpreting ratios. A clean export does not prove validity.

Common command elements, not a universal ready-to-run invocation:

| Element | Use and check |
| --- | --- |
| `--target-processes all` | Include child workers when supported; verify the correct worker was captured |
| `--kernel-name` / NVTX filter | Match an observed signature/range; graph/eager names may differ |
| `--launch-skip` / `--launch-count` | Select warmed occurrences; validate their installed matching semantics |
| `--section` | Request supported sections that answer the current question |
| `--replay-mode` | Kernel, application or range treatment selected for the diagnostic scope |
| `--cache-control` / `--clock-control` | Explicitly record treatment; do not silently change shared-device policy |
| `--import`, `--page`, `--csv` | Export an existing report; an export is not a new measurement |

## Replay and graph decisions

- Kernel replay can save/restore memory and alter normal concurrency/cache state.
  It is useful for an isolated launch, not a natural pipeline schedule.
- Application replay can preserve the application's preparation path across runs,
  but requires reproducible launch matching. A variable path can mix incomparable
  occurrences; inspect match warnings and per-pass stability.
- Range/graph approaches may preserve concurrency relevant to a region. Their
  aggregate metrics are not per-kernel counters. Attribute at the collected scope.
- Switching to eager to make a graph node selectable is a diagnostic variant.
  Record kernel-generation and scheduling differences; never label it graph-native.
- Warm-cache versus flushed-cache and controlled versus variable clocks are
  different treatments. Compare like-for-like and state deployment relevance.

Keep NCU durations diagnostic. Its wall-clock execution can include substantial
replay overhead; it is never the normal request benchmark.

## Troubleshooting branches

| Failure/symptom | Next action |
| --- | --- |
| Counter permission denied | Record exact error; use authorized profiler-only privilege if available, otherwise report UNKNOWN |
| No matching kernels | Check process, range, graph mode, names and occurrence filters against the timeline |
| Unexpected match count | Reconfirm regime/shape and eliminate ambiguous name-only attribution |
| Missing metric/section | Query installed architecture/tool support; do not replace absence with zero |
| Contradictory or implausible counters | Reduce collection to relevant metrics; inspect replay determinism, sample size and treatment |
| Timeout/hang or enormous replay cost | Bound selection further; inspect dependencies/replay compatibility before another limited attempt |
| Duration differs from Nsys | Inspect clock/cache/replay/concurrency and timing definition, not just arithmetic |

Do not change driver-wide profiling policy, clocks, kill another process or install
tools without the required authority. Sudo, when authorized, is for the task's
profiler command; preserve scheduling/device restrictions and never log passwords.

## Reference

[Official profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
defines replay, metrics and perturbations. Read the installed-version documentation
when behavior differs. Record unavailable source correlation rather than inventing
source-line explanations for opaque library kernels.
