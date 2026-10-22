# CUDA Graph and compile coverage

Use when repeated small launches, graph breaks, capture overhead or unexplained
graph envelopes affect the selected workload. Compilation and CUDA Graph replay
are related but distinct mechanisms; one flag does not prove either coverage.

## Procedure

1. Identify the actual eager/compiled/captured regions in the runner and backend.
   Record shape keys, graph variants and warmup/capture policy for this workload.
2. Compare expected step/replay counts with observed graph launches and node events.
   Separate initial capture/compilation from warmed replay and recurring misses.
3. Inspect work outside the captured region: host preparation, scheduler updates,
   QKV/layout wrappers, transfers and output handling. A graph may cover only part
   of the expensive path.
4. For breaks/misses, inspect dynamic shapes, addresses, control flow, allocation
   and host-device round trips. Record the exact unsupported operation or guard
   rather than labeling all Python code uncapturable.
5. Test the hypothesis analytically against exposed submission gaps. If host
   already supplies work ahead of the device, graph coverage may have little
   single-request leverage even when CPU cost is visible.

## Collection pitfalls

Graph envelope duration cannot reveal the internal busy union. Require node-level
coverage for internal-gap claims. An eager NCU diagnostic changes scheduling and
possibly kernel generation; match implementations and label its applicability.
Do not switch the actual benchmark to eager merely to obtain easier profiler output.

## Recommendation and acceptance

Suggest expanding capture only for the identified eligible region. Include shape
coverage, recapture cost, persistent-buffer memory, concurrent request safety and
output correctness in the proposed experiment. Compile/fusion/graph candidates
that remove the same launches share one overlap group in L6.

A healthy result is bounded: observed repeated regions are captured as intended,
misses are understood, and exposed submission cost is small at this load. It does
not prove zero internal gaps, high MFU, or absence of L3/L4 opportunities.
