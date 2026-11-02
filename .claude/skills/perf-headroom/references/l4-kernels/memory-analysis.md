# Memory analysis: traffic, hierarchy and latency

## Question and prerequisites

Determine which memory mechanism limits the selected launch/region and whether
the traffic is necessary. Require known shapes/strides, valid NCU treatment and
an L3 producer-consumer view. Low FLOP efficiency does not by itself imply a
memory-bandwidth bottleneck.

## Step 1: predict the useful traffic

Count logical reads/writes for the semantic operation and identify possible reuse.
For a simple out-of-place elementwise operation, inputs plus output give a useful
logical-byte model. For reductions, GEMMs, tiled attention and convolutions,
reuse, padding and cache residency need explicit treatment.

This model is not measured HBM traffic or an unavoidable lower bound. Preserve
three quantities: logical bytes, observed bytes at each available memory tier,
and separately justified minimum bytes if any.

## Step 2: locate the limiting tier or path

Use MemoryWorkloadAnalysis and its available tables/chart, expanding aggregates.
Record exact counters, units, scope and elapsed/active denominator. Example DRAM
read/write byte counters may be available as `dram__bytes_read.sum` and
`dram__bytes_write.sum`; query the installed tool rather than assuming support.

| Observation | Distinguishing evidence | Next branch |
| --- | --- | --- |
| High HBM traffic/throughput | Read/write bytes versus useful model and matched bandwidth reference | Necessary traffic -> limited current bandwidth leverage; excess traffic -> L3/reuse |
| Low HBM, high L2 activity | Cache hits and actual L2 traffic, same cache treatment | L2 traffic/reuse or L2 throughput, not automatically poor HBM utilization |
| Low bus throughput with memory waits | Eligible warps, independent requests, grid size, access pattern | Latency/insufficient concurrency; investigate scheduling and work granularity |
| Large transaction amplification | Access width, alignment, active lanes, strides and ideal sectors for this instruction | Layout/coalescing rather than simply demanding more bandwidth |
| Shared-memory serialization | Bank/access pattern and actual wavefront/conflict evidence | Investigate indexing/layout, with resource tradeoffs |
| Local-memory traffic | Register pressure, generated loads/stores, private arrays and source correlation | Possible spills or explicit local storage; do not equate every local access with spilling |
| Off-device traffic | Peer/system aperture and placement evidence | L5 transport/residency, not local HBM tuning |

## Step 3: follow instruction -> request -> transfer

For a suspicious access, identify the hot source/instruction when symbols allow,
the participating lanes, access width and stride. Compare actual transfer sectors
or wavefronts to the expectation for that access. Do not use one sectors/request
threshold across instruction widths, predication patterns or L1/L2 definitions.

Inspect repeated loads/stores, write amplification, unnecessary casts/packing,
cache thrashing and lifetimes. A high cache hit ratio can coexist with too much
traffic; a low hit ratio can be correct for a streaming operation.

## Step 4: choose a falsifiable recommendation

- Excess intermediate traffic -> L3 fusion/layout region including all consumers.
- Inefficient transaction shape -> producer/consumer layout or vectorization
  investigation, including conversion costs and alignment limitations.
- Memory latency without saturation -> independent work, tiling/prefetch or backend
  investigation; check registers/shared memory and eligible work before proposing.
- Required traffic already near a matched reference -> low current execution
  leverage, while work-volume reduction remains a separate question.

Recommendations are hypotheses, not instructions to modify code in an analysis
task. Compare the whole region; a faster core can lose to extra packing.

## Evidence handoff and pitfalls

Provide useful/observed bytes, tier-specific rates, treatment, limiter evidence,
competing explanations and candidate ownership. Never call logical bytes/time
"HBM utilization". Do not derive an E2E gain from a bandwidth percentage alone.
Replay cache flushing, short kernels and different concurrent work can invalidate
a comparison; return to collection validity before concluding inefficiency.
