# CosyVoice3 B2 follow-up measurements

The expanded study supports a small local C=1 AR latency reduction, but does
not establish a consistent C=4 or TTFA benefit. Enabling asynchronous output
materialization on every step while retaining hidden-state payloads was
slower in all four ablation rounds. The current token-only inline path avoids
that additional work.

Measurements were collected on 2026-09-15 using the same pinned
vLLM 0.28.0-compatible sources, RTX 4090, and Torch flow backend as the
[initial experiment](../README.md). This is not current-main runtime validation.
The implementation did not change during these experiments. This directory
publishes the report and summaries afterward.

## Repeated A/B comparisons

Each run contains eight measured requests. Natural-EOS runs have two
single-request warmups. The fixed-length control forces
`min_tokens=max_tokens=128` and warms two full batches of four requests.
All timing runs have profiling disabled.

| Scenario | Paired rounds | Baseline AR ms/token | B2 AR ms/token | Mean reduction | Approximate paired 95% interval for B2 minus baseline, ms |
| --- | ---: | ---: | ---: | ---: | --- |
| Natural EOS, C=1 | 5 | 4.8624 | 4.8381 | 0.50% | [-0.0436, -0.0049] |
| Natural EOS, C=4 | 3 | 7.2892 | 7.2333 | 0.77% | [-0.2592, 0.1474] |
| Fixed 128 tokens, C=4 | 3 | 7.5864 | 7.5473 | 0.52% | [-0.4303, 0.3519] |

AR estimates are weighted by token-interval count within each run, then
averaged across paired rounds. The intervals use paired Student-t estimates
with small samples; rounds, not individual tokens, are the statistical units.
They do not establish general performance guarantees.

- C=1 improved in four of five pairs. All 40 paired waveforms, lengths, and
  chunk boundaries matched exactly; every run had 1092 measured AR intervals.
- Natural C=4 workloads differed in every pair, and baseline repeats also
  differed. Its aggregate numbers cannot establish a controlled speedup.
- Fixed C=4 aligned work: 1016 AR intervals per run and 122880 audio samples
  per request. Two of three pairs had slower AR timing, while one improved.
  It did not show a consistent benefit.
- C=4 waveform equality was not established: 1/24 natural and 16/24 fixed
  pairs matched bitwise, with variations also present between baseline runs.
  Fixed-length generation is a workload control, not a speech-quality test.
- C=4 uses offline waves of four requests, not continuously refilled online
  serving. Both languages and the reference audio remain as documented in
  the initial experiment.

| Scenario | Mean of run-median TTFA, baseline → B2 ms | Mean last-audio latency, baseline → B2 ms | Audio throughput, baseline → B2 audio-s/s |
| --- | ---: | ---: | ---: |
| Natural C=1 | 461.72 → 462.28 | 980.68 → 975.49 | 5.562 → 5.582 |
| Natural C=4 | 1474.89 → 1484.95 | 3214.20 → 3207.04 | 7.204 → 7.085 |
| Fixed C=4 | 1476.85 → 1480.82 | 3072.74 → 3117.89 | 6.659 → 6.562 |

TTFA changes were mixed. End-to-end changes at C=1 were small, and neither
C=4 workload demonstrated a consistent throughput improvement.

## Component ablation

All cells keep the same B2 scheduler, token-update contract, and runner guard.
AR async scheduling and async chunk remain enabled. This common-framework
control is distinct from the untouched baseline in the preceding comparison.

| Cell | Async output-materialization flag | Include hidden payload |
| --- | --- | --- |
| `control` | Off | Yes |
| `async_only` | On | Yes |
| `payload_only` | Off | No |
| `b2` | On; empty decode payloads use the inline path | No |

| Paired comparison | Usable rounds | AR change, ms/token | Reduction |
| --- | ---: | ---: | ---: |
| `control` → `async_only` | 4 | +0.38385 | -7.89% |
| `control` → `payload_only` | 4 | -0.02019 | +0.42% |
| `payload_only` → `b2` | 2 | +0.00312 | -0.06% |
| `async_only` → `b2` | 2 | -0.43248 | +8.21% |
| `control` → `b2` | 2 | -0.01535 | +0.32% |

Async-only regressed in all four rounds; its approximate paired interval was
+0.3015 to +0.4662 ms/token. Omitting hidden payloads alone had a small,
variable effect with an interval spanning zero. Adding prefill async
materialization to that path did not show a clear additional gain.

## Resource interference

Two early `b2` ablation runs had anomalous device activity: peak GPU memory
was 36836 and 37578 MiB, versus about 24760 MiB in the other runs. Available
memory at model startup was already lower: 38.63 and 34.75 GiB, versus the
usual 47.12 GiB. One run's AR latency rose to 7.85 ms/token. The source of
the extra occupancy was not conclusively identified.

These observations are retained but excluded from code-performance attribution.
The exclusion follows device activity and startup logs, not the direction of
the latency result. Later rounds waited for an idle GPU before launch and
recorded process memory. B2 returned to 4.82–4.85 ms/token with normal memory
usage. Other workloads were not stopped.

[results.json](results.json) contains both all-observation and filtered
summaries, including usable pair labels. The interference evidence and
selection rule are in [resource_interference.json](resource_interference.json).

## Separate profiling

Three diagnostic captures disable stack, shape, and memory recording.
These inclusive CPU spans are affected by profiling and must not be used as
unprofiled latency estimates or summed into a claimed net speedup.

| Cell | Bookkeeping steps | Async snapshots | Main-thread decode output building, ms | Main-thread decode output wrapping, ms |
| --- | ---: | ---: | ---: | ---: |
| `control` | 132 | 0 | 0.2359 | 0.1152 |
| `async_only` | 132 | 132 | Background work not captured | 0.4280 |
| `b2` | 132 | 1 | 0.1443 | 0.1262 |

B2 takes one prefill snapshot and constructs 131 lightweight outputs inline.
Async-only takes snapshots on all 132 captured steps and has greater wrapping
overhead. Together with the unprofiled ablation, this supports keeping
empty token-only steps out of the background materialization path.
Missing background-thread annotations do not mean the background did no work.
See [profile_summary.json](profile_summary.json) for scope details.

## Reproduction and scope

The study completed 38 unprofiled runs (304 measured requests) and three
profile captures. Two unprofiled runs are resource-flagged as described above.
Use the same environment and source hashes in the [parent manifest](../manifest.json).
The common B2 source can be reconstructed with the [parent patch](../v028.patch).

The expanded harness, configuration patches, full logs, and raw traces are
provided in `b2-followup-evidence.zip`. Extract that archive and follow its
`REPRODUCE.md` to configure source/model/cache paths. The recorded commands,
run sequentially from its experiment directory, are:

```bash
python3 -u experiment.py
python3 -u fixed_length.py --pairs 3
python3 -u ablation.py --rounds 2
python3 -u ablation.py --rounds 4
python3 -u profile_cases.py control async_only b2
python3 analyze.py
```

The later ablation rounds add GPU process monitoring for both compared cells.
Models, caches, and generated waveform arrays are not checked in. Generated
audio is regression-test output, not a ground-truth recording or independent
speech-quality evidence. Attach the follow-up archive to the issue update
for raw data and exact reproduction details.

The results favor minimizing unnecessary output work and retaining the
token-only guard. They do not establish B2 as a solution to the larger AR
host-overhead or concurrency-scaling problem; current-main runtime validation
remains pending.
