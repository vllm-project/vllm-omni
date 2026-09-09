# MRV2 comparison after rebasing onto main

Measured source: `07da1fed7ded6ebbf9d1892ed98b28a5db98df48`, based on main
`2f1845fa4e74d78609a2c301d3b843628cbf6fb8`. Subsequent publish preparation
adds sign-offs, preserves published ancestry, and applies formatting/SPDX changes;
it does not change the measured Python ASTs.

Qwen3-TTS-12Hz-1.7B-Base, Base voice_clone, one H200 shared by both stages.
Full SeedTTS English set (1,088 rows, 666 distinct references), unchanged reference
inputs, no request seed, no retries, maximum generation length 1,024.
Stage0 max_num_seqs=64, Stage1=10, decoder B4. MRV2 generation completion
events enabled; async index staging disabled. Each runner: 8 c1 requests,
three complete c64 warmup passes, then two complete c64 measured passes
(2,176 measured, 5,448 total). The reference cache remains at its existing
capacity; warmup does not imply all references are cached.

Each throughput denominator includes the full measured wall time. Audio-s/s uses
successfully decoded audio duration. TTFA excludes a standalone WAV header.
P95 is computed from the combined request samples, not averaged across phases.
Change is `(V2 / V1 - 1) * 100`; negative is better for latency/gaps.

| Metric | V1 | V2 | Change |
|---|---:|---:|---:|
| Audio throughput (audio-s/s) | 64.22 | 89.23 | +38.9% |
| TTFA mean (ms) | 447.2 | 447.1 | -0.0% |
| TTFA P95 (ms) | 1098.4 | 1040.5 | -5.3% |
| E2E mean (s) | 4.082 | 2.945 | -27.8% |
| Simulated playback gap (s/request) | 1.703 | 1.104 | -35.2% |
| Failed / total requests, including warmup | 0 / 5,448 | 0 / 5,448 | N/A |

The old-base warmup comparison was 63.08 / 86.66 audio-s/s (V1 / V2).
The throughput advantage is therefore similar after rebase. However, old-base
TTFA P95 was 1,011 / 923 ms, lower than both new-base measurements. This one
ordered comparison cannot attribute that change to a particular main commit.
Do not interpret it as proof that latency is unchanged or EOS caps are fixed.

All measured and warmup ownership samples detected no foreign process on the
selected GPU. Shared-host CPU load was not controlled. This is a single ordered
run without randomized repetitions or confidence intervals. The simulated gap
starts playback immediately at first audio with no startup buffer; it is not an
actual player underrun or subjective audio quality measurement.

Validation: 798 CPU tests passed, one missing-HF-config case skipped, one GPU
case deselected; five targeted TTS seed/EOS tests passed. Source migration retains
main's TTS adapters and connector module split, while keeping explicit decode
span accumulation in MRV2's data plane. Both servers were cleaned up.

The first launch on main rejected the old served-model-name alias. The successful
pair omitted the alias and used the identical local weight path as the API model
name for both runners. Deployment YAMLs otherwise match the old-base experiment.

See `tts_rebase_comparison.json` for phase summaries, counts and ownership audits.
Full per-request measurements and raw logs are retained in the local experiment
workspace; they are not bundled here. Qwen3-Omni is a separate workload and its
results must not be inferred from this TTS table.
