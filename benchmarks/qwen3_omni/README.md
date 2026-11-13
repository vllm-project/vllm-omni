# Qwen3-Omni Mooncake M1 validation

Measured 2026-11-12 at source `83b8c3102`. Mooncake retains full-prefix hits after the local KV working set overflows; local prefix caching is faster when its working set fits. These are bounded turn-serving measurements, not production tail-latency estimates.

The initial performance and quality measurements precede the change that submits a successor to the save executor only after its predecessor finishes. The Store fault test and cold concurrency-2 benchmark were rerun at `a67b7a721` on 2026-11-13 and are reported separately below; the earlier measurements remain evidence for their recorded revision.

## Configuration and workload

- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`, checkpoint `26291f793822fb6be9555850f06dfe95f2d7e695`, loaded from a local copy.
- Runtime: Python 3.12.15, vLLM 0.33.0 (`4ba787ec9959e566d3b587a796dff07673324da5`), PyTorch 2.14.0, Transformers 5.19.0, Mooncake CUDA 13 package 0.3.12.post1; NVIDIA driver 590.48.01.
- Hardware: two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, 97,887 MiB each. `qwen3_omni_moe_mrv2.yaml`: TP=1 Thinker on GPU 0; Talker and Code2Wav on GPU 1; memory-utilization settings 0.85/0.30/0.20. Normal asynchronous scheduling and CUDA graphs remain enabled.
- Identical Thinker settings: 128 MiB KV budget, measured capacity 1,360 tokens, block size 16, maximum model length 1,024, two sequences, chunked prefill budget 256. Store: single-host TCP, embedded 8 GiB segment, 256 MiB client buffer, no disk offload, fresh namespace per Mooncake process.
- A = cache off; B = local prefix cache with its existing stage-output restoration; D = Mooncake only. C = combined local/Mooncake caching is unsupported; see the [feature contract](../../docs/design/feature/mooncake_kv_hs_cache.md).
- Two fresh processes per mode, in forward A/B/D then reverse D/B/A order. Each process has two untimed text warmups, three repeats at concurrency 1 and 2, cold/full/partial/overflow/revisit cases, serial references, and image/audio cold/full/changed-media cases: 72 requests per process, 432 total.
- All request stages use explicit greedy sampling, seed 42, and maximum 48 tokens; residual Talker codebooks use `do_sample: false`. `VLLM_BATCH_INVARIANT=1`. Client timing uses AsyncOmni DELTA streams, excluding HTTP. Output hashing, memory snapshots, and RPC observations run after timed waves; no tensor dumps or save barriers run in the timed path.

## Latency and throughput

Mean ± sample standard deviation, pooled across the two processes. Latencies are milliseconds. Each serial row has six requests; each concurrent row has twelve requests across six waves. Throughput is successful requests per wave-second; RTF is request latency divided by output-audio duration. Concurrent per-request variation includes queueing within each submitted pair. No P99 estimate is made.

### Concurrency 1

| Case | Mode | TTFT ms | First audio ms | E2E ms | RTF | Requests/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cold | A | 167.9 ± 0.5 | 200.1 ± 1.9 | 735.4 ± 0.7 | 0.1927 ± 0.0002 | 1.360 ± 0.001 |
| cold | B | 168.5 ± 0.6 | 200.9 ± 1.9 | 737.1 ± 0.6 | 0.1931 ± 0.0001 | 1.356 ± 0.001 |
| cold | D | 170.2 ± 0.6 | 202.1 ± 1.7 | 737.8 ± 0.7 | 0.1933 ± 0.0002 | 1.355 ± 0.001 |
| full | A | 167.6 ± 0.6 | 199.7 ± 2.7 | 736.3 ± 2.1 | 0.1929 ± 0.0005 | 1.358 ± 0.004 |
| full | B | 36.1 ± 1.1 | 66.7 ± 2.9 | 603.0 ± 0.5 | 0.1580 ± 0.0001 | 1.658 ± 0.001 |
| full | D | 54.6 ± 2.0 | 86.4 ± 3.5 | 622.5 ± 2.9 | 0.1631 ± 0.0007 | 1.606 ± 0.007 |
| partial | A | 169.8 ± 0.9 | 201.5 ± 2.7 | 737.6 ± 1.0 | 0.1932 ± 0.0002 | 1.356 ± 0.002 |
| partial | B | 111.9 ± 0.3 | 142.9 ± 0.9 | 679.6 ± 0.9 | 0.1780 ± 0.0002 | 1.471 ± 0.002 |
| partial | D | 124.4 ± 3.1 | 158.6 ± 2.8 | 691.8 ± 2.9 | 0.1812 ± 0.0008 | 1.445 ± 0.006 |
| after overflow | A | 167.7 ± 0.4 | 200.9 ± 4.7 | 736.8 ± 3.5 | 0.1930 ± 0.0009 | 1.357 ± 0.006 |
| after overflow | B | 168.2 ± 0.3 | 200.8 ± 1.9 | 736.3 ± 0.7 | 0.1929 ± 0.0002 | 1.358 ± 0.001 |
| after overflow | D | 56.7 ± 4.6 | 86.4 ± 2.8 | 622.9 ± 2.5 | 0.1632 ± 0.0006 | 1.605 ± 0.006 |

### Concurrency 2

| Case | Mode | TTFT ms | First audio ms | E2E ms | RTF | Requests/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cold | A | 239.5 ± 53.4 | 269.7 ± 56.8 | 926.3 ± 19.4 | 0.2427 ± 0.0051 | 2.116 ± 0.003 |
| cold | B | 239.5 ± 54.0 | 271.9 ± 55.6 | 927.2 ± 18.9 | 0.2429 ± 0.0049 | 2.115 ± 0.002 |
| cold | D | 255.1 ± 82.2 | 284.1 ± 86.4 | 960.7 ± 26.7 | 0.2517 ± 0.0070 | 2.028 ± 0.001 |
| full | A | 239.4 ± 53.5 | 269.7 ± 55.6 | 927.1 ± 19.1 | 0.2429 ± 0.0050 | 2.115 ± 0.002 |
| full | B | 69.5 ± 2.8 | 86.0 ± 18.8 | 713.8 ± 12.4 | 0.1870 ± 0.0032 | 2.756 ± 0.009 |
| full | D | 83.2 ± 23.4 | 115.0 ± 23.6 | 740.4 ± 20.2 | 0.1940 ± 0.0053 | 2.632 ± 0.006 |
| partial | A | 240.9 ± 53.8 | 272.6 ± 55.9 | 928.4 ± 19.1 | 0.2432 ± 0.0050 | 2.112 ± 0.002 |
| partial | B | 152.4 ± 24.1 | 185.0 ± 25.1 | 819.0 ± 12.5 | 0.2146 ± 0.0033 | 2.406 ± 0.004 |
| partial | D | 181.4 ± 51.3 | 213.9 ± 54.5 | 866.7 ± 18.9 | 0.2271 ± 0.0050 | 2.260 ± 0.004 |
| after overflow | A | 239.5 ± 53.5 | 271.6 ± 54.3 | 926.5 ± 19.6 | 0.2427 ± 0.0051 | 2.116 ± 0.002 |
| after overflow | B | 239.7 ± 53.7 | 270.7 ± 56.4 | 926.7 ± 19.2 | 0.2428 ± 0.0050 | 2.116 ± 0.002 |
| after overflow | D | 82.0 ± 23.7 | 114.1 ± 23.6 | 741.8 ± 17.3 | 0.1943 ± 0.0045 | 2.638 ± 0.009 |

At concurrency 2, Mooncake's cold E2E latency was 3.7% higher and throughput 4.2% lower than cache off. The run did not isolate HS staging, serialization, background saves, or native KV transfer costs, so it does not establish which operation caused this overhead.

### Concurrency-2 recheck after the save-scheduling fix

At `a67b7a721`, four fresh processes ran in A/D/D/A order with the same checkpoint, environment, sampling, cache capacity, two warmups, and three repeats described above, restricted to text inputs at concurrency 2. All 168 timed requests completed. Each cold row below contains 12 requests across six waves; values are mean ± sample standard deviation.

| Mode | Cold TTFT ms | Cold E2E ms | Requests/s |
| --- | ---: | ---: | ---: |
| A: cache off | 237.6 ± 53.8 | 921.6 ± 19.6 | 2.1266 ± 0.0023 |
| D: Mooncake | 253.4 ± 82.8 | 957.5 ± 26.6 | 2.0344 ± 0.0030 |

Mooncake's measured cold overhead was 6.67% in TTFT, 3.90% in E2E latency, and 4.33% lower throughput. The E2E/throughput cost remains around 4%, close to the earlier 3.7%/4.2% measurements; this bounded recheck does not establish zero overhead or a production tail-latency bound. All four processes exited cleanly. Reports and commands are under `results/mooncake_m1_review_20261113/performance/`.

Matched inputs had identical token IDs and audio sample counts across all 168 observations. One input had varying waveform hashes between cache-off runs; no cached output differed from its selected baseline. Universal waveform-hash equality is therefore not established.

## Store fault injection

The primary rerun at `a67b7a721` used the original Python 3.12.15 environment, `--repeats 3 --delay 5 --lookup-timeout 1`, the same model/hardware above, and a real single-host Store. It requested Thinker text only: a 256-token ongoing decode, a cold 48-token writer, and a 48-token healthy request. KV-write trials also submitted a cold 48-token peer after the writer queued a successor. A zero-delay control preceded each delayed trial.

**All 12 trials, containing 42 requests, passed the progress checks.** In all three delayed KV-write trials, the cold peer started its HS save while the writer still had an unfinished save and a queued successor. The table reports mean ± sample standard deviation across three trials, in milliseconds; each cell is **control / five-second delay**. HS-save start is measured from that request's submission; a dash means no save started.

| Delayed operation and request | TTFT ms | E2E ms | First HS-save start ms |
| --- | ---: | ---: | ---: |
| Lookup, healthy request | 242.7 ± 0.5 / 230.3 ± 0.2 | 1156.3 ± 0.5 / 1266.3 ± 6.4 | — / 117.7 ± 0.7 |
| KV write, cached healthy request | 357.7 ± 21.4 / 316.3 ± 2.1 | 1476.2 ± 34.5 / 1417.4 ± 35.5 | — / — |
| KV write, cold peer | 293.9 ± 20.8 / 263.5 ± 3.4 | 1409.5 ± 29.0 / 1364.6 ± 36.8 | 152.1 ± 3.7 / 156.4 ± 3.4 |

All 42 requests had **zero admission drops and zero pending save batches** at the final snapshots. The following nonzero counts distinguish blocks skipped before execution from started batches that failed:

| Delayed trial | Request | Skipped blocks per request | Failed batches per request |
| --- | --- | ---: | ---: |
| KV write, each of three repeats | Writer | 19 | 1 |
| KV write, each of three repeats | Cold peer | 19 | 1 |

All other request counters, including all controls and lookup trials, were zero. Totals across the delayed KV-write trials were **114 skipped blocks / 6 failed batches**. The pinned native connector uses one KV sending thread: the injected five-second write delay holds subsequent KV writes beyond the test's one-second HS readiness deadline. The cold peer can start its HS save but still fail waiting for its KV. Thus the run establishes serving and HS-worker progress, **not successful cache population for unrelated requests under a stalled KV writer**. No per-attempt queue cap was added; this workload measured no admission drops.

Ongoing and healthy requests retained identical token IDs against their controls. Each cold peer computed 550 prompt tokens, restored zero rows, and generated 48 tokens. Its prompt contains a distinct control/delay case prefix, so paired peers are not an output-parity comparison. Healthy requests produced text before each five-second delay ended; ongoing decode progressed during lookup delays, and cached HS reads completed during KV-write delays. The lookup deadline changes the healthy request from a cache hit to recomputation. These small-sample timing differences are not speedup claims. Delays occur at Store API entry and do not reproduce Store-internal transport or mutex stalls.

A supplemental run at the same source in a separate Python 3.12.16 environment also passed all 12 trials. Cold-peer mean HS-save start was 152.8/136.2 ms, TTFT 280.0/248.8 ms, and E2E 1377.2/1362.5 ms (control/delay). KV-write counters were the same; delayed lookup repeats 0 and 1 each additionally had 18 skipped blocks and one failed batch for the healthy request (36/2 total), with zero in repeat 2. Admission drops and pending batches remained zero. These environments are reported separately, not pooled.

For comparison, the earlier `bb18ac89a` run passed 12 trials without a cold peer or save counters. Healthy TTFT/E2E means were 221.1/1152.6 ms (lookup control), 230.9/1269.8 ms (lookup delay), 332.4/1356.5 ms (KV-write control), and 301.7/1328.2 ms (KV-write delay). Those results established serving/read progress only.

## Cache and memory observations

Text full hits restored 544 prefix tokens; partial hits restored 256. Four divergent overflow documents require 2,240 rounded prompt tokens, exceeding the measured 1,360-token local pool. Every local revisit then missed; every Mooncake revisit retained its 544-token hit. Each unchanged image/audio hit reused 560 tokens; changing only the media yielded zero cached tokens and zero Mooncake restored rows.

| Mode | Observed full / partial / miss | Mooncake restored rows | GPU 0 / GPU 1 GiB | Thinker RSS / USS GiB | Store allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0 / 0 / 144 | 0 | 60.893 / 38.789 | 2.940 / 2.216 | 0 |
| B | 34 / 18 / 92 | 0* | 60.895 / 38.789 | 3.020 / 2.281 | 0 |
| D | 52 / 18 / 74 | 32,960 | 60.893 / 38.789 | 5.429 / 4.681 | 2.368 |

Memory values are maximum **observed snapshots**, not peaks. GPU usage covers each physical device; host RSS/USS covers the Thinker process. Store allocation is the master gauge, excluding its reserved 8 GiB capacity. *The restored-row probe measures Mooncake only; zero for B does not mean local stage outputs were omitted.

## Process variability and startup

| Process | Full-hit TTFT mean, C=1 / C=2 ms | Maximum observed TTFT / E2E ms | Engine initialization s | Inference JIT warnings |
| --- | ---: | ---: | ---: | ---: |
| off03 | 168.09 / 238.63 | 292.83 / 947.30 | 47.89 | 2 |
| local03 | 36.09 / 69.28 | 291.90 / 946.25 | 46.76 | 2 |
| mooncake03 | 55.73 / 81.82 | 335.51 / 988.86 | 48.20 | 2 |
| mooncake04 | 53.44 / 84.52 | 336.13 / 987.83 | 47.81 | 2 |
| local04 | 36.13 / 69.78 | 292.62 / 947.20 | 47.67 | 2 |
| off04 | 167.20 / 240.21 | 292.62 / 947.10 | 48.40 | 2 |

Every process reported first-image inference JIT for `matmul_kernel_persistent` and `rotary_kernel`; cold-media latency includes this first-shape setup. Text warmups preceded measurement. Initialization above is excluded from request latency. Maximum observed request latency is not a tail percentile or a bound.

## Correctness and limitations

The exact GPU acceptance test was rerun at `a67b7a721` and passed all nine text/image/audio scenarios: cold/full/partial text hits and unchanged/changed media. Restored embeddings, hidden states, packed Talker inputs, token IDs, and audio matched two independent cache-off engines exactly. This is a separate correctness run; its tensor dumps and save barriers do not contribute timing results. Evidence is under `results/mooncake_m1_review_20261113/acceptance/`.

In the original `83b8c3102` comparison, all 432 requests completed. Matched text token IDs and audio sample counts agree; all eight image/audio changed-media checks pass. Strict waveform-SHA equality is **not established**: the two cache-off processes show five input digests with multiple hashes, and cached arms have four observations differing from the first baseline hash, all on the same concurrent overflow cache miss. Every cached waveform hash occurs among cache-off observations for that input. Strict per-observation hash equality therefore remains qualified by baseline variability.

Separate quality-only runs saved client waveforms after each wave. Two independent cache-off engines supplied 144 requests and 260 same-input pairs. Their maximum absolute difference was `0.001953125`, maximum RMSE `4.9504e-5`, and minimum finite SNR 59.75 dB. Before either cached quality run, tolerances were frozen at twice those absolute/RMSE maxima (`0.00390625`, `9.9008e-5`).

All 144 cached quality requests passed those fixed limits with exact tokens and sample counts. Each mode matched its first baseline waveform bit-for-bit in 69 of 72 cases. Local/Mooncake maximum RMSE was `4.8679e-5` / `4.3211e-5`, minimum finite SNR 59.75 / 61.23 dB, and maximum absolute difference `0.001953125` for both—within the observed baseline maxima. These checks establish numeric waveform agreement; no perceptual-quality claim or timing result is taken from the diagnostic runs.

The audio full-hit workload exposed a pre-existing PIECEWISE capture bug: flattening auxiliary outputs for FULL-graph buffers also changed the PIECEWISE replay payload from a dictionary to a list. The measured source preserves the original auxiliary pytree for PIECEWISE and retains flattening for NONE warmup/FULL. Earlier failed local/Mooncake audio runs are retained as diagnostic artifacts and excluded from these tables.

Exact source-embedding, hidden-state, and Talker-input checks use the separate [acceptance test](../../tests/engine/test_mooncake_turn_consistency_qwen3.py). Fault injection uses [mooncake_faults.py](mooncake_faults.py); the primary raw report is `results/mooncake_m1_review_20261113/fault-local/fault-data/report.json`, the supplemental run is `results/mooncake_m1_review_20261113/fault-data/report.json`, and the earlier run is `results/mooncake_m1_review_20261112/fault-final-data/report.json`. CPU failure coverage and feature constraints are documented in the [feature contract](../../docs/design/feature/mooncake_kv_hs_cache.md).

## Reproduction

Run from the repository root with the pinned environment, local checkpoint, and a running Store configured above. `MOONCAKE_METRICS_URL` names its Prometheus endpoint. Reports, logs, source manifests, per-process statistics, and failed diagnostics are local artifacts under ignored `results/mooncake_m1_review_20261112/` and `results/mooncake_m1_review_20261113/`; no public artifact URL is provided.

```bash
export VLLM_TARGET_DEVICE=cuda VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MOONCAKE_TEST_MODEL=/path/to/pinned/checkpoint
export MOONCAKE_CONFIG_PATH=/path/to/store.json
export MOONCAKE_METRICS_URL=http://127.0.0.1:PORT/metrics
```

The experiments allow 15 seconds for parent process cleanup. This uses a test wrapper around the pinned process manager; it does not rely on a public AsyncOmni shutdown-timeout parameter. The same wrapper applies to all modes and does not alter the timed serving path.

```bash
cat > /tmp/mooncake_benchmark_runner.py <<'PY'
import os, runpy, sys
sys.path.insert(0, os.getcwd())
def main():
    from vllm.v1.engine.utils import CoreEngineProcManager
    original = CoreEngineProcManager.shutdown
    def shutdown(manager, timeout=None):
        return original(manager, 15 if timeout is None else timeout)
    CoreEngineProcManager.shutdown = shutdown
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name="__main__")
if __name__ == "__main__":
    main()
PY
for run in off03 local03 mooncake03 mooncake04 local04 off04; do
  python /tmp/mooncake_benchmark_runner.py benchmarks.qwen3_omni.mooncake_cache_benchmark \
    --mode "${run%??}" --model "$MOONCAKE_TEST_MODEL" \
    --store-config "$MOONCAKE_CONFIG_PATH" --store-metrics-url "$MOONCAKE_METRICS_URL" \
    --output "compare-${run}.json" --repeats 3 --warmup 2 --concurrency 1 2 \
    --overflow-documents 4 --kv-mib 128 --media image audio
done
python -m benchmarks.qwen3_omni.mooncake_cache_benchmark \
  --compare compare-off03.json compare-off04.json compare-local03.json \
  compare-local04.json compare-mooncake03.json compare-mooncake04.json \
  --output comparison.json
```

For the `a67b7a721` concurrency-2 recheck, use process order `off01 mooncake01 mooncake02 off02`, set `--concurrency 2`, and omit `--media image audio`.

```bash
python /tmp/mooncake_benchmark_runner.py pytest \
  tests/engine/test_mooncake_turn_consistency_qwen3.py -vv -s --run-level advanced_model
python /tmp/mooncake_benchmark_runner.py benchmarks.qwen3_omni.mooncake_faults \
  --model "$MOONCAKE_TEST_MODEL" --store-config "$MOONCAKE_CONFIG_PATH" \
  --output-dir fault-results --repeats 3 --delay 5 --lookup-timeout 1
```
