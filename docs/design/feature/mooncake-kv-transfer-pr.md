# Native Mooncake KV Transfer: PR and Experiment Report

## Executive verdict

The optimized route is faster than the matched legacy dense route on the
primary metrics in both tested transports:

- TCP: throughput `+1.146%`, mean E2E latency `-1.035%`.
- RDMA: throughput `+1.137%`, mean E2E latency `-1.032%` in the five-round
  short test; throughput `+1.478%`, mean E2E latency `-1.443%` over the
  60-request steady-state aggregate.

The result is not that RDMA is measurably faster than TCP end to end. The
two-step workload is model-dominated and the native TCP/RDMA short results are
within `0.04%`. RDMA's demonstrated advantage is instead the transfer path:
the imported KV payload reaches registered GPU pages without a payload H2D,
whereas Mooncake TCP stages received socket data through host memory.

The implementation also satisfies the compatibility and reuse requirements:

- Native vLLM `MooncakeConnector` remains page-granular and feeds paged
  attention directly.
- The pre-existing `OmniKVTransferManager` remains tensor-granular and feeds
  the dense attention/cache-reuse path.
- The two configurations are mutually exclusive; neither path silently falls
  back to the other.
- Connector construction, registration, page grouping, transfer submission,
  completion, and ACK handling are reused from vLLM. There is no Omni
  `MooncakeConnector` subclass or private-method monkeypatch.
- No benchmark hook, profiler hook, temporary path, debug print, or custom
  transport implementation remains in the production diff.

The strict limitation is tail latency: short-run p95/p99 gains are positive
but below `1%`, and the 60-request RDMA p99 gain is `0.979%`. The benchmark is
also not evidence for DiT CUDA-graph behavior because DiT was eager on both
sides.

## Path contract and backward compatibility

The final routing contract is explicit:

```text
kv_transfer_config + diffusion_kv_mode=paged_scheduler
    AR Scheduler pages
      -> vLLM MooncakeConnector (TCP or RDMA)
      -> DiT Scheduler pages
      -> vLLM paged attention

omni_kv_config only
    AR per-layer tensors
      -> OmniKVTransferManager
      -> DiT dense prompt cache/reuse
      -> dense attention
```

Native mode always enters `ImageKVCacheManager._forward_paged`. The received
prefix remains in Scheduler-owned pages and is addressed with vLLM block
tables. It does not call the removed dense page materializer, does not build
the legacy dense prompt cache, and does not expand imported KV heads with
`repeat_kv`.

Legacy mode still enters `ImageKVCacheManager._forward_dense_legacy`. It keeps
the original `_injected_ar_kv`, `_cache_prompt_kv`, `_reuse_prompt_kv`, and
`repeat_kv` behavior. The legacy manager itself was not replaced by a paged
adapter.

The following regression boundaries cover the split:

- `test_native_config_requires_paged_scheduler`
- `test_kv_and_legacy_transfer_configs_are_exclusive`
- `test_native_connector_keeps_imported_prefix_in_paged_attention`
- `test_legacy_tensor_prefix_still_uses_dense_cache_and_reuse`

The end-to-end benchmark also completed every formal legacy TCP and RDMA run,
so the compatibility conclusion is not based on unit routing alone.

## vLLM reuse boundary

The implementation uses `KVConnectorFactory.create_connector` to construct
vLLM's unmodified `MooncakeConnector`. The previous local connector subclass,
dynamic `__getattr__`, private `_build_transfer_params` replacement, and
`kv_connector_module_path` override have been removed.

| Responsibility | Implementation owner |
| --- | --- |
| Connector lifecycle and endpoint setup | vLLM `MooncakeConnector` |
| GPU-memory registration | vLLM `MooncakeConnector` |
| Adjacent page-run discovery | vLLM `group_concurrent_contiguous` |
| Safety check for descriptor coalescing | vLLM `_can_coalesce_block_transfers` |
| Batched transfer submission | vLLM call to Mooncake `batch_transfer_sync_write` |
| Completion and ACK protocol | vLLM connector Scheduler/Worker contracts |
| Page allocation and block tables | vLLM `KVCacheManager` |
| Cache update and attention kernel | vLLM attention backend |
| AR/DiT request translation and logical prefix boundary | vLLM-Omni glue |

The “merged page transfer” is an upstream MooncakeConnector optimization, not
an Omni conversion from pages to per-tensor buffers. vLLM groups a run only
when both source and destination block IDs are consecutive and the per-page
copy layouts are coalescible. It then submits fewer descriptors while retaining
the original block IDs and page lifetime.

One small Omni allocator policy is retained: non-cached AR and DiT pages are
returned to vLLM's `BlockPool` in physical-ID order. vLLM's non-caching pool is
LIFO; its normal reverse free order otherwise makes the next allocation
descending, while the upstream grouping helper detects ascending consecutive
IDs. The Omni helper only controls free-list order through vLLM's
`pop_blocks_for_free` and `free_blocks` APIs. It does not merge memory, alter
page size, or run for caching-enabled pools.

With the current 32-layer HND registration, vLLM exposes one K and one V
region per layer. Consecutive pages therefore reduce to 64 transfer regions,
not one dense tensor per layer and not thousands of one-page descriptors.

The consumer also advertises the producer's complete physical page envelope
while marking only the exact logical reusable prefix as computed. This is a
narrow vLLM 0.28 compatibility rule: a shorter consumer block list is
interpreted by the connector as a partial-hit decode suffix. Extra bytes in
the final advertised page are not reused and are overwritten by DiT prefill.
No connector internals are patched to obtain this behavior.

The local upstream reference checkout used for this audit is
`/root/rsync/prs/vllm`. The relevant implementation is in
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`.

## Removed overhead

### AR side

- Native mode transfers directly from registered Scheduler KV pages.
- It no longer gathers selected AR blocks into a per-layer tensor payload.
- The transfer boundary excludes optimistic async output placeholders before
  vLLM derives the physical block list.
- Recycled non-cached pages are ordered so vLLM's existing contiguous-run
  grouping remains effective.

### DiT side

- The dense paged-prefix materializer and dense native fallback were removed.
- Imported pages are consumed directly by paged FlashAttention.
- K/V is written to the paged cache once; piecewise attention then reads the
  cache instead of repacking the same projected K/V for every segment.
- Homogeneous CFG rows use strided segment slices instead of indexed row
  gathers and scatter updates.
- The full projected Q tensor is not made contiguous before those segment
  slices; each segment packs only what its native call consumes.
- FA2 uses its existing `max_num_splits` metadata field with value `1` for
  these small piecewise calls, avoiding unnecessary SplitKV partial outputs
  and combine kernels.

### Control plane

The two object-collective sites use the world coordinator's existing Gloo CPU
group. The default NCCL object-collective path serializes Python objects
through temporary CUDA tensors and introduces unrelated H2D/DtoH copies.
This change reuses vLLM's CPU control group; it does not add a new process
group.

## H2D findings

The final retained RDMA trace covers one DiT rank, two unique 1024x1024
requests, and two denoising steps:

| Metric | Native RDMA trace |
| --- | ---: |
| H2D calls | 60 |
| H2D bytes | 855,592 |
| H2D time | 0.055104 ms |
| Imported KV payload H2D | 0 |
| DtoD calls / bytes / time | 6 / 5,574,688 / 0.021472 ms |
| DtoH calls / bytes / time | 18 / 12,583,008 / 0.234272 ms |

The remaining small H2D copies are model inputs and attention/cache metadata,
not imported KV payload. Three traced `all_gather_object` calls have zero H2D
scoped beneath them after using the Gloo group.

TCP cannot make the same zero-copy claim. Mooncake 0.3.12 TCP receives into a
host buffer and then copies into registered CUDA memory. A trace that retained
the asynchronous callback cycle recorded:

| Metric | Native TCP callback trace |
| --- | ---: |
| Payload H2D copies | 64 x 2,785,280 bytes |
| Payload H2D bytes | 178,257,920 |
| Total H2D bytes | 179,087,868 |
| Total H2D time | 4.7571 ms |

A separate final TCP model/control window recorded only 60 small H2D calls,
850,206 bytes, and 0.051329 ms, but that profiler schedule did not retain the
background TCP callback. It must not be interpreted as TCP zero-copy.

Therefore the precise conclusion is:

- Extra Omni dense materialization/gather H2D is eliminated in native mode.
- RDMA has zero imported-payload H2D in the retained trace.
- TCP still has transport-intrinsic host staging inside Mooncake; removing it
  requires RDMA or a Mooncake transport change, not another Omni page adapter.

## Profiler route evidence

The final native RDMA route trace contains:

| Function/event | Calls |
| --- | ---: |
| `ImageKVCacheManager._forward_paged` | 128 |
| paged backend `forward_paged` | 128 |
| native `do_kv_cache_update` | 128 |
| `ImageKVCacheManager._forward_dense_legacy` | 0 |
| dense page materialization | 0 |
| `_cache_prompt_kv` / `_reuse_prompt_kv` | 0 / 0 |
| `_build_neg_ar_kv` / `repeat_kv` | 0 / 0 |
| `wait_for_kv_load` | 2 |

The separate legacy trace contains zero paged forwards and zero native cache
updates, while recording dense forwards, prompt-cache/reuse, negative-AR-KV
construction, and `repeat_kv`. That trace used a different profiler workload,
so its call counts are used only as route-exclusivity evidence and are not
compared numerically with native timing.

## Benchmark methodology

| Item | Setting |
| --- | --- |
| Host | One node |
| GPUs | 4 x NVIDIA RTX PRO 6000 Blackwell Server Edition |
| AR | One replica, TP=2, devices 0-1 |
| DiT | One replica, TP=2, devices 2-3, EP enabled |
| Model | HunyuanImage-3.0, bfloat16 |
| Image | 1024x1024 |
| Denoising steps | 2 |
| AR output cap | 100 tokens |
| Client concurrency | 2 |
| Mooncake | 0.3.12.post1 |
| Short repetition | 5 independent rounds x 4 requests per path |
| Steady-state repetition | RDMA only, 3 rounds x 20 requests per path |
| Warmup | Excluded |

Every native/legacy pair used the same prompts, seeds, model, attention
backend, topology, and transport. The prompt dataset hash was
`8685c2b47f64a57ff1457d7f208fe30045ef6e2bcc12847e27edb87481ca362a`;
the source dataset hash was
`d35a58376672f3c65b2f50486c255037d1353a006e7efb6b99b0bb2a87f4b51c`.

The legacy DiT dense route was explicitly bound to installed vLLM FA2 with
`MOONCAKE_BENCH_VLLM_FA2=1`, matching the native vLLM FA2 backend. That avoids
comparing native paged FA2 against a weaker unrelated dense backend.

### CUDA graph fairness

The baseline was not accidentally deprived of CUDA graphs:

| Stage | Legacy | Native | Runtime evidence |
| --- | --- | --- | --- |
| AR | `enforce_eager=false`, `FULL_DECODE_ONLY`, capture size 1 | Same | Both logs contain CUDA graph capture start and completion |
| DiT | `enforce_eager=true` | Same | Eager on both paths |

The comparison is fair for “AR decode graph enabled, DiT eager.” It does not
establish behavior with DiT CUDA graphs enabled. The repository's original
Hunyuan deployment also configures both AR and DiT as eager; the benchmark
changed AR symmetrically on both sides to test the stronger baseline.

### Source matching and transport comparison caveat

Within each transport, native and legacy used an identical production diff:

| Pair | Production diff SHA-256 |
| --- | --- |
| RDMA native vs legacy | `e5d23b018ec07e0ac5cc31460dfcc8d80a04839e540c4941d891b03b431056d9` |
| TCP native vs legacy | `41feb862fc70ea8b70dc2446812c33f57781f9c24d42b96c4e7f50d8111c93f9` |

The current production diff is the TCP hash. Relative to the RDMA benchmark
hash, it additionally routes the second rank-status object collective through
the same existing Gloo group. It changes neither connector nor data path and
applies to both native and legacy, but the exact final hash was not rerun for
RDMA. Per instruction, no replacement benchmark was added. Consequently the
report makes native-vs-legacy claims within each matched pair and does not use
the short rows as a strict TCP-vs-RDMA transport A/B.

## TCP results

Values are arithmetic means over five independent four-request rounds.

| Path | Batch duration | Mean E2E | Median E2E | p95 E2E | p99 E2E | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy dense | 8.337102 s | 3.765501 s | 3.370011 s | 4.782041 s | 4.973673 s | 0.479783 req/s |
| Native paged | 8.242605 s | 3.726540 s | 3.332225 s | 4.738717 s | 4.929830 s | 0.485284 req/s |
| Native improvement | 1.133% | 1.035% | 1.121% | 0.906% | 0.881% | 1.146% |

TCP passes the `1%` target for batch duration, mean E2E, median E2E, and
throughput. It does not pass `1%` for p95 or p99.

TCP used Mooncake's connection pool and a 4 MiB TCP slice on both sides.

## RDMA results

### Five-round short test

| Path | Batch duration | Mean E2E | Median E2E | p95 E2E | p99 E2E | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy dense | 8.339363 s | 3.766672 s | 3.373347 s | 4.782215 s | 4.973224 s | 0.479653 req/s |
| Native paged | 8.245588 s | 3.727803 s | 3.333252 s | 4.740390 s | 4.931542 s | 0.485108 req/s |
| Native improvement | 1.124% | 1.032% | 1.189% | 0.875% | 0.838% | 1.137% |

### Three-round, 60-request aggregate

| Path | Batch duration | Mean E2E | Median E2E | p95 E2E | p99 E2E | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy dense | 35.676724 s | 3.481303 s | 3.439452 s | 3.568046 s | 4.733591 s | 0.560591 req/s |
| Native paged | 35.157068 s | 3.431059 s | 3.390789 s | 3.520102 s | 4.687254 s | 0.568876 req/s |
| Native improvement | 1.457% | 1.443% | 1.415% | 1.344% | 0.979% | 1.478% |

The longer aggregate confirms a stable greater-than-`1%` RDMA gain for batch
duration, mean, median, p95, and throughput. p99 remains narrowly below the
threshold.

## Correctness

- The matched TCP images were pixel-bitwise identical for all four requests.
- The matched RDMA images differed by at most `1/255` per channel. MAE was
  `0.0084-0.0106` in raw `0-255` units and PSNR was `67.9-68.9 dB`.
- Both native traces entered the connector load path; results are not local
  recomputation mislabeled as transfer.
- Source and destination pages remain held until connector completion/ACK.
- The native logical computed prefix is not extended to the physical page
  envelope.

No latent-level or RDMA bitwise-equivalence claim is made.

## Diff and debug-code audit

The audit deliberately separates necessary correctness glue from performance
changes. Correctness code is not removed merely because it does not by itself
increase throughput.

| Retained area | Reason it remains |
| --- | --- |
| Confirmed AR token boundary | Prevents transferring async placeholders that have no valid GPU KV |
| Consumer physical-envelope alignment | Required by vLLM 0.28 Mooncake partial-hit semantics |
| Page-lifetime fences and failure-close behavior | Prevents use-after-free and partial-rank success |
| Mooncake 0.3.12.post1 pin | Fixes callback CUDA-device binding in TP=2 TCP |
| Physical free-list ordering | Enables upstream page-run coalescing without custom merging |
| Direct paged DiT route and dense materializer removal | Removes dense reconstruction and repeated cache work |
| Homogeneous slicing, one K/V update, FA2 single split | Removes observed gather/packing/SplitKV overhead |
| Gloo object collectives | Removes temporary control-plane CUDA copies |
| Empty pending-load fast path | Avoids a redundant connector completion poll |

The following redundant changes were removed from the final working tree:

- The main `mooncake-kv-transfer.md` was restored to `HEAD`; the detailed
  experiment belongs only in this requested PR report.
- A TCP/RDMA configuration-passthrough parametrization was restored to its
  original single test because it only retested vLLM config storage.
- The custom Mooncake connector subclass and private method patch were removed.
- The dense paged-prefix materializer and native-to-dense fallback were removed.
- Rejected benchmark branches for query CPU indexing, async metadata H2D,
  stage-specific NIC affinity, RDMA slice tuning, path round-robin, and custom
  connector worker count are not present in production code.

Current production diff size (`vllm_omni` plus `pyproject.toml`) is 130 lines
added and 156 removed: net `-26` lines. Tests add 313 and remove 7 lines. The
larger test delta is regression coverage and is not imported at runtime.

A scan of production additions found no `MOONCAKE_BENCH` switch, `/tmp` path,
`print`, `breakpoint`, `pdb`, profiler-only hook, `sitecustomize`, monkeypatch,
custom connector class, or connector module-path override. All profiler and
benchmark instrumentation lives under `/tmp/mooncake-perf-qg2SQV`, outside the
repository.

## Rejected performance experiments

The following changes were measured and not retained:

| Experiment | Observed batch duration | Decision |
| --- | ---: | --- |
| Final native RDMA configuration | 8.2456 s | Retain |
| Restrict RDMA NICs | 8.2573 s | Reject |
| RDMA `MC_SLICE_SIZE=4 MiB` | 8.2553 s | Reject |
| RDMA path round-robin | 8.2660 s | Reject |
| Connector workers = 1 | 8.2697 s | Reject |
| Move homogeneous query indexing to CPU | 8.2727 s | Reject |
| Async metadata H2D helper | 8.2625 s | Reject |
| Stage-specific NIC affinity | 8.2609 s vs 8.2707 s same-window default | Reject; only 0.119% and slower than final |

`VLLM_BATCH_INVARIANT` was also rejected because the vision batch exercised a
dtype failure. A 128-token page size was incompatible with the model/cache
contract, and row-wise piecewise execution was slower.

## Validation status

The last completed relevant suite for the identical production code reported:

```text
298 passed, 19 warnings in 7.21s
```

It covered the AR scheduler cleanup, diffusion attention, diffusion KV
connector/manager/adapter, Hunyuan image KV cache manager, and multiprocessing
engine concurrency. Ruff check, Ruff format check, and `git diff --check` also
passed at that point.

Afterward, only the duplicate documentation diff and one unrelated config
passthrough test parametrization were pruned. Per instruction, the suite was
not rerun. The current `git diff --check` remains clean.

## Acceptance matrix

| Requirement | Result |
| --- | --- |
| Reuse vLLM MooncakeConnector rather than fork it | Pass |
| Native route stays page-granular | Pass |
| Legacy route stays per-tensor/dense | Pass |
| No custom page-to-tensor merge | Pass |
| Remove extra Omni dense/gather H2D | Pass |
| RDMA imported-payload H2D | Zero in retained trace |
| TCP imported-payload H2D | Still present inside Mooncake TCP staging |
| Native faster than legacy on primary TCP metrics | Pass, about 1.0-1.15% |
| Native faster than legacy on primary RDMA metrics | Pass, about 1.0-1.48% |
| At least 1% improvement on every tail metric | Not met |
| Fair CUDA-graph baseline | Pass for AR-graph-on/DiT-eager configuration |
| DiT CUDA-graph validation | Not covered |
| Debug/benchmark code absent from production diff | Pass |

## Scope limitations

- The measured topology is single-node AR TP=2 plus DiT TP=2. Multi-node and
  multi-replica throughput are not claimed.
- The workload uses two denoising steps, not a 50-step production-quality
  image run.
- TCP has no 60-request steady-state aggregate because no additional test was
  run after the request to stop testing.
- Exact-final-source RDMA was not rerun after the second object collective was
  moved to the same Gloo control group.
- Native transfer currently requires `async_chunk: false`.

## Local artifacts

Benchmark evidence is stored outside the repository under
`/tmp/mooncake-perf-qg2SQV` on the test host:

- TCP short results:
  `legacy-tcp-312-final-cpugroup-matched-formal-r[1-5].json` and
  `native-tcp-312-final-cpugroup-formal-r[1-5].json`
- RDMA short results:
  `legacy-rdma-312-cpugroup-matched-formal-r[1-5].json` and
  `native-rdma-312-cpugroup-formal-r[1-5].json`
- RDMA steady-state results:
  `legacy-rdma-312-cpugroup-matched-long20-r[1-3].json` and
  `native-rdma-312-cpugroup-long-long20-r[1-3].json`
- Route/H2D evidence:
  `native-rdma-312-final-cpugroup-profile-counts.json`,
  `native-tcp-312-final-cpugroup-unique-{ar,dit}-profile-counts.json`, and
  `legacy-rdma-dense-fa2-profile-counts.json`
- Launch/source manifests:
  `native-rdma-312-cpugroup-launch.json`,
  `legacy-rdma-312-cpugroup-matched-launch.json`,
  `native-tcp-312-final-cpugroup-launch.json`, and
  `legacy-tcp-312-final-cpugroup-matched-launch.json`

Related issue: https://github.com/vllm-project/vllm-omni/issues/5244
