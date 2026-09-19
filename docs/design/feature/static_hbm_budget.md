# Static cross-stage HBM budgets

`hbm_limit_gb` is a **total profiled operating envelope per replica, per worker
GPU**, in GiB (2^30 bytes). It is not a KV-only budget and is not a CUDA allocator
hard limit. This upstream checkout previously had no `hbm_limit_gb`; configurations
from experimental forks where it meant KV capacity must migrate to
`kv_cache_memory_bytes` to preserve their old meaning.

## Deployment

Apply these entries to an existing deployment with two AR stages on an 80 GiB GPU:

```yaml
stages:
  - stage_id: 0
    devices: "0"
    hbm_limit_gb: 24
    hbm_reserved_gb: 2
    max_num_seqs: 8
    engine_extras:
      watermark: 0.05
      scheduler_reserve_full_isl: true
  - stage_id: 1
    devices: "0"
    hbm_limit_gb: 48
    hbm_reserved_gb: 4
    max_num_seqs: 4
    engine_extras:
      watermark: 0.05
      scheduler_reserve_full_isl: true
```

The existing physical-device startup ledger is reused. It runs whenever any stage
sets a total budget, even if `parallel_stage_init` is false. Every replica in that
deployment must have an explicit total budget. On GPU 0 the above configuration
claims 24 + 48 + 1 external reserve + 1 device safety margin = 74 GiB. Stage-local
reserves are **inside** the 24/48 GiB totals, not added a second time. Device
reserves retain the existing admission module defaults. Budgets are counted on
each device of a TP replica and for each co-located replica. They are not divided
by the rank count.

Admission uses the launcher's existing physical-device mapping. This initial
implementation rejects unresolved placement, UUID/MIG mappings unsupported by
that resolver, remote replicas, and the Ray/multi-node executors rejected by the
existing startup guard. It does not coordinate independent server deployments.
Capacity accounting does not reserve memory against unrelated processes; each
worker additionally checks its budget against initial free memory.

## Worker calculation

AR workers always profile in total-budget mode, even with an explicit KV byte
budget. If a stage has a 24 GiB total, 14 GiB accounted non-KV peak, and 2 GiB
reserve, it receives at most 8 GiB of KV storage:

```
KV bytes = total bytes - profiled non-KV bytes - stage reserve bytes
KV bytes = min(KV bytes, explicit kv_cache_memory_bytes)  # if configured
```

The existing non-KV accounting is retained (weights + peak activation increase +
non-Torch increase for AR). Device-level profiling can conservatively include
other consumers; it is not claimed to be exact per-process attribution. The
existing initialization locks are retained. Neither the ledger nor a profile
proves that every future allocation fits.

`hbm_reserved_gb` defaults to 2 GiB and covers unprofiled graph pools, transfers,
fragmentation and slack. This is a configurable allowance, **not a measured or
universally sufficient bound**. Validate it with the model's configured batch,
token limits and graph modes, including simultaneous startup and transfer peaks.
Total-budget mode takes precedence over `gpu_memory_utilization` for deriving
requested memory. `num_gpu_blocks_override` is rejected because it can defeat
byte-budget sizing. Nonpositive derived KV capacity fails startup. Native KV
initialization remains responsible for model/block-layout capacity validation.

## Scheduler

No new AR queue gate or allocation algorithm is introduced. Native
`Scheduler.schedule()` and `KVCacheManager.allocate_slots()` retain their
watermark, full-input checks, prefix reuse, reservations and preemption behavior.
A free KV block remains allocated GPU storage; pausing admissions does not lend
that storage to another stage. Static mode does not resize pools or implement
runtime budget borrowing.

## Diffusion support

Total budgets are accepted only for `diffusion_kv_mode: paged_scheduler`.
The existing maximum-shape profile and request-shape validation are reused; the
worker deducts `non_kv_cache_memory` and the stage reserve before deriving KV
capacity. An explicit KV override cannot bypass this calculation. Existing
scheduler execution-count limits remain active.

Dense diffusion and non-AR generation workers are rejected in this version:
they do not share the supported end-to-end profiled KV capacity contract. A
model-specific profile/envelope integration is required before enabling total
budgets for them. This does not claim generic AR + arbitrary DiT protection.

## Verification

CPU-only tests (no CUDA/vLLM installation needed):

```sh
python3 tests/engine/test_static_hbm_budget.py
```

These execute the production budget arithmetic and device ledger with injected
logging helpers. Worker tests execute the production profiling method with a fake
runner and memory profile. They do not validate vLLM object construction, model
execution, graph capture, or physical allocations.

Before production use, run full configuration projection tests with the pinned
vLLM version, AR and supported paged diffusion GPU integration tests, and
co-located startup/serving pressure tests. Compare equivalent native KV settings
for scheduling parity; inspect peak device/process memory, completion rate,
preemption, throughput and tail latency. Test input shapes outside the diffusion
profile envelope and verify rejection. Deployments without `hbm_limit_gb` retain
the previous profiling and parallel-init admission paths.
