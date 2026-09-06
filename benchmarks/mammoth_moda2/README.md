# MammothModa2 Phase 1 Validation

This directory records the validation status of the MammothModa2 AR-to-DiT
request-end payload change. The change is intentionally scoped to the
MammothModa2 integration. It does not add a D2D connector or alter generic
scheduler, coordinator, transfer-manager, or connector behavior.

## Scope

The optimized path is:

```text
AR hidden states
  -> device-resident request-end accumulator
  -> ar2dit_full_payload() one-time D2H + float32 materialization
  -> existing OmniPayload / transfer manager / SharedMemoryConnector
  -> DiT
```

The existing connector remains D2H -> transport -> H2D. "Device-resident" in
this document only means that per-step AR hidden snapshots are retained on the
producer GPU until the request completes; it is not a D2D transport claim.

## Code Changes

The Phase 1 implementation changes only MammothModa2-specific integration plus
the AR-runner hook that owns request-end hidden accumulation:

- `mammoth_moda2.py`: AR stages opt into request-end payload accumulation.
- `pipeline.py`: stage 0 uses `ar2dit_full_payload`; stage 1 declares the
  existing full-payload input contract.
- `stage_input_processors/mammoth_moda2.py`: adapts request-end hidden states,
  token ids, image metadata, and T2I placeholder ids to the legacy DiT input
  schema. The legacy `ar2dit()` entry point remains available.
- `gpu_ar_model_runner.py`: request-end mode snapshots scheduled hidden slices
  on GPU instead of calling per-step CPU materialization.
- Focused unit/regression tests cover the adapter contract and both deferred and
  non-deferred AR-runner paths.

## Test Environment

| Item | Value |
| --- | --- |
| Hardware | 2 x NVIDIA A100 40 GB |
| Model | `/data/vllm-workspace/models/MammothModa2-Preview` |
| Topology | AR stage on GPU 0; DiT stage on GPU 1 |
| Workload | One 512x512 image, 20 DiT steps, seed 42 |
| AR output | 1,057 tokens (`32 * (32 + 1) + 1`) |
| Connector | Existing `SharedMemoryConnector` |
| Optimized server commit | `469b733a` (applied equivalent of local `226af4cc`) |
| Baseline server commit | `44d3ae10` |

The A100 smoke configuration uses `max_num_seqs: 1`, `max_model_len: 2048`,
`max_num_batched_tokens: 2048`, and stage-0
`gpu_memory_utilization: 0.85`. This is a 512x512 smoke configuration, not a
general 1024x1024 deployment setting.

## Functional Results

| Check | Result |
| --- | --- |
| Focused local tests | `133 passed, 17 warnings` |
| AR stage startup | Passed; 11.72 GiB KV cache available |
| DiT stage startup | Passed |
| End-to-end output | PNG produced successfully |
| Baseline vs optimized output | Exact SHA-256 match for all captured profile PNGs |
| Runtime errors in captured runs | None |

The exact image hash is meaningful here because the same prompt, dimensions,
sampling parameters, and seed were used. It demonstrates that the request-end
payload adaptation preserves the generated result for this workload.

## Profiling Results

### Valid operation-count evidence

PyTorch profiler captured both spawned stage workers without CUPTI conflicts.
For the stage-0 worker:

| Metric | Baseline | Request-end path | Delta |
| --- | ---: | ---: | ---: |
| `aten::to` calls | 15,857 | 14,802 | -1,055 |

The 1,055-call reduction closely matches the 1,057-token AR generation shape:
the optimized path no longer materializes the hidden slice to host during each
decode iteration, retaining the request-end transfer instead.

### Results that are not performance claims

Single profiled wall-clock samples were collected, but profiler overhead and
run-to-run variation make them unsuitable for an end-to-end latency claim. In
the PyTorch-profiler run, total generation time was 39.15 s baseline and 40.02
s optimized. In the Nsight run, it was 32.54 s baseline and 32.85 s optimized.
Neither pair should be interpreted as a regression or speedup.

The prior Nsight Systems D2H/H2D aggregate reports are not comparable. They
captured unequal CUDA process sets:

| Nsight metric | Baseline | Request-end path |
| --- | ---: | ---: |
| `cudaLaunchKernel` calls | 79,319 | 410,185 |
| Device-to-host memcpy calls | 46 | 3,174 |

The 5.17x kernel-count disparity means the reports did not cover equivalent
spawned workers. Its memcpy counts, bytes, and times must not be used as AR
payload-transfer results.

## Reproduction

Run from the optimized checkout on a two-GPU host after sourcing its environment:

```bash
RESULTS_DIR=/data/vllm-workspace/results/mammoth_torch \
PROFILE_BACKEND=torch \
bash benchmarks/mammoth_moda2/compare_phase1_payload.sh
```

This mode checks exact output equality and reports stage-0 `aten::to` calls.
It deliberately does not start Nsight.

To attempt a separate Nsight run:

```bash
RESULTS_DIR=/data/vllm-workspace/results/mammoth_nsys \
PROFILE_BACKEND=nsys \
bash benchmarks/mammoth_moda2/compare_phase1_payload.sh
```

This mode deliberately does not enable the PyTorch profiler. It uses the
existing CUDA-profiler backend to delimit spawned-worker GPU activity and sets
`VLLM_OMNI_MAMMOTH_MODA2_NVTX=1` for a MammothModa2-only request-end D2H range.
It writes `nsys_transfer_attribution.json` only after checking, for each
variant, that the stage-0 AR and stage-1 DiT worker PIDs from the application
log are present in the trace and have CUDA activity on their configured
physical GPUs.

The JSON separates CUDA API (`cudaMemcpy*`, synchronization), memory-operation
(`DtoH`, `HtoD`, `DtoD`) counts/bytes/durations, and the largest individual
transfers by stage. It also reports DtoH copies overlapping the optimized
request-end materialization range. A CUDA trace cannot directly attribute CPU
serialization inside `SharedMemoryConnector`; its stage-1 HtoD data identifies
the receive-side GPU copy, while host-side connector cost needs CPU sampling if
it remains a suspected bottleneck.

By default the script compares the parent of the commit named `Optimize
MammothModa2 request-end AR to DiT payload` against the current checkout. Set
`BASE_COMMIT` explicitly only when comparing a different baseline.

## Remaining Work

Phase 1 implementation and functional correctness are complete for the tested
path. Before publishing a quantitative performance claim, collect a worker-aware
Nsight trace with equivalent baseline/optimized process coverage and attribute:

1. AR hidden D2H count, bytes, and duration.
2. Request-end full-payload D2H count, bytes, and duration.
3. Existing connector H2D count, bytes, and duration.

The script uses an opt-in MammothModa2-only NVTX range around
`ar2dit_full_payload()` and does not modify the generic connector or scheduler
to collect this evidence. Actual D2D
transport (NCCL, UCX, CUDA IPC, or equivalent) is a separate Phase 2 design
question and is out of scope for this change.
