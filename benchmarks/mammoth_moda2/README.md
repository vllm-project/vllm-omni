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
  -> ar2dit_full_payload() GPU token-mask selection
  -> selected text/image conditions, one-time BF16 D2H
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
- `stage_input_processors/mammoth_moda2.py`: resolves T2I placeholder ids,
  selects the DiT text/image conditions on the AR GPU, and sends the selected
  tensors with their producer dtype. The legacy `ar2dit()` entry point remains
  available.
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

The results in this section validate the request-end accumulation path. The
current producer-side condition selection and BF16 wire payload were exercised
in the single-request A/B run recorded below.

| Check | Result |
| --- | --- |
| Focused local tests before the producer-side payload follow-up | `133 passed, 17 warnings` |
| AR stage startup | Passed; 11.72 GiB KV cache available |
| DiT stage startup | Passed |
| End-to-end output | PNG produced successfully |
| Baseline vs optimized output | Exact SHA-256 match for all captured profile PNGs |
| Current producer-side helper checks | Python AST parsing and CPU BF16 selection check passed |

The exact image hash is meaningful here because the same prompt, dimensions,
sampling parameters, and seed were used. It demonstrates that the request-end
payload adaptation preserves the generated result for this workload. It does
not replace lifecycle coverage for cancellation or preemption/resume.

### Single-request A/B result

The following reproducible A/B smoke result was collected on 2026-09-06. It
compares the pinned baseline `44d3ae10` with optimized server checkout
`469b733a` using the same model revision, two A100 40GB devices (AR on GPU 0,
DiT on GPU 1), `SharedMemoryConnector`, 512x512 output, 20 DiT steps, seed 42,
one request, eager mode, and disabled prefix caching.

| Check | Baseline | Optimized |
| --- | ---: | ---: |
| Output PNG SHA-256 | `A4F3AD...CA9FAED2` | `A4F3AD...CA9FAED2` |
| Total generation time, one profiled sample | 27.9765 s | 26.5668 s |
| AR stage time, one profiled sample | 25.8807 s | 24.4465 s |
| DiT stage time, one profiled sample | 1.9955 s | 2.0496 s |

These wall-clock values are recorded for transparency only. They are one
sample per variant and the requested CUDA-profiler RPC was not enabled in the
per-stage deploy YAML of that run, so they are **not** a latency, throughput,
or regression claim.

The optimized producer reported the following payload facts for this workload:

| Payload property | Value |
| --- | ---: |
| Full AR trajectory | 1,094 BF16 rows, 7,841,792 bytes |
| Text condition | 36 rows |
| Image condition | 1,056 rows |
| Selected BF16 conditions | 1,092 rows, 7,827,456 bytes |
| Selected/full BF16 ratio | 0.998172 |
| Prior full FP32 wire payload | 15,683,584 bytes |

Consequently, producer-side selection itself removes only two rows (about
0.18%) for this Preview workload. The material reduction comes from preserving
BF16 across the existing host-mediated wire path, which halves the payload
relative to the previous full FP32 materialization. The exact output hash
confirms that this representation change did not alter this deterministic
workload.

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

The script refuses to start by default when `nvidia-smi` reports an existing
compute process. This prevents a previous failed worker from consuming GPU
memory during the baseline/head comparison. Inspect the reported PID before
stopping it; set `REQUIRE_IDLE_GPUS=0` only for an intentionally shared host.

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

Phase 1 implementation and single-request functional correctness are complete
for the tested path. Before publishing a quantitative performance claim,
collect a worker-aware Nsight trace with equivalent baseline/optimized process
coverage and attribute:

1. AR hidden D2H count, bytes, and duration.
2. Request-end full-payload D2H count, bytes, and duration.
3. Existing connector H2D count, bytes, and duration.

The script uses an opt-in MammothModa2-only NVTX range around
`ar2dit_full_payload()` and does not modify the generic connector or scheduler
to collect this evidence. Actual D2D
transport (NCCL, UCX, CUDA IPC, or equivalent) is a separate Phase 2 design
question and is out of scope for this change.

Within the current connector architecture, the meaningful model-specific work
is intentionally exhausted by request-end batching, producer-side condition
selection, and BF16 payload preservation. Additional clone/cat allocator work
may change small local overheads but cannot reduce the remaining transport
bytes; it is deferred until measurements show that it matters. Meaningful
further transfer reduction requires a GPU-aware connector/transport capability,
which is not supplied by the current generic interface.

For `PROFILE_BACKEND=nsys`, the generated profile deploy YAML declares
`profiler_config: {profiler: cuda}` on both stages. The top-level CLI option
only starts/stops profiling; without the per-stage declaration, Omni rejects
the profiling RPC and Nsight has no valid capture. The script also checks that
`analyze_nsys_transfer.py` exists before it starts either workload.

Set `VLLM_OMNI_MAMMOTH_MODA2_PAYLOAD_STATS=1` on the optimized checkout to
emit one producer-side line per request with the full hidden-state bytes, the
selected condition bytes, and `retained_ratio`. This establishes how much of
the AR trajectory remains after the canonical DiT token masks before drawing
any transport-performance conclusion.
