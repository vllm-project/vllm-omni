# MiniMax-H3 on SM120: 1, 2, 4, or 8 GPUs

This example benchmarks the MiniMax-H3 `FL2VA` partition with reproducible
parallel topologies. The default is the recommended starting point for four
72 GB NVIDIA RTX PRO 5000 Blackwell (SM120) cards:

- DiT TP2 x Ulysses2, Ring1;
- text-encoder TP4;
- tiled VAE patch parallel degree 4;
- fully resident weights and regional `torch.compile`;
- cuDNN BF16 attention as the accuracy/performance baseline;
- T2V followed by first-frame I2V, with one checkpoint load.

The runner provides one launch script per GPU count. The recipes below make
the hardware, placement, workload, and acceptance criteria explicit, in the
same spirit as a deployment cookbook rather than a single benchmark command.

## SM120 deployment cookbook

Scope: these are single-node RTX PRO 5000 Blackwell (SM120, 73,415 MiB per
GPU) recipes. The target host has two PCIe/NUMA islands: GPUs `0-3` on NUMA 0
and GPUs `4-7` on NUMA 1. Keep a run inside one island whenever possible.

### Select a profile

| GPUs | DiT parallelism | Weight placement | Script | Intended use | Status |
| ---: | --- | --- | --- | --- | --- |
| 1 | TP1 x Ulysses1 | DLO, 20 resident layers | `run_sm120_1gpu.sh` | Capacity baseline | Recipe; requires offload |
| 2 | TP2 x Ulysses1 | DLO, 20 resident layers | `run_sm120_2gpu.sh` | Smallest two-GPU deployment | Required: resident OOM on 72 GiB |
| 4 | TP2 x Ulysses2 | Resident | `run_sm120_4gpu.sh` | PCIe-balanced default | Validated on this host |
| 8 | TP2 x Ulysses4 | Resident | `run_sm120_8gpu.sh` | First 8-GPU latency candidate | Recipe; compare with TP4 x Ulysses2 |

Do not use TP1 x Ulysses4 on this 72 GiB target: it does not shard DiT
weights and the B300 preflight peak was about 98 GiB. For every profile, the
reference comparison backend is BF16 `CUDNN_ATTN`. FlashAttention-4 is not a
supported MiniMax-H3 SM120 baseline yet: its packed/varlen DiT path fails in
the current upstream CuTe implementation.

### Common workload

All scripts default to the reproducible `FL2VA` workload: T2VA plus
first-frame I2VA, 1344x768, 124 frames (about five seconds), BF16, 50 denoise
steps, and two warmups. `RUN_REF2VA=1` additionally runs the `Ref2VA`
partition; it requires the corresponding model checkpoint under `MODEL_ROOT`.

For a one-GPU deployment, use `DLO_RESIDENT_LAYERS=auto` to select the largest
candidate from `35 30 25 20` that completes a five-step probe while retaining
6 GiB of measured external-GPU headroom. The autotuner writes its decision to
`summary.json` under `dlo_autotune`, then runs the requested 50-step workload.

```bash
DLO_RESIDENT_LAYERS=auto \
  DLO_AUTO_HEADROOM_MIB=6144 \
  OUTPUT_DIR="$TEST_ROOT/results/sm120-1gpu-auto" \
  bash examples/offline_inference/minimax_h3/run_sm120_1gpu.sh
```

Override `DLO_AUTO_CANDIDATES` or `DLO_AUTO_PROBE_STEPS` only when evaluating a
new card or model revision. This is launch-time selection; resident layers are
not changed during a generation.

Set the shared paths once, then run exactly one recipe:

```bash
export TEST_ROOT=/path/to/minimax-h3-native
cd "$TEST_ROOT/vllm-omni"
export PATH="$TEST_ROOT/ffmpeg-tools/bin:$PATH"

# Choose one: 1, 2, 4, or 8 GPUs.
bash examples/offline_inference/minimax_h3/run_sm120_1gpu.sh
bash examples/offline_inference/minimax_h3/run_sm120_2gpu.sh
bash examples/offline_inference/minimax_h3/run_sm120_4gpu.sh
bash examples/offline_inference/minimax_h3/run_sm120_8gpu.sh
```

Each wrapper passes its topology to `run_all_tasks.sh`; callers can override
only the inputs that vary per deployment. For example:

```bash
# Run the full three-workload matrix on NUMA 1 using physical GPUs 4--7.
CUDA_VISIBLE_DEVICES=4,6,5,7 NUMA_NODE=1 RUN_REF2VA=1 \
  OUTPUT_DIR="$TEST_ROOT/results/sm120-4gpu-full" \
  bash examples/offline_inference/minimax_h3/run_sm120_4gpu.sh
```

The four-GPU ordering `0,2,1,3` (or `4,6,5,7`) deliberately maps each
Ulysses pair to a close PCIe pair. The 8-GPU script interleaves the two NUMA
islands and uses `numactl --interleave=0,1`; do not pin it to one memory node.

### Resolve the remaining topology choices

Use the screen before committing a 50-step result when more than one topology
is viable. It runs the common workload at five steps and writes one output
directory per candidate. It is deliberately limited to candidates that fit
the 72 GiB target.

```bash
# Set SCREEN_GPU_COUNT to one of 1, 2, 4, or 8.
# The default is 4. Use a free NUMA island by overriding CUDA_VISIBLE_DEVICES.
SCREEN_GPU_COUNT=4 \
  SCREEN_ROOT="/results/sm120-4gpu-screen" \
  bash examples/offline_inference/minimax_h3/run_sm120_topology_screen.sh
```

| GPU count | Screened cases | Decision from the screen |
| ---: | --- | --- |
| 1 | TP1 x Ulysses1, DLO resident layers 20, 35, and 50 | Lowest steady-state T2VA/I2VA latency that fits |
| 2 | TP1 x Ulysses2 vs TP2 x Ulysses1, each at DLO resident layers 20, 35, and 50 | Lowest maximum T2VA/I2VA latency that fits; fully resident OOMs during VAE decode |
| 4 | TP2 x Ulysses2 vs TP4 x Ulysses1 | Lowest maximum T2VA/I2VA latency that passes memory |
| 8 | TP2 x Ulysses4 vs TP4 x Ulysses2 vs TP8 x Ulysses1 | Lowest maximum T2VA/I2VA latency that passes memory |

Promote only the winning candidate to the corresponding 50-step wrapper.
Keep the loser directories: their `summary.json`, peak CSV, and logs document
why that topology was rejected. The screen is a topology decision, not a
reported latency benchmark.

For a complete, directly reportable 50-step matrix, use the four dedicated
wrappers instead. They run every viable candidate for that GPU count with two
warmups and keep `RUN_REF2VA=0` so all topology comparisons use the same
FL2VA T2VA/I2VA workload.

```bash
SCREEN_ROOT="$TEST_ROOT/results/sm120-1gpu-matrix" bash examples/offline_inference/minimax_h3/run_sm120_1gpu_matrix.sh
SCREEN_ROOT="$TEST_ROOT/results/sm120-2gpu-matrix" bash examples/offline_inference/minimax_h3/run_sm120_2gpu_matrix.sh
SCREEN_ROOT="$TEST_ROOT/results/sm120-4gpu-matrix" bash examples/offline_inference/minimax_h3/run_sm120_4gpu_matrix.sh
SCREEN_ROOT="$TEST_ROOT/results/sm120-8gpu-matrix" bash examples/offline_inference/minimax_h3/run_sm120_8gpu_matrix.sh
```

To test resident layers 20 through 50 across both two-GPU topologies and then
one GPU, select the lowest worst-case T2VA/FL2VA latency with at least 4 GiB
headroom, and fully validate both winners at 50 steps in one invocation:

```bash
RESULT_ROOT="$TEST_ROOT/results/sm120-1gpu-2gpu-residency" \
  bash examples/offline_inference/minimax_h3/run_sm120_1gpu_2gpu_residency_sweep.sh
```

The sweep is resumable and writes `probe_summary.csv`, `selection.json`, and
`conclusion.txt` below `RESULT_ROOT`.

Run these one at a time on idle GPUs. Select the candidate with the lowest
maximum steady-state T2VA/I2VA latency that also passes the external peak
memory gate; only then run that winner with `RUN_REF2VA=1`.

The one- and two-GPU DLO wrappers default to eager execution because regional
compilation is unstable with this offload path. Keep the execution mode fixed
across candidates in the same matrix. The four- and eight-GPU fully resident
wrappers retain regional compilation.

### Record and accept a run

Every output directory contains the evidence needed to populate a results
matrix:

| File | Use |
| --- | --- |
| `summary.json` | Per-workload E2E and stage timings, hashes, dimensions, and worker peak |
| `gpu_peak_memory.csv` | External per-physical-GPU peak memory |
| `run.log` | Backend resolution, topology, warnings, and failures |
| generated `.mp4` | Output artifact for visual/audio validation |

Mark a row as passed only when all selected GPUs are below capacity, the
expected `TASK_RESULT` records are present, output hashes/dimensions are
recorded, and the chosen attention backend is logged. Use the maximum entry
in `gpu_peak_memory.csv` as **Peak Memory (GiB)** and label its scope as
`per-GPU (external nvidia-smi)`.

For a kernel breakdown, profile the final chosen topology with the existing
Nsight recipe below. Treat a five-step trace as a kernel-mix and load-balance
measurement; use the 50-step steady-state runs from `summary.json` for the
reported E2E and per-step timing.

## Why start with TP2 x Ulysses2

The target server's PCIe bus IDs naturally form close pairs. TP2 x Ulysses2
keeps TP collectives and sequence exchange in two-rank groups, while avoiding
the four-rank per-layer TP collectives of TP4. It is the leading PCIe candidate,
not a substitute for measurement. Compare it with both endpoints:

| Candidate | Environment | Expected trade-off |
| --- | --- | --- |
| TP2 x Ulysses2 | `TP_SIZE=2 ULYSSES_DEGREE=2` | Leading RTX PCIe candidate; pair-sized collectives |
| TP1 x Ulysses4 | `TP_SIZE=1 ULYSSES_DEGREE=4` | No DiT TP reductions; four-rank AllToAll/SendRecv |
| TP4 x Ulysses1 | `TP_SIZE=4 ULYSSES_DEGREE=1` | No sequence AllToAll; four-rank TP communication in every block |

The B300 preflight below makes TP1 x Ulysses4 the fastest NVLink topology,
but its 98 GB peak makes it invalid for a 72 GB target. PCIe can also change the
winner, so the final RTX PRO 5000 choice is the lowest steady-state T2V/I2V
latency that passes memory, quality, and load-balance gates. Fully resident
execution has no FSDP/DLO weight AllGather. If the reference FSDP run spends
23.1% in NCCL AllGather, that component should fall sharply; Ulysses will still
appear as NCCL SendRecv/AllToAll.

## B300 preflight results (2026-08-06)

These are single-run selection measurements on four B300 SXM6 GPUs with cuDNN
BF16 attention, 1344x768 output, 124 frames, five requested denoising steps, and
two warmups. They validate the runner and eliminate memory-invalid candidates;
they are not RTX PRO 5000 performance claims.

| DiT topology | T2V wall time | I2V wall time | Maximum peak per rank | 72 GB target status |
| --- | ---: | ---: | ---: | --- |
| TP2 x Ulysses2 | 11.060 s | 6.470 s | 66,930 MiB | Fits, about 6.3 GiB margin |
| TP1 x Ulysses4 | 7.298 s | 6.137 s | 98,214 MiB | Rejected: OOM |
| TP4 x Ulysses1 | 7.545 s | 6.441 s | 49,332 MiB | Fits; fallback candidate |

The primary RTX PRO 5000 plan remains TP2 x Ulysses2 because every DiT TP
collective stays within a close two-GPU PCIe pair. TP4 x Ulysses1 is the
lower-memory fallback and must be measured on the PCIe target: B300 NVLink
hides the four-rank TP cost. The TP2 memory margin is narrow enough that no
other process should occupy the selected cards.

A five-step Nsight trace of TP2 x Ulysses2 on the same B300 host, after two
warmups, produced the following aggregate GPU-kernel shares:

### T2V preflight

- NCCL AllGather: 0.60%
- NCCL SendRecv: 12.18%
- NCCL other: 5.01%
- NCCL total: 17.79%
- Dense FlashAttention/FMHA: 26.54%
- Other GEMM, norm, elementwise, and VAE: 55.67%
- Maximum GPU deviation: 4.63%; max-min/mean: 8.77%

### I2V preflight

- NCCL AllGather: 0.46%
- NCCL SendRecv: 6.30%
- NCCL other: 7.63%
- NCCL total: 14.39%
- Dense FlashAttention/FMHA: 29.35%
- Other GEMM, norm, elementwise, and VAE: 56.26%
- Maximum GPU deviation: 0.61%; max-min/mean: 0.96%

This is directionally better than the supplied FSDP kernel mix because weight
AllGather is almost absent, but it is not an apples-to-apples speedup claim:
the hardware, interconnect, and step count differ. Use the exact profiling
workflow below for the final 50-step RTX result.

## Prerequisites

Use Python 3.12, a CUDA 13-compatible PyTorch/vLLM environment, and install this
checkout:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.26.0 --torch-backend=auto
uv pip install -e .
```

Download `FL2VA`; `Ref2VA` is optional unless `RUN_REF2VA=1` is set:

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 \
  --include 'FL2VA/**' \
  --local-dir "${MODEL_ROOT}"
```

`ffmpeg`, `ffprobe`, `nvidia-smi`, and Nsight Systems (`nsys`) are required for
the complete output/profile workflow. The launch preflight rejects selected
GPUs that are busy or have less than 70,000 MiB physical memory by default.

## Run the leading four-card plan

On the measured dual-socket RTX PRO 5000 host there is no GPU NVLink. GPU
pairs 0-1 and 2-3 are PXB-local in NUMA node 0; pairs 4-5 and 6-7 are
PXB-local in NUMA node 1. With TP2 x Ulysses2, logical TP groups are 0-1 and
2-3 while Ulysses groups are 0-2 and 1-3. The profiled SendRecv share is larger
than the TP collective share, so the first target candidate maps the Ulysses
groups to PXB-local physical pairs:

| NUMA node | Preferred physical order | TP-PXB control order |
| --- | --- | --- |
| 0 | `CUDA_VISIBLE_DEVICES=0,2,1,3` | `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| 1 | `CUDA_VISIBLE_DEVICES=4,6,5,7` | `CUDA_VISIBLE_DEVICES=4,5,6,7` |

Never mix GPUs from the two NUMA nodes in a four-card run; those paths are
`SYS`. Run both orders on the target and retain the lower median latency.

From the repository root:

```bash
MODEL_ROOT=/path/to/MiniMax-H3 \
CUDA_VISIBLE_DEVICES=0,2,1,3 \
bash examples/offline_inference/minimax_h3/run_all_tasks.sh
```

Defaults are 1344x768, 124 frames at 24 FPS (about 5.17 seconds after H3 frame
alignment), and 50 denoising steps. The default run produces only the requested
T2V and first-frame I2V outputs. Set `RUN_REF2VA=1` for the other two paths.

Use the same prompt, seed, shape, step count, GPU order, and attention backend
for topology A/B runs:

```bash
# Candidate A: TP2 x Ulysses2 (default)
OUTPUT_DIR=/results/h3-tp2-u2 \
TP_SIZE=2 ULYSSES_DEGREE=2 \
bash examples/offline_inference/minimax_h3/run_all_tasks.sh

# Candidate B: TP1 x Ulysses4 (B300/NVLink baseline)
OUTPUT_DIR=/results/h3-tp1-u4 \
TP_SIZE=1 ULYSSES_DEGREE=4 \
bash examples/offline_inference/minimax_h3/run_all_tasks.sh

# Candidate C: TP4 without sequence parallelism
OUTPUT_DIR=/results/h3-tp4-u1 \
TP_SIZE=4 ULYSSES_DEGREE=1 \
bash examples/offline_inference/minimax_h3/run_all_tasks.sh
```

The 72 GB target should start fully resident. If a target-only OOM occurs, use
the merged #5764 no-AllGather DLO path as a capacity fallback, not as the first
performance result:

```bash
ENABLE_DLO=1 DLO_RESIDENT_LAYERS=20 \
TP_SIZE=2 ULYSSES_DEGREE=2 \
bash examples/offline_inference/minimax_h3/run_all_tasks.sh
```

Useful controls include `PYTHON`, `INSTALL_EDITABLE=0`, `ENFORCE_EAGER`,
`NUM_INFERENCE_STEPS`, `WARMUP_STEPS`, `DURATION_SECONDS`, and the preflight
memory/utilization thresholds. Keep the selected execution mode fixed for all
rows being compared.

## SM120 FlashInfer FP8 attention experiment

The optional `FLASHINFER_SM120_ATTN` backend targets the implementation in
FlashInfer commit `4a2345906256da0849d7e1e4681db514ab9b800e`. It uses
`BatchPrefillWithRaggedKVCacheWrapper(..., backend="cute-dsl-prims")` with FP8
E4M3 Q/K/V and BF16/FP16 output. MiniMax-H3 is MHA with head dimension 128, so
TP2 x Ulysses2 presents 14 local heads and satisfies the kernel contract.

Build the exact FlashInfer revision on the RTX PRO 5000 host:

```bash
git clone --recursive https://github.com/Tom-Zheng/flashinfer.git
cd flashinfer
git checkout 4a2345906256da0849d7e1e4681db514ab9b800e
python -m pip install -v '.[cu13]'
python -m pytest -q tests/attention/test_sm120_prims_prefill_backend.py
```

Then replace only the attention backend in the same topology:

```bash
ATTENTION_BACKEND=FLASHINFER_SM120_ATTN \
WARMUP_STEPS=2 \
OUTPUT_DIR=/results/h3-tp2-u2-sm120-fp8 \
bash examples/offline_inference/minimax_h3/run_all_tasks.sh
```

The adapter converts BF16 activations to FP8. If scales are omitted, every
attention layer calibrates Q/K/V once on its first invocation with 2x headroom
and caches the Python scalar scales; `WARMUP_STEPS=2` keeps that calibration and
regional compilation outside the measured NVTX task ranges. Optional global
static values can be supplied as `FP8_Q_SCALE`, `FP8_K_SCALE`, and
`FP8_V_SCALE`, but should be used only after target accuracy validation.
Explicit attention masks fall back to SDPA. Selecting this backend on B300 or
any non-SM120 GPU fails before the first kernel launch.

Keep cuDNN as the baseline until the complete BF16-to-FP8 conversion plus FP8
attention wins end to end. On the development B300, a compute-only proxy at
`S=39,232, H=14, D=128` measured 6.105 ms median for cuDNN BF16 and 26.399 ms
for the existing FlashInfer BF16 path. This result selects the B300 baseline;
it is not a result for the new SM120 kernel.

## Capture the requested kernel proportions

Use a short warmup outside the marked task ranges, but keep the measured
request at the production 50 steps:

```bash
TRACE_ROOT=/results/h3-tp2-u2-nsys
mkdir -p "${TRACE_ROOT}"

MODEL_ROOT=/path/to/MiniMax-H3 \
OUTPUT_DIR="${TRACE_ROOT}/outputs" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=2 ULYSSES_DEGREE=2 WARMUP_STEPS=2 \
INSTALL_EDITABLE=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="${TRACE_ROOT}/minimax-h3" \
  bash examples/offline_inference/minimax_h3/run_all_tasks.sh

nsys export \
  --type=sqlite \
  --force-overwrite=true \
  --output="${TRACE_ROOT}/minimax-h3.sqlite" \
  "${TRACE_ROOT}/minimax-h3.nsys-rep"

python examples/offline_inference/minimax_h3/analyze_nsys.py \
  "${TRACE_ROOT}/minimax-h3.sqlite" \
  --json-output "${TRACE_ROOT}/kernel-breakdown.json" \
  | tee "${TRACE_ROOT}/kernel-breakdown.md"
```

The runner places `minimax_h3_task:t2va` and
`minimax_h3_task:fl2va_first_frame` NVTX ranges around only the measured
requests. The analyzer sums `end-start` for every CUDA kernel on every GPU
inside each range. This is aggregate GPU kernel time, matching reports such as
“NCCL 35.7%”; it is intentionally not wall-clock percentage because kernels on
different GPUs overlap.

The Markdown report contains:

```text
### T2V

GPU kernel 聚合时间占比：

- NCCL AllGather：...
- NCCL SendRecv：...
- NCCL 其他：...
- NCCL 总计：...
- Dense FlashAttention/FMHA：...
- 其余 GEMM、norm、elementwise、VAE：...

GPU 负载均衡（累计 kernel time）：

- GPU 0：... ms（相对均值 ...%）
...
- 最大偏离均值：...
- max-min/mean：...
```

Use `max deviation <= 5%` and `max-min/mean <= 10%` as initial balance gates.
Compare topology medians only after one warmup, and retain `summary.json`,
`gpu_peak_memory.csv`, the Nsight report, and the generated media. For the FP8
backend, also compare output validity and a representative quality suite; a
lower attention-kernel time alone is not sufficient.

## Outputs

The default output directory contains:

- `01_t2va.mp4` and `02_fl2va_first_frame.mp4`;
- `summary.json` with topology, attention backend, latency, shapes, and hashes;
- one `ffprobe.json` file per MP4;
- `gpu_peak_memory.csv`, `nvidia-smi.csv`, and `artifact_sha256.txt`.

The compatibility entry point `run_4gpu_all_tasks.sh` delegates to the generic
runner. Set `RUN_REF2VA=1` to additionally produce the two reference-conditioned
MP4 files.
