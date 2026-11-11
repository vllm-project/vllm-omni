# Ceilings on Ascend NPU

What the skill's method keeps on Ascend, what it loses, and the replacements.

## Contents
- What carries over unchanged
- What does not exist on Ascend
- Device reservation
- Utilisation sampling
- Per-stage attribution
- Device peaks and bound terms
- Pitfalls

## What carries over unchanged

The ledger is a queueing and work argument, not a CUDA one. These hold on any
accelerator:

- RTF definitions, and the gap between per-request mean and aggregate.
- Little's law in a closed-loop bench: `aggregate RTF(C) = C / X`.
- Shared capacity accumulates demand per resource and takes the maximum;
  independent stage durations must not be summed. Serial stage constraints
  and path gates remain separate (see [bound-models.md](bound-models.md)).
- The roofline terms: decode bounded by bytes read per step over measured
  bandwidth, compute stages bounded by FLOPs per audio second over measured
  dense throughput.
- `vllm bench serve --omni` and `--print-stage`: the client is device-neutral,
  so cell commands in [ceiling-experiments.md](ceiling-experiments.md) are
  unchanged.

Measured peaks calibrate estimates, not strict hardware bounds. Mean RTF
requires per-request work, and residual attribution requires profiling.
So the workflow and the report template stay. Only the three measurement
mechanics below are CUDA-specific.

## What does not exist on Ascend

| CUDA mechanism | Ascend status | Replacement |
|---|---|---|
| `cuda_mps` co-location | Not available. `OmniStageRuntime._mps_environment` requires `current_omni_platform.is_cuda()` and logs `cuda_mps disabled` otherwise | None. Stages time-share the device through the driver; there is no MPS-equivalent to toggle |
| `nvidia-smi pmon -s u` per-process SM share | No per-process AI Core counter. `npu-smi info -t proc-mem` reports memory only | `--stage-util` with one device per stage, or `msprof` per stage |
| `nvidia-smi --query-gpu` CSV | Different tool and format | `npu-smi info watch`, parsed by `ceiling_report.py` |
| CUDA graph capture budgets | ACL graph (`ACLGraphWrapper`) or torchair graph mode | Same knob, different keys and log lines; read startup logs for the Ascend fallback message |
| TF32 matmul | HF32 | `measure_peaks.py` reports `hf32` in place of `tf32`; pass `--compute-bound NAME=TFLOP,hf32` |
## Device reservation

Every cell of one comparison must run inside a single reservation on fixed
cards, as in the `npu-ab-experiment` skill:

```bash
npu status   # pick a card that is AVAILABLE, not UNRESERVED-in-use
npu run --gpu-ids 2 --timeout 6h --idle-timeout 0 \
  --note "omni-perf-ceiling <model> C1/C64" -- bash cells.sh
```

`npu run` sets `ASCEND_RT_VISIBLE_DEVICES`; never set it yourself. A card shown
as `UNRESERVED` already has another tenant's process on it, and its utilisation
samples are a sum over both tenants, which silently inflates every `g_i`.

## Utilisation sampling

`npu-smi info watch` has no timestamp column, so prefix one; without it
`ceiling_report.py` cannot trim to the measured window and says so.

```bash
npu-smi info watch -i $CARD -c $CHIP -d 1 -s abmn \
  | while IFS= read -r line; do
      printf '%s, %s\n' "$(date '+%Y/%m/%d %H:%M:%S.%3N')" "$line"
    done > $OUT/c${C}_npu.txt &
```

`-s abmn` selects AI Core, memory, memory bandwidth and NPU utilisation. Pass
the file in the cell's third slot exactly like the CUDA CSV:

```bash
python .claude/skills/omni-perf-ceiling/scripts/ceiling_report.py \
  --cell c64=$OUT/c64.json:$OUT/c64_bench.log:$OUT/c64_npu.txt
```

The default column is `AI Core(%)`. It counts cube occupancy, so a stage
dominated by vector or AI CPU work reads low and its `g_i` is understated;
`--npu-util-column 'NPU Util(%)'` is the closer analogue of `utilization.gpu`.
Report which column a ledger used. When the two disagree by much, the stage is
not cube bound and the compute roofline is the wrong bound for it.

`Memory BW(%)` is a cross-check CUDA's `utilization.gpu` cannot give: a decode
stage claimed to be bandwidth bound should sample near its bandwidth ceiling.
If it does not, the decode bound is not the binding term.

## Per-stage attribution

Method 1 from the skill (one device per stage) is the only one that gives
absolute `g_i` on Ascend. Place each stage on its own card, sample each card,
and pass one file per stage:

```bash
python .../ceiling_report.py \
  --cell c64=$OUT/c64.json:$OUT/c64_bench.log:$OUT/c64_npu0.txt \
  --stage-util c64=0:$OUT/c64_npu0.txt \
  --stage-util c64=1:$OUT/c64_npu1.txt \
  --stage-util c64=2:$OUT/c64_npu2.txt
```

Inter-card transfer differs from the shared-card deployment, so these `g_i` do
not reproduce the shared-card RTF; use them for the ledger only.

Method 3 (stage isolation at fixed batch) is unchanged and is the way to get
the `g_i(B)` curve. Profiler-based attribution uses `msprof` or
`torch_npu.profiler`; keep it out of any run that produces timing numbers.

## Device peaks and bound terms

```bash
python .claude/skills/omni-perf-ceiling/scripts/measure_peaks.py \
  --device npu --json $OUT/peaks.json
```

Measured on one Ascend 910B1 (CANN 8.5.1, driver 25.5.0, `torch_npu`, matmul
n=8192), as an order-of-magnitude reference only; measure your own card:

| Term | Value |
|---|---|
| Copy bandwidth (read+write) | 1253 GB/s |
| Dense matmul bf16 | 299 TFLOPS |
| Dense matmul fp16 | 297 TFLOPS |
| Dense matmul hf32 | 137 TFLOPS |
| Dense matmul fp32 | 89 TFLOPS |

Two consequences for the bounds:

- The bf16 to fp32 ratio is about 3.4x on 910B1 against roughly 15x on L20X, so
  an fp32 vocoder or flow-matching stage is far less penalised here. A "move
  this stage off fp32" recommendation ported from a CUDA ledger can be worth
  almost nothing on Ascend; recompute the term before repeating it.
- HF32 sits between fp32 and bf16 rather than near bf16. Name the dtype the
  stage actually runs in, and check whether HF32 is enabled at all
  (`torch.npu.matmul.allow_hf32`), since the bound changes by 1.5x with it.
- The launch term is re-measured, not ported. `measure_peaks.py` times eager
  `torch_npu` dispatch and, when `torch.npu.NPUGraph` capture works on the
  installed release, ACL graph replay; a null `graph` means only the eager
  value is grounded. Kernel counts per step also change with the backend's
  kernels and fusion, so recount them from an `msprof` capture of the stage.
  The table above predates the launch measurement.

Weight bytes, codec token rates and FLOPs per audio second are properties of
the model, not the device, so they carry over from a CUDA ledger unmodified.
Only the peaks, the launch latency, the kernel counts and the measured `g_i`
need re-measuring.

## Pitfalls

- Reading a CUDA ceiling as an Ascend one. Bounds scale with the measured peaks
  of the card in hand; a bound is not portable.
- Sampling a card that another tenant is using. Check `npu status` first.
- Treating `AI Core(%)` as total device busy time. It is cube occupancy.
- Summing a fused kernel's cube work and vector work as two rows. The two
  pipes overlap, so the kernel is one row at `max(cube, vector)`; two rows
  overstate the floor and can push a stage past 100% of SOL.
- Keeping `cuda_mps: true` in an overlay and reporting the run as MPS on. It
  is accepted and ignored.
- Profiling and timing in one run: `msprof` perturbs the numbers like any other
  profiler.
