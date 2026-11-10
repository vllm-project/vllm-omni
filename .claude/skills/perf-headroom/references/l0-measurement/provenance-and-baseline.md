# Provenance, baseline and quality

## Runtime identity procedure

Record repository HEAD and dirty/untracked runtime files on both the requested
source and execution host. Record the interpreter, imported `vllm_omni` and
`vllm` locations in the actual worker environment, dependency versions, container
identity, launch command and effective config. Trace the selected optional backend
to its installed implementation. Backend configuration is intent; executed kernels
and dispatch branches provide runtime confirmation.

Inspect before aligning code. Do not overwrite a shared dirty checkout or replace
the requested revision with upstream main. If alignment requires another checkout,
use an authorized dedicated location. Record the instrumented diff/hash separately
from the production revision. Never embed host credentials in the skill or report.

Useful read-only inventory, adapted to the selected environment:

```bash
git rev-parse HEAD
git status --short
python -c "import sys, vllm_omni; print(sys.executable); print(vllm_omni.__file__)"
python -m pip show vllm vllm-omni torch
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem --format=csv
```

Run interpreter checks where the workers run, not only in the SSH login shell.
Avoid dumping all environment variables; selected version/placement variables
are sufficient and avoid leaking tokens. Read-only health checks do not authorize
changing clocks, power policy, CPU affinity or other users' jobs.

## Baseline recipe

1. Resolve the native runner and inspect its help/config. Use the same successful
   request specification for all evidence products. Save the exact command.
2. Choose cold-start or steady-state treatment explicitly. Warm until the relevant
   compilation/capture/loading paths stabilize; a fixed count alone is not proof.
3. Measure to the actual completion boundary. An asynchronous GPU return needs a
   completion observation; do not insert synchronization at every internal op.
4. Save multiple cohorts of raw samples, success/failure status, output shape,
   hardware observations, elapsed wall time and memory peaks with their definitions.
5. Describe median/mean, spread and cohort drift. Quantiles need enough samples
   for their claimed precision; five requests cannot establish a robust P99.
6. For a later authorized A/B experiment, interleave or rotate order and preserve
   matching inputs/cache state. Analyze paired results when the design supports it.

Peak allocated, peak reserved and device-used memory are different measures.
Record the owner/time window of each. Do not silently disable required offload or
tiling to get a simpler baseline, or reduce steps/frames to fit the profiling budget.
Capture fewer representative requests/steps instead and state sampling limits.

## Four evidence products

| Product | Primary purpose | Not valid as |
| --- | --- | --- |
| Unprofiled cohorts | Actual latency/throughput/quality result | Operator root-cause evidence on their own |
| Nsys / bounded timeline | Ordering, activity, waits and overlap | A complete counter-based efficiency diagnosis |
| Source/shape/stack inventory | Semantics, population and path attribution | Proof that unobserved code executed |
| Targeted NCU | Hardware mechanism for selected launches/ranges | Authoritative request wall time |

Compare captures to the baseline; never splice NCU durations, instrumented phase
times and unprofiled wall time into one closed ledger. Instrumented phase timers
that synchronize deliberately change the schedule.

## Noise and quality decisions

The helper's cohort-spread proxy is a screening measure, not a confidence
interval. Match the statistical design to the objective and minimum useful gain;
multiple attempts and repeated stopping checks can invalidate naive fixed-test
confidence statements. If uncertainty is larger than the remaining-space claim,
collect better evidence or leave the claim unresolved.

For quality, retain prompt/seed set and output evidence privately in the campaign.
Numerical kernel comparisons and end-to-end image/video acceptance are distinct.
Do not call a faster candidate a validated gain until the declared quality gate
passes. A user allowing small differences has not authorized fewer steps,
quantization, approximate attention or approximate denoising reuse.
