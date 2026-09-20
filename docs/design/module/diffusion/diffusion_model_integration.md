---
title: Diffusion Model Integration
kind: module
status: draft
owners:
  - "@fhfuih"
  - "@Bounty-hunter"
  - "@wtomin"
  - "@RuixiangMa"
primary_code_paths:
  - vllm_omni/diffusion/models/**
  - vllm_omni/diffusion/model_loader/**
related_code_paths:
  - vllm_omni/diffusion/layers/**
  - vllm_omni/diffusion/lora/**
  - vllm_omni/diffusion/utils/**
depends_on:
  - diffusion_runtime.md
  - ../input_output_modality_contracts.md
validation_paths:
  - tests/diffusion/models/**
  - tests/diffusion/model_loader/**
  - tests/diffusion/layers/**
  - tests/diffusion/lora/**
upstream_refs:
  - diffusers.DiffusionPipeline
last_reviewed: 2026-07-16
---

# Diffusion model integration

Diffusion model integration owns pipeline contracts, registration, checkpoint
loading, adapters, shared layers, and model-specific processing.

## Candidate invariants

### DIFF-MODEL-INV-001: Pipelines implement one runtime contract

**Rule:** A pipeline MUST declare its supported modalities, configuration,
inputs, outputs, loading path, and runtime capabilities.

### DIFF-MODEL-INV-002: Registration is the selection boundary

**Rule:** Runtime code MUST select model implementations through the registry or
loader contract rather than scattered model-name conditionals.

### DIFF-MODEL-INV-003: Model code does not schedule requests

**Rule:** Pipeline code MUST NOT own admission, batching, cancellation, or
cross-stage routing.

### DIFF-MODEL-INV-004: Shared behavior stays shared

**Rule:** Model directories SHOULD contain only genuine model differences.

## Safe-change guide

Test registry selection, checkpoint loading, minimal inference, input and output
contracts, and every declared optional capability.

## FlowUniPC solver plans

`FlowUniPCMultistepScheduler` retains one schedule's predictor/corrector
coefficients per scheduler instance. The first step prepares the plan after the
starting index, device, and sample dtype are known. Subsequent requests with an
identical schedule reuse these read-only coefficients; `set_timesteps` still
resets model-output history and the previous sample.

The plan preserves the existing device, dtype, and scalar evaluation order.
Its identity includes the complete sigma schedule, starting index, sample and
default dtypes, device, solver order/type, prediction mode, corrector exclusions,
and final-order policy. Changing these inputs replaces the plan. It is not a
process-global cache and does not share mutable request state.

Planning applies to CPU/CUDA samples, orders 1–3, and strictly decreasing,
finite CPU sigma schedules with at most 128 timesteps. Longer or unsupported
schedules, other devices, and external predictor solvers evaluate coefficients
on demand. The first request pays plan construction; steady-state savings come
from reusing it. This does not implement a general solver-plan registry or
cross-instance cache.

For a scheduler-only comparison against a trusted base revision:

```bash
git show <base-sha>:vllm_omni/diffusion/models/schedulers/scheduling_flow_unipc_multistep.py > /tmp/base_unipc.py
PYTHONPATH=. python benchmarks/diffusion/bench_flow_unipc_solver_plan.py \
  --reference /tmp/base_unipc.py --device cuda --steps 30 --repeats 20
```

The benchmark alternates base/plan ordering, excludes two warmups, checks exact
output equality, and reports cold and warm durations separately. Scheduler-only
savings must not be presented as end-to-end generation speedups.
