---
title: Autoregressive Runtime
kind: module
status: draft
owners:
  - "@tzhouam"
  - "@yinpeiqi"
  - "@fake0fan"
  - "@Sy0307"
  - "@Gaohan123"
primary_code_paths:
  - vllm_omni/core/**
  - vllm_omni/worker/**
related_code_paths:
  - vllm_omni/model_executor/**
depends_on:
  - engine_orchestration.md
  - model_integration.md
  - input_output_modality_contracts.md
validation_paths:
  - tests/core/**
  - tests/worker/**
upstream_refs:
  - vllm.v1.core
  - vllm.v1.worker
last_reviewed: 2026-07-16
---

# Autoregressive runtime

The AR runtime extends vLLM scheduling and worker execution for omni-stage
inputs and outputs while preserving vLLM scheduling and cache semantics.

## Candidate invariants

### AR-INV-001: vLLM owns base scheduling semantics

**Rule:** Omni schedulers MUST preserve upstream request-state and cache
transitions unless an Omni-specific difference is documented and tested.

### AR-INV-002: Omni data crosses explicit adapters

**Rule:** Modality-specific stage data MUST be converted at an input or output
adapter, not injected through unrelated scheduler state.

### AR-INV-003: Workers execute assigned work

**Rule:** Workers and model runners MUST NOT implement cross-stage routing.

## Safe-change guide

Test request lifecycle, abort, cache state, and every affected worker execution
mode against the supported upstream vLLM contract.

## Stateful non-AR generation

The IndexTTS-2.5 continuous recipe selects a model-owned generation scheduler
and worker. Initial admission retains ordinary token and KV accounting. Once
a request has consumed its input tokens, subsequent CFM steps travel in
`OmniSchedulerOutput.stepwise_req_ids` and consume no additional token or KV
slots. Resident CFM work can continue when token admission exhausts its budget;
a full scheduler pause still prevents execution.

`IndexTTS2GenerationModelRunner` executes those request IDs against resident
stage payloads. It returns sparse audio rows and
`OmniModelRunnerOutput.generation_finished_req_ids`; an empty audio row does
not finish an active request. The scheduler retains each request until the
model reports completion or normal cancellation/error handling finishes it.
Finished-request cleanup runs before the worker's no-work return, including
when there are no remaining CFM requests. Reusing a request ID clears the
previous model state before initializing the replacement.

The ordinary generation path retains upstream encoder-transfer, KV-transfer,
and external-launcher data-parallel handling for zero-token steps. The
continuous recipe opts into recurrent work without changing that default.
See the
[IndexTTS-2.5 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/IndexTeam/IndexTTS-2_5.md)
for configuration and the disabled experimental vocoder overlap and admission delay.
