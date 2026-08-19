---
title: vLLM-Omni Configuration
kind: module
status: draft
architecture_state: deferred-pending-refactor
ownership_status: provisional
owners:
  - "@lishunyang12"
  - "@alex-jw-brooks"
primary_code_paths:
  - vllm_omni/config/**
  - vllm_omni/deploy/**
  - vllm_omni/model_executor/stage_configs/**
related_code_paths:
  - vllm_omni/platforms/*/stage_configs/**
depends_on: []
validation_paths:
  - tests/config/**
upstream_refs:
  - vllm.config
last_reviewed: 2026-08-11
last_verified_commit: e356708da4cd39992405686dacfc32b18fcccc7f
---

# vLLM-Omni configuration

This page intentionally records discovery paths only. Substantive
configuration ownership and contract work is deferred until the active
configuration refactoring has settled.

## Deferred contract scope

The current code and tests remain authoritative for parsing, defaults,
overrides, deployment topology, stage construction, and runtime projection.
This draft does not define a stable precedence model, environment-variable
capture rule, canonical configuration object, or invariant namespace.

The PR review proposal to capture environment-derived values in configuration
objects at initialization is an open design question, not a current
invariant.

## Current structured-stage reuse behavior

The current RFC #4021 implementation uses upstream vLLM config classes as the
schema source for AR and generation stages:

- `OmniStageLoadConfig` inherits `vllm.config.LoadConfig`.
- `OmniStageCacheConfig` inherits `vllm.config.CacheConfig`.
- `OmniStageSchedulerConfig` inherits `vllm.config.SchedulerConfig`.
- `OmniStageParallelConfig` inherits `vllm.config.ParallelConfig`.

This inheritance preserves the existing load, cache, scheduler, and parallel
concern boundaries while reducing duplicate downstream declarations. It does
not, by itself, make every inherited field an effective vLLM-Omni runtime
option. Construction and runtime consumption are separate surfaces:

| Surface | Current behavior |
| --- | --- |
| Structured schema | Upstream dataclass fields are inherited. Their defaults, default factories, and applicable Pydantic validation can participate when the structured object is constructed. |
| Omni input ownership | Pipeline/deploy/CLI construction continues to accept only fields with an existing structured owner. Direct construction of an inherited sub-config can expose a wider upstream schema. |
| Engine projection | For AR and generation stages, reusable fields are discovered by intersecting each upstream config dataclass with upstream `EngineArgs`. Only constructor-explicit values are emitted; inherited defaults remain deferred to terminal materialization. |
| Terminal materialization | The engine-owning process constructs the final upstream `VllmConfig` and performs model-, platform-, rank-, port-, and backend-dependent initialization. |

The generated projection maps include known upstream naming differences such
as `cache_dtype` to `kv_cache_dtype`, `policy` to `scheduling_policy`, and
`data_parallel_master_ip` to `data_parallel_address`. A field added to both an
upstream concern config and `EngineArgs` therefore enters the AR/generation
projection without requiring a second downstream allowlist. Explicit inherited
fields that have no projection raise an error rather than being silently
dropped.

Ownership exclusions remain deliberate. Stage topology owns `scheduler_cls`,
cache owns `disable_hybrid_kv_cache_manager`, and Omni runtime owns
`distributed_executor_backend` and `worker_cls`; vLLM's private API-process
fields are terminal internals. These exclusions prevent one input from being
constructed or projected through two config concerns. Effective-engine-argument
tests cover both the dynamic mapping and this exclusion boundary.

`CompilationConfig` and `ProfilerConfig` are direct structured-stage fields:
`stage.compilation_config` and `stage.profiler_config`. They accept mapping/YAML
inputs and complete upstream value objects. Mapping inputs are validated and
materialized as their concrete upstream types when the structured config is
constructed, so downstream consumers see one resolved representation.
Prebuilt upstream objects retain their type across the structured projection
boundary.

Quantization remains on its existing Omni-owned transport contract. This
reuse step does not adopt upstream `QuantizationConfigArgs`, because the
current engine-specific quantization split needs to be resolved separately.

Diffusion keeps its existing engine-field ownership and projection surface.
Shared Python inheritance must not be treated as evidence that an LLM-only
parallel input is effective for a diffusion stage.

## Safe-change guide while deferred

Changes should trace every affected producer to its runtime consumer and test
the structured, legacy, CLI, and deployment paths that are actually supported.
For inherited vLLM fields, tests should separately cover constructor behavior
and effective `EngineArgs` projection. Do not infer a stable contract from
this placeholder page.

## Promotion gate

- Finish or stabilize the configuration refactor and identify the canonical
  runtime configuration representation.
- Confirm technical owners, primary path exceptions, and validation owners.
- Document parsing and override precedence, including when environment values
  are captured.
- Allocate an invariant namespace only after those behaviors are enforced and
  owner-approved.
