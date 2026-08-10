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
last_reviewed: 2026-08-07
last_verified_commit: 3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc
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

## Safe-change guide while deferred

Changes should trace every affected producer to its runtime consumer and test
the structured, legacy, CLI, and deployment paths that are actually supported.
Do not infer a stable contract from this placeholder page.

## Promotion gate

- Finish or stabilize the configuration refactor and identify the canonical
  runtime configuration representation.
- Confirm technical owners, primary path exceptions, and validation owners.
- Document parsing and override precedence, including when environment values
  are captured.
- Allocate an invariant namespace only after those behaviors are enforced and
  owner-approved.
