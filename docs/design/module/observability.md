---
title: Observability
kind: module
status: draft
owners:
  - "@lishunyang12"
primary_code_paths:
  - vllm_omni/metrics/**
related_code_paths:
  - vllm_omni/engine/**
  - vllm_omni/entrypoints/**
depends_on:
  - engine_orchestration.md
validation_paths:
  - tests/metrics/**
upstream_refs:
  - vllm.v1.metrics
last_reviewed: 2026-07-16
---

# Observability

Observability defines stable metrics, logs, and correlation fields for tracing a
request across entrypoints, orchestration, stages, and workers.

## Candidate invariants

### OBS-INV-001: Request correlation survives stage boundaries

**Rule:** Logs and metrics for the same request SHOULD carry a stable request
identifier and stage identity.

### OBS-INV-002: Metric meaning is stable

**Rule:** A metric's unit, labels, aggregation, and lifecycle MUST be documented;
changing any of them is a contract change.

### OBS-INV-003: Labels are bounded

**Rule:** Metrics MUST NOT use request IDs, prompts, outputs, or other unbounded
values as labels.

## Safe-change guide

Test registration, label sets, units, lifecycle resets, and representative
multi-stage emission.
