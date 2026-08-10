---
title: Diffusion Parallelism
kind: module
status: draft
owners:
  - "@princepride"
  - "@yuanheng-zhao"
  - "@xuechendi"
primary_code_paths:
  - vllm_omni/diffusion/distributed/**
  - vllm_omni/diffusion/attention/parallel/**
related_code_paths:
  - vllm_omni/distributed/**
  - vllm_omni/config/composable_parallel/**
depends_on:
  - diffusion_runtime.md
  - ../vllm_omni_config.md
validation_paths:
  - tests/diffusion/distributed/**
  - tests/diffusion/attention/**
upstream_refs:
  - torch.distributed
last_reviewed: 2026-07-16
---

# Diffusion parallelism

Parallelism owns diffusion rank topology, process groups, tensor and sequence
sharding, collectives, and distributed execution.

## Candidate invariants

### PARALLEL-INV-001: Topology has one source of truth

**Rule:** Rank coordinates, group membership, and parallel dimensions MUST be
derived from validated configuration.

### PARALLEL-INV-002: Sharding contracts are explicit

**Rule:** Every distributed boundary MUST define tensor shape, sharded
dimension, placement, dtype, producer, and consumer.

### PARALLEL-INV-003: Collectives are symmetric

**Rule:** Process-group members MUST invoke compatible collectives in the same
logical order.

### PARALLEL-INV-004: Single-rank behavior remains valid

**Rule:** Distributed implementations SHOULD preserve a supported single-rank
path without distributed initialization.

## Safe-change guide

Test topology validation, single-rank execution, affected parallel modes,
combined modes, invalid shapes, and orderly teardown.
