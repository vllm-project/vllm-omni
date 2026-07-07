# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm_omni.engine.arg_utils — invariants that must
hold for the orchestrator/engine/server CLI flag partition."""

from __future__ import annotations

from dataclasses import fields

import pytest

from vllm_omni.engine.arg_utils import (
    SHARED_FIELDS,
    OmniEngineArgs,
    internal_blacklist_keys,
    orchestrator_field_names,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_no_ambiguous_overlap_with_real_engine():
    """OrchestratorArgs ∩ OmniEngineArgs must be ⊆ SHARED_FIELDS ∪ orchestrator-captured.

    Fields that the orchestrator intentionally captures to prevent uniform
    propagation (async_chunk, tokenizer) are excluded — they exist in both
    classes by design but flow through DeployConfig, not kwargs forwarding.
    """
    # Fields on both classes by design: orchestrator captures them to prevent
    # uniform per-stage propagation; redistributed via DeployConfig.
    orchestrator_captured = {"async_chunk", "tokenizer"}

    orch = orchestrator_field_names()
    engine = {f.name for f in fields(OmniEngineArgs)}
    overlap = orch & engine
    unexpected = overlap - SHARED_FIELDS - orchestrator_captured
    assert not unexpected, (
        f"OmniEngineArgs has ambiguous overlap with OrchestratorArgs: "
        f"{sorted(unexpected)}. Update SHARED_FIELDS or remove duplication."
    )


def test_internal_blacklist_keys_derived_from_orchestrator():
    """Blacklist is exactly OrchestratorArgs fields minus SHARED_FIELDS.

    This function replaces the old hardcoded INTERNAL_STAGE_OVERRIDE_KEYS
    frozenset. Asserts the contract so future changes to OrchestratorArgs
    automatically propagate to the blacklist.
    """
    blacklist = internal_blacklist_keys()
    assert blacklist == orchestrator_field_names() - SHARED_FIELDS
