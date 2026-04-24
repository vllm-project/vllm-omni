# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm_omni.engine.arg_utils — invariants that must
hold for the orchestrator/engine/server CLI flag partition."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import pytest

from vllm_omni.engine.arg_utils import (
    SHARED_FIELDS,
    derive_server_dests_from_vllm_parser,
    internal_blacklist_keys,
    orchestrator_args_from_argparse,
    orchestrator_field_names,
    split_kwargs,
)

# ---------------------------------------------------------------------------
# Fake engine class for unit testing — avoids pulling in the full vllm
# EngineArgs and its heavy __post_init__ at test time.
# ---------------------------------------------------------------------------


@dataclass
class _FakeEngineArgs:
    """Stand-in for OmniEngineArgs with a representative subset of fields."""

    model: str = ""
    stage_id: int = 0
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.9
    async_chunk: bool = False  # also in OrchestratorArgs → shared
    log_stats: bool = False  # also in OrchestratorArgs → shared
    stage_configs_path: str | None = None


# ============================================================================
# Invariant 1 — OrchestratorArgs and engine must not ambiguously overlap.
# ============================================================================


def test_no_ambiguous_overlap_with_fake_engine():
    """OrchestratorArgs ∩ engine fields must be ⊆ SHARED_FIELDS."""
    orch = orchestrator_field_names()
    engine = {f.name for f in fields(_FakeEngineArgs)}
    overlap = orch & engine
    unexpected = overlap - SHARED_FIELDS
    assert not unexpected, (
        f"Fields declared in both OrchestratorArgs and the engine class "
        f"but not in SHARED_FIELDS: {sorted(unexpected)}. These cause "
        f"double-routing — either remove the duplicate declaration or add "
        f"to SHARED_FIELDS if sharing is intentional."
    )


def test_no_ambiguous_overlap_with_real_engine():
    """Same check, but against the real OmniEngineArgs."""
    try:
        from vllm_omni.engine.arg_utils import OmniEngineArgs
    except Exception as exc:
        pytest.skip(f"OmniEngineArgs not importable: {exc}")

    orch = orchestrator_field_names()
    engine = {f.name for f in fields(OmniEngineArgs)}
    overlap = orch & engine
    unexpected = overlap - SHARED_FIELDS
    assert not unexpected, (
        f"Real OmniEngineArgs has ambiguous overlap with OrchestratorArgs: "
        f"{sorted(unexpected)}. Update SHARED_FIELDS or remove duplication."
    )


# ============================================================================
# Invariant 2 — split_kwargs partitions correctly.
# ============================================================================


def test_split_orchestrator_only():
    """Pure orchestrator fields go to OrchestratorArgs, not engine_kwargs."""
    raw = {"stage_init_timeout": 500, "worker_backend": "ray"}
    orch, engine = split_kwargs(raw, engine_cls=_FakeEngineArgs)
    assert orch.stage_init_timeout == 500
    assert orch.worker_backend == "ray"
    assert "stage_init_timeout" not in engine
    assert "worker_backend" not in engine


def test_split_engine_only():
    """Pure engine fields go to engine_kwargs, not OrchestratorArgs."""
    raw = {"max_num_seqs": 128, "gpu_memory_utilization": 0.85}
    orch, engine = split_kwargs(raw, engine_cls=_FakeEngineArgs)
    assert engine["max_num_seqs"] == 128
    assert engine["gpu_memory_utilization"] == 0.85
    # These fields don't exist on OrchestratorArgs at all.


def test_split_shared_fields_go_to_both():
    """Fields in SHARED_FIELDS are copied to both buckets."""
    raw = {"model": "Qwen/Qwen2.5-Omni-7B", "log_stats": True}
    orch, engine = split_kwargs(raw, engine_cls=_FakeEngineArgs)
    assert orch.log_stats is True
    assert engine["model"] == "Qwen/Qwen2.5-Omni-7B"
    assert engine["log_stats"] is True


def test_split_drops_unclassified():
    """Unclassified fields (uvicorn/server) are dropped silently."""
    raw = {
        "max_num_seqs": 64,  # engine
        "host": "0.0.0.0",  # unclassified (server)
        "port": 8091,  # unclassified (server)
        "ssl_keyfile": "key.pem",  # unclassified (server)
    }
    orch, engine = split_kwargs(raw, engine_cls=_FakeEngineArgs)
    assert engine == {"max_num_seqs": 64}
    assert "host" not in engine
    assert "port" not in engine
    assert "ssl_keyfile" not in engine


def test_split_mixed_real_world():
    """End-to-end: raw CLI kwargs with all three classes present."""
    raw = {
        # orchestrator
        "stage_init_timeout": 400,
        "deploy_config": "/tmp/deploy.yaml",
        "worker_backend": "multi_process",
        "async_chunk": True,
        # engine
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.8,
        # shared
        "model": "Qwen/Qwen3-Omni",
        "log_stats": False,
        # server / unclassified
        "host": "0.0.0.0",
        "port": 8091,
        "api_key": "secret",
        # None values
        "ray_address": None,
    }
    orch, engine = split_kwargs(raw, engine_cls=_FakeEngineArgs)

    # Orchestrator side
    assert orch.stage_init_timeout == 400
    assert orch.deploy_config == "/tmp/deploy.yaml"
    assert orch.worker_backend == "multi_process"
    assert orch.async_chunk is True
    assert orch.log_stats is False  # shared, read from raw
    assert orch.ray_address is None  # default preserved

    # Engine side
    assert engine["max_num_seqs"] == 32
    assert engine["gpu_memory_utilization"] == 0.8
    assert engine["model"] == "Qwen/Qwen3-Omni"
    assert engine["log_stats"] is False
    assert "host" not in engine
    assert "port" not in engine
    assert "api_key" not in engine
    # orchestrator-only keys never reach engine
    assert "stage_init_timeout" not in engine
    assert "deploy_config" not in engine
    assert "async_chunk" not in engine


# ============================================================================
# Invariant 3 — user-typed unclassifiable flags warn (don't fail silently).
# ============================================================================


def test_user_typed_unclassified_warns(caplog):
    """If the user types a flag we can't route, warn — don't silently drop."""
    raw = {"bogus_flag": "value", "max_num_seqs": 64}
    with caplog.at_level(logging.WARNING, logger="vllm_omni.engine.arg_utils"):
        split_kwargs(raw, engine_cls=_FakeEngineArgs, user_typed={"bogus_flag"})
    assert any("bogus_flag" in rec.message for rec in caplog.records), (
        f"Expected warning mentioning 'bogus_flag', got: {[rec.message for rec in caplog.records]}"
    )


def test_unclassified_without_user_typed_silent(caplog):
    """Without user_typed, unclassified keys drop silently (argparse defaults
    for server flags shouldn't spam logs on every launch)."""
    raw = {"host": "0.0.0.0", "port": 8091, "max_num_seqs": 64}
    with caplog.at_level(logging.WARNING, logger="vllm_omni.engine.arg_utils"):
        split_kwargs(raw, engine_cls=_FakeEngineArgs, user_typed=None)
    # No warnings because we don't know these were user-typed.
    assert not any("host" in rec.message or "port" in rec.message for rec in caplog.records)


# ============================================================================
# Invariant 4 — CLI flag classification completeness.
# Catches new flags added without updating OrchestratorArgs or OmniEngineArgs.
# ============================================================================


def test_all_omni_cli_flags_classified():
    """Every vllm-omni-added CLI flag must be classifiable.

    Runs ``OmniServeCommand.subparser_init`` and checks that every new
    argument (compared to vllm's base parser) is either:
      - a field on OrchestratorArgs, OR
      - a field on OmniEngineArgs, OR
      - in SHARED_FIELDS
    """
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    except Exception as exc:
        pytest.skip(f"Cannot build parser in this environment: {exc}")

    # Build the serve parser
    root = FlexibleArgumentParser()
    subparsers = root.add_subparsers()
    cmd = OmniServeCommand()
    try:
        parser = cmd.subparser_init(subparsers)
    except Exception as exc:
        pytest.skip(f"subparser_init failed (dev env issue): {exc}")

    all_dests = {a.dest for a in parser._actions if a.dest and a.dest not in {"help", "model_tag"}}

    orch = orchestrator_field_names()
    engine = {f.name for f in fields(OmniEngineArgs)}
    server_derived = derive_server_dests_from_vllm_parser()

    unclassified = all_dests - orch - engine - SHARED_FIELDS - server_derived
    # Some argparse-internal dests (suppressed, private) may not match —
    # filter those out.
    unclassified = {d for d in unclassified if not d.startswith("_")}

    assert not unclassified, (
        f"These CLI flags are not classified as "
        f"orchestrator/engine/shared/server: {sorted(unclassified)}. "
        f"Add them to OrchestratorArgs (if consumed by orchestrator), "
        f"OmniEngineArgs (if consumed by per-stage engine), or the known-server "
        f"allowlist (if they're vllm frontend flags). "
        f"If intentional (e.g. a new CLI-only flag that doesn't map to either "
        f"dataclass), add it to a KNOWN_UNROUTED allowlist."
    )


# ============================================================================
# argparse interop (Phase 3).
# ============================================================================


def test_orchestrator_args_from_argparse():
    """Can build OrchestratorArgs from an argparse.Namespace."""
    import argparse

    ns = argparse.Namespace(
        stage_init_timeout=500,
        deploy_config="/tmp/x.yaml",
        max_num_seqs=64,  # engine field — ignored
        host="0.0.0.0",  # server field — ignored
    )
    orch = orchestrator_args_from_argparse(ns)
    assert orch.stage_init_timeout == 500
    assert orch.deploy_config == "/tmp/x.yaml"
    assert orch.worker_backend == "multi_process"  # default


def test_derive_server_dests_returns_frozenset():
    """Server-dest derivation returns a frozenset (possibly empty)."""
    result = derive_server_dests_from_vllm_parser()
    assert isinstance(result, frozenset)


# ============================================================================
# internal_blacklist_keys — single source of truth for per-stage forwarding.
# ============================================================================


def test_internal_blacklist_keys_derived_from_orchestrator():
    """Blacklist is exactly OrchestratorArgs fields minus SHARED_FIELDS.

    This function replaces the old hardcoded INTERNAL_STAGE_OVERRIDE_KEYS
    frozenset. Asserts the contract so future changes to OrchestratorArgs
    automatically propagate to the blacklist.
    """
    blacklist = internal_blacklist_keys()
    assert blacklist == orchestrator_field_names() - SHARED_FIELDS
    # Spot-check expected entries
    assert "stage_init_timeout" in blacklist
    assert "deploy_config" in blacklist
    assert "async_chunk" in blacklist
    # Shared fields must NOT appear — they flow to both orchestrator and engine
    assert "model" not in blacklist
    assert "log_stats" not in blacklist


# ============================================================================
# Boundary value analysis — edge cases around split_kwargs.
# ============================================================================


def test_split_empty_kwargs():
    """Empty kwargs yields default OrchestratorArgs and empty engine dict."""
    orch, engine = split_kwargs({}, engine_cls=_FakeEngineArgs)
    assert orch.stage_init_timeout == 300  # dataclass default
    assert orch.worker_backend == "multi_process"  # dataclass default
    assert engine == {}


def test_split_all_none_values_preserved_on_orchestrator():
    """None values for orchestrator fields are kept (represents 'not set')."""
    raw = {"ray_address": None, "deploy_config": None, "max_num_seqs": None}
    orch, engine = split_kwargs(raw, engine_cls=_FakeEngineArgs)
    assert orch.ray_address is None
    assert orch.deploy_config is None
    # Engine-side None still passes through; caller decides semantics downstream.
    assert engine.get("max_num_seqs") is None


def test_split_user_typed_with_empty_kwargs_no_warn(caplog):
    """user_typed non-empty but kwargs empty — no warnings emitted."""
    with caplog.at_level(logging.WARNING, logger="vllm_omni.engine.arg_utils"):
        split_kwargs({}, engine_cls=_FakeEngineArgs, user_typed={"nothing"})
    assert not caplog.records


def test_ambiguous_field_strict_raises():
    """strict=True raises ValueError on overlap outside SHARED_FIELDS."""

    # deploy_config is on OrchestratorArgs; declaring it on the engine class
    # too (without adding to SHARED_FIELDS) creates an ambiguous route.
    @dataclass
    class _AmbiguousEngine:
        deploy_config: str | None = None

    with pytest.raises(ValueError, match="both OrchestratorArgs and"):
        split_kwargs({"deploy_config": "x"}, engine_cls=_AmbiguousEngine, strict=True)


def test_ambiguous_field_non_strict_routes_to_orchestrator(caplog):
    """strict=False logs ERROR but routes the ambiguous field to orchestrator."""

    @dataclass
    class _AmbiguousEngine:
        deploy_config: str | None = None

    with caplog.at_level(logging.ERROR, logger="vllm_omni.engine.arg_utils"):
        orch, engine = split_kwargs({"deploy_config": "x"}, engine_cls=_AmbiguousEngine, strict=False)
    assert orch.deploy_config == "x"
    assert "deploy_config" not in engine
    assert any("both OrchestratorArgs" in r.message for r in caplog.records)
