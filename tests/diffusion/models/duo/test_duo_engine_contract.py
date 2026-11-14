# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duo must only use duplex-engine names that the shared engine declares.

The ported Duo code drives the shared duplex engine, so every kwarg it passes to
an engine dataclass and every base method it overrides has to exist on `main`.
This went wrong once: the plugin passed `supports_output_video=` and
`forward=` to dataclasses that did not declare those fields, and called an
orchestrator method that did not exist, so the duplex path raised `TypeError` /
`AttributeError` on the first chunk. Keyword-argument shape is not checkable by
importing the module, hence the explicit checks below.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, PartialStageForward
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator, DuplexOrchestratorRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DUO_ROOT = Path("vllm_omni/model_executor/models/duo")


def _call_keywords(source: str, callee: str) -> set[str]:
    """Collect every keyword argument used at any call site of `callee`."""
    keywords: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == callee:
                keywords.update(kw.arg for kw in node.keywords if kw.arg)
    return keywords


def _duo_sources() -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in sorted(DUO_ROOT.rglob("*.py"))}


def test_duo_engine_dataclass_kwargs_are_declared():
    # Every kwarg Duo passes to a duplex-engine dataclass must be declared by it,
    # otherwise construction raises TypeError on the live path.
    targets = {"DuplexCapabilities": DuplexCapabilities, "PartialStageForward": PartialStageForward}
    sources = _duo_sources().values()
    for callee, dataclass_type in targets.items():
        used: set[str] = set()
        for source in sources:
            used |= _call_keywords(source, callee)
        assert used, f"expected Duo to construct {callee}"
        declared = {field.name for field in dataclasses.fields(dataclass_type)}
        undeclared = used - declared
        assert not undeclared, (
            f"Duo passes kwargs that {callee} does not declare: {sorted(undeclared)}. Declared: {sorted(declared)}"
        )


def test_partial_stage_forward_exposes_the_forward_gate():
    # The plugin marks settle-only updates with forward=False, and the engine
    # only skips forwarding because it reads this field.
    assert dataclasses.fields(PartialStageForward)
    assert "forward" in {f.name for f in dataclasses.fields(PartialStageForward)}
    assert PartialStageForward(output=object(), is_final_update=True).forward is True


def test_duo_duplex_engine_members_exist():
    # Every self.<member> / orchestrator.<member> the plugin uses on the engine
    # side has to be a real attribute of the class that owns it.
    plugin_source = (DUO_ROOT / "duplex" / "plugin.py").read_text(encoding="utf-8")
    for member in ("fork_stage_request",):
        assert hasattr(DuplexOrchestrator, member), (
            f"Duo duplex plugin calls orchestrator.{member}(), which DuplexOrchestrator does not define"
        )
    assert "parent_request_id" in {f.name for f in dataclasses.fields(DuplexOrchestratorRequestState)}

    # Duo must not override a hook name that the base plugin never declared: the
    # engine can only call what it knows about, so such an override is dead code
    # that misleads readers into believing the behaviour is wired.
    duo_plugin_class = next(
        node
        for node in ast.walk(ast.parse(plugin_source))
        if isinstance(node, ast.ClassDef) and node.name == "DuoDuplexPlugin"
    )
    overridden = {
        node.name
        for node in duo_plugin_class.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert overridden, "expected DuoDuplexPlugin to override base hooks"

    # Hooks that upstream relies on but `main` never calls are tracked in
    # docs/models/duo.md rather than asserted here: they are inert, not broken.
    # Duo must piggyback on existing rollback/lease hooks instead.
    inert = {"stage_is_resumable", "model_state_session_key", "model_state_stage_ids"}
    inert |= {"start_transcription_preparation", "record_stage_timing"}
    unknown = overridden - set(dir(DuplexModelPlugin)) - inert
    assert not unknown, (
        f"DuoDuplexPlugin overrides public hooks the base DuplexModelPlugin does not declare: "
        f"{sorted(unknown)}. The engine has no call site for them, so they are dead code."
    )
