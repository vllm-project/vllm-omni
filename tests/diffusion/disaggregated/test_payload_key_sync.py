# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard: every consumer of the cross-stage payload keys agrees with the canonical source.

The canonical definitions of ``STAGE_PAYLOAD_OUTPUT_KEY`` / ``STAGE_PAYLOAD_PROMPT_KEY``
live in ``vllm_omni.diffusion.stage_payload`` (the torch-free transport leaf). The
torch-heavy consumers (``diffusion/worker/diffusion_model_runner.py`` and
``experimental/ar_diffusion/runner.py``) import them directly, so they cannot drift.

The generic transition processor (``model_executor/stage_input_processors/diffusion.py``)
is the one exception: it is loaded by file path in the torch-free foundation tests
(see ``conftest.py``), so it cannot ``from vllm_omni...`` import at module scope without
pulling in the package ``__init__`` (and torch). It therefore re-declares the two keys as
literals. This test parses both source files (no torch import) and asserts the processor's
literals still equal the canonical constants, so that one hand-copy cannot drift.
"""

from __future__ import annotations

import ast
from pathlib import Path

VLLM_OMNI_ROOT = Path(__file__).resolve().parents[3] / "vllm_omni"

_CANONICAL = VLLM_OMNI_ROOT / "diffusion" / "stage_payload.py"
_PROCESSOR = VLLM_OMNI_ROOT / "model_executor" / "stage_input_processors" / "diffusion.py"

_KEY_NAMES = {"STAGE_PAYLOAD_OUTPUT_KEY", "STAGE_PAYLOAD_PROMPT_KEY"}

_INTERFACE = VLLM_OMNI_ROOT / "diffusion" / "models" / "interface.py"


def _module_string_constants(path: Path, names: set[str]) -> dict[str, str]:
    """Extract top-level ``NAME = "literal"`` assignments for the given names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names and isinstance(node.value, ast.Constant):
                found[target.id] = node.value.value
    return found


def _module_int_constant(path: Path, name: str) -> int | None:
    """Extract a top-level ``NAME = <int>`` assignment (no import needed)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id == name
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
            ):
                return node.value.value
    return None


def test_schema_version_constants_stay_in_lockstep():
    """The payload schema version is duplicated in the torch-free leaf and in
    interface.py (interface must stay importable without the leaf). They MUST be
    equal — a drift would let one side accept a payload the other rejects. Parsed
    from source (no torch) so this runs anywhere.
    """
    leaf = _module_int_constant(_CANONICAL, "DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION")
    iface = _module_int_constant(_INTERFACE, "STAGE_PAYLOAD_SCHEMA_VERSION")
    assert leaf is not None, "stage_payload schema-version constant not found"
    assert iface is not None, "interface schema-version constant not found"
    assert leaf == iface, f"schema-version drift: stage_payload={leaf} interface={iface}"


def test_processor_literals_match_canonical_source():
    canonical = _module_string_constants(_CANONICAL, _KEY_NAMES)
    processor = _module_string_constants(_PROCESSOR, _KEY_NAMES)
    assert canonical == {
        "STAGE_PAYLOAD_OUTPUT_KEY": "__diffusion_stage_payload__",
        "STAGE_PAYLOAD_PROMPT_KEY": "diffusion_stage_payload",
    }, f"canonical stage_payload keys changed unexpectedly: {canonical}"
    assert processor == canonical, f"payload key drift: processor={processor} canonical={canonical}"


def test_torch_heavy_consumers_do_not_redeclare_the_keys():
    """The runner and AR-Diffusion runner must import the keys, not re-declare them.

    A local re-declaration would reintroduce exactly the drift this single-source-of-truth
    refactor removed, so assert those two files carry no top-level literal copy.
    """
    for rel in (
        Path("diffusion") / "worker" / "diffusion_model_runner.py",
        Path("experimental") / "ar_diffusion" / "runner.py",
    ):
        redeclared = _module_string_constants(VLLM_OMNI_ROOT / rel, _KEY_NAMES)
        assert redeclared == {}, f"{rel} re-declares payload keys instead of importing them: {redeclared}"
