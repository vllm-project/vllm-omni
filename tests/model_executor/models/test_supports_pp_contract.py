# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Every model declaring ``SupportsPP`` must supply ``make_empty_intermediate_tensors``.

vLLM 0.28 turned ``SupportsPP.make_empty_intermediate_tensors`` from a method on
the Protocol class into a bare annotation, so declaring ``SupportsPP`` no longer
supplies one. A class that declares the interface without providing the
attribute is advertised as pipeline-parallel capable by
``vllm.model_executor.models.interfaces.supports_pp`` and then raises
``AttributeError`` wherever the attribute is read (#6790, #6859).

The check is static: models import torch and vLLM layers, so building them is far
too heavy for a unit test, and the repo-wide convention is to assign the
attribute on the instance inside ``__init__`` (mirroring upstream
``Qwen2ForCausalLM``), which no class-level ``hasattr`` would see.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model]

MODELS_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "models"
ATTR = "make_empty_intermediate_tensors"


def _declares_supports_pp(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "SupportsPP":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "SupportsPP":
            return True
    return False


def _provides_attr(node: ast.ClassDef) -> bool:
    """True if the class defines the attribute or binds it on the class or ``self``.

    An annotation without a value (``make_empty_intermediate_tensors: Callable``)
    binds nothing at runtime -- that is exactly how the attribute went missing in
    vLLM 0.28 -- so only an ``AnnAssign`` carrying a value counts.
    """
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == ATTR:
            return True
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Attribute) and target.attr == ATTR:
                    return True
                if isinstance(target, ast.Name) and target.id == ATTR:
                    return True
        if isinstance(child, ast.AnnAssign) and child.value is not None:
            target = child.target
            if isinstance(target, ast.Attribute) and target.attr == ATTR:
                return True
            if isinstance(target, ast.Name) and target.id == ATTR:
                return True
    return False


def _supports_pp_classes() -> list[tuple[str, str, bool]]:
    """Return ``(relative_path, class_name, provides_attr)`` for every SupportsPP class."""
    found: list[tuple[str, str, bool]] = []
    for path in sorted(MODELS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - vendored files are still valid python
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _declares_supports_pp(node):
                found.append(
                    (
                        str(path.relative_to(MODELS_DIR.parents[2])),
                        node.name,
                        _provides_attr(node),
                    )
                )
    return found


@pytest.mark.cpu
def test_supports_pp_classes_provide_make_empty_intermediate_tensors():
    classes = _supports_pp_classes()
    assert classes, f"no SupportsPP classes found under {MODELS_DIR}; the scan is broken"

    missing = [f"{path}::{name}" for path, name, provides in classes if not provides]
    assert not missing, (
        "These classes declare SupportsPP but never provide "
        f"`{ATTR}`, so vLLM advertises them as pipeline-parallel capable and any "
        "read of the attribute raises AttributeError. Either assign it (usually "
        "`self." + ATTR + " = self.model." + ATTR + "` after building the inner "
        "model) or drop SupportsPP if the model does not implement pipeline "
        "parallelism:\n  " + "\n  ".join(missing)
    )


BINDS_ATTR = [
    pytest.param("def " + ATTR + "(self, *a, **kw): ...", id="method"),
    pytest.param("def __init__(self):\n        self." + ATTR + " = self.model." + ATTR, id="self-assign"),
    pytest.param(
        "def __init__(self):\n        self." + ATTR + ": Callable = self.model." + ATTR,
        id="self-annotated-assign",
    ),
    pytest.param(ATTR + " = staticmethod(_make_empty)", id="class-assign"),
    pytest.param(ATTR + ": Callable = staticmethod(_make_empty)", id="class-annotated-assign"),
]

BINDS_NOTHING = [
    pytest.param(ATTR + ": Callable", id="bare-class-annotation"),
    pytest.param("def __init__(self):\n        self." + ATTR + ": Callable", id="bare-self-annotation"),
    pytest.param("def forward(self, intermediate_tensors=None): ...", id="unrelated-method"),
]


def _parse_class(body: str) -> ast.ClassDef:
    node = ast.parse(f"class M(nn.Module, SupportsPP):\n    {body}\n").body[0]
    assert isinstance(node, ast.ClassDef) and _declares_supports_pp(node)
    return node


@pytest.mark.cpu
@pytest.mark.parametrize("body", BINDS_ATTR)
def test_provides_attr_accepts_real_bindings(body):
    assert _provides_attr(_parse_class(body))


@pytest.mark.cpu
@pytest.mark.parametrize("body", BINDS_NOTHING)
def test_provides_attr_rejects_bare_annotations(body):
    assert not _provides_attr(_parse_class(body))
