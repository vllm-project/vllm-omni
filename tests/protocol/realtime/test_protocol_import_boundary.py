# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The shared Realtime codec must not depend on any runtime that uses it.

``vllm_omni.protocol.realtime`` exists so a second Realtime surface can reuse
the wire codec without adopting the duplex session control plane. That only
holds while the arrows point one way: the engine, the entrypoints and the model
code import the protocol, never the reverse. Nothing in the package's own code
enforces that, so it is asserted here --- once on the source (which catches a
lazy import inside a function body) and once on a real import (which catches a
dependency arriving through a third module).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_DIR = REPO_ROOT / "vllm_omni" / "protocol"

#: Packages that consume the protocol. A dependency on any of them would make
#: the codec unusable outside the runtime it accidentally reached into.
#: ``vllm_omni.utils`` is deliberately absent: it is leaf helpers (the audio
#: resampler) that consume nothing, so depending on it costs a reuser nothing.
FORBIDDEN_PREFIXES = (
    "vllm_omni.engine",
    "vllm_omni.entrypoints",
    "vllm_omni.model_executor",
    "vllm_omni.worker",
    "vllm_omni.clients",
)


def _protocol_sources() -> list[Path]:
    sources = sorted(PROTOCOL_DIR.rglob("*.py"))
    assert sources, f"no python sources under {PROTOCOL_DIR}"
    return sources


def _imported_modules(source: Path) -> set[str]:
    """Every module name imported anywhere in the file, lazy imports included."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("source", _protocol_sources(), ids=lambda path: path.name)
def test_protocol_sources_do_not_import_a_consumer(source: Path) -> None:
    offenders = sorted(
        name
        for name in _imported_modules(source)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    )
    assert not offenders, f"{source.relative_to(REPO_ROOT)} imports {', '.join(offenders)}"


def test_importing_the_protocol_package_loads_no_consumer() -> None:
    # A child interpreter, because this test session has already imported the
    # duplex engine: only a fresh process can tell what the package pulls in.
    # ``vllm_omni`` itself is imported first, because its ``__init__`` loads a
    # baseline graph (patches, the model registry) that has nothing to do with
    # this package; only what the protocol import *adds* is measured.
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    script = f"""
import sys

import vllm_omni  # noqa: F401

baseline = set(sys.modules)

import vllm_omni.protocol.realtime  # noqa: F401

forbidden_prefixes = {FORBIDDEN_PREFIXES!r}
added = sorted(
    name
    for name in set(sys.modules) - baseline
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
)
if added:
    raise SystemExit("the realtime protocol package loaded consumers: " + ", ".join(added))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
