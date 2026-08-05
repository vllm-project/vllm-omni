# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_isolated_import_succeeds(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_stable_engine_imports_do_not_load_experimental_duplex() -> None:
    _assert_isolated_import_succeeds("""
import sys

import vllm_omni.engine.async_omni_engine
import vllm_omni.engine.orchestrator
import vllm_omni.entrypoints.async_omni

loaded = sorted(
    name
    for name in sys.modules
    if name == "vllm_omni.experimental.fullduplex"
    or name.startswith("vllm_omni.experimental.fullduplex.")
)
if loaded:
    raise SystemExit("stable imports loaded experimental duplex: " + ", ".join(loaded))
""")


def test_stable_engine_does_not_expose_duplex_contract_modules() -> None:
    _assert_isolated_import_succeeds("""
import importlib.util

from vllm_omni.engine import messages

legacy_modules = (
    "vllm_omni.engine.duplex_contracts",
    "vllm_omni.engine.duplex_lease",
    "vllm_omni.engine.resumable",
)
present = [name for name in legacy_modules if importlib.util.find_spec(name) is not None]
if present:
    raise SystemExit("stable duplex modules still exist: " + ", ".join(present))

duplex_exports = sorted(name for name in vars(messages) if name.startswith("Duplex"))
if duplex_exports:
    raise SystemExit("stable messages still expose duplex contracts: " + ", ".join(duplex_exports))
""")


def test_stable_outputs_do_not_declare_duplex_decision_field() -> None:
    _assert_isolated_import_succeeds("""
import dataclasses

from vllm_omni.outputs import OmniRequestOutput

fields = {field.name for field in dataclasses.fields(OmniRequestOutput)}
if "duplex_output_decision" in fields:
    raise SystemExit("stable output declares duplex_output_decision")
""")


def test_stable_model_executor_does_not_expose_duplex_helper_module() -> None:
    _assert_isolated_import_succeeds("""
import importlib.util

if importlib.util.find_spec("vllm_omni.model_executor.duplex") is not None:
    raise SystemExit("stable model_executor still exposes duplex helper module")
""")


def test_runtime_package_does_not_bundle_the_browser_demo() -> None:
    assert not (REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "web").exists()


def test_experimental_engine_uses_canonical_contract_module_names() -> None:
    engine_dir = REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "engine"
    core_dir = REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "core"

    assert (engine_dir / "contracts.py").is_file()
    assert (engine_dir / "lease.py").is_file()
    assert (engine_dir / "messages.py").is_file()
    assert not (engine_dir / "duplex_lease.py").exists()
    assert not (engine_dir / "duplex_types.py").exists()
    assert not (core_dir / "identity.py").exists()
