# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the lazy import of the optional Mooncake connectors.

``mooncake`` is only needed when a stage selects a Mooncake connector, but an
eager re-export in ``vllm_omni.distributed.omni_connectors`` used to load it in
every process that imported ``vllm_omni`` -- including the short-lived
subprocesses vLLM spawns to inspect model architectures, where a broken
``mooncake`` build aborts the interpreter and surfaces as
``Model architectures [...] failed to be inspected``.
"""

import subprocess
import sys
import textwrap

import pytest

import vllm_omni.distributed as distributed
from vllm_omni.distributed import omni_connectors

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LAZY_NAMES = (
    "MooncakeConnector",
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_import_vllm_omni_does_not_load_mooncake():
    proc = _run(
        """
        import sys

        import vllm_omni

        leaked = [
            m
            for m in sys.modules
            if m == "mooncake"
            or m.startswith("mooncake.")
            or m.endswith((".mooncake_store_connector", ".mooncake_transfer_engine_connector"))
        ]
        assert not leaked, f"importing vllm_omni pulled in a Mooncake connector: {leaked}"
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_import_distributed_does_not_load_mooncake():
    proc = _run(
        """
        import sys

        import vllm_omni.distributed

        leaked = [
            m
            for m in sys.modules
            if m == "mooncake"
            or m.startswith("mooncake.")
            or m.endswith((".mooncake_store_connector", ".mooncake_transfer_engine_connector"))
        ]
        assert not leaked, f"importing vllm_omni.distributed pulled in a Mooncake connector: {leaked}"
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_mooncake_connector_alias_still_resolves():
    proc = _run(
        """
        from vllm_omni.distributed import MooncakeConnector, MooncakeStoreConnector

        assert MooncakeConnector is MooncakeStoreConnector
        print("OK")
        """
    )
    if proc.returncode < 0 or proc.returncode == 134:
        pytest.skip(f"native mooncake build aborts on import here (rc={proc.returncode})")
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_accessing_a_connector_does_import_its_module():
    """Negative control: the leak assertion above is not vacuous.

    Accessing a lazy name must pull its module into ``sys.modules``, so a
    regression that re-adds an eager import would show up in the leak list even
    where ``mooncake`` itself is not installed (its own modules swallow the
    ``ImportError`` and degrade to ``None``).
    """
    proc = _run(
        """
        import sys

        from vllm_omni.distributed import MooncakeStoreConnector

        name = "vllm_omni.distributed.omni_connectors.connectors.mooncake_store_connector"
        assert name in sys.modules, f"{MooncakeStoreConnector!r} did not import {name}"
        print("OK")
        """
    )
    if proc.returncode < 0 or proc.returncode == 134:
        pytest.skip(f"native mooncake build aborts on import here (rc={proc.returncode})")
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_dir_lists_lazy_mooncake_names():
    for name in _LAZY_NAMES:
        assert name in dir(distributed), f"{name} missing from dir(vllm_omni.distributed)"
        assert name in dir(omni_connectors), f"{name} missing from dir(vllm_omni.distributed.omni_connectors)"
