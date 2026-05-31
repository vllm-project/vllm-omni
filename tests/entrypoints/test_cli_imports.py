import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cli_package_import_does_not_load_benchmark_patch():
    script = textwrap.dedent(
        """
        import importlib
        import sys

        importlib.import_module("vllm_omni.entrypoints.cli")

        forbidden = sorted(
            name
            for name in sys.modules
            if name.startswith("vllm.benchmarks")
            or name == "vllm_omni.benchmarks.patch"
            or name.startswith("vllm_omni.benchmarks.patch.")
        )
        assert forbidden == [], forbidden
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
