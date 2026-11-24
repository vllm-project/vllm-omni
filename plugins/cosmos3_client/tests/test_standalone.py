"""Check the client boundary and bundled resources without loading a model."""

import os
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_client_import_does_not_use_other_checkouts_or_user_libero_config(tmp_path):
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cosmos_framework', 'libero', 'vllm', 'vllm_omni'}:
            raise AssertionError(f'external import: {fullname}')

sys.meta_path.insert(0, BlockExternal())
from vllm_omni_cosmos3_client.closed_loop_eval import ActionEnvironmentClient
from vllm_omni_cosmos3_client._vendor.libero import get_libero_path
root = Path(get_libero_path('benchmark_root'))
assert root.is_relative_to(Path(sys.argv[1]))
assert len(list((root / 'bddl_files/libero_10').glob('*.bddl'))) == 10
assert len(list((root / 'init_files/libero_10').glob('*.pruned_init'))) == 10
assert (root / 'assets/textures/ceramic.png').is_file()
assert ActionEnvironmentClient('http://localhost:1', 'libero', '', 256, 1).timeout == 1
"""
    subprocess.run(
        [sys.executable, "-c", script, str(PLUGIN_ROOT)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(PLUGIN_ROOT), "LIBERO_CONFIG_PATH": str(tmp_path / "missing")},
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert not (tmp_path / "missing").exists()
