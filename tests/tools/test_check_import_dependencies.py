# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for the offline declaration gate, using synthetic repos."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from tools.pre_commit.check_import_dependencies import BASELINE, CONFIG, Checker, Imports, main, requirements

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    write(tmp_path, "pyproject.toml", '[project.optional-dependencies]\ndev = ["pytest", "decord"]\n')
    write(tmp_path, "requirements/common.txt", "Pillow>=10\n-r nested.in\n-c constraints.txt\n")
    write(tmp_path, "requirements/nested.in", "numpy[example]>=1; python_version >= '3.10'\n")
    write(tmp_path, "requirements/constraints.txt", "decord==0.6.0\n")
    config = {
        "default_group": "runtime",
        "internal_modules": ["vllm_omni"],
        "groups": {
            "runtime": {"files": ["requirements/common.txt"]},
            "dev": {"extends": ["runtime"], "extras": ["dev"]},
        },
        "rules": [{"paths": ["tests/*"], "group": "dev"}],
        "import_names": {"PIL": ["Pillow"], "google.protobuf": ["protobuf"]},
    }
    write(tmp_path, CONFIG, json.dumps(config))
    write(tmp_path, BASELINE, "[]")
    return tmp_path


def test_walks_all_imports():
    visitor = Imports()
    visitor.visit(
        ast.parse("""
import os
from .local import helper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import type_only
class Model:
    def load(self):
        try:
            from decord import (
                VideoReader,
                cpu,
            )
        except ImportError:
            import fallback
""")
    )
    found = {(module, scope) for module, scope, _, _ in visitor.found}
    assert ("type_only", "") in found
    assert ("decord.VideoReader", "Model.load") in found
    assert ("decord.cpu", "Model.load") in found
    assert ("fallback", "Model.load") in found
    assert not any(module.startswith("local") for module, _ in found)


@pytest.mark.parametrize(
    "source",
    [
        "import decord",
        "def load():\n    from decord import VideoReader, cpu",
        "try:\n    import decord\nexcept ImportError:\n    pass",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import decord",
    ],
)
def test_runtime_cannot_borrow_test_extra_or_constraint(repo, source):
    write(repo, "vllm_omni/new.py", source)
    _, errors = Checker(repo).scan(["vllm_omni/new.py"])
    assert errors and "decord" in errors[0] and "runtime" in errors[0]
    write(repo, "tests/new.py", source)
    assert not Checker(repo).scan(["tests/new.py"])[1]


def test_aliases_internal_stdlib_and_siblings(repo):
    write(repo, "vllm_omni/helper.py", "")
    write(repo, "vllm_omni/new.py", "import os, numpy, helper, vllm_omni\nfrom PIL import Image\nfrom . import local")
    assert not Checker(repo).scan(["vllm_omni/new.py"])[1]
    # An unrelated package elsewhere in the repository does not mask a dependency.
    write(repo, "elsewhere/missing.py", "")
    write(repo, "vllm_omni/new.py", "import missing")
    assert Checker(repo).scan(["vllm_omni/new.py"])[1]


def test_namespace_import_ownership(repo):
    write(repo, "requirements/common.txt", "protobuf\n")
    write(repo, "new.py", "from google import protobuf\nfrom google.protobuf import message")
    assert not Checker(repo).scan(["new.py"])[1]
    write(repo, "new.py", "import google.cloud.storage")
    assert Checker(repo).scan(["new.py"])[1]


def test_local_search_roots_are_scoped_and_require_real_files(repo):
    config = json.loads((repo / CONFIG).read_text())
    config["local_roots"] = [
        {"paths": ["tests/*"], "roots": ["ci_helpers"], "reason": "Tests add these repository scripts to sys.path."}
    ]
    write(repo, CONFIG, json.dumps(config))
    write(repo, "ci_helpers/upload_pipeline.py", "")
    write(repo, "tests/new.py", "import upload_pipeline")
    assert not Checker(repo).scan(["tests/new.py"])[1]
    write(repo, "new.py", "import upload_pipeline")
    assert Checker(repo).scan(["new.py"])[1]
    write(repo, "tests/new.py", "import imaginary_helper")
    assert Checker(repo).scan(["tests/new.py"])[1]


def test_requirements_recursion_markers_direct_urls_and_constraints(repo):
    write(
        repo,
        "requirements/common.txt",
        "--requirement=nested.in\n--constraint constraints.txt\nFoo[bar] @ https://example.org/foo.whl#sha256=abc\n",
    )
    reqs = requirements(repo / "requirements/common.txt")
    assert {r.name for r in reqs} == {"numpy", "Foo"}
    assert next(r for r in reqs if r.name == "numpy").marker is not None
    write(repo, "requirements/nested.in", "-r common.txt")
    with pytest.raises(ValueError, match="Recursive requirements"):
        requirements(repo / "requirements/common.txt")


def test_markers_use_configured_platform_not_host(repo):
    write(
        repo, "requirements/common.txt", "linux_only; sys_platform == 'linux'\nwindows_only; sys_platform == 'win32'\n"
    )
    write(repo, "new.py", "import windows_only")
    assert Checker(repo).scan(["new.py"])[1]
    write(repo, "new.py", "import linux_only")
    assert not Checker(repo).scan(["new.py"])[1]


def test_baseline_scoped_to_file_function_statement_and_count(repo):
    write(repo, "old.py", "def load():\n    import decord\n")
    entries, _ = Checker(repo).scan(["old.py"])
    entries[0]["reason"] = "Existing optional video integration; tracked debt."
    write(repo, BASELINE, json.dumps(entries))
    assert not Checker(repo).scan(["old.py"])[1]
    for path, source in [
        ("other.py", "def load():\n    import decord\n"),
        ("old.py", "def other():\n    import decord\n"),
        ("old.py", "def load():\n    import decord as reader\n"),
        ("old.py", "def load():\n    import decord\n    import decord\n"),
    ]:
        write(repo, path, source)
        assert Checker(repo).scan([path])[1]


def test_manifest_edit_scans_unchanged_python(repo, capsys):
    write(repo, "old.py", "import numpy")
    assert main(["--root", str(repo), "--all-files"]) == 0
    write(repo, "requirements/nested.in", "")
    assert main(["--root", str(repo), "requirements/nested.in"]) == 1
    assert "numpy" in capsys.readouterr().err


def test_deleted_manifest_triggers_rescan_even_with_python_filenames(repo, capsys):
    write(repo, "old.py", "import numpy")
    write(repo, "changed.py", "import os")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Gao Han", "-c", "user.email=hgaoaf@connect.ust.hk", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "rm", "-q", "requirements/nested.in"], cwd=repo, check=True)
    assert main(["--root", str(repo), "changed.py"]) == 1
    assert "nested.in" in capsys.readouterr().err


def test_changed_file_complete_scan_and_syntax_errors(repo):
    write(repo, "new.py", "def old_unchanged_function():\n    import decord\n\nnew_variable = 1")
    assert main(["--root", str(repo), "new.py"]) == 1
    write(repo, "new.py", "this is invalid python!!!")
    assert main(["--root", str(repo), "new.py"]) == 1


def test_stale_baseline_requires_cleanup(repo):
    write(repo, "old.py", "import decord")
    entries, _ = Checker(repo).scan(["old.py"])
    entries[0]["reason"] = "Existing debt"
    write(repo, BASELINE, json.dumps(entries))
    assert main(["--root", str(repo), "--all-files"]) == 0
    write(repo, "old.py", "import os")
    assert main(["--root", str(repo), "--all-files"]) == 1


def test_invalid_manifest_is_not_silently_ignored(repo):
    write(repo, "requirements/common.txt", "--unsupported-install-option foo")
    write(repo, "new.py", "import os")
    assert main(["--root", str(repo), "new.py"]) == 1
