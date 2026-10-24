# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the forbidden-import gate.

This gate runs in GitHub Actions (it is not on the workflow's SKIP list), so
the line number it prints is the only pointer a contributor gets. These pin
that number against the layouts that used to shift it, plus the allowlist
mechanics the rules rely on.
"""

from __future__ import annotations

import re as stdlib_re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "pre_commit"))

from check_forbidden_imports import (  # noqa: E402
    CHECK_IMPORTS,
    check_file,
    main,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write(tmp_path: Path, body: str, name: str = "mod.py") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="UTF-8")
    return path


def _reported_lines(capsys) -> list[int]:
    out = capsys.readouterr().out
    return [int(n) for n in stdlib_re.findall(r"^.*?:(\d+): ", out, stdlib_re.MULTILINE)]


# --------------------------------------------------------------------------
# Reported line numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "body", "expected"),
    [
        ("import on line 1", "import pickle\n", 1),
        ("no blank line before", "x = 1\nimport pickle\n", 2),
        ("one blank line before", '"""Doc."""\n\nimport pickle\n', 3),
        ("two blank lines before", "x = 1\n\n\nimport pickle\n", 4),
        ("three blank lines before", "x = 1\n\n\n\nimport pickle\n", 5),
        # PEP 8 puts two blank lines after a top-level def, so a module-level
        # import that follows one lands in exactly this shape.
        ("after a def", "def f():\n    pass\n\n\nimport pickle\n", 5),
        ("indented inside a function", "def f():\n\n    import pickle\n", 3),
    ],
)
def test_line_number_points_at_the_import(tmp_path, capsys, label, body, expected):
    assert check_file(str(_write(tmp_path, body))) == 1, label
    assert _reported_lines(capsys) == [expected], label


@pytest.mark.parametrize(
    ("rule_body", "expected"),
    [
        ("x = 1\n\n\nimport base64\n", 4),
        ("x = 1\n\n\nimport re\n", 4),
        ("x = 1\n\n\nfrom huggingface_hub import HfApi\n", 4),
    ],
)
def test_line_number_holds_for_every_rule_with_leading_whitespace(tmp_path, capsys, rule_body, expected):
    """The `re`, `base64` and huggingface_hub rules share the same prefix."""
    assert check_file(str(_write(tmp_path, rule_body, "vllm_omni/mod.py"))) == 1
    assert _reported_lines(capsys) == [expected]


def test_every_violation_in_a_file_is_reported(tmp_path, capsys):
    body = "import pickle\n\n\nimport cloudpickle\n\n\n\nfrom pickle import loads\n"
    assert check_file(str(_write(tmp_path, body))) == 1
    assert _reported_lines(capsys) == [1, 4, 8]


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "body", "flagged"),
    [
        ("plain import", "import pickle\n", True),
        ("aliased", "import pickle as pkl\n", True),
        ("in a list", "import os, pickle\n", True),
        ("from-import", "from pickle import loads\n", True),
        ("cloudpickle", "import cloudpickle\n", True),
        ("indented", "def f():\n    import pickle\n", True),
        ("commented out", "# import pickle\n", False),
        ("inside a string", "x = 'import pickle'\n", False),
        ("different module", "import pickletools\n", False),
        ("imported from elsewhere", "from mymod import pickle\n", False),
    ],
)
def test_pickle_rule(tmp_path, label, body, flagged):
    assert check_file(str(_write(tmp_path, body))) == (1 if flagged else 0), label


@pytest.mark.parametrize(
    ("label", "body", "flagged"),
    [
        ("stdlib re is banned", "import re\n", True),
        ("from re import is banned", "from re import match\n", True),
        ("regex is the replacement", "import regex\n", False),
        ("regex aliased as re", "import regex as re\n", False),
        ("stdlib base64 is banned", "import base64\n", True),
        ("pybase64 is the replacement", "import pybase64\n", False),
        ("pybase64 aliased", "import pybase64 as base64\n", False),
    ],
)
def test_replacement_imports_are_accepted(tmp_path, label, body, flagged):
    path = _write(tmp_path, body, "vllm_omni/mod.py")
    assert check_file(str(path)) == (1 if flagged else 0), label


@pytest.mark.parametrize(
    ("label", "body", "flagged"),
    [
        ("bare triton", "import triton\n", True),
        ("triton submodule", "from triton import language\n", True),
        ("the sanctioned shim", "from vllm.triton_utils import triton\n", False),
        ("bare tilelang", "import tilelang\n", True),
        ("the sanctioned shim", "from vllm.tilelang_utils import tilelang, T\n", False),
        ("unrelated prefix", "import tilelang_kernels\n", False),
    ],
)
def test_kernel_shim_rules(tmp_path, label, body, flagged):
    assert check_file(str(_write(tmp_path, body))) == (1 if flagged else 0), label


@pytest.mark.parametrize(
    ("label", "body", "flagged"),
    [
        ("repo API name", "from huggingface_hub import snapshot_download\n", True),
        ("parenthesized multi-line", "from huggingface_hub import (\n    HfApi,\n)\n", True),
        ("mixed with an allowed name", "from huggingface_hub import PyTorchModelHubMixin, HfApi\n", True),
        ("non-repo name alone", "from huggingface_hub import PyTorchModelHubMixin\n", False),
        ("module import", "import huggingface_hub\n", False),
        ("submodule constant", "from huggingface_hub.constants import HF_HUB_CACHE\n", False),
    ],
)
def test_huggingface_repo_api_rule(tmp_path, label, body, flagged):
    path = _write(tmp_path, body, "vllm_omni/mod.py")
    assert check_file(str(path)) == (1 if flagged else 0), label


# --------------------------------------------------------------------------
# Allowlists
# --------------------------------------------------------------------------


@pytest.mark.parametrize("directory", ["tests", "examples", "benchmarks", "tools", "docs"])
def test_non_library_dirs_may_keep_stdlib_imports(tmp_path, monkeypatch, directory):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "import re\nimport base64\n", f"{directory}/thing.py")
    assert check_file(f"{directory}/thing.py") == 0


def test_the_same_import_is_flagged_inside_the_library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "import re\n", "vllm_omni/thing.py")
    assert check_file("vllm_omni/thing.py") == 1


def test_allowed_dirs_do_not_exempt_pickle(tmp_path, monkeypatch):
    """pickle has no allowed_dirs -- only an explicit per-file allowlist."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "import pickle\n", "examples/thing.py")
    assert check_file("examples/thing.py") == 1


def test_allowed_files_exempts_an_exact_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    allowed = sorted(CHECK_IMPORTS["pickle/cloudpickle"].allowed_files)[0]
    _write(tmp_path, "import pickle\n", allowed)
    assert check_file(allowed) == 0

    _write(tmp_path, "import pickle\n", "vllm_omni/not_allowlisted.py")
    assert check_file("vllm_omni/not_allowlisted.py") == 1


def test_windows_separators_are_normalized(tmp_path, monkeypatch):
    """allowed_files stores posix paths; check_file normalizes before matching."""
    monkeypatch.chdir(tmp_path)
    allowed = sorted(CHECK_IMPORTS["pickle/cloudpickle"].allowed_files)[0]
    _write(tmp_path, "import pickle\n", allowed)
    assert check_file(allowed.replace("/", "\\")) == 0


# --------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------


def test_main_reports_across_files_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    clean = _write(tmp_path, "import os\n", "clean.py")
    dirty = _write(tmp_path, "x = 1\n\n\nimport pickle\n", "dirty.py")
    monkeypatch.setattr(sys, "argv", ["check_forbidden_imports.py", str(clean), str(dirty)])

    assert main() == 1
    assert _reported_lines(capsys) == [4]


def test_main_exits_zero_when_everything_is_clean(tmp_path, monkeypatch):
    path = _write(tmp_path, "import os\nimport regex as re\n")
    monkeypatch.setattr(sys, "argv", ["check_forbidden_imports.py", str(path)])

    assert main() == 0
