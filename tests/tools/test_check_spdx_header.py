# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the SPDX header gate.

This hook rewrites source files in place, so its contract deserves pinning:
which comment prefix each suffix gets, that a shebang stays on line 1, and --
the reason these tests exist -- that a file carrying the stale upstream vLLM
copyright line ends up with exactly one copyright line, never two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "pre_commit"))

from check_spdx_header import (  # noqa: E402
    COPYRIGHT_TEXT,
    LEGACY_COPYRIGHT_TEXT,
    LICENSE_TEXT,
    SPDXStatus,
    add_header,
    check_spdx_header_status,
    file_style,
    header_insertion_index,
    main,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _lines(prefix: str) -> tuple[str, str, str]:
    """Return the (license, omni copyright, stale upstream copyright) lines."""
    return (
        f"{prefix} {LICENSE_TEXT}",
        f"{prefix} {COPYRIGHT_TEXT}",
        f"{prefix} {LEGACY_COPYRIGHT_TEXT}",
    )


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="UTF-8")
    return path


def _fix(path: Path) -> list[str]:
    """Run the hook's detect-then-rewrite cycle once; return the new lines."""
    add_header(path, check_spdx_header_status(path))
    return path.read_text(encoding="UTF-8").splitlines()


# --------------------------------------------------------------------------
# Status detection
# --------------------------------------------------------------------------


def test_empty_file_is_exempt(tmp_path):
    """An empty __init__.py needs no header."""
    assert check_spdx_header_status(_write(tmp_path, "__init__.py", "")) is SPDXStatus.EMPTY


@pytest.mark.parametrize(
    ("label", "header", "expected"),
    [
        ("both lines", "{lic}\n{omni}\n", SPDXStatus.COMPLETE),
        ("license only", "{lic}\n", SPDXStatus.MISSING_COPYRIGHT),
        ("license plus stale upstream copyright", "{lic}\n{legacy}\n", SPDXStatus.MISSING_COPYRIGHT),
        ("omni copyright only", "{omni}\n", SPDXStatus.MISSING_LICENSE),
        ("stale upstream copyright only", "{legacy}\n", SPDXStatus.MISSING_BOTH),
        ("no header at all", "", SPDXStatus.MISSING_BOTH),
    ],
)
def test_status_detection(tmp_path, label, header, expected):
    lic, omni, legacy = _lines("#")
    body = header.format(lic=lic, omni=omni, legacy=legacy) + "import os\n"
    assert check_spdx_header_status(_write(tmp_path, "mod.py", body)) is expected, label


# --------------------------------------------------------------------------
# Rewrites
# --------------------------------------------------------------------------


def test_missing_both_inserts_header_and_blank_line(tmp_path):
    lic, omni, _ = _lines("#")
    lines = _fix(_write(tmp_path, "mod.py", "import os\n"))
    assert lines == [lic, omni, "", "import os"]


def test_shebang_stays_on_the_first_line(tmp_path):
    lic, omni, _ = _lines("#")
    lines = _fix(_write(tmp_path, "run.sh", "#!/usr/bin/env bash\necho hi\n"))
    assert lines == ["#!/usr/bin/env bash", lic, omni, "", "echo hi"]


def test_missing_license_is_inserted_above_the_copyright(tmp_path):
    lic, omni, _ = _lines("#")
    lines = _fix(_write(tmp_path, "mod.py", f"{omni}\nimport os\n"))
    assert lines == [lic, omni, "import os"]


def test_stale_copyright_beside_a_license_is_rewritten_in_place(tmp_path):
    lic, omni, legacy = _lines("#")
    lines = _fix(_write(tmp_path, "mod.py", f"{lic}\n{legacy}\nimport os\n"))
    assert lines == [lic, omni, "import os"]
    assert legacy not in lines


def test_stale_copyright_without_a_license_does_not_duplicate(tmp_path):
    """A stale upstream copyright line must be rewritten, not shadowed.

    The file has the upstream vLLM copyright but no license line, so detection
    reports MISSING_BOTH. Prepending a full header there would leave the stale
    line in place and the file would carry two conflicting SPDX-FileCopyrightText
    lines -- and the next run would call it COMPLETE, so the conflict sticks.
    """
    lic, omni, legacy = _lines("#")
    lines = _fix(_write(tmp_path, "mod.py", f"{legacy}\nimport os\n"))

    copyrights = [line for line in lines if "SPDX-FileCopyrightText" in line]
    assert copyrights == [omni], f"expected exactly one copyright line, got {copyrights}"
    assert lines == [lic, omni, "import os"]


def test_stale_copyright_without_a_license_keeps_the_shebang(tmp_path):
    lic, omni, legacy = _lines("#")
    lines = _fix(_write(tmp_path, "run.sh", f"#!/usr/bin/env bash\n{legacy}\necho hi\n"))
    assert lines == ["#!/usr/bin/env bash", lic, omni, "echo hi"]


@pytest.mark.parametrize(
    ("name", "prefix"), [("mod.py", "#"), ("stub.pyi", "#"), ("lib.rs", "//"), ("api.proto", "//")]
)
def test_comment_prefix_per_suffix(tmp_path, name, prefix):
    lic, omni, _ = _lines(prefix)
    assert _fix(_write(tmp_path, name, "code\n"))[:2] == [lic, omni]


def test_unsupported_suffix_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        file_style(tmp_path / "notes.md")


def test_header_insertion_index_only_skips_a_shebang_when_the_style_allows(tmp_path):
    shebang = ["#!/usr/bin/env python3\n", "import os\n"]
    assert header_insertion_index(file_style("mod.py"), shebang) == 1
    # .pyi stubs never carry a shebang, so the style does not preserve one.
    assert header_insertion_index(file_style("stub.pyi"), shebang) == 0


# --------------------------------------------------------------------------
# Idempotence and CLI contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    ["import os\n", "{legacy}\nimport os\n", "{lic}\nimport os\n", "{omni}\nimport os\n"],
)
def test_second_run_is_a_no_op(tmp_path, monkeypatch, body):
    """Driven through main(), the way pre-commit invokes the gate."""
    lic, omni, legacy = _lines("#")
    path = _write(tmp_path, "mod.py", body.format(lic=lic, omni=omni, legacy=legacy))
    monkeypatch.setattr(sys, "argv", ["check_spdx_header.py", str(path)])

    with pytest.raises(SystemExit):
        main()
    settled = path.read_text(encoding="UTF-8")
    assert check_spdx_header_status(path) is SPDXStatus.COMPLETE

    with pytest.raises(SystemExit) as second:
        main()
    assert second.value.code == 0
    assert path.read_text(encoding="UTF-8") == settled


@pytest.mark.parametrize("status", [SPDXStatus.COMPLETE, SPDXStatus.EMPTY])
def test_add_header_leaves_a_file_it_cannot_fix_untouched(tmp_path, status):
    """add_header truncates before it branches, so an unhandled status must bail.

    main() only dispatches the three MISSING_* statuses, but nothing in the
    signature says so: a caller that hands over a COMPLETE file would otherwise
    get an empty one back.
    """
    lic, omni, _ = _lines("#")
    body = f"{lic}\n{omni}\n\ndef important():\n    return 42\n"
    path = _write(tmp_path, "mod.py", body)

    add_header(path, status)

    assert path.read_text(encoding="UTF-8") == body


def test_main_exits_nonzero_when_it_rewrites_and_zero_once_clean(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, "mod.py", "import os\n")
    monkeypatch.setattr(sys, "argv", ["check_spdx_header.py", str(path)])

    with pytest.raises(SystemExit) as first:
        main()
    assert first.value.code == 1
    assert str(path) in capsys.readouterr().out

    with pytest.raises(SystemExit) as second:
        main()
    assert second.value.code == 0


def test_main_skips_unsupported_suffixes(tmp_path, monkeypatch):
    path = _write(tmp_path, "notes.md", "# heading\n")
    monkeypatch.setattr(sys, "argv", ["check_spdx_header.py", str(path)])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 0
    assert path.read_text(encoding="UTF-8") == "# heading\n"
