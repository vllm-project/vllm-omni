# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Protect retained exporters and native connectors during bootstrap migration."""

import json
import os
import runpy
import sys
import sysconfig
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).resolve().parents[2] / "tools/prepare_ci_telemetry.py"
COMMON = "opentelemetry-exporter-otlp-common"
OLD_FAMILY = (
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "opentelemetry-exporter-otlp-proto-grpc",
    "opentelemetry-exporter-otlp-proto-http",
)


@pytest.fixture(autouse=True)
def _dpkg_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "bin"
    directory.mkdir()
    tool = directory / "dpkg-query"
    tool.write_text(
        "#!/bin/sh\n"
        'if [ "${DPKG_QUERY_ERROR:-0}" = 1 ]; then exit 2; fi\n'
        'case "$3" in "${DPKG_APT_ROOT:-/no-fixture}"/*)\n'
        '  if [ "${DPKG_METADATA_OWNED:-1}" = 1 ]; then printf "os-package: %s\\n" "$3"; exit 0; fi\n'
        '  printf "dpkg-query: no path found matching pattern %s\\n" "$3" >&2; exit 1;;\n'
        "esac\n"
        'if [ "${DPKG_PAYLOAD_OWNED:-0}" = 1 ]; then\n'
        '  printf "os-package: %s\\n" "$3"; exit "${DPKG_OWNER_CODE:-0}"\n'
        "fi\n"
        'if [ "${DPKG_QUERY_WARNING:-0}" = 1 ]; then printf "dpkg-query: warning: files list missing\\n" >&2; fi\n'
        'printf "dpkg-query: no path found matching pattern %s\\n" "$3" >&2\n'
        "exit 1\n"
    )
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])


def _distribution(root: Path, name: str, version: str, *requirements: str) -> Path:
    directory = root / f"{name.replace('-', '_')}-{version}.dist-info"
    directory.mkdir()
    (directory / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
    )
    payload = f"fixture_payload/{name}.py"
    (root / "fixture_payload").mkdir(exist_ok=True)
    (root / payload).touch()
    (directory / "RECORD").write_text(f"{payload},,\n{directory.name}/METADATA,,\n{directory.name}/RECORD,,\n")
    return directory


def _old_family(root: Path) -> None:
    for name in OLD_FAMILY:
        _distribution(root, name, "1.40.0")
    _distribution(root, COMMON, "0.66b1", "opentelemetry-sdk~=1.45.1")
    _distribution(root, "lmcache", "0.5.6", "opentelemetry-api>=1.20,<=1.40")


def _prepare(root: Path, monkeypatch: pytest.MonkeyPatch, outside: Path | None = None) -> tuple[set[str], str]:
    prepare = runpy.run_path(str(SCRIPT))["prepare"]
    roots, orphans = root / "roots.txt", root / "orphans.txt"
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "path", [str(root)] + ([str(outside)] if outside else []))
        scoped.setattr(sysconfig, "get_path", lambda _name: str(root))
        prepare(roots, orphans)
    return set(roots.read_text().splitlines()), orphans.read_text()


def test_only_verified_old_family_orphan_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _old_family(tmp_path)
    roots, orphans = _prepare(tmp_path, monkeypatch)
    assert roots == {*OLD_FAMILY, "lmcache==0.5.6"}
    assert orphans == COMMON + "\n"
    decision = json.loads(capsys.readouterr().out)
    assert decision["common_consumers"] == decision["common_payload_overlaps"] == decision["unknown_file_records"] == []
    assert decision["installed"]["lmcache"] == "0.5.6"


@pytest.mark.parametrize(
    "guard",
    [
        "optional-consumer",
        "new-family",
        "common-record",
        "retained-record",
        "empty-record",
        "shared-file",
        "absent-shared",
    ],
)
def test_uncertain_or_needed_common_is_retained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: str) -> None:
    _old_family(tmp_path)
    if guard == "optional-consumer":
        _distribution(tmp_path, "retained-feature", "1.0", "OpenTelemetry_Exporter_OTLP_Common; extra == 'optional'")
    elif guard == "new-family":
        path = next(tmp_path.glob("opentelemetry_sdk-*.dist-info/METADATA"))
        path.write_text(path.read_text().replace("1.40.0", "1.45.2"))
    elif guard == "common-record":
        next(tmp_path.glob("opentelemetry_exporter_otlp_common-*.dist-info/RECORD")).unlink()
    elif guard == "retained-record":
        next(tmp_path.glob("lmcache-*.dist-info/RECORD")).unlink()
    elif guard == "empty-record":
        next(tmp_path.glob("lmcache-*.dist-info/RECORD")).write_text("")
    elif guard == "shared-file":
        record = next(tmp_path.glob("lmcache-*.dist-info/RECORD"))
        record.write_text(record.read_text() + f"fixture_payload/{COMMON}.py,,\n")
    elif guard == "absent-shared":
        for pattern in ("lmcache-*.dist-info/RECORD", "opentelemetry_exporter_otlp_common-*.dist-info/RECORD"):
            record = next(tmp_path.glob(pattern))
            record.write_text(record.read_text() + "absent_but_declared.py,,\n")
    roots, orphans = _prepare(tmp_path, monkeypatch)
    assert COMMON in roots
    assert "lmcache==0.5.6" in roots
    assert orphans == ""


@pytest.mark.parametrize("invalid", ["duplicate", "missing-name", "malformed-requirement", "malformed-record"])
def test_ambiguous_metadata_fails_before_prune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str) -> None:
    _old_family(tmp_path)
    if invalid == "duplicate":
        _distribution(tmp_path, "LMCache", "0.5.6", COMMON)
    elif invalid == "missing-name":
        directory = _distribution(tmp_path, "retained-feature", "1.0")
        metadata = directory / "METADATA"
        metadata.write_text(metadata.read_text().replace("Name: retained-feature\n", ""))
    elif invalid == "malformed-requirement":
        _distribution(tmp_path, "retained-feature", "1.0", "broken dependency !")
    elif invalid == "malformed-record":
        next(tmp_path.glob("opentelemetry_exporter_otlp_common-*.dist-info/RECORD")).write_text("two,columns\n")
    with pytest.raises(ValueError if invalid == "malformed-requirement" else RuntimeError):
        _prepare(tmp_path, monkeypatch)
    assert not (tmp_path / "orphans.txt").exists()


@pytest.mark.parametrize("outside_kind", ["consumer", "common-owner"])
def test_outside_metadata_cannot_be_silently_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_kind: str
) -> None:
    _old_family(tmp_path)
    outside = tmp_path / "os-site"
    outside.mkdir()
    if outside_kind == "consumer":
        # Same canonical name as a managed root must not erase this reverse edge.
        _distribution(outside, "LMCache", "0.5.6", COMMON + "; extra == 'optional'")
    else:
        _distribution(outside, COMMON, "0.66b1")
    roots, orphans = _prepare(tmp_path, monkeypatch, outside)
    assert COMMON in roots
    assert "lmcache==0.5.6" in roots
    assert orphans == ""


@pytest.mark.parametrize(
    "decision", ["unowned", "metadata-unowned", "payload-owned", "partial-owner", "query-error", "incomplete-database"]
)
def test_os_manifests_must_prove_metadata_and_payload_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, decision: str
) -> None:
    _old_family(tmp_path)
    outside = tmp_path / "os-site"
    outside.mkdir()
    legacy = _distribution(outside, "cryptography", "41.0.7")
    (legacy / "RECORD").unlink()
    _distribution(tmp_path, "cryptography", "44.0.0")
    monkeypatch.setenv("DPKG_APT_ROOT", str(outside))
    if decision == "metadata-unowned":
        monkeypatch.setenv("DPKG_METADATA_OWNED", "0")
    elif decision in {"payload-owned", "partial-owner"}:
        monkeypatch.setenv("DPKG_PAYLOAD_OWNED", "1")
        if decision == "partial-owner":
            monkeypatch.setenv("DPKG_OWNER_CODE", "1")
    elif decision == "query-error":
        monkeypatch.setenv("DPKG_QUERY_ERROR", "1")
    elif decision == "incomplete-database":
        monkeypatch.setenv("DPKG_QUERY_WARNING", "1")
    roots, orphans = _prepare(tmp_path, monkeypatch, outside)
    receipt = json.loads(capsys.readouterr().out)
    if decision == "unowned":
        assert COMMON not in roots
        assert orphans == COMMON + "\n"
        assert receipt["outside_unknown_metadata_os_owned"] is True
        assert receipt["common_payload_os_unowned"] is True
    else:
        assert COMMON in roots
        assert orphans == ""
    assert receipt["os_ownership_queries"]


def test_unavailable_os_manager_cannot_authorize_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _old_family(tmp_path)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    roots, orphans = _prepare(tmp_path, monkeypatch)
    assert COMMON in roots
    assert orphans == ""
