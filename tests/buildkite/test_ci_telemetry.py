# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Protect retained exporters and native connectors during bootstrap migration."""

import json
import runpy
import sys
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


def _prepare(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[set[str], str]:
    prepare = runpy.run_path(str(SCRIPT))["prepare"]
    roots, orphans = root / "roots.txt", root / "orphans.txt"
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "path", [str(root)])
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
    "guard", ["optional-consumer", "new-family", "common-record", "retained-record", "empty-record", "shared-file"]
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
    roots, orphans = _prepare(tmp_path, monkeypatch)
    assert COMMON in roots
    assert "lmcache==0.5.6" in roots
    assert orphans == ""


@pytest.mark.parametrize("invalid", ["duplicate", "missing-name", "malformed-requirement"])
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
    with pytest.raises(ValueError if invalid == "malformed-requirement" else RuntimeError):
        _prepare(tmp_path, monkeypatch)
    assert not (tmp_path / "orphans.txt").exists()
