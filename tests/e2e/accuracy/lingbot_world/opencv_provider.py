# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify the canonical cv2 provider in the isolated evaluator interpreter."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
from typing import TypedDict

PROVIDER = "opencv-python-headless"


class OpenCVProviderRecord(TypedDict):
    schema_version: int
    provider: str
    provider_version: str
    opencv_version: str
    opencv_build_sha256: str
    native_module_sha256: str
    module_path: str
    python: str
    verified_files: int


def verify_opencv_provider(expected_version: str) -> OpenCVProviderRecord:
    import cv2

    try:
        distribution = metadata.distribution(PROVIDER)
    except metadata.PackageNotFoundError as error:
        raise ValueError("Canonical OpenCV provider is unavailable") from error
    if distribution.version != expected_version:
        raise ValueError("OpenCV provider version differs from the pinned dependency")
    if cv2.__version__ != expected_version.rsplit(".", 1)[0]:
        raise ValueError("Imported OpenCV version differs from its provider")
    module_path = Path(cv2.__file__).resolve()
    expected_module = Path(str(distribution.locate_file("cv2/__init__.py"))).resolve()
    if module_path != expected_module:
        raise ValueError("OpenCV imported module belongs to another installation")
    files = distribution.files
    if not files:
        raise ValueError("OpenCV provider RECORD is missing")
    native_hashes = []
    verified = 0
    for entry in files:
        if entry.parts[0] not in {"cv2", "opencv_python_headless.libs"} or entry.suffix == ".pyc":
            continue
        recorded = entry.hash
        if recorded is None or recorded.mode != "sha256":
            raise ValueError(f"OpenCV provider RECORD needs a SHA256 hash: {entry}")
        value = recorded.value
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        if len(value) != 43 or any(character not in alphabet for character in value):
            raise ValueError(f"OpenCV provider RECORD has an invalid digest: {entry}")
        expected = binascii.a2b_base64(value.replace("-", "+").replace("_", "/") + "=")
        digest = hashlib.sha256()
        with Path(str(distribution.locate_file(entry))).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
        if digest.digest() != expected:
            raise ValueError(f"OpenCV provider file mismatch: {entry}")
        verified += 1
        if len(entry.parts) == 2 and entry.parts[0] == "cv2" and entry.suffix in {".so", ".pyd"}:
            native_hashes.append(digest.hexdigest())
    if len(native_hashes) != 1:
        raise ValueError("OpenCV provider needs one recorded native module")
    return {
        "schema_version": 1,
        "provider": PROVIDER,
        "provider_version": expected_version,
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        "native_module_sha256": native_hashes[0],
        "module_path": str(module_path),
        "python": sys.executable,
        "verified_files": verified,
    }


def validate_opencv_provider_record(record: object, *, expected_version: str, python: str) -> OpenCVProviderRecord:
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("OpenCV provider record schema is missing or changed")
    if record.get("provider") != PROVIDER or record.get("provider_version") != expected_version:
        raise ValueError("OpenCV provider record does not match the pinned provider")
    if record.get("python") != python or record.get("opencv_version") != expected_version.rsplit(".", 1)[0]:
        raise ValueError("OpenCV provider record does not match the selected interpreter")
    module = record.get("module_path")
    count = record.get("verified_files")
    if not isinstance(module, str) or not Path(module).is_absolute() or type(count) is not int or count < 2:
        raise ValueError("OpenCV provider record has incomplete file verification")
    for key in ("opencv_build_sha256", "native_module_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("OpenCV provider record has an invalid binary fingerprint")
    return {
        "schema_version": 1,
        "provider": PROVIDER,
        "provider_version": expected_version,
        "opencv_version": str(record["opencv_version"]),
        "opencv_build_sha256": str(record["opencv_build_sha256"]),
        "native_module_sha256": str(record["native_module_sha256"]),
        "module_path": module,
        "python": python,
        "verified_files": count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_opencv_provider(args.expected_version), allow_nan=False))
