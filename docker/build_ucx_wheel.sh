#!/usr/bin/env bash
# Build a relocatable UCX runtime/development wheel with Level Zero support.
set -euo pipefail

UCX_VERSION="${1:-v1.21.0}"
OUTPUT_DIR="${2:-$PWD/dist}"
INSTALL_PREFIX="${3:-/opt/venv}"
UCX_PACKAGE_VERSION="${UCX_VERSION#v}"
BUILD_ROOT="$(mktemp -d)"
SOURCE_DIR="${BUILD_ROOT}/ucx"
STAGE_DIR="${BUILD_ROOT}/stage"

cleanup() {
  rm -rf "${BUILD_ROOT}"
}
trap cleanup EXIT

mkdir -p "${OUTPUT_DIR}" "${STAGE_DIR}"

git clone --depth 1 --branch "${UCX_VERSION}" https://github.com/openucx/ucx.git "${SOURCE_DIR}"
cd "${SOURCE_DIR}"
./autogen.sh
./contrib/configure-release-mt \
  --prefix="${INSTALL_PREFIX}" \
  --enable-shared \
  --disable-static \
  --disable-doxygen-doc \
  --enable-optimizations \
  --enable-cma \
  --enable-devel-headers \
  --enable-mt \
  --with-ze \
  --with-verbs \
  --with-rdmacm \
  --without-cuda \
  --without-rocm \
  --without-gdrcopy
make -j"$(nproc)" DESTDIR="${STAGE_DIR}" install-strip

python3 - "${STAGE_DIR}${INSTALL_PREFIX}" "${OUTPUT_DIR}" "${UCX_PACKAGE_VERSION}" <<'PY'
import base64
import csv
import hashlib
import io
import os
import platform
import stat
import sys
import zipfile
from pathlib import Path

prefix = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
version = sys.argv[3]
architecture = platform.machine().lower().replace("-", "_")
platform_tag = f"linux_{architecture}"
distribution = "ucx_runtime"
dist_info = f"{distribution}-{version}.dist-info"
data_root = f"{distribution}-{version}.data/data"
wheel_name = f"{distribution}-{version}-py3-none-{platform_tag}.whl"
wheel_path = output_dir / wheel_name

metadata = (
    "Metadata-Version: 2.1\n"
    "Name: ucx-runtime\n"
    f"Version: {version}\n"
    "Summary: UCX runtime and development files with Level Zero support\n"
)
wheel_metadata = (
    "Wheel-Version: 1.0\n"
    "Generator: build_ucx_wheel.sh\n"
    "Root-Is-Purelib: false\n"
    f"Tag: py3-none-{platform_tag}\n"
)

records: list[tuple[str, str, str]] = []


def add_bytes(archive: zipfile.ZipFile, name: str, content: bytes, mode: int = 0o644) -> None:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, content)
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    records.append((name, f"sha256={digest}", str(len(content))))


with zipfile.ZipFile(wheel_path, "w", allowZip64=True) as archive:
    for path in sorted(prefix.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(prefix).as_posix()
        archive_name = f"{data_root}/{relative}"
        content = path.read_bytes()
        mode = path.stat().st_mode & 0o777
        add_bytes(archive, archive_name, content, mode)

    add_bytes(archive, f"{dist_info}/METADATA", metadata.encode())
    add_bytes(archive, f"{dist_info}/WHEEL", wheel_metadata.encode())

    record_name = f"{dist_info}/RECORD"
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    writer.writerows(records)
    writer.writerow((record_name, "", ""))
    add_bytes(archive, record_name, record_buffer.getvalue().encode())

print(wheel_path)
PY
