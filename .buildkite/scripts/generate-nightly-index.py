#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import regex as re


def normalize_package_name(name: str) -> str:
    """Normalize package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


if not sys.version_info >= (3, 12):
    raise RuntimeError("This script requires Python 3.12 or higher.")

INDEX_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
  <!-- {comment} -->
  <meta name="pypi:repository-version" content="1.0">
  <body>
{items}
  </body>
</html>
"""


@dataclass
class WheelFileInfo:
    package_name: str
    version: str
    build_tag: str | None
    python_tag: str
    abi_tag: str
    platform_tag: str
    filename: str


def parse_from_filename(file: str) -> WheelFileInfo:
    """
    Parse wheel filename per PEP 427:
        {package_name}-{version}(-{build_tag})?-{python_tag}-{abi_tag}-{platform_tag}.whl
    """
    wheel_file_re = re.compile(
        r"^(?P<package_name>.+)-(?P<version>[^-]+?)(-(?P<build_tag>[^-]+))?-(?P<python_tag>[^-]+)-(?P<abi_tag>[^-]+)-(?P<platform_tag>[^-]+)\.whl$"
    )
    match = wheel_file_re.match(file)
    if not match:
        raise ValueError(f"Invalid wheel file name: {file}")

    return WheelFileInfo(
        package_name=match.group("package_name"),
        version=match.group("version"),
        build_tag=match.group("build_tag"),
        python_tag=match.group("python_tag"),
        abi_tag=match.group("abi_tag"),
        platform_tag=match.group("platform_tag"),
        filename=file,
    )


def generate_project_list(package_names: list[str], comment: str = "") -> str:
    """Generate top-level PEP 503 project list HTML."""
    href_tags = []
    for name in sorted(package_names):
        href_tags.append(f'    <a href="{name}/">{name}/</a><br/>')
    return INDEX_HTML_TEMPLATE.format(items="\n".join(href_tags), comment=comment)


def generate_package_index(
    wheel_files: list[WheelFileInfo],
    wheel_base_dir: Path,
    index_base_dir: Path,
    comment: str = "",
) -> tuple[str, str]:
    """Generate package index HTML and metadata JSON linking to wheel files."""
    href_tags = []
    metadata = []
    for file in sorted(wheel_files, key=lambda x: x.filename):
        relative_path = wheel_base_dir.relative_to(index_base_dir, walk_up=True) / file.filename
        # handle '+' in URL; avoid double-encoding '/' and '%2B' (AWS S3 behavior)
        file_path_quoted = quote(relative_path.as_posix(), safe=":%/")
        href_tags.append(f'    <a href="{file_path_quoted}">{file.filename}</a><br/>')
        file_meta = asdict(file)
        file_meta["path"] = file_path_quoted
        metadata.append(file_meta)
    index_str = INDEX_HTML_TEMPLATE.format(items="\n".join(href_tags), comment=comment)
    metadata_str = json.dumps(metadata, indent=2)
    return index_str, metadata_str


def generate_index(
    whl_files: list[str],
    wheel_base_dir: Path,
    index_base_dir: Path,
    comment: str = "",
):
    """
    Generate PEP 503 index for all wheel files.

    Output structure:
        index_base_dir/
            index.html          # project list linking to vllm-omni/
            vllm-omni/
                index.html      # package index linking to wheel files
                metadata.json   # machine-readable metadata
    """
    parsed_files = [parse_from_filename(f) for f in whl_files]

    if not parsed_files:
        print("No wheel files found, skipping index generation.")
        return

    comment_str = f" ({comment})" if comment else ""
    comment_tmpl = f"Generated on {datetime.now().isoformat()}{comment_str}"

    # Group by normalized package name
    packages: dict[str, list[WheelFileInfo]] = {}
    for file in parsed_files:
        name = normalize_package_name(file.package_name)
        packages.setdefault(name, []).append(file)

    print(f"Found packages: {list(packages.keys())}")

    # Generate per-package index
    for package, files in packages.items():
        package_dir = index_base_dir / package
        package_dir.mkdir(parents=True, exist_ok=True)
        index_str, metadata_str = generate_package_index(files, wheel_base_dir, package_dir, comment)
        with open(package_dir / "index.html", "w") as f:
            f.write(index_str)
        with open(package_dir / "metadata.json", "w") as f:
            f.write(metadata_str)

    # Generate top-level project list
    project_list_str = generate_project_list(sorted(packages.keys()), comment_tmpl)
    with open(index_base_dir / "index.html", "w") as f:
        f.write(project_list_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PEP 503 wheel index from S3 object listing.")
    parser.add_argument("--version", type=str, required=True, help="Version string (e.g., commit hash)")
    parser.add_argument("--current-objects", type=str, required=True, help="Path to JSON from S3 list-objects-v2")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write index files")
    parser.add_argument("--wheel-dir", type=str, default=None, help="Wheel directory (defaults to --version)")
    parser.add_argument("--comment", type=str, default="", help="Comment for generated HTML")

    args = parser.parse_args()

    version = args.version
    if "\\" in version or "/" in version:
        raise ValueError("Version string must not contain slashes or backslashes.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.current_objects) as f:
        current_objects: dict[str, list[dict[str, Any]]] = json.load(f)

    wheel_files = [
        item["Key"].split("/")[-1] for item in current_objects.get("Contents", []) if item["Key"].endswith(".whl")
    ]

    print(f"Found {len(wheel_files)} wheel files for version {version}: {wheel_files}")

    # For release versions, filter to only matching non-dev wheels
    PY_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+([a-zA-Z0-9.+-]*)?$")
    if PY_VERSION_RE.match(version):
        wheel_files = [f for f in wheel_files if version in f and "dev" not in f]
        print(f"Non-nightly version detected, wheel files used: {wheel_files}")
    else:
        print("Nightly version detected, keeping all wheel files.")

    wheel_dir = (args.wheel_dir or version).strip().rstrip("/")
    wheel_base_dir = Path(output_dir).parent / wheel_dir
    index_base_dir = Path(output_dir)

    generate_index(
        whl_files=wheel_files,
        wheel_base_dir=wheel_base_dir,
        index_base_dir=index_base_dir,
        comment=args.comment.strip(),
    )
    print(f"Successfully generated index in {output_dir}")
