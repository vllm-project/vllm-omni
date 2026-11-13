# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep the PAN 2 guardrail install out of published package metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements" / "ci-pan2-guardrail.txt"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.ci"
IMAGE_SCRIPT = REPO_ROOT / ".buildkite" / "cuda" / "scripts" / "build-ci-image.sh"

PAN2_URL = "git+https://github.com/ifm-ai/pan2-guardrail@267f95a943162e56d835f3897217a7d14ee7edce"
CLIP_URL = "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"


def _requirement_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _is_direct_url(requirement: str) -> bool:
    return "git+" in requirement or "https://" in requirement or " @ " in requirement


def test_published_metadata_has_no_direct_url_dependencies() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    offenders = [requirement for requirement in project.get("dependencies", []) if _is_direct_url(requirement)]
    offenders.extend(
        f"{extra}: {requirement}"
        for extra, requirements in project["optional-dependencies"].items()
        for requirement in requirements
        if _is_direct_url(requirement)
    )
    assert offenders == []


def test_ci_image_hashes_the_pinned_guardrail_requirements() -> None:
    lines = _requirement_lines(REQUIREMENTS.read_text(encoding="utf-8"))
    assert f"pan2-guardrail @ {PAN2_URL}" in lines
    assert f"clip @ {CLIP_URL}" in lines
    # build-ci-image.sh hashes requirements/*.txt, so this file changes the image.
    assert "requirements/*.txt" in IMAGE_SCRIPT.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    install_at = dockerfile.index("requirements/ci-pan2-guardrail.txt")
    check_at = dockerfile.index("uv pip check --system", install_at)
    assert check_at > install_at
