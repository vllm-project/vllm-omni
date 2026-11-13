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


def test_published_extras_have_no_direct_url_dependencies() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    offenders = [
        f"{extra}: {requirement}"
        for extra, requirements in extras.items()
        for requirement in requirements
        if "git+" in requirement or "https://" in requirement or " @ " in requirement
    ]
    assert offenders == []


def test_ci_image_installs_pinned_pan2_guardrail_without_upgrading_torch() -> None:
    lines = _requirement_lines(REQUIREMENTS.read_text(encoding="utf-8"))
    assert lines == [
        f"pan2-guardrail @ {PAN2_URL}",
        f"clip @ {CLIP_URL}",
    ]
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "uv pip install --system --no-deps -r requirements/ci-pan2-guardrail.txt" in dockerfile
    assert 'uv pip install --system "ftfy>=6.1.1" "regex>=2022.1.18"' in dockerfile
    # requirements/*.txt is already part of the image dependency hash.
    assert "requirements/*.txt" in IMAGE_SCRIPT.read_text(encoding="utf-8")
