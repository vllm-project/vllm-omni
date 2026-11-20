# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]


def test_common_requirements_pin_openai_realtime_reasoning_floor() -> None:
    requirements = []
    requirements_path = ROOT / "requirements" / "common.txt"
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.partition("#")[0].strip()
        if line:
            requirements.append(Requirement(line))

    matches = [requirement for requirement in requirements if requirement.name.lower() == "openai"]

    assert len(matches) == 1
    requirement = matches[0]
    assert requirement.marker is None
    assert str(requirement.specifier) == ">=2.36.0"
