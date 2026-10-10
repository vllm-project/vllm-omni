# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The generic diffusion layers must not name a model.

A model may depend on the generic layers, never the reverse. This guards the
dependency direction mechanically, because a single import or environment lookup
is enough to reverse it and nothing else would fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GENERIC_DIRS = [
    Path("vllm_omni/diffusion/distributed"),
    Path("vllm_omni/diffusion/attention/parallel"),
]
# Model names that must not appear in those layers. Extend as models are added.
MODEL_TOKENS = ("minimax_h3", "H3_")


def _offending_lines(path: Path) -> list[tuple[int, str]]:
    return [
        (number, line.rstrip())
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if any(token in line for token in MODEL_TOKENS)
    ]


@pytest.mark.parametrize("directory", GENERIC_DIRS, ids=lambda d: d.name)
def test_generic_layer_names_no_model(directory: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / directory
    assert target.is_dir(), f"expected a generic layer at {target}"

    offenders = {
        str(source.relative_to(repo_root)): lines
        for source in sorted(target.rglob("*.py"))
        if (lines := _offending_lines(source))
    }
    assert not offenders, (
        "a generic diffusion layer refers to a model; move the model-specific piece behind "
        f"the model's own registration: {offenders}"
    )
