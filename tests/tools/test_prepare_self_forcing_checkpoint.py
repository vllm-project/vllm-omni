# SPDX-License-Identifier: Apache-2.0
"""Tests for the Self-Forcing checkpoint assembly script.

The script links or copies the base model's components into the output directory, removing
whatever already sits at each target. If the output directory *is* the base model directory,
every target names its own source, so the removal deletes the user's tokenizer, text encoder
and VAE before anything is written. That is unrecoverable, so it is refused up front.

These run without network access: the guard is checked before any checkpoint is fetched, and
the Hub client is stubbed so a regression that reorders the two would fail here rather than
pull several gigabytes on a CI worker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from prepare_self_forcing_checkpoint import BASE_COMPONENTS, main  # noqa: E402

import vllm_omni.transformers_utils.repo_utils as repo_utils  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def base_model(tmp_path: Path) -> Path:
    """A base model directory holding one recognizable file per component."""
    root = tmp_path / "wan"
    for name in BASE_COMPONENTS:
        (root / name).mkdir(parents=True)
        (root / name / "config.json").write_text(f'{{"component": "{name}"}}')
    return root


@pytest.fixture
def no_downloads(monkeypatch) -> list[str]:
    """Fail any Hub access, and record what was attempted.

    `main()` imports `hf_api` from `repo_utils` when it runs, so the patch goes on that
    module rather than on the script.
    """
    attempts: list[str] = []

    class _Forbidden:
        def __getattr__(self, name):
            def _fail(*args, **kwargs):
                attempts.append(name)
                raise AssertionError(f"{name} must not be called: these tests are offline")

            return _fail

    monkeypatch.setattr(repo_utils, "hf_api", lambda: _Forbidden())
    return attempts


@pytest.mark.parametrize("copy", [False, True])
def test_output_dir_equal_to_base_model_is_refused(base_model: Path, no_downloads: list[str], monkeypatch, copy: bool):
    """Assembling into the base model directory fails before anything is deleted or fetched."""
    argv = [
        "prepare_self_forcing_checkpoint.py",
        "--output-dir",
        str(base_model),
        "--base-model",
        str(base_model),
    ]
    if copy:
        argv.append("--copy")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="must not be the base model directory"):
        main()

    assert no_downloads == [], f"the guard ran after a Hub call: {no_downloads}"
    for name in BASE_COMPONENTS:
        assert (base_model / name / "config.json").read_text() == f'{{"component": "{name}"}}', (
            f"{name} was modified before the check rejected the run"
        )


def test_output_dir_aliasing_the_base_model_is_refused(
    base_model: Path, tmp_path: Path, no_downloads: list[str], monkeypatch
):
    """A symlinked path to the base model directory is the same hazard and is refused too."""
    alias = tmp_path / "wan-alias"
    alias.symlink_to(base_model, target_is_directory=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_self_forcing_checkpoint.py",
            "--output-dir",
            str(alias),
            "--base-model",
            str(base_model),
        ],
    )

    with pytest.raises(ValueError, match="must not be the base model directory"):
        main()

    assert no_downloads == [], f"the guard ran after a Hub call: {no_downloads}"
    for name in BASE_COMPONENTS:
        assert (base_model / name / "config.json").exists(), f"{name} was deleted through the alias"
