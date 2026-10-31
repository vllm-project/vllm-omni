# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import load_samples

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "dataset, repo, revision",
    [
        ("Hothan/Omni-DuplexEval@fixed-revision", "Hothan/Omni-DuplexEval", "fixed-revision"),
        ("org/repo@refs/pr/1", "org/repo", "refs/pr/1"),
        ("legacy_repo@rev", "legacy_repo", "rev"),
    ],
)
def test_remote_dataset_revision_is_passed_to_loader(monkeypatch, dataset, repo, revision):
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return [{"id": "0", "split": "PR_correction", "question_text": "The hat is blue."}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    samples = load_samples(dataset, split="PR_correction")
    assert len(samples) == 1
    assert calls == [(repo, {"revision": revision, "split": "PR_correction"})]


@pytest.mark.parametrize("dataset, message", [("@rev", "Dataset ID"), ("org/repo@", "Dataset revision")])
def test_invalid_revision_selector_is_rejected_before_loading(monkeypatch, dataset, message):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid revision selector reached datasets.load_dataset")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=unexpected_load))
    with pytest.raises(ValueError, match=message):
        load_samples(dataset, split="PR_correction")


@pytest.mark.parametrize("relative", [False, True])
def test_missing_explicit_local_path_with_at_is_not_a_remote_revision(monkeypatch, tmp_path, relative):
    monkeypatch.chdir(tmp_path)
    dataset = "./missing@1/ds" if relative else str(tmp_path / "missing@1" / "ds")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Missing local path was interpreted as a remote repository")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=unexpected_load))
    with pytest.raises(FileNotFoundError, match="Local dataset path does not exist"):
        load_samples(dataset, split="PR_correction")


def test_existing_local_directory_with_at_preserves_literal_name(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset@revision"
    dataset.mkdir()
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return [{"id": "0", "split": "PR_correction", "question_text": "The hat is blue."}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    assert len(load_samples(str(dataset), split="PR_correction")) == 1
    assert calls == [(str(dataset), {"split": "PR_correction"})]


@pytest.mark.parametrize("collapsed", [False, True])
def test_home_relative_local_directory_preserves_at_and_split_fallback(monkeypatch, tmp_path, collapsed):
    dataset = tmp_path / "dataset@revision"
    dataset.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    home_relative = "~/dataset@revision"
    calls = []

    def fake_load_dataset(name, **kwargs):
        assert Path(name).is_absolute() and Path(name).resolve() == dataset
        calls.append(kwargs)
        if collapsed and kwargs.get("split"):
            raise ValueError("Unknown split PR_correction")
        return [{"id": "0", "split": "PR_correction", "question_text": "The hat is blue."}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    assert len(load_samples(home_relative, split="PR_correction")) == 1
    assert calls == ([{"split": "PR_correction"}, {}] if collapsed else [{"split": "PR_correction"}])
