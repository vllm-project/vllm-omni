# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import load_samples

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_remote_dataset_revision_is_passed_to_loader(monkeypatch):
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return [{"id": "0", "split": "PR_correction", "question_text": "The hat is blue."}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    samples = load_samples("Hothan/Omni-DuplexEval@fixed-revision", split="PR_correction")
    assert len(samples) == 1
    assert calls == [("Hothan/Omni-DuplexEval", {"revision": "fixed-revision", "split": "PR_correction"})]
