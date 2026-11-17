# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Source images retain honest candidate client/server measurement identity."""

import pytest

from tests.e2e.accuracy.lingbot_world import runner
from vllm_omni.benchmarks.realtime_video import client as realtime
from vllm_omni.benchmarks.realtime_video.metrics import RegressionError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def source_image(tmp_path, monkeypatch):
    image = tmp_path / "image"
    image.mkdir()
    monkeypatch.setattr(runner, "ROOT", image)
    monkeypatch.setattr(realtime, "REPO_ROOT", image)
    monkeypatch.setenv("BUILDKITE_COMMIT", "b" * 40)
    assert realtime.git_revision() is None


@pytest.mark.parametrize("client_revision", [None, "b" * 40])
def test_source_image_attests_client_and_single_candidate_revision(source_image, client_revision):
    document = {"git_revision": client_revision}
    runner.record_benchmark_revisions(document)
    assert document == {"git_revision": "b" * 40, "server_revision": "b" * 40}


@pytest.mark.parametrize("revision", [None, "", "HEAD", "a" * 39, "a" * 41, "A" * 40, "g" * 40])
def test_source_image_requires_full_validated_commit(source_image, monkeypatch, revision):
    if revision is None:
        monkeypatch.delenv("BUILDKITE_COMMIT")
    else:
        monkeypatch.setenv("BUILDKITE_COMMIT", revision)
    document = {"git_revision": None}
    with pytest.raises(RegressionError, match="full BUILDKITE_COMMIT"):
        runner.record_benchmark_revisions(document)
    assert document == {"git_revision": None}


@pytest.mark.parametrize("document", [{}, {"git_revision": "c" * 40}, {"git_revision": ""}])
def test_source_image_rejects_conflicting_or_absent_client_claim(source_image, document):
    original = document.copy()
    with pytest.raises(RegressionError, match="client revision differs"):
        runner.record_benchmark_revisions(document)
    assert document == original
