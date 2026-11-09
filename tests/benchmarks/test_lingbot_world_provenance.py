# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Source images must retain honest client and server measurement identities."""

import subprocess

import pytest

from benchmarks.lingbot_world import benchmark_lingbot_world_realtime as realtime
from benchmarks.lingbot_world.regression import RegressionError
from tests.dfx.lingbot_world import runner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def source_image(tmp_path, monkeypatch):
    image = tmp_path / "image"
    image.mkdir()
    reference = tmp_path / "reference"
    subprocess.run(["git", "init", "-q", str(reference)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(reference),
            "-c",
            "user.name=CI fixture",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Frozen reference",
        ],
        check=True,
    )
    reference_revision = runner.git_revision(reference)
    monkeypatch.setattr(runner, "ROOT", image)
    monkeypatch.setattr(realtime, "REPO_ROOT", image)
    monkeypatch.setenv("BUILDKITE_COMMIT", "b" * 40)
    assert not (image / ".git").exists()
    # Standalone installed clients must still refuse to claim a caller's SHA.
    assert realtime._git_revision() is None
    return image, reference, reference_revision


@pytest.mark.parametrize("client_revision", [None, "b" * 40])
def test_source_image_attests_missing_client_revision_without_changing_server(source_image, client_revision):
    _, reference, reference_revision = source_image
    document = {"git_revision": client_revision}
    runner.record_benchmark_revisions(document, reference)
    assert document == {"git_revision": "b" * 40, "server_revision": reference_revision}
    assert reference_revision != document["git_revision"]


@pytest.mark.parametrize("revision", [None, "", "HEAD", "a" * 39, "a" * 41, "A" * 40, "g" * 40])
def test_source_image_requires_full_validated_commit(source_image, monkeypatch, revision):
    _, reference, _ = source_image
    if revision is None:
        monkeypatch.delenv("BUILDKITE_COMMIT")
    else:
        monkeypatch.setenv("BUILDKITE_COMMIT", revision)
    document = {"git_revision": None}
    with pytest.raises(RegressionError, match="full BUILDKITE_COMMIT"):
        runner.record_benchmark_revisions(document, reference)
    assert document == {"git_revision": None}


@pytest.mark.parametrize("document", [{}, {"git_revision": "c" * 40}, {"git_revision": ""}])
def test_source_image_rejects_conflicting_or_absent_client_claim(source_image, document):
    _, reference, _ = source_image
    original = document.copy()
    with pytest.raises(RegressionError, match="client revision differs"):
        runner.record_benchmark_revisions(document, reference)
    assert document == original


def test_real_checkout_wins_over_buildkite_and_rejects_conflicting_client(source_image, monkeypatch):
    _, reference, reference_revision = source_image
    monkeypatch.setattr(runner, "ROOT", reference)
    assert runner.git_revision(reference) == reference_revision
    document = {"git_revision": "b" * 40}
    with pytest.raises(RegressionError, match="client revision differs"):
        runner.record_benchmark_revisions(document, reference)
    assert document == {"git_revision": "b" * 40}
    document["git_revision"] = reference_revision
    runner.record_benchmark_revisions(document, reference)
    assert document == {"git_revision": reference_revision, "server_revision": reference_revision}
