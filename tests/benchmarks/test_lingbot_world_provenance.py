# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Source images must retain honest client and server measurement identities."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from tests.dfx.lingbot_world import runner
from vllm_omni.benchmarks.lingbot_world import realtime

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


@pytest.mark.parametrize(("platform", "repeat"), [("cuda", None), ("npu", 1)])
def test_reference_generation_from_source_image_retains_strict_raw_provenance(
    source_image, monkeypatch, platform, repeat
):
    import psutil

    from tests.helpers import runtime

    image, reference, reference_revision = source_image
    (image / "deploy.yaml").write_text("stages: []\n")
    protocol = image / "tests/dfx/lingbot_world/calibration/v033_reference_protocol.json"
    write_json(protocol, {})
    profile = {
        "platform": platform,
        "reference_revision": reference_revision,
        "deploy_config": "deploy.yaml",
        "cases": [{"id": "scene", "image": "image", "prompt": "test", "camera_pattern": "hold", "seed": 42}],
        "height": 24,
        "width": 32,
        "mux_fps": 16,
        "target_fps": 12,
        "num_chunks": 3,
        "warmup_chunks": 1,
        "warmup_num_chunks": 1,
        "first_chunk_timeout_s": 10,
        "chunk_timeout_s": 10,
        "startup_timeout_s": 10,
        "dies": 1,
        "lifecycle_checks": False,
    }
    profile_path = image / "profile.json"
    write_json(profile_path, profile)
    settings = {**profile, "profile_path": profile_path, "output": image / "output"}
    starts = []

    class Server:
        def __init__(self, *args, source_root, **kwargs):
            starts.append(Path(source_root))
            self.proc = SimpleNamespace(pid=1)
            self.port = 8181

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    async def benchmark(args):
        video = Path(args[args.index("--save-video") + 1])
        video.write_bytes(b"reference video")
        document = {"git_revision": realtime._git_revision(), "sessions": [{"request_id": str(video)}]}
        write_json(video.with_name("benchmark.json"), document)
        return document

    monkeypatch.setattr(runtime, "OmniServer", Server)
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(children=lambda recursive: []))
    monkeypatch.setattr(runner, "checked_assets", lambda settings: (image / "model", {"image": image / "image"}))
    monkeypatch.setattr(runner, "validate_reference", lambda settings: reference)
    monkeypatch.setattr(runner, "hardware_contract", lambda settings: {})
    monkeypatch.setattr(runner, "server_arguments", lambda settings: [])
    monkeypatch.setattr(runner, "_capture", lambda source, model, path, settings: path.write_bytes(b"precision"))
    monkeypatch.setattr(runner, "diagnostic_session", lambda server, settings, image, path: write_json(path, {}))
    monkeypatch.setattr(runner, "assert_device_cleanup", lambda dies, path: write_json(path, {}))
    monkeypatch.setattr(runner, "_measured_metrics", lambda document, settings: {"ttfc_ms": 10.0})
    monkeypatch.setattr(runner, "parse_args", lambda args: args)
    monkeypatch.setattr(runner, "run_benchmark", benchmark)
    report = runner.generate(settings, reference_only=True, measurement_repeat=repeat)
    assert report["harness_revision"] == "b" * 40
    assert report["complete"] is True and report["passed"] is False
    assert starts == [reference] * (3 if repeat is None else 1)
    assert not (settings["output"] / "candidate").exists()
    runner.validate_reference_collection(settings, settings["output"], report, measurement_repeat=repeat)
    raw_path = settings["output"] / "reference" / str(0 if repeat is None else repeat) / "scene/benchmark.json"
    raw = read_json(raw_path)
    assert raw["git_revision"] == "b" * 40
    assert raw["server_revision"] == reference_revision
    # A fully rehashed conflicting source claim must still fail strict validation.
    raw["git_revision"] = "c" * 40
    write_json(raw_path, raw)
    report["artifacts"][str(raw_path.relative_to(settings["output"]))] = sha256(raw_path)
    with pytest.raises(RegressionError, match="source or raw measurements changed"):
        runner.validate_reference_collection(settings, settings["output"], report, measurement_repeat=repeat)
