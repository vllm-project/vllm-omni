# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime migrations and reference collection must preserve acceptance boundaries."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.lingbot_world.regression import RegressionError, contract_digest, read_json, sha256, write_json
from tests.dfx.lingbot_world import prepare, runner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("torch_version", ["2.14.0", "2.14.0+cu130", "2.13.0", "2.14.0rc1", "2.14.0.dev1"])
def test_cuda_runtime_requires_matching_final_torch(monkeypatch, torch_version):
    monkeypatch.setattr(
        prepare.importlib.metadata, "version", lambda package: "0.33.0" if package == "vllm" else torch_version
    )
    if torch_version in {"2.14.0", "2.14.0+cu130"}:
        assert prepare.verify_inference_runtime("cuda")["packages"] == {"vllm": "0.33.0", "torch": torch_version}
    else:
        with pytest.raises(RegressionError, match="requires torch 2.14.0"):
            prepare.verify_inference_runtime("cuda")


@pytest.mark.parametrize("missing", [None, "torch-npu", "vllm-ascend"])
def test_npu_checks_integrated_runtime_without_installing_cuda_packages(monkeypatch, missing):
    versions = {
        "vllm": "0.33.0+empty",
        "torch": "2.10.0+cpu",
        "torch-npu": "2.10.0.post7",
        "vllm-ascend": "0.19.1rc2.dev3349+g221fb096c.d20261104",
    }

    def version(package):
        if package == missing:
            raise importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(prepare.importlib.metadata, "version", version)
    attestation = {"source_snapshot_sha256": prepare.NPU_SOURCE_SNAPSHOT_SHA256}
    monkeypatch.setattr(prepare, "attest_runtime_sources", lambda expected, **kwargs: attestation)
    if missing:
        with pytest.raises(RegressionError, match=f"integrated {missing}"):
            prepare.verify_inference_runtime("npu")
    else:
        observed = prepare.verify_inference_runtime("npu")
        assert observed == {
            "packages": versions,
            "source_revisions": prepare.NPU_RUNTIME["source_revisions"],
            "source_attestations": attestation,
        }


@pytest.mark.parametrize("mutation", [None, "missing", "wrong", "boolean"])
def test_npu_profile_requires_reviewed_source_packages_digest(monkeypatch, mutation):
    lock = read_json(prepare.ROOT / "tests/dfx/lingbot_world/runtime_sources.lock.json")
    assert prepare.NPU_SOURCE_PACKAGES_SHA256 == contract_digest(lock["packages"])
    profile = json.loads(json.dumps(prepare.NPU_RUNTIME))
    if mutation == "missing":
        profile.pop("source_packages_sha256")
    elif mutation is not None:
        profile["source_packages_sha256"] = True if mutation == "boolean" else "0" * 64
    calls = []

    def attest(expected, *, expected_sha256):
        calls.append((expected, expected_sha256))
        return {"packages": lock["packages"]}

    monkeypatch.setattr(prepare.importlib.metadata, "version", prepare.NPU_PACKAGES.__getitem__)
    monkeypatch.setattr(prepare, "attest_runtime_sources", attest)
    if mutation is None:
        observed = prepare.verify_inference_runtime("npu", profile_runtime=profile)
        assert observed["source_attestations"] == {"packages": lock["packages"]}
        assert calls == [(prepare.NPU_SOURCE_REVISIONS, prepare.NPU_SOURCE_SNAPSHOT_SHA256)]
    else:
        with pytest.raises(RegressionError, match="pinned runtime contract"):
            prepare.verify_inference_runtime("npu", profile_runtime=profile)
        assert calls == []


@pytest.mark.parametrize("mutation", [None, "source", "dirty", "missing_provenance", "remote_provenance"])
def test_npu_runtime_requires_pinned_clean_editable_source(tmp_path, monkeypatch, mutation):
    expected = {"vllm": "a" * 40}
    provenance = {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    if mutation == "remote_provenance":
        provenance["url"] = "https://example.test/vendor-runtime"
    monkeypatch.setattr(
        prepare.importlib.metadata,
        "distribution",
        lambda package: SimpleNamespace(
            read_text=lambda name: None if mutation == "missing_provenance" else json.dumps(provenance)
        ),
    )

    def source(command, **kwargs):
        assert command[:3] == ["git", "-C", str(tmp_path)]
        if "rev-parse" in command:
            return ("b" if mutation == "source" else "a") * 40
        return " M runtime.py" if mutation == "dirty" else ""

    monkeypatch.setattr(prepare.subprocess, "check_output", source)
    if mutation is None:
        assert prepare.verify_editable_runtime_sources(expected) == expected
    else:
        with pytest.raises(RegressionError, match="runtime source"):
            prepare.verify_editable_runtime_sources(expected)


@pytest.mark.parametrize("broken", ["lazy_cli", "pipeline"])
def test_source_preflight_exercises_model_and_lazy_cli_imports(tmp_path, monkeypatch, broken):
    for role in ("reference", "candidate"):
        source = tmp_path / role / "vllm_omni"
        cli = source / "entrypoints/cli"
        cli.mkdir(parents=True)
        (source / "__init__.py").write_text("")
        (cli / "main.py").write_text(
            "def main():\n    raise ImportError('removed lazy startup API')\n"
            if role == "reference" and broken == "lazy_cli"
            else "def main(): pass\n"
        )
        model = source / "diffusion/models/lingbot_world"
        model.mkdir(parents=True)
        (model / "pipeline.py").write_text(
            "raise ImportError('removed model runtime API')\n" if role == "reference" and broken == "pipeline" else ""
        )
    monkeypatch.setattr(prepare, "ROOT", tmp_path / "candidate")
    with pytest.raises(RegressionError, match="reference serving import preflight failed"):
        prepare.verify_source_imports(tmp_path / "reference")


@pytest.fixture
def reference_collection(tmp_path, monkeypatch):
    """Drive all three real collector iterations with synthetic server outputs."""
    from tests.helpers import runtime

    root = tmp_path / "harness"
    root.mkdir()
    deployment = root / "deploy.yaml"
    deployment.write_text("stages: []\n")
    protocol = root / "tests/dfx/lingbot_world/calibration/v033_reference_protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}\n")
    reference = tmp_path / "frozen-reference"
    reference.mkdir()
    profile = {
        "platform": "cuda",
        "reference_revision": "a" * 40,
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
        "dies": 4,
        "lifecycle_checks": False,
    }
    profile_path = tmp_path / "profile.json"
    write_json(profile_path, profile)
    settings = {**profile, "output": tmp_path / "generation", "profile_path": profile_path}
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "git_revision", lambda source: "a" * 40 if source == reference else "b" * 40)
    monkeypatch.setattr(runner, "checked_assets", lambda settings: (tmp_path / "model", {"image": tmp_path / "image"}))
    monkeypatch.setattr(runner, "validate_reference", lambda settings: reference)
    monkeypatch.setattr(runner, "hardware_contract", lambda settings: {"platform": "cuda"})
    monkeypatch.setattr(runner, "server_arguments", lambda settings: [])
    monkeypatch.setattr(runner, "_measured_metrics", lambda document, settings: {"ttfc_ms": 10.0})
    monkeypatch.setattr(runner, "_finalize_measurements", lambda *args: pytest.fail("acceptance invoked while fitting"))
    captures = []
    starts = []

    def capture(source, model, destination, settings):
        captures.append(source)
        destination.write_bytes(b"fixed precision payload")

    class Server:
        def __init__(self, *args, source_root, **kwargs):
            starts.append(Path(source_root))
            self.proc = SimpleNamespace(pid=1)
            self.port = 8181

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    import psutil

    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(children=lambda recursive: []))
    monkeypatch.setattr(runtime, "OmniServer", Server)
    monkeypatch.setattr(runner, "_capture", capture)
    monkeypatch.setattr(runner, "diagnostic_session", lambda server, settings, image, path: write_json(path, {}))
    monkeypatch.setattr(runner, "assert_device_cleanup", lambda dies, path: write_json(path, {}))
    monkeypatch.setattr(runner, "parse_args", lambda args: args)
    requests: list[str] = []

    async def benchmark(args):
        video = Path(args[args.index("--save-video") + 1])
        video.write_bytes(b"reference-only video")
        identity = f"independent-request-{len(requests)}"
        requests.append(identity)
        return {"git_revision": "b" * 40, "sessions": [{"request_id": identity}]}

    monkeypatch.setattr(runner, "run_benchmark", benchmark)
    report = runner.generate(settings, reference_only=True)
    return settings, report, reference, captures, starts


def test_reference_collection_never_runs_candidate_or_acceptance(reference_collection):
    settings, report, reference, captures, starts = reference_collection
    assert captures == starts == [reference] * 3
    assert set(report["records"]) == {"reference"}
    assert report["complete"] and report["measurement_only"] and report["reference_only"]
    assert report["passed"] is False
    assert not (settings["output"] / "candidate").exists()
    runner.validate_reference_collection(settings, settings["output"], report)


@pytest.mark.parametrize(
    "mutation", ["accepted", "candidate", "missing", "reused", "changed", "escape", "identity", "backend"]
)
def test_reference_collection_rejects_incomplete_or_contaminated_inputs(reference_collection, mutation):
    settings, report, *_ = reference_collection
    output = settings["output"]
    if mutation == "accepted":
        report["passed"] = True
    elif mutation == "candidate":
        (output / "candidate").mkdir()
    elif mutation == "missing":
        del report["artifacts"]["reference/2/scene/rollout.mp4"]
    elif mutation == "reused":
        path = output / "reference/2/scene/benchmark.json"
        raw = read_json(path)
        raw["sessions"][0]["request_id"] = "independent-request-0"
        write_json(path, raw)
        report["artifacts"][str(path.relative_to(output))] = sha256(path)
    elif mutation == "changed":
        (output / "reference/1/scene/rollout.mp4").write_bytes(b"changed")
    elif mutation == "identity":
        report["collection_id"] = "unrecorded"
    elif mutation == "backend":
        report["collection_backend"] = "buildkite"
        report["buildkite_job_id"] = None
    else:
        report["artifacts"]["reference/../../outside"] = "0" * 64
    with pytest.raises(RegressionError):
        runner.validate_reference_collection(settings, output, report)


def test_paired_acceptance_rejects_reference_collection(reference_collection, monkeypatch, tmp_path):
    from tests.dfx.lingbot_world import test_quality

    settings, report, *_ = reference_collection
    monkeypatch.setattr(test_quality, "git_revision", lambda root: report["harness_revision"])
    with pytest.raises(RegressionError, match="cannot establish candidate acceptance"):
        test_quality.evaluate_quality(
            {**settings, "generation": settings["output"], "output": tmp_path / "quality"}, monkeypatch
        )
