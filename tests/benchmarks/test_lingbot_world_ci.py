# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CI wiring and failure propagation, without substituting model inference."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from benchmarks.lingbot_world.regression import RegressionError, check_binary_activity, check_quality
from tests.dfx.lingbot_world.runner import server_arguments, server_environment

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


# AMD coverage lives in test_amd_cpu_timing.py with main's shared CPU runner.


class _DeviceProperties(NamedTuple):
    uuid: str


@pytest.mark.parametrize("device_env", ["CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"])
@pytest.mark.parametrize("state", ["released", "leaked", "unavailable"])
def test_device_cleanup_is_a_blocking_single_probe(tmp_path, monkeypatch, mocker, device_env, state):
    from tests.dfx.lingbot_world import runner
    from tests.helpers import clean

    queried = []

    @contextmanager
    def device(index):
        queried.append(index)
        yield

    def memory():
        if state == "unavailable":
            raise OSError("device disappeared")
        used = 4 if state == "leaked" and queried[-1] == 1 else 1
        return ((64 - used) * 2**30, 64 * 2**30)

    monkeypatch.setenv(device_env, "4,5")
    monkeypatch.setattr(clean.current_omni_platform, "device_control_env_var", device_env)
    mocker.patch.object(clean.current_omni_platform, "device", side_effect=device)
    mocker.patch.object(clean.current_omni_platform, "mem_get_info", side_effect=memory)
    monkeypatch.setattr(clean, "_whisper_vram_allowance", lambda: (0.0, None))
    wall = iter((100.0, 100.01))
    monkeypatch.setattr(
        clean,
        "time",
        SimpleNamespace(
            time=lambda: next(wall),
            monotonic=lambda: 100.0,
            sleep=lambda _: pytest.fail("post-cleanup assertion must not add another wait"),
        ),
    )
    report = tmp_path / "cleanup.json"
    if state == "released":
        runner.assert_device_cleanup(2, report)
    else:
        expected = OSError if state == "unavailable" else ValueError
        with pytest.raises(expected, match="device disappeared|Device Memory Timeout"):
            runner.assert_device_cleanup(2, report)
    result = json.loads(report.read_text())
    assert result["passed"] is (state == "released")
    assert result["logical_devices"] == [0, 1] and result["threshold_ratio"] == 0.05
    assert queried == ([0] if state == "unavailable" else [0, 1])


@pytest.mark.parametrize("dies", [0, True, 1.0])
def test_device_cleanup_cannot_accept_empty_or_invalid_device_populations(tmp_path, dies):
    from tests.dfx.lingbot_world import runner

    report = tmp_path / "cleanup.json"
    with pytest.raises(RegressionError, match="positive integer die count"):
        runner.assert_device_cleanup(dies, report)
    assert json.loads(report.read_text())["passed"] is False


@pytest.mark.parametrize("timeout", [37, 1200])
def test_server_initialization_and_stage_deadlines_use_the_explicit_profile_budget(timeout):
    settings = {"deploy_config": "benchmarks/lingbot_world/configs/npu_usp4_eager.yaml", "startup_timeout_s": timeout}
    arguments = server_arguments(settings)
    assert dict(zip(arguments[::2], arguments[1::2])) == {
        "--deploy-config": str(ROOT / settings["deploy_config"]),
        "--init-timeout": str(timeout),
        "--stage-init-timeout": str(timeout),
    }


@pytest.mark.parametrize("mask", [None, "7,6,5,4,3,2,1,0", ",".join(f"GPU-{index}" for index in range(8))])
def test_reserved_cuda_allocation_preserves_the_original_device_mapping(mask):
    from tests.dfx.lingbot_world.prepare import select_cuda_devices

    expected = "0,1,2,3" if mask is None else ",".join(mask.split(",")[:4])
    assert select_cuda_devices(8, mask, dies=4, reserved_cards=8) == expected


@pytest.mark.parametrize(
    "available,mask,dies,reserved",
    [(4, None, 4, 8), (16, None, 4, 8), (8, None, 9, 8), (8, "0,1,2,3", 4, 8), (8, "0," * 7 + "0", 4, 8)],
)
def test_reserved_cuda_allocation_rejects_incomplete_or_ambiguous_visibility(available, mask, dies, reserved):
    from tests.dfx.lingbot_world.prepare import select_cuda_devices

    with pytest.raises(RegressionError):
        select_cuda_devices(available, mask, dies=dies, reserved_cards=reserved)


def test_exclusive_allocation_preflight_records_all_reserved_devices(tmp_path, monkeypatch):
    import torch

    from tests.dfx.lingbot_world.prepare import prepare_cuda_allocation

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "H100")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _DeviceProperties(uuid=f"GPU-{index}"))
    output = tmp_path / "allocation.json"
    assert (
        prepare_cuda_allocation({"platform": "cuda", "cards": 4, "dies": 4, "reserved_cards": 8}, output) == "7,6,5,4"
    )
    evidence = json.loads(output.read_text())
    assert evidence["reserved_cards"] == 8 and evidence["visible_dies"] == 4
    assert [device["uuid"] for device in evidence["devices"]] == [f"GPU-{index}" for index in range(8)]


@pytest.mark.parametrize("mask", [None, "7,6,5,4,3,2,1,0", "8,9,10,11,12,13,14,15"])
def test_npu_allocation_preserves_cann_runtime_mapping(mask):
    from tests.dfx.lingbot_world.prepare import select_npu_devices

    expected = "0,1,2,3" if mask is None else ",".join(mask.split(",")[:4])
    assert select_npu_devices(8, mask, dies=4, reserved_cards=8) == expected


@pytest.mark.parametrize(
    "available,mask", [(4, None), (16, None), (8, "0,1,2,3"), (8, "0," * 7 + "0"), (8, "-1,1,2,3,4,5,6,7")]
)
def test_npu_allocation_rejects_partial_or_ambiguous_reservations(available, mask):
    from tests.dfx.lingbot_world.prepare import select_npu_devices

    with pytest.raises(RegressionError):
        select_npu_devices(available, mask, dies=4, reserved_cards=8)


def test_reference_server_keeps_backend_import_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = tmp_path / "cann"
    backend.mkdir()
    (backend / "compiler_backend.py").write_text('origin = "backend"\n')
    for name in ("reference", "candidate"):
        package = tmp_path / name / "vllm_omni"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'origin = "{name}"\n')
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path / "candidate"), str(backend))))
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_omni, compiler_backend; print(vllm_omni.origin, compiler_backend.origin)"],
        cwd=tmp_path,
        env={**os.environ, **server_environment(tmp_path / "reference")},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "reference backend"


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_engine_switches_override_inherited_environment_before_import(tmp_path, monkeypatch, role):
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")
    monkeypatch.setenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "999")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,runpy,sys; config=runpy.run_path(sys.argv[1]); "
            "print(config['contiguous_kv_gather_enabled']()); "
            "print(os.environ['AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH'])",
            str(ROOT / "vllm_omni/experimental/ar_diffusion/kv_cache/config.py"),
        ],
        env={**os.environ, **server_environment(tmp_path / role)},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.splitlines() == ["False", "0"]


def test_fixed_worker_threads_apply_before_torch_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "19")
    monkeypatch.setenv("MKL_NUM_THREADS", "23")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "29")
    env = server_environment(tmp_path / "reference", cpu_threads=1)
    assert all(env[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "1"


@pytest.mark.parametrize("threads", [0, True])
def test_invalid_worker_thread_count_fails_before_launch(tmp_path: Path, threads):
    with pytest.raises(RegressionError, match="positive integer"):
        server_environment(tmp_path, cpu_threads=threads)


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_fixed_worker_threads_also_reach_precision_capture(tmp_path: Path, monkeypatch, role: str):
    from tests.dfx.lingbot_world import runner

    calls = []
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")
    monkeypatch.setenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "999")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    source = tmp_path / role
    runner._capture(
        source,
        tmp_path / "model",
        tmp_path / "precision.npz",
        {
            "height": 480,
            "width": 832,
            "precision_timeout_s": 600,
            "cpu_threads_per_worker": 1,
        },
    )
    assert len(calls) == 1
    options = calls[0][1]
    assert options["cwd"] == source
    assert all(options["env"][name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert options["env"]["VLLM_OMNI_AR_DIFFUSION_KV_GATHER"] == "0"
    assert options["env"]["AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH"] == "0"


def evaluator_repository(root: Path):
    directory = root / "tests/dfx/lingbot_world"
    directory.mkdir(parents=True)
    (directory / "evaluators.lock.json").write_text('{"repositories": {}, "files": {}}')
    (directory / "opencv_provider.py").write_text("# canonical provider policy\n")
    (root / "requirements").mkdir()
    (root / "requirements/lingbot-quality.txt").write_text("example==1\n")


@pytest.mark.parametrize("broken_role", [None, "reference", "candidate"])
def test_source_import_preflight_uses_each_checkout(tmp_path, monkeypatch, broken_role):
    from tests.dfx.lingbot_world import prepare

    for role in ("reference", "candidate"):
        package = tmp_path / role / "vllm_omni"
        cli = package / "entrypoints/cli"
        cli.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        model = package / "diffusion/models/lingbot_world"
        model.mkdir(parents=True)
        (model / "pipeline.py").write_text("")
        (cli / "main.py").write_text("raise ImportError('removed runtime API')\n" if role == broken_role else "")
        if role != broken_role:
            (cli / "main.py").write_text("def main(): pass\n")
    monkeypatch.setattr(prepare, "ROOT", tmp_path / "candidate")
    # A candidate inherited from the job must not shadow the reference.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "candidate"))
    if broken_role:
        with pytest.raises(RegressionError, match=f"{broken_role} serving import preflight failed"):
            prepare.verify_source_imports(tmp_path / "reference")
    else:
        prepare.verify_source_imports(tmp_path / "reference")


def test_evaluator_cache_isolated_by_code_weights_and_dependencies(tmp_path: Path):
    from tests.dfx.lingbot_world.prepare import evaluator_cache_key

    evaluator_repository(tmp_path)
    initial = evaluator_cache_key(tmp_path)
    (tmp_path / "requirements/lingbot-quality.txt").write_text("example==2\n")
    assert evaluator_cache_key(tmp_path) != initial
    (tmp_path / "requirements/lingbot-quality.txt").write_text("example==1\n")
    (tmp_path / "tests/dfx/lingbot_world/evaluators.lock.json").write_text('{"revision": "different"}')
    assert evaluator_cache_key(tmp_path) != initial
    (tmp_path / "tests/dfx/lingbot_world/evaluators.lock.json").write_text('{"repositories": {}, "files": {}}')
    (tmp_path / "tests/dfx/lingbot_world/opencv_provider.py").write_text("# changed canonical provider policy\n")
    assert evaluator_cache_key(tmp_path) != initial


def test_ready_evaluator_is_never_resynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.dfx.lingbot_world import prepare

    repository = tmp_path / "repository"
    evaluator_repository(repository)
    monkeypatch.setattr(prepare, "ROOT", repository)
    monkeypatch.setattr(
        prepare,
        "verify_environment",
        lambda *args: {"opencv-python": "4.11.0.86", "opencv-python-headless": "4.11.0.86"},
    )
    calls = []

    def provision(command, **kwargs):
        calls.append(command)
        if command[:2] == ["uv", "venv"]:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        if len(command) > 2 and command[1] == "-I":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "opencv-python-headless",
                        "provider_version": "4.11.0.86",
                        "opencv_version": "4.11.0",
                        "opencv_build_sha256": "a" * 64,
                        "native_module_sha256": "b" * 64,
                        "module_path": str(tmp_path / "cv2/__init__.py"),
                        "python": command[0],
                        "verified_files": 2,
                    }
                ),
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare.subprocess, "run", provision)
    root = tmp_path / "cache"
    prepare._prepare_evaluators(root)
    prepare._prepare_evaluators(root)
    assert sum(command[:3] == ["uv", "pip", "sync"] for command in calls) == 1
    assert sum(command[:3] == ["uv", "pip", "install"] for command in calls) == 1
    (repository / "requirements/lingbot-quality.txt").write_text("example==2\n")
    with pytest.raises(RegressionError, match="fresh evaluator root"):
        prepare._prepare_evaluators(root)
    assert sum(command[:3] == ["uv", "pip", "sync"] for command in calls) == 1


def test_activity_is_not_a_higher_is_better_quality_score():
    result = check_quality(
        [{"dynamic_degree": 0.1}] * 3,
        {"dynamic_degree": 0.9},
        tolerances={"dynamic_degree": 0.1},
        activity_dimensions=("dynamic_degree",),
    )
    assert not result["passed"]


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_every_configured_quality_dimension_rejects_a_collapse(platform):
    limits = json.loads((ROOT / "tests/dfx/lingbot_world/quality.json").read_text())
    # Both pinned evaluators return these dimensions on a [0, 1] scale.
    # A tolerance accidentally expressed as percentage points must not make
    # a dimension incapable of blocking a regression.
    for suite in ("vbench", "wbench"):
        binary_dimensions = set(limits["binary_activity_dimensions"])
        assert binary_dimensions == {"dynamic_degree"}
        tolerances = limits["candidate_tolerances"][platform][suite]
        assert set(tolerances) - binary_dimensions == set(limits["max_reference_spreads"][platform][suite])
        for dimension, tolerance in tolerances.items():
            if dimension in binary_dimensions:
                assert not check_binary_activity([1, 1, 1], [0, 0, 0], tolerance=tolerance)["passed"]
                assert not check_binary_activity([0, 0, 0], [1, 1, 1], tolerance=tolerance)["passed"]
                continue
            result = check_quality(
                [{dimension: 0.9}] * 3,
                {dimension: 0.0},
                tolerances={dimension: tolerance},
                activity_dimensions=("dynamic_degree",),
                max_reference_spreads={dimension: limits["max_reference_spreads"][platform][suite][dimension]},
            )
            assert not result["passed"], (suite, dimension)


def opencv_provider_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import binascii
    from importlib.metadata import PathDistribution
    from types import ModuleType

    from tests.dfx.lingbot_world import opencv_provider

    site = tmp_path / "site"
    module = site / "cv2"
    module.mkdir(parents=True)
    expected = {"__init__.py": b"# headless loader\n", "cv2.abi3.so": b"headless native binary\n"}
    for name, value in expected.items():
        (module / name).write_bytes(value)
    info = site / "opencv_python_headless-4.11.0.86.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: opencv-python-headless\nVersion: 4.11.0.86\n")
    records = []
    for name, value in expected.items():
        digest = binascii.b2a_base64(hashlib.sha256(value).digest(), newline=False).decode().rstrip("=")
        digest = digest.replace("+", "-").replace("/", "_")
        records.append(f"cv2/{name},sha256={digest},{len(value)}")
    (info / "RECORD").write_text("\n".join(records) + "\n")
    distribution = PathDistribution(info)
    lookup = opencv_provider.metadata.distribution
    monkeypatch.setattr(
        opencv_provider.metadata,
        "distribution",
        lambda name: distribution if name == opencv_provider.PROVIDER else lookup(name),
    )
    cv2 = ModuleType("cv2")
    cv2.__dict__.update(
        __file__=str(module / "__init__.py"),
        __version__="4.11.0",
        getBuildInformation=lambda: "canonical headless build",
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return opencv_provider, module, info, cv2


def test_opencv_provider_verifies_imported_files_not_only_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    provider, module, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    result = provider.verify_opencv_provider("4.11.0.86")
    assert result["provider"] == "opencv-python-headless"
    assert result["provider_version"] == "4.11.0.86"
    assert result["module_path"] == str((module / "__init__.py").resolve())
    assert result["native_module_sha256"] == hashlib.sha256((module / "cv2.abi3.so").read_bytes()).hexdigest()
    assert result["verified_files"] == 2


@pytest.mark.parametrize("overwritten", ["__init__.py", "cv2.abi3.so"])
def test_opencv_provider_rejects_shared_namespace_overwrite(tmp_path, monkeypatch, overwritten):
    provider, module, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    (module / overwritten).write_bytes(b"GUI distribution overwrote this file; package versions stayed the same")
    with pytest.raises(ValueError, match="provider file mismatch"):
        provider.verify_opencv_provider("4.11.0.86")


def test_opencv_provider_rejects_import_from_another_location(tmp_path, monkeypatch):
    provider, _, _, cv2 = opencv_provider_installation(tmp_path, monkeypatch)
    cv2.__file__ = str(tmp_path / "other/cv2/__init__.py")
    with pytest.raises(ValueError, match="imported module"):
        provider.verify_opencv_provider("4.11.0.86")


def test_opencv_provider_requires_recorded_native_binary(tmp_path, monkeypatch):
    provider, _, info, _ = opencv_provider_installation(tmp_path, monkeypatch)
    record = info / "RECORD"
    record.write_text("\n".join(row for row in record.read_text().splitlines() if "cv2.abi3.so" not in row) + "\n")
    with pytest.raises(ValueError, match="native module"):
        provider.verify_opencv_provider("4.11.0.86")


def test_opencv_provider_rejects_wrong_pinned_version(tmp_path, monkeypatch):
    provider, _, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="provider version"):
        provider.verify_opencv_provider("4.12.0.88")


@pytest.mark.parametrize("field,value", [("provider", "opencv-python"), ("verified_files", 0), ("python", "other")])
def test_opencv_provider_record_rejects_incompatible_interpreter_or_owner(tmp_path, monkeypatch, field, value):
    provider, _, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    record = provider.verify_opencv_provider("4.11.0.86")
    record[field] = value
    with pytest.raises(ValueError, match="OpenCV provider"):
        provider.validate_opencv_provider_record(record, expected_version="4.11.0.86", python=sys.executable)


@pytest.mark.parametrize("mutation", ["binary", "record"])
def test_ready_evaluator_rejects_changed_opencv_without_reinstall(tmp_path, monkeypatch, mutation):
    from tests.dfx.lingbot_world import prepare

    repository = tmp_path / "repository"
    evaluator_repository(repository)
    monkeypatch.setattr(prepare, "ROOT", repository)
    monkeypatch.setattr(
        prepare,
        "verify_environment",
        lambda *args: {"opencv-python": "4.11.0.86", "opencv-python-headless": "4.11.0.86"},
    )
    root = tmp_path / "ready"
    python = root / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    (root / "environment-ready").write_text(prepare.evaluator_cache_key(repository) + "\n")
    provider = {
        "schema_version": 1,
        "provider": "opencv-python-headless",
        "provider_version": "4.11.0.86",
        "opencv_version": "4.11.0",
        "opencv_build_sha256": "a" * 64,
        "native_module_sha256": "b" * 64,
        "module_path": str(tmp_path / "cv2/__init__.py"),
        "python": str(python),
        "verified_files": 2,
    }
    stored = {**provider, "opencv_build_sha256": "c" * 64} if mutation == "record" else provider
    (root / "opencv-provider.json").write_text(json.dumps(stored))
    original = {p: p.read_bytes() for p in (root / "environment-ready", root / "opencv-provider.json")}
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        if len(command) > 2 and command[1] == "-I":
            if mutation == "binary":
                raise subprocess.CalledProcessError(1, command, stderr="OpenCV provider file mismatch: cv2/cv2.abi3.so")
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(provider))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare.subprocess, "run", probe)
    with pytest.raises(RegressionError, match="OpenCV provider"):
        prepare._prepare_evaluators(root)
    assert not any(command[:2] == ["uv", "pip"] for command in calls)
    assert all(p.read_bytes() == value for p, value in original.items())
