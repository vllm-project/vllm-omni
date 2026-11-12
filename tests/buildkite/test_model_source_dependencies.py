# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise model job selection through the production CUDA/NPU/ROCm filters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".buildkite/common/scripts"))
sys.path.insert(0, str(ROOT / ".buildkite/amd/scripts"))

from filter_test_suite import filter_amd_nightly_suite  # noqa: E402
from upload_pipeline import filter_pipeline_by_source_dependencies  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINES = {
    "cuda": Path(".buildkite/cuda/test-nightly.yml"),
    "npu": Path(".buildkite/npu/test-npu-nightly.yml"),
    "rocm": Path(".buildkite/amd/test-amd-nightly.yml"),
}
QWEN = "vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py"
WAN = "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"


def _leaves(doc):
    for step in doc.get("steps", []):
        if not isinstance(step, dict):
            continue
        if "steps" in step:
            yield from _leaves(step)
        else:
            yield step


def _keys(step):
    value = step.get("source_file_dependencies", [])
    return [value] if isinstance(value, str) else value


def _selected(path, doc, changes):
    if path == PIPELINES["rocm"] and changes is not None:
        filtered = filter_amd_nightly_suite(doc, changes, pipeline_path=path)
    else:
        filtered = filter_pipeline_by_source_dependencies(doc, changes, pipeline_path=path)
    leaves = list(_leaves(filtered))
    assert all("source_file_dependencies" not in step for step in leaves)
    return {step["label"] for step in leaves}


@pytest.fixture(scope="module", params=PIPELINES)
def nightly(request):
    platform = request.param
    path = PIPELINES[platform]
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    families: dict[str, set[str]] = {}
    for step in _leaves(doc):
        for key in _keys(step):
            if not key.startswith("omni_qwen3_omni_") and (
                key.endswith(("_accuracy", "_perf")) or key == "omni_duplex_eval_ci"
            ):
                families.setdefault(key, set()).add(step["label"])
    # Assert the intended model families exist, without snapshotting job counts.
    for prefix in (
        "omni_minicpmo_4_5_",
        "tts_qwen3_tts_",
        "diffusion_qwen_image_21_",
        "diffusion_hunyuan_image3_",
        "diffusion_wan22_",
        "diffusion_minimax_h3_",
        "diffusion_hunyuan_video_",
    ):
        assert any(key.startswith(prefix) for key in families), (platform, prefix)
    assert bool(families.get("omni_duplex_eval_ci")) == (platform != "npu")
    return platform, path, doc, families


def _labels(families, predicate=lambda key: True):
    return set().union(*(labels for key, labels in families.items() if predicate(key)))


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/engine/orchestrator.py",
        "vllm_omni/engine/stage_pool.py",
        "vllm_omni/config/resolver.py",
        "vllm_omni/outputs/multimodal_accumulation.py",
        "vllm_omni/entrypoints/openai/api_server.py",
        "tests/helpers/runtime.py",
        "tests/helpers/fixtures/runtime.py",
        "tests/conftest.py",
        "pyproject.toml",
        "requirements/common.txt",
    ],
)
def test_shared_runtime_and_infrastructure_keep_evaluations(nightly, changed_file):
    _, path, doc, families = nightly
    # The foreign-model change disables fallback; only direct dependencies count.
    assert _labels(families) <= _selected(path, doc, [changed_file, QWEN])


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/request.py",
        "vllm_omni/core/sched/omni_ar_scheduler.py",
        "vllm_omni/worker/sampling_utils.py",
        "vllm_omni/distributed/omni_connectors/connectors/shm_connector.py",
    ],
)
def test_omni_runtime_keeps_speech_evaluations(nightly, changed_file):
    _, path, doc, families = nightly
    expected = _labels(families, lambda key: not key.startswith("diffusion_"))
    assert expected <= _selected(path, doc, [changed_file, QWEN])


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/engine/duplex_omni_engine.py",
        "vllm_omni/engine/duplex_orchestrator.py",
        "vllm_omni/entrypoints/duplex_omni.py",
    ],
)
def test_minicpm_evaluations_follow_their_duplex_deployment(nightly, changed_file):
    _, path, doc, families = nightly
    expected = _labels(families, lambda key: key.startswith("omni_minicpmo_4_5_") or key == "omni_duplex_eval_ci")
    assert expected <= _selected(path, doc, [changed_file, QWEN])


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/diffusion/worker/diffusion_model_runner.py",
        "vllm_omni/diffusion/worker/diffusion_worker.py",
        "vllm_omni/diffusion/diffusion_engine.py",
        "vllm_omni/diffusion/attention/layer.py",
        "vllm_omni/diffusion/offloader/base.py",
        "vllm_omni/diffusion/model_loader/diffusers_loader.py",
        "vllm_omni/lora/manager.py",
    ],
)
def test_diffusion_runtime_keeps_diffusion_evaluations(nightly, changed_file):
    _, path, doc, families = nightly
    expected = _labels(families, lambda key: key.startswith("diffusion_"))
    assert expected <= _selected(path, doc, [changed_file, QWEN])


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/benchmarks/serve.py",
        "vllm_omni/benchmarks/patch/patch.py",
        "vllm_omni/benchmarks/metrics/metrics.py",
        "vllm_omni/entrypoints/cli/benchmark/serve.py",
    ],
)
def test_serving_benchmark_changes_keep_consumers(nightly, changed_file):
    _, path, doc, families = nightly
    labels = _labels(families)
    expected = {
        step["label"]
        for step in _leaves(doc)
        if step["label"] in labels
        and (
            "tests/dfx/perf/scripts/run_benchmark.py" in "\n".join(step.get("commands", []))
            or "omni_minicpmo_4_5_accuracy" in _keys(step)
        )
    }
    assert expected
    assert expected <= _selected(path, doc, [changed_file, QWEN])


@pytest.mark.parametrize("changed_file", ["tests/e2e/accuracy/helpers.py", "tests/e2e/accuracy/conftest.py"])
def test_accuracy_helpers_select_accuracy_consumers(nightly, changed_file):
    _, path, doc, families = nightly
    expected = _labels(families, lambda key: key.endswith("_accuracy"))
    assert _selected(path, doc, [changed_file, QWEN]) & _labels(families) == expected


def test_diffusion_benchmark_client_selects_its_own_jobs(nightly):
    _, path, doc, families = nightly
    expected = families.get("diffusion_hunyuan_image3_perf", set())
    selected = _selected(path, doc, ["benchmarks/diffusion/diffusion_benchmark_serving.py", QWEN])
    assert selected & _labels(families) == expected


def test_tts_adapters_keep_model_isolation(nightly):
    _, path, doc, families = nightly
    speech = _labels(families, lambda key: not key.startswith("diffusion_"))
    expected = _labels(families, lambda key: key.startswith("tts_qwen3_tts_"))
    selected = _selected(path, doc, ["vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py", WAN])
    assert selected & speech == expected


@pytest.mark.parametrize(
    "changed_file,platform",
    [
        ("docker/Dockerfile.ci", "cuda"),
        ("requirements/cuda.txt", "cuda"),
        (".buildkite/cuda/scripts/build-ci-image.sh", "cuda"),
        ("docker/Dockerfile.npu.ci", "npu"),
        ("requirements/npu.txt", "npu"),
        (".buildkite/npu/scripts/build-npu-image.sh", "npu"),
        ("docker/Dockerfile.rocm", "rocm"),
        ("requirements/rocm.txt", "rocm"),
        (".buildkite/amd/scripts/build-ci-image.sh", "rocm"),
    ],
)
def test_platform_changes_select_only_their_platform(nightly, changed_file, platform):
    actual_platform, path, doc, families = nightly
    labels = _labels(families)
    selected = _selected(path, doc, [changed_file, QWEN]) & labels
    assert selected == (labels if platform == actual_platform else set())


@pytest.mark.parametrize(
    "changed_file",
    [
        "tests/e2e/accuracy/omni_duplex_eval/test_omni_duplex_eval_ci.py",
        "tests/e2e/accuracy/omni_duplex_eval/validation.py",
        "vllm_omni/benchmarks/duplex/omni_duplex_eval_runner.py",
    ],
)
def test_duplex_directories_match_real_files(nightly, changed_file):
    _, path, doc, families = nightly
    expected = families.get("omni_duplex_eval_ci", set())
    assert expected <= _selected(path, doc, [changed_file, QWEN])


def test_v2_runtime_follows_deployment_profiles(nightly):
    platform, path, doc, families = nightly
    speech = _labels(families, lambda key: not key.startswith("diffusion_"))
    expected = set()
    if platform == "cuda":
        expected = speech
    elif platform == "npu":
        expected = _labels(families, lambda key: key.startswith("omni_minicpmo_4_5_"))
    selected = _selected(path, doc, ["vllm_omni/worker_v2/omni_model_runner.py", WAN])
    assert selected & speech == expected


def test_npu_v2_adapter_selects_only_its_consumers(nightly):
    platform, path, doc, families = nightly
    speech = _labels(families, lambda key: not key.startswith("diffusion_"))
    expected = _labels(families, lambda key: key.startswith("omni_minicpmo_4_5_")) if platform == "npu" else set()
    selected = _selected(path, doc, ["vllm_omni/platforms/npu/worker/npu_omni_v2_model_runner.py", WAN])
    assert selected & speech == expected
    qwen = {
        step["label"]
        for step in _leaves(doc)
        if any(key.startswith("omni_qwen3_omni_") and key.endswith(("_accuracy", "_perf")) for key in _keys(step))
    }
    assert not selected & qwen


def test_gpu_v1_workers_do_not_select_diffusion_or_npu_jobs(nightly):
    platform, path, doc, families = nightly
    labels = _labels(families)
    expected = _labels(families, lambda key: not key.startswith("diffusion_")) if platform != "npu" else set()
    selected = _selected(path, doc, ["vllm_omni/worker/gpu_ar_model_runner.py", QWEN])
    assert selected & labels == expected


def test_unrelated_model_changes_do_not_select_evaluations(nightly):
    _, path, doc, families = nightly
    labels = _labels(families)
    assert not (_selected(path, doc, [QWEN]) & labels)
    expected = _labels(families, lambda key: key.startswith("diffusion_wan22_"))
    assert _selected(path, doc, [WAN]) & labels == expected


def test_full_nightly_preserves_every_job(nightly):
    _, path, doc, _ = nightly
    assert _selected(path, doc, None) == {step["label"] for step in _leaves(doc)}


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/engine/orchestrator.py",
        "vllm_omni/worker/gpu_ar_model_runner.py",
        "vllm_omni/worker_v2/omni_model_runner.py",
        "vllm_omni/benchmarks/serve.py",
        "requirements/cuda.txt",
    ],
)
def test_cuda_ready_minicpm_perf_keeps_its_runtime_dependencies(changed_file):
    path = Path(".buildkite/cuda/test-ready.yml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {step["label"] for step in _leaves(doc) if "omni_minicpmo_4_5_perf" in _keys(step)}
    assert expected
    assert expected <= _selected(path, doc, [changed_file, QWEN])


@pytest.mark.parametrize("level", ["ready", "merge"])
def test_cuda_shared_diffusion_runner_keeps_model_e2e(level):
    path = Path(f".buildkite/cuda/test-{level}.yml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    families = {
        key: {step["label"] for step in _leaves(doc) if key in _keys(step)}
        for key in (
            "diffusion_wan22_function",
            "diffusion_hunyuan_video_function",
            "diffusion_lingbot_function",
            "diffusion_minimax_h3_function",
        )
    }
    assert all(families.values())
    assert _labels(families) <= _selected(path, doc, ["vllm_omni/diffusion/worker/diffusion_model_runner.py", QWEN])
