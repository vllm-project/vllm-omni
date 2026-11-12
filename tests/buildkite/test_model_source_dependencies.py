# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise model job selection through the production CUDA/NPU/ROCm filters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.buildkite.source_dependency_helpers import (
    PIPELINES,
    evaluation_families,
    load_pipeline,
)
from tests.buildkite.source_dependency_helpers import dependency_keys as _keys
from tests.buildkite.source_dependency_helpers import leaf_steps as _leaves
from tests.buildkite.source_dependency_helpers import nightly_pipeline_fixture as nightly_pipeline_fixture
from tests.buildkite.source_dependency_helpers import selected_labels as _selected

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

QWEN = "vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py"
MINICPM = "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py"
WAN = "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"


@pytest.fixture(scope="module")
def nightly(nightly_pipeline):
    """Discover non-Qwen3-Omni evaluations from the shared live inventory."""
    platform, path, doc = nightly_pipeline
    families = {
        key: labels for key, labels in evaluation_families(doc).items() if not key.startswith("omni_qwen3_omni_")
    }
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
            "tests/dfx/perf/scripts/run_benchmark.py"
            in json.dumps({"commands": step.get("commands", []), "env": step.get("env", {})})
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
    "changed_file,platforms",
    [
        ("vllm_omni/platforms/cuda/platform.py", {"cuda"}),
        ("docker/Dockerfile.ci", {"cuda"}),
        ("requirements/cuda.txt", {"cuda"}),
        (".buildkite/cuda/scripts/build-ci-image.sh", {"cuda"}),
        (".buildkite/cuda/test-nightly.yml", {"cuda"}),
        ("vllm_omni/platforms/npu/attention/static_shape_decode.py", {"npu"}),
        ("vllm_omni/platforms/npu/platform.py", {"npu"}),
        ("docker/Dockerfile.npu.ci", {"npu"}),
        ("requirements/npu.txt", {"npu"}),
        (".buildkite/npu/scripts/build-npu-image.sh", {"npu"}),
        (".buildkite/npu/test-npu-nightly.yml", {"npu"}),
        ("vllm_omni/platforms/rocm/platform.py", {"rocm"}),
        ("docker/Dockerfile.rocm", {"rocm"}),
        ("requirements/rocm.txt", {"rocm"}),
        (".buildkite/amd/scripts/build-ci-image.sh", {"rocm"}),
        (".buildkite/amd/test-amd-nightly.yml", {"rocm"}),
        (".buildkite/amd/test-template-amd-omni.j2", {"rocm"}),
    ],
)
@pytest.mark.parametrize("qwen_family", [False, True], ids=["other-models", "qwen3-omni"])
def test_platform_changes_select_only_their_platform(nightly_pipeline, changed_file, platforms, qwen_family):
    actual_platform, path, doc = nightly_pipeline
    families = evaluation_families(doc)
    labels = _labels(families, lambda key: key.startswith("omni_qwen3_omni_") == qwen_family)
    # Use a different model to disable fallback without selecting the tested family.
    foreign_model = MINICPM if qwen_family else QWEN
    selected = _selected(path, doc, [changed_file, foreign_model]) & labels
    assert selected == (labels if actual_platform in platforms else set())


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
    "routing_file",
    [
        ".buildkite/common/ci_source_file_dependencies.yml",
        ".buildkite/common/scripts/upload_pipeline.py",
        ".buildkite/common/scripts/ci_conditions.py",
    ],
)
@pytest.mark.parametrize(
    "model_file",
    [QWEN, WAN, "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py"],
)
def test_routing_metadata_does_not_expand_model_evaluations(nightly_pipeline, routing_file, model_file):
    """Registry edits accompanying a model change retain the model's selection."""
    _, path, doc = nightly_pipeline
    labels = _labels(evaluation_families(doc))
    model_selection = _selected(path, doc, [model_file]) & labels
    assert model_selection
    assert _selected(path, doc, [routing_file, model_file]) & labels == model_selection


@pytest.fixture(
    scope="module",
    params=[
        "tts_qwen3_tts_function",
        "tts_qwen3_tts_perf",
        "tts_qwen3_tts_cov",
        "tts_voxtral_tts_perf",
        "reliability_invalid_param_l4",
    ],
)
def tts_consumers(request):
    """Find every live consumer of keys narrowed by the shared-runtime extraction."""
    consumers = []
    paths = [
        *PIPELINES.values(),
        *(Path(f".buildkite/cuda/test-{level}.yml") for level in ("ready", "merge", "weekly")),
    ]
    for path in paths:
        doc = load_pipeline(path)
        labels = {step["label"] for step in _leaves(doc) if request.param in _keys(step)}
        if labels:
            consumers.append((path, doc, labels))
    assert consumers, request.param
    return consumers


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/core/sched/__init__.py",
        "vllm_omni/distributed/omni_connectors/__init__.py",
        "vllm_omni/distributed/omni_connectors/connectors/__init__.py",
        "vllm_omni/distributed/omni_connectors/connectors/yuanrong_connector.py",
        "vllm_omni/entrypoints/openai/tts_adapters/__init__.py",
        "vllm_omni/entrypoints/openai/tts_adapters/capabilities.py",
    ],
)
def test_package_entries_keep_tts_consumers(tts_consumers, changed_file):
    """Package imports remain dependencies when broad directories are narrowed."""
    for path, doc, labels in tts_consumers:
        assert labels <= _selected(path, doc, [changed_file, WAN]), (path, changed_file)


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/engine/orchestrator.py",
        "vllm_omni/engine/stage_pool.py",
        "vllm_omni/worker/sampling_utils.py",
        "vllm_omni/worker_v2/omni_model_runner.py",
        "vllm_omni/worker/gpu_ar_model_runner.py",
        "vllm_omni/core/prefix_cache/manager.py",
        "vllm_omni/diffusion/worker/diffusion_model_runner.py",
        "vllm_omni/entrypoints/openai/tts_adapters/higgs_audio_v3.py",
        "vllm_omni/entrypoints/openai/tts_adapters/audex_tta.py",
        "tests/helpers/runtime.py",
        "tests/conftest.py",
    ],
)
def test_weekly_l4_keeps_execution_dependencies(changed_file):
    """Keep the shared runtimes and model adapters used by the L4 deployments."""
    path = Path(".buildkite/cuda/test-weekly.yml")
    doc = load_pipeline(path)
    expected = {step["label"] for step in _leaves(doc) if "reliability_invalid_param_l4" in _keys(step)}
    assert expected
    assert expected <= _selected(path, doc, [changed_file, QWEN])


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
    doc = load_pipeline(path)
    expected = {step["label"] for step in _leaves(doc) if "omni_minicpmo_4_5_perf" in _keys(step)}
    assert expected
    assert expected <= _selected(path, doc, [changed_file, QWEN])


@pytest.mark.parametrize("level", ["ready", "merge"])
def test_cuda_shared_diffusion_runner_keeps_model_e2e(level):
    path = Path(f".buildkite/cuda/test-{level}.yml")
    doc = load_pipeline(path)
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
