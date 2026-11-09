# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Qwen3 nightly selection with real jobs, including mixed-model diffs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".buildkite" / "common" / "scripts"))

from upload_pipeline import filter_pipeline_by_source_dependencies  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINES = {
    "cuda": Path(".buildkite/cuda/test-nightly.yml"),
    "npu": Path(".buildkite/npu/test-npu-nightly.yml"),
    "rocm": Path(".buildkite/amd/test-amd-nightly.yml"),
}
MINICPM = "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py"
FAMILIES = ("accuracy", "perf", "duplex_perf")


def _leaves(doc):
    for step in doc.get("steps", []):
        if isinstance(step, dict):
            if "steps" in step:
                yield from _leaves(step)
            else:
                yield step


@pytest.fixture(scope="module", params=PIPELINES)
def nightly(request):
    platform = request.param
    path = PIPELINES[platform]
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    families: dict[str, set[str]] = {family: set() for family in FAMILIES}
    for step in _leaves(doc):
        dependencies = step.get("source_file_dependencies", [])
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        for family in FAMILIES:
            if f"omni_qwen3_omni_{family}" in dependencies:
                families[family].add(step["label"])
    assert [len(families[family]) for family in FAMILIES] == {
        "cuda": [1, 2, 2],
        "npu": [1, 2, 0],
        "rocm": [1, 2, 0],
    }[platform]
    return platform, path, doc, families


def _selected(nightly, changed_files):
    _, path, doc, _ = nightly
    filtered = filter_pipeline_by_source_dependencies(doc, changed_files, pipeline_path=path)
    leaves = list(_leaves(filtered))
    assert all("source_file_dependencies" not in step for step in leaves)
    return {step["label"] for step in leaves}


@pytest.mark.parametrize("mixed", [False, True], ids=["alone", "with-minicpm"])
@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/core/sched/output.py",
        "vllm_omni/request.py",
        "vllm_omni/engine/orchestrator.py",
        "vllm_omni/engine/stage_engine_core_proc.py",
        "vllm_omni/distributed/omni_connectors/connectors/shm_connector.py",
        "vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py",
        # NPU inherits this runner despite its GPU name.
        "vllm_omni/worker/gpu_model_runner.py",
        "vllm_omni/worker/sampling_utils.py",
        "vllm_omni/worker/runner_assisted_metadata.py",
        "vllm_omni/engine/serialization.py",
        "vllm_omni/outputs/multimodal_accumulation.py",
        "vllm_omni/model_executor/stage_input_processors/tts_utils.py",
        "vllm_omni/config/resolver.py",
        "vllm_omni/model_executor/models/registry.py",
        "vllm_omni/patch.py",
        "vllm_omni/platforms/interface.py",
        "vllm_omni/entrypoints/openai/api_server.py",
        "vllm_omni/inputs/preprocess.py",
        "vllm_omni/utils/audio_resample.py",
        "vllm_omni/model_executor/layers/rotary_embedding/mrope.py",
        "vllm_omni/diffusion/layers/custom_op.py",
        "vllm_omni/model_executor/model_loader/weight_utils.py",
        "vllm_omni/quantization/component_config.py",
        "vllm_omni/benchmarks/serve.py",
        "vllm_omni/benchmarks/patch/patch.py",
        "vllm_omni/benchmarks/metrics/metrics.py",
        "vllm_omni/entrypoints/cli/benchmark/cli_args.py",
        "tests/helpers/runtime.py",
        "tests/helpers/fixtures/pytest_collection.py",
        "tests/conftest.py",
        "pyproject.toml",
        "setup.py",
        "requirements/common.txt",
        ".buildkite/common/ci_mirror_hardwares.yml",
    ],
)
def test_shared_changes_keep_accuracy_and_perf(nightly, changed_file, mixed):
    changes = [changed_file, MINICPM] if mixed else [changed_file]
    expected = set().union(*nightly[3].values())
    assert expected <= _selected(nightly, changes)


@pytest.mark.parametrize(
    "changed_file,platforms",
    [
        ("vllm_omni/worker/gpu_ar_model_runner.py", {"cuda", "rocm"}),
        ("vllm_omni/worker_v2/omni_model_runner.py", {"cuda"}),
        ("vllm_omni/core/prefix_cache/manager.py", {"cuda"}),
        ("vllm_omni/platforms/cuda/platform.py", {"cuda"}),
        ("docker/Dockerfile.ci", {"cuda"}),
        ("requirements/cuda.txt", {"cuda"}),
        (".buildkite/cuda/scripts/build-ci-image.sh", {"cuda"}),
        (".buildkite/cuda/test-nightly.yml", {"cuda"}),
        ("vllm_omni/platforms/npu/worker/npu_ar_model_runner.py", {"npu"}),
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
def test_platform_dependencies_are_scoped(nightly, changed_file, platforms):
    platform, _, _, families = nightly
    qwen_labels = set().union(*families.values())
    # A real model match disables fallback, including for changed pipeline YAML.
    selected = _selected(nightly, [changed_file, MINICPM]) & qwen_labels
    assert selected == (qwen_labels if platform in platforms else set())


@pytest.mark.parametrize(
    "changed_file,expected_families",
    [
        ("tests/e2e/accuracy/helpers.py", {"accuracy"}),
        ("tests/e2e/accuracy/conftest.py", {"accuracy"}),
        ("vllm_omni/benchmarks/data_modules/daily_omni_eval.py", {"accuracy", "duplex_perf"}),
        ("vllm_omni/benchmarks/data_modules/seed_tts_dataset.py", {"accuracy", "duplex_perf"}),
        ("tests/dfx/conftest.py", {"perf", "duplex_perf"}),
        ("tests/dfx/perf/scripts/run_benchmark.py", {"perf", "duplex_perf"}),
        ("vllm_omni/benchmarks/data_modules/random_multi_modal_dataset.py", {"perf", "duplex_perf"}),
        ("vllm_omni/model_executor/duplex_sampling.py", {"duplex_perf"}),
        ("vllm_omni/outputs/duplex.py", {"duplex_perf"}),
        ("vllm_omni/core/sched/duplex_stage_work.py", {"duplex_perf"}),
        ("vllm_omni/deploy/qwen3_omni_duplex_mrv2.yaml", {"duplex_perf"}),
        (MINICPM, set()),
        ("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py", set()),
        ("vllm_omni/platforms/npu/graph_tools.py", set()),
        ("vllm_omni/distributed/omni_connectors/connectors/mooncake_store_connector.py", set()),
    ],
)
def test_tooling_and_optional_features_select_relevant_jobs(nightly, changed_file, expected_families):
    families = nightly[3]
    qwen_labels = set().union(*families.values())
    expected = set().union(*(families[family] for family in expected_families))
    assert _selected(nightly, [changed_file, MINICPM]) & qwen_labels == expected


def test_full_nightly_keeps_all_jobs(nightly):
    doc = nightly[2]
    assert _selected(nightly, None) == {step["label"] for step in _leaves(doc)}
