# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Non-duplex nightly: reuse c64 inference for quality and performance."""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pytest

from tests.e2e.accuracy import helpers
from tests.e2e.accuracy.qwen3_omni import qwen3_omni_acc_bench_core as core
from tests.e2e.accuracy.qwen3_omni.nightly import (
    DATASETS,
    MANIFEST,
    MODEL,
    NightlyCase,
    accuracy_coverage_errors,
    dataset_args,
    nightly_cases,
    performance_gate,
    quality_baseline,
    resolve_dataset,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.benchmarks.validation import coverage_errors

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

_SERVERS = {
    config: OmniServerParams(
        model=MODEL,
        stage_config_path=get_deploy_config_path(f"ci/qwen3_omni_nightly_{config.replace('-', '_')}.yaml"),
    )
    for config in ("default", "async-prefix")
}


def _params(platform: str) -> list:
    return [pytest.param(_SERVERS[case.config], case, id=case.id) for case in nightly_cases(platform)]


@pytest.fixture(scope="module", autouse=True)
def nightly_sources() -> dict[str, Path]:
    # Resolve once, outside measured inference. A missing offline snapshot fails
    # with a staging error instead of silently selecting a different revision.
    return {dataset: resolve_dataset(dataset) for dataset in DATASETS}


def _run_case(server, case: NightlyCase, platform: str, sources: dict[str, Path], work: Path) -> None:
    hardware = helpers._accuracy_hardware_label()
    assert hardware, "Nightly baseline needs an identified hardware profile"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    root = Path(os.environ.get("ACC_BENCH_RESULT_DIR", "tests/e2e/accuracy/qwen3_omni/results/qwen_omni_acc"))
    root.mkdir(parents=True, exist_ok=True)
    prefix = "result_acc" if case.accuracy else "result_test"
    filename = f"{prefix}_Qwen3-Omni_{hardware}_{case.id}_{stamp}.json"
    argv = core.build_serve_common_argv(
        host=server.host,
        port=server.port,
        model=server.model,
        num_prompts=case.count,
        max_concurrency=case.concurrency,
        num_warmups=max(2, case.concurrency),
        result_dir=root,
        result_filename=filename,
        percentile_metrics="ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration",
        temperature=0.0,
    )
    argv += dataset_args(case, sources[case.dataset], work / "dataset")
    argv += [
        "--disable-shuffle",
        "--omni-isolate-prefix-cache",
        "--omni-fail-fast-on-timeout",
        "--extra-body",
        json.dumps({"modalities": ["text", "audio"]}),
    ]
    score_seed = case.accuracy and case.dataset == "seed-tts"
    env = {
        "SEED_TTS_WER_EVAL": str(int(score_seed)),
        "SEED_TTS_SIM_EVAL": str(int(score_seed)),
        "SEED_TTS_UTMOS_EVAL": str(int(score_seed and platform == "cuda")),
    }
    if score_seed:
        argv.append("--seed-tts-wer-eval")
    core.run_vllm_bench_subprocess(core.find_vllm_cli(), argv, extra_env=env)
    path = root / filename
    result = core.load_benchmark_result(path)
    limits = quality_baseline(case, platform)
    errors = coverage_errors(result, case.count, require_audio=True)
    errors += accuracy_coverage_errors(result, case, platform)
    if case.accuracy:
        if case.dataset == "daily-omni":
            errors += helpers._validate_daily_omni(result, min_accuracy=limits["min_daily_omni_accuracy"])
        else:
            errors += helpers._validate_seed_tts(
                result,
                max_mean_wer=limits["max_seed_tts_mean_wer"],
                min_mean_sim=limits["min_seed_tts_mean_sim"],
                min_mean_utmos=limits.get("min_seed_tts_mean_utmos"),
            )
    manifest = json.loads(MANIFEST.read_text())
    mode = os.environ.get("QWEN3_OMNI_BASELINE_MODE", manifest["mode"])
    errors += performance_gate(result, case, hardware, manifest, mode)
    perf = result.get("performance_baseline") or {}
    perf_limits = {
        key: spec["value"] * (1 + spec["relative_tolerance"])
        for key, spec in perf.get("metrics", {}).items()
        if isinstance(spec, dict)
        and all(
            isinstance(spec.get(k), (int, float)) and math.isfinite(spec[k]) for k in ("value", "relative_tolerance")
        )
    }
    result.update(
        {
            "nightly_suite": "qwen3-omni-non-duplex",
            "test_name": f"qwen3_omni_{hardware}_{case.config}",
            "name": case.id,
            "configuration": case.config,
            "accuracy_evaluated": case.accuracy,
            "async_chunk": case.config == "async-prefix",
            "thinker_prefix_caching": case.config == "async-prefix",
            "model_runner": "v2",
            "num_devices": 4 if platform == "npu" else 2,
            "dataset_name": case.dataset,
            "dataset": case.dataset.replace("-", "_"),
            "dataset_revision": DATASETS[case.dataset]["revision"],
            "dataset_manifest_sha256": DATASETS[case.dataset]["sha256"],
            "modalities": ["text", "audio"],
            "max_concurrency": case.concurrency,
            "num_prompts": case.count,
            "num_warmups": max(2, case.concurrency),
            "Hardware": hardware,
            "branch": os.environ.get("BUILDKITE_BRANCH", ""),
            "date": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "software_versions": {package: version(package) for package in ("vllm", "vllm-omni", "torch")},
            "baseline": {hardware: {**limits, **perf_limits}},
            "validation_errors": errors,
        }
    )
    if result.get("num_tpot_samples") == 0:
        for key in result:
            if key.endswith("_tpot_ms"):
                result[key] = None
    # Persist diagnostics and both metric families before failing the test.
    path.write_text(
        json.dumps(helpers._json_safe_metric_value(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    assert not errors, "\n".join(errors)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server,case", _params("cuda"), indirect=["omni_server"], scope="module")
def test_cuda_nightly(omni_server, case, nightly_sources, tmp_path):
    _run_case(omni_server, case, "cuda", nightly_sources, tmp_path)


@hardware_test(res={"npu": "A3"}, num_cards=4)
@pytest.mark.parametrize("omni_server,case", _params("npu"), indirect=["omni_server"], scope="module")
def test_npu_nightly(omni_server, case, nightly_sources, tmp_path):
    _run_case(omni_server, case, "npu", nightly_sources, tmp_path)
